import torch
import torch.nn as nn
import numpy as np
import time
import os
import warnings
import pandas as pd
import copy

from dataloader.generation.action_genome.ag_dataset import AG, cuda_collate_fn
from constants import Constants as const
from lib.object_detector import detector
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

np.set_printoptions(precision=3)
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def prepare_optimizer(model):
	# optimizer
	if conf.optimizer == const.ADAMW:
		optimizer = AdamW(model.parameters(), lr=conf.lr)
	elif conf.optimizer == const.ADAM:
		optimizer = optim.Adam(model.parameters(), lr=conf.lr)
	elif conf.optimizer == const.SGD:
		optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)
	else:
		raise NotImplementedError
	
	scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4,
	                              threshold_mode="abs", min_lr=1e-7)
	return optimizer, scheduler


def train_dsg_detr():
	from lib.biased.dsgdetr.dsgdetr import DsgDETR
	from lib.biased.dsgdetr.track import get_sequence
	from lib.biased.dsgdetr.matcher import HungarianMatcher
	
	model = DsgDETR(mode=conf.mode,
	                attention_class_num=len(AG_dataset_train.attention_relationships),
	                spatial_class_num=len(AG_dataset_train.spatial_relationships),
	                contact_class_num=len(AG_dataset_train.contacting_relationships),
	                obj_classes=AG_dataset_train.object_classes,
	                enc_layer_num=conf.enc_layer,
	                dec_layer_num=conf.dec_layer).to(device=gpu_device)
	
	if conf.ckpt:
		ckpt = torch.load(conf.ckpt, map_location=gpu_device)
		model.load_state_dict(ckpt[const.STATE_DICT], strict=False)
	
	optimizer, scheduler = prepare_optimizer(model)
	
	tr = []
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	
	for epoch in range(conf.nepoch):
		object_detector.is_train = True
		model.train()
		object_detector.train_x = True
		start = time.time()
		train_iter = iter(dataloader_train)
		test_iter = iter(dataloader_test)
		for b in range(len(dataloader_train)):
			data = next(train_iter)
			
			im_data = copy.deepcopy(data[0].cuda(0))
			im_info = copy.deepcopy(data[1].cuda(0))
			gt_boxes = copy.deepcopy(data[2].cuda(0))
			num_boxes = copy.deepcopy(data[3].cuda(0))
			gt_annotation = AG_dataset_train.gt_annotations[data[4]]
			
			# prevent gradients to FasterRCNN
			with torch.no_grad():
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
			get_sequence(entry, gt_annotation, matcher, (im_info[0][:2] / im_info[0, 2]).cpu().data, conf.mode)
			pred = model(entry)
			
			attention_distribution = pred[const.ATTENTION_DISTRIBUTION]
			spatial_distribution = pred[const.SPATIAL_DISTRIBUTION]
			contact_distribution = pred[const.CONTACTING_DISTRIBUTION]
			
			attention_label = torch.tensor(pred[const.ATTENTION_GT], dtype=torch.long).to(
				device=attention_distribution.device).squeeze()
			if not conf.bce_loss:
				# multi-label margin loss or adaptive loss
				spatial_label = -torch.ones([len(pred[const.SPATIAL_GT]), 6], dtype=torch.long).to(
					device=attention_distribution.device)
				contact_label = -torch.ones([len(pred[const.CONTACTING_GT]), 17], dtype=torch.long).to(
					device=attention_distribution.device)
				for i in range(len(pred[const.SPATIAL_GT])):
					spatial_label[i, : len(pred[const.SPATIAL_GT][i])] = torch.tensor(pred[const.SPATIAL_GT][i])
					contact_label[i, : len(pred[const.CONTACTING_GT][i])] = torch.tensor(pred[const.CONTACTING_GT][i])
			else:
				# bce loss
				spatial_label = torch.zeros([len(pred[const.SPATIAL_GT]), 6], dtype=torch.float32).to(
					device=attention_distribution.device)
				contact_label = torch.zeros([len(pred[const.CONTACTING_GT]), 17], dtype=torch.float32).to(
					device=attention_distribution.device)
				for i in range(len(pred[const.SPATIAL_GT])):
					spatial_label[i, pred[const.SPATIAL_GT][i]] = 1
					contact_label[i, pred[const.CONTACTING_GT][i]] = 1
			
			losses = {}
			if conf.mode == const.SGCLS or conf.mode == const.SGDET:
				losses[const.OBJECT_LOSS] = ce_loss(pred[const.DISTRIBUTION], pred[const.LABELS])
			
			losses[const.ATTENTION_RELATION_LOSS] = ce_loss(attention_distribution, attention_label)
			if not conf.bce_loss:
				losses[const.SPATIAL_RELATION_LOSS] = mlm_loss(spatial_distribution, spatial_label)
				losses[const.CONTACTING_RELATION_LOSS] = mlm_loss(contact_distribution, contact_label)
			else:
				losses[const.SPATIAL_RELATION_LOSS] = bce_loss(spatial_distribution, spatial_label)
				losses[const.CONTACTING_RELATION_LOSS] = bce_loss(contact_distribution, contact_label)
			
			optimizer.zero_grad()
			loss = sum(losses.values())
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
			optimizer.step()
			
			tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
			
			if b % 1000 == 0 and b >= 1000:
				time_per_batch = (time.time() - start) / 1000
				print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
				                                                                    time_per_batch,
				                                                                    len(dataloader_train) * time_per_batch / 60))
				
				mn = pd.concat(tr[-1000:], axis=1).mean(1)
				print(mn)
				start = time.time()
		
		torch.save({const.STATE_DICT: model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
		print("*" * 40)
		print("save the checkpoint after {} epochs".format(epoch))
		with open(evaluator.save_file, "a") as f:
			f.write("save the checkpoint after {} epochs\n".format(epoch))
		
		model.eval()
		object_detector.is_train = False
		with torch.no_grad():
			for b in range(len(dataloader_test)):
				data = next(test_iter)
				im_data = copy.deepcopy(data[0].cuda(0))
				im_info = copy.deepcopy(data[1].cuda(0))
				gt_boxes = copy.deepcopy(data[2].cuda(0))
				num_boxes = copy.deepcopy(data[3].cuda(0))
				gt_annotation = AG_dataset_test.gt_annotations[data[4]]
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
				get_sequence(entry, gt_annotation, matcher, (im_info[0][:2] / im_info[0, 2]).cpu().data, conf.mode)
				pred = model(entry)
				evaluator.evaluate_scene_graph(gt_annotation, pred)
			print('-----------------------------------------------------------------------------------', flush=True)
		score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
		evaluator.print_stats()
		evaluator.reset_result()
		scheduler.step(score)


def train_sttran():
	from lib.biased.sttran.sttran import STTran
	
	model = STTran(mode=conf.mode,
	               attention_class_num=len(AG_dataset_train.attention_relationships),
	               spatial_class_num=len(AG_dataset_train.spatial_relationships),
	               contact_class_num=len(AG_dataset_train.contacting_relationships),
	               obj_classes=AG_dataset_train.object_classes,
	               enc_layer_num=conf.enc_layer,
	               dec_layer_num=conf.dec_layer).to(device=gpu_device)
	
	optimizer, scheduler = prepare_optimizer(model)
	
	tr = []
	for epoch in range(conf.nepoch):
		model.train()
		object_detector.is_train = True
		start = time.time()
		train_iter = iter(dataloader_train)
		test_iter = iter(dataloader_test)
		for b in range(len(dataloader_train)):
			data = next(train_iter)
			
			im_data = copy.deepcopy(data[0].cuda(0))
			im_info = copy.deepcopy(data[1].cuda(0))
			gt_boxes = copy.deepcopy(data[2].cuda(0))
			num_boxes = copy.deepcopy(data[3].cuda(0))
			gt_annotation = AG_dataset_train.gt_annotations[data[4]]
			
			# prevent gradients to FasterRCNN
			with torch.no_grad():
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
			
			pred = model(entry)
			
			attention_distribution = pred[const.ATTENTION_DISTRIBUTION]
			spatial_distribution = pred[const.SPATIAL_DISTRIBUTION]
			contact_distribution = pred[const.CONTACTING_DISTRIBUTION]
			
			attention_label = torch.tensor(pred[const.ATTENTION_GT], dtype=torch.long).to(
				device=attention_distribution.device).squeeze()
			if not conf.bce_loss:
				# multi-label margin loss or adaptive loss
				spatial_label = -torch.ones([len(pred[const.SPATIAL_GT]), 6], dtype=torch.long).to(
					device=attention_distribution.device)
				contact_label = -torch.ones([len(pred[const.CONTACTING_GT]), 17], dtype=torch.long).to(
					device=attention_distribution.device)
				for i in range(len(pred[const.SPATIAL_GT])):
					spatial_label[i, : len(pred[const.SPATIAL_GT][i])] = torch.tensor(pred[const.SPATIAL_GT][i])
					contact_label[i, : len(pred[const.CONTACTING_GT][i])] = torch.tensor(pred[const.CONTACTING_GT][i])
			
			else:
				# bce loss
				spatial_label = torch.zeros([len(pred[const.SPATIAL_GT]), 6], dtype=torch.float32).to(
					device=attention_distribution.device)
				contact_label = torch.zeros([len(pred[const.CONTACTING_GT]), 17], dtype=torch.float32).to(
					device=attention_distribution.device)
				for i in range(len(pred[const.SPATIAL_GT])):
					spatial_label[i, pred[const.SPATIAL_GT][i]] = 1
					contact_label[i, pred[const.CONTACTING_GT][i]] = 1
			
			losses = {}
			if conf.mode == const.SGCLS or conf.mode == const.SGDET:
				losses[const.OBJECT_LOSS] = ce_loss(pred[const.DISTRIBUTION], pred[const.LABELS])
			
			losses[const.ATTENTION_RELATION_LOSS] = ce_loss(attention_distribution, attention_label)
			if not conf.bce_loss:
				losses[const.SPATIAL_RELATION_LOSS] = mlm_loss(spatial_distribution, spatial_label)
				losses[const.CONTACTING_RELATION_LOSS] = mlm_loss(contact_distribution, contact_label)
			
			else:
				losses[const.SPATIAL_RELATION_LOSS] = bce_loss(spatial_distribution, spatial_label)
				losses[const.CONTACTING_RELATION_LOSS] = bce_loss(contact_distribution, contact_label)
			
			optimizer.zero_grad()
			loss = sum(losses.values())
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
			optimizer.step()
			
			tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
			
			if b % 1000 == 0 and b >= 1000:
				time_per_batch = (time.time() - start) / 1000
				print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
				                                                                    time_per_batch,
				                                                                    len(dataloader_train) * time_per_batch / 60))
				
				mn = pd.concat(tr[-1000:], axis=1).mean(1)
				print(mn)
				start = time.time()
		
		torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
		print("*" * 40)
		print("save the checkpoint after {} epochs".format(epoch))
		
		model.eval()
		object_detector.is_train = False
		with torch.no_grad():
			for b in range(len(dataloader_test)):
				data = next(test_iter)
				
				im_data = copy.deepcopy(data[0].cuda(0))
				im_info = copy.deepcopy(data[1].cuda(0))
				gt_boxes = copy.deepcopy(data[2].cuda(0))
				num_boxes = copy.deepcopy(data[3].cuda(0))
				gt_annotation = AG_dataset_test.gt_annotations[data[4]]
				
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
				pred = model(entry)
				evaluator.evaluate_scene_graph(gt_annotation, pred)
			print('-----------', flush=True)
		score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
		evaluator.print_stats()
		evaluator.reset_result()
		scheduler.step(score)


def train_tempura():
	pass


if __name__ == '__main__':
	conf = Config()
	print('The CKPT saved here:', conf.save_path)
	if not os.path.exists(conf.save_path):
		os.mkdir(conf.save_path)
	print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
	for i in conf.args:
		print(i, ':', conf.args[i])
	
	# 	TODO: Add code for saving method specific model
	
	AG_dataset_train = AG(
		mode=const.TRAIN,
		datasize=conf.datasize,
		data_path=conf.data_path,
		filter_nonperson_box_frame=True,
		filter_small_box=False if conf.mode == const.PREDCLS else True
	)
	
	dataloader_train = DataLoader(
		AG_dataset_train,
		shuffle=True,
		num_workers=4,
		collate_fn=cuda_collate_fn,
		pin_memory=False
	)
	
	AG_dataset_test = AG(
		mode=const.TEST,
		datasize=conf.datasize,
		data_path=conf.data_path,
		filter_nonperson_box_frame=True,
		filter_small_box=False if conf.mode == 'predcls' else True
	)
	
	dataloader_test = DataLoader(
		AG_dataset_test,
		shuffle=False,
		num_workers=4,
		collate_fn=cuda_collate_fn,
		pin_memory=False
	)
	
	gpu_device = torch.device("cuda:0")
	
	object_detector = detector(
		train=True,
		object_classes=AG_dataset_train.object_classes,
		use_SUPPLY=True,
		mode=conf.mode
	).to(device=gpu_device)
	object_detector.eval()
	
	evaluator = BasicSceneGraphEvaluator(
		mode=conf.mode,
		AG_object_classes=AG_dataset_train.object_classes,
		AG_all_predicates=AG_dataset_train.relationship_classes,
		AG_attention_predicates=AG_dataset_train.attention_relationships,
		AG_spatial_predicates=AG_dataset_train.spatial_relationships,
		AG_contacting_predicates=AG_dataset_train.contacting_relationships,
		iou_threshold=0.5,
		save_file=os.path.join(conf.save_path, const.PROGRESS_TEXT_FILE),
		constraint='with'
	)
	
	# loss function, default Multi-label margin loss
	if conf.bce_loss:
		ce_loss = nn.CrossEntropyLoss()
		bce_loss = nn.BCELoss()
	else:
		ce_loss = nn.CrossEntropyLoss()
		mlm_loss = nn.MultiLabelMarginLoss()
