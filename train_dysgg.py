import copy
import os
import time

import numpy as np
import pandas as pd
import torch

from constants import Constants as const
from lib.object_detector import Detector
from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
from lib.supervised.dysgg import DyDsgDETR
from lib.supervised.dysgg import DySTTran
from train_base import fetch_train_basic_config, fetch_loss_functions, save_model, get_sequence_no_tracking, \
	prepare_optimizer


def load_DySTTran(conf, ag_train_data, gpu_device):
	model = DySTTran(mode=conf.mode,
	                 attention_class_num=len(ag_train_data.attention_relationships),
	                 spatial_class_num=len(ag_train_data.spatial_relationships),
	                 contact_class_num=len(ag_train_data.contacting_relationships),
	                 obj_classes=ag_train_data.object_classes,
	                 enc_layer_num=conf.enc_layer,
	                 dec_layer_num=conf.dec_layer).to(device=gpu_device)
	
	if conf.ckpt:
		ckpt = torch.load(conf.ckpt, map_location=gpu_device)
		model.load_state_dict(ckpt["dysttran_state_dict"], strict=False)
		print(f"Loaded checkpoint {conf.ckpt}")
	
	optimizer, scheduler = prepare_optimizer(conf, model)
	return model, optimizer, scheduler


def load_DyDsgDETR(conf, ag_train_data, gpu_device):
	model = DyDsgDETR(mode=conf.mode,
	                  attention_class_num=len(ag_train_data.attention_relationships),
	                  spatial_class_num=len(ag_train_data.spatial_relationships),
	                  contact_class_num=len(ag_train_data.contacting_relationships),
	                  obj_classes=ag_train_data.object_classes,
	                  enc_layer_num=conf.enc_layer,
	                  dec_layer_num=conf.dec_layer).to(device=gpu_device)
	
	if conf.ckpt:
		ckpt = torch.load(conf.ckpt, map_location=gpu_device)
		model.load_state_dict(ckpt["dydsgdetr_state_dict"], strict=False)
		print(f"Loaded checkpoint {conf.ckpt}")
	
	optimizer, scheduler = prepare_optimizer(conf, model)
	return model, optimizer, scheduler


def train_model(conf, model, object_detector, matcher, optimizer, ag_train_data, dataloader_train, tr, epoch, is_tracking_enabled=False):
	bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss = fetch_loss_functions()
	train_iter = iter(dataloader_train)
	object_detector.is_train = True
	model.train()
	object_detector.train_x = True
	start = time.time()
	for b in range(len(dataloader_train)):
		data = next(train_iter)
		im_data = copy.deepcopy(data[0].cuda(0))
		im_info = copy.deepcopy(data[1].cuda(0))
		gt_boxes = copy.deepcopy(data[2].cuda(0))
		num_boxes = copy.deepcopy(data[3].cuda(0))
		gt_annotation = ag_train_data.gt_annotations[data[4]]
		with torch.no_grad():
			entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
		
		if is_tracking_enabled:
			get_sequence_with_tracking(
				entry,
				gt_annotation,
				matcher,
				(im_info[0][:2] / im_info[0, 2]).cpu().data,
				conf.mode
			)
		else:
			get_sequence_no_tracking(entry, task=conf.mode)
		
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
		if b % 50 == 0:
			print("epoch {:2d}  batch {:5d}/{:5d}  loss {:.4f}".format(epoch, b, len(dataloader_train),
			                                                           loss.item()))
		
		if b % 1000 == 0 and b >= 1000:
			time_per_batch = (time.time() - start) / 1000
			print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
			                                                                    time_per_batch,
			                                                                    len(dataloader_train) * time_per_batch / 60))
			
			mn = pd.concat(tr[-1000:], axis=1).mean(1)
			print(mn)
			start = time.time()


def test_model(model, object_detector, dataloader_test, ag_test_data, evaluator, conf, matcher, is_tracking_enabled=False):
	model.eval()
	object_detector.is_train = False
	test_iter = iter(dataloader_test)
	with torch.no_grad():
		for b in range(len(dataloader_test)):
			data = next(test_iter)
			im_data = copy.deepcopy(data[0].cuda(0))
			im_info = copy.deepcopy(data[1].cuda(0))
			gt_boxes = copy.deepcopy(data[2].cuda(0))
			num_boxes = copy.deepcopy(data[3].cuda(0))
			gt_annotation = ag_test_data.gt_annotations[data[4]]
			entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
			
			if is_tracking_enabled:
				get_sequence_with_tracking(entry, gt_annotation, matcher, (im_info[0][:2] / im_info[0, 2]).cpu().data, conf.mode)
			else:
				get_sequence_no_tracking(entry, task=conf.mode)
			
			pred = model(entry)
			
			evaluator.evaluate_scene_graph(gt_annotation, pred)
			
			if b % 50 == 0:
				print(f"Finished processing {b} of {len(dataloader_test)} batches")


def load_object_detector(conf, gpu_device, ag_train_data):
	object_detector = Detector(
		train=True,
		object_classes=ag_train_data.object_classes,
		use_SUPPLY=True,
		mode=conf.mode
	).to(device=gpu_device)
	object_detector.eval()
	print("Finished loading object detector", flush=True)
	return object_detector


def main():
	conf, dataloader_train, dataloader_test, gpu_device, evaluator, ag_train_data, ag_test_data = fetch_train_basic_config()
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	method_name = conf.method_name
	checkpoint_name = f"{method_name}_{conf.mode}"
	checkpoint_save_file_path = os.path.join(conf.save_path, method_name)
	os.makedirs(checkpoint_save_file_path, exist_ok=True)
	evaluator_save_file_path = os.path.join(os.path.abspath('.'), conf.results_path, method_name,
	                                        f"train_{method_name}_{conf.mode}.txt")
	os.makedirs(os.path.dirname(evaluator_save_file_path), exist_ok=True)
	evaluator.save_file = evaluator_save_file_path
	
	model, optimizer, scheduler = None, None, None
	model_name = None
	
	is_tracking_enabled = False
	if method_name == "dysttran":
		model, optimizer, scheduler = load_DySTTran(conf, ag_train_data, gpu_device)
		model_name = "dysttran"
	elif method_name == "dydsgdetr":
		model, optimizer, scheduler = load_DyDsgDETR(conf, ag_train_data, gpu_device)
		model_name = "dydsgdetr"
		is_tracking_enabled = True
	
	assert model is not None and optimizer is not None and scheduler is not None
	assert model_name is not None

	print(f"Training model with name {model_name}")
	object_detector = load_object_detector(conf, gpu_device, ag_train_data)
	tr = []
	for epoch in range(conf.nepoch):
		train_model(conf, model, object_detector, matcher, optimizer, ag_train_data, dataloader_train, tr, epoch, is_tracking_enabled=is_tracking_enabled)
		save_model(model, epoch, checkpoint_save_file_path, checkpoint_name, model_name)
		test_model(model, object_detector, dataloader_test, ag_test_data, evaluator, conf, matcher, is_tracking_enabled=is_tracking_enabled)
		score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
		evaluator.print_stats()
		evaluator.reset_result()
		scheduler.step(score)


if __name__ == '__main__':
	# Should set method_name appropriately
	main()


# python train_dysgg.py -method_name dysttran -use_raw_data -mode sgdet -save_path /home/rxp190007/DATA/ag/checkpoints -datasize large -data_path /home/rxp190007/DATA/ag
# python train_dysgg.py -method_name dydsgdetr -use_raw_data -mode sgdet -save_path /home/rxp190007/DATA/ag/checkpoints -datasize large -data_path /home/rxp190007/DATA/ag

# python train_try.py -mode sgcls -ckpt /home/cse/msr/csy227518/scratch/DSG/DSG-DETR/sgcls/model_9.tar -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/
""" python train_obj_mask.py -mode sgdet -save_path forecasting/sgcls_full_context_f5/ -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """
