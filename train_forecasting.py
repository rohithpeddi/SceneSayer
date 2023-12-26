import torch
import numpy as np

np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import copy
from lib.object_detector import detector
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator
from lib.supervised.biased.sga.forecasting import STTran
from lib.supervised.biased.dsgdetr.track import get_sequence
from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher
from train_base import fetch_train_basic_config, prepare_optimizer, fetch_loss_functions

CONTEXT = 4
FUTURE = 5


def train_forecasting():
	object_detector = detector(train=True, object_classes=ag_features_train.object_classes, use_SUPPLY=True,
	                           mode=conf.mode).to(device=gpu_device)
	object_detector.eval()
	
	model = STTran(mode=conf.mode,
	               attention_class_num=len(ag_features_train.attention_relationships),
	               spatial_class_num=len(ag_features_train.spatial_relationships),
	               contact_class_num=len(ag_features_train.contacting_relationships),
	               obj_classes=ag_features_train.object_classes,
	               enc_layer_num=conf.enc_layer,
	               dec_layer_num=conf.dec_layer).to(device=gpu_device)
	if conf.ckpt:
		ckpt = torch.load(conf.ckpt, map_location=gpu_device)
		model.load_state_dict(ckpt['state_dict'], strict=False)
	
	evaluator = BasicSceneGraphEvaluator(mode=conf.mode,
	                                     AG_object_classes=ag_features_train.object_classes,
	                                     AG_all_predicates=ag_features_train.relationship_classes,
	                                     AG_attention_predicates=ag_features_train.attention_relationships,
	                                     AG_spatial_predicates=ag_features_train.spatial_relationships,
	                                     AG_contacting_predicates=ag_features_train.contacting_relationships,
	                                     iou_threshold=0.5,
	                                     save_file=os.path.join(conf.save_path, "progress.txt"),
	                                     constraint='with')
	
	optimizer, scheduler = prepare_optimizer(conf, model)
	
	# some parameters
	tr = []
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	for epoch in range(10):
		object_detector.is_train = True
		model.train()
		object_detector.train_x = True
		start_time = time.time()
		train_iter = iter(dataloader_train)
		test_iter = iter(dataloader_test)
		counter = 0
		for b in range(len(dataloader_train)):
			data = next(train_iter)
			im_data = copy.deepcopy(data[0].cuda(0))
			im_info = copy.deepcopy(data[1].cuda(0))
			gt_boxes = copy.deepcopy(data[2].cuda(0))
			num_boxes = copy.deepcopy(data[3].cuda(0))
			gt_annotation = ag_features_train.gt_annotations[data[4]]
			
			# prevent gradients to FasterRCNN
			with torch.no_grad():
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
			get_sequence(entry, gt_annotation, matcher, (im_info[0][:2] / im_info[0, 2]).cpu().data, conf.mode)
			
			pred = model(entry, CONTEXT, FUTURE)
			
			start = 0
			context = CONTEXT
			future = FUTURE
			count = 0
			losses = {}
			
			if conf.mode == 'sgcls' or conf.mode == 'sgdet':
				losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])
			
			losses["attention_relation_loss"] = 0
			losses["spatial_relation_loss"] = 0
			losses["contact_relation_loss"] = 0
			
			if (start + context + 1 > len(entry["im_idx"].unique())):
				while (start + context + 1 != len(entry["im_idx"].unique()) and context > 1):
					context -= 1
				future = 1
			
			if (start + context + future > len(entry["im_idx"].unique()) and start + context < len(
					entry["im_idx"].unique())):
				future = len(entry["im_idx"].unique()) - (start + context)
			
			while (start + context + 1 <= len(entry["im_idx"].unique())):
				
				future_frame_start_id = entry["im_idx"].unique()[context]
				
				if (start + context + future > len(entry["im_idx"].unique()) and start + context < len(
						entry["im_idx"].unique())):
					future = len(entry["im_idx"].unique()) - (start + context)
				
				future_frame_end_id = entry["im_idx"].unique()[context + future - 1]
				
				context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
				context_idx = entry["im_idx"][:context_end_idx]
				context_len = context_idx.shape[0]
				
				future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
				future_idx = entry["im_idx"][context_end_idx:future_end_idx]
				future_len = future_idx.shape[0]
				
				attention_distribution = pred["output"][count]["attention_distribution"]
				spatial_distribution = pred["output"][count]["spatial_distribution"]
				contact_distribution = pred["output"][count]["contacting_distribution"]
				
				attention_label = torch.tensor(pred["attention_gt"][context_end_idx:future_end_idx],
				                               dtype=torch.long).to(
					device=attention_distribution.device).squeeze()
				if not conf.bce_loss:
					# multi-label margin loss or adaptive loss
					spatial_label = -torch.ones([len(pred["spatial_gt"][context_end_idx:future_end_idx]), 6],
					                            dtype=torch.long).to(device=attention_distribution.device)
					contact_label = -torch.ones([len(pred["contacting_gt"][context_end_idx:future_end_idx]), 17],
					                            dtype=torch.long).to(device=attention_distribution.device)
					for i in range(len(pred["spatial_gt"][context_end_idx:future_end_idx])):
						spatial_label[i, : len(pred["spatial_gt"][context_end_idx:future_end_idx][i])] = torch.tensor(
							pred["spatial_gt"][context_end_idx:future_end_idx][i])
						contact_label[i,
						: len(pred["contacting_gt"][context_end_idx:future_end_idx][i])] = torch.tensor(
							pred["contacting_gt"][context_end_idx:future_end_idx][i])
				
				else:
					spatial_label = torch.zeros([len(pred["spatial_gt"][context_end_idx:future_end_idx]), 6],
					                            dtype=torch.float32).to(device=attention_distribution.device)
					contact_label = torch.zeros([len(pred["contacting_gt"][context_end_idx:future_end_idx]), 17],
					                            dtype=torch.float32).to(device=attention_distribution.device)
					for i in range(len(pred["spatial_gt"][context_end_idx:future_end_idx])):
						spatial_label[i, pred["spatial_gt"][context_end_idx:future_end_idx][i]] = 1
						contact_label[i, pred["contacting_gt"][context_end_idx:future_end_idx][i]] = 1
				
				vid_no = gt_annotation[0][0]["frame"].split('.')[0]
				try:
					losses["attention_relation_loss"] += ce_loss(attention_distribution, attention_label)
				except ValueError:
					attention_label = attention_label.unsqueeze(0)
					losses["attention_relation_loss"] += ce_loss(attention_distribution, attention_label)
				
				if not conf.bce_loss:
					losses["spatial_relation_loss"] += mlm_loss(spatial_distribution, spatial_label)
					losses["contact_relation_loss"] += mlm_loss(contact_distribution, contact_label)
				
				else:
					losses["spatial_relation_loss"] += bce_loss(spatial_distribution, spatial_label)
					losses["contact_relation_loss"] += bce_loss(contact_distribution, contact_label)
				
				context += 1
				count += 1
				
				if start + context + future > len(entry["im_idx"].unique()):
					break
			
			losses["attention_relation_loss"] = losses["attention_relation_loss"] / count
			losses["spatial_relation_loss"] = losses["spatial_relation_loss"] / count
			losses["contact_relation_loss"] = losses["contact_relation_loss"] / count
			optimizer.zero_grad()
			loss = sum(losses.values())
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
			optimizer.step()
			
			tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
			
			if b % 1000 == 0 and b >= 1000:
				time_per_batch = (time.time() - start_time) / 1000
				print("\ne{:2d}encoder_tran  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b,
				                                                                                len(dataloader_train),
				                                                                                time_per_batch,
				                                                                                len(dataloader_train) * time_per_batch / 60))
				
				mn = pd.concat(tr[-1000:], axis=1).mean(1)
				print(mn)
				start_time = time.time()
		
		torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "forecast_{}.tar".format(epoch)))
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
				gt_annotation = ag_features_test.gt_annotations[data[4]]
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
				get_sequence(entry, gt_annotation, matcher, (im_info[0][:2] / im_info[0, 2]).cpu().data, conf.mode)
				
				start = 0
				context = CONTEXT
				future = FUTURE
				count = 0
				pred = model(entry, context, future)
				
				if (start + context + 1 > len(entry["im_idx"].unique())):
					while (start + context + 1 != len(entry["im_idx"].unique()) and context > 1):
						context -= 1
					future = 1
				
				if (start + context + future > len(entry["im_idx"].unique()) and start + context < len(
						entry["im_idx"].unique())):
					future = len(entry["im_idx"].unique()) - (start + context)
				while (start + context + 1 <= len(entry["im_idx"].unique())):
					
					future_frame_start_id = entry["im_idx"].unique()[context]
					
					if (start + context + future > len(entry["im_idx"].unique()) and start + context < len(
							entry["im_idx"].unique())):
						future = len(entry["im_idx"].unique()) - (start + context)
					
					future_frame_end_id = entry["im_idx"].unique()[context + future - 1]
					
					context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
					context_idx = entry["im_idx"][:context_end_idx]
					context_len = context_idx.shape[0]
					
					future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
					future_idx = entry["im_idx"][context_end_idx:future_end_idx]
					future_len = future_idx.shape[0]
					
					gt_future = gt_annotation[start + context:start + context + future]
					
					vid_no = gt_annotation[0][0]["frame"].split('.')[0]
					# pickle.dump(pred,open('/home/cse/msr/csy227518/Dsg_masked_output/sgdet/test'+'/'+vid_no+'.pkl','wb'))
					evaluator.evaluate_scene_graph_forecasting(gt_future, pred, context_end_idx, future_end_idx, future_idx, count)
					# evaluator.print_stats()
					count += 1
					context += 1
					
					if start + context + future > len(entry["im_idx"].unique()):
						break
			
			print('-----------', flush=True)
		score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
		evaluator.print_stats()
		evaluator.reset_result()
		scheduler.step(score)


if __name__ == '__main__':
	conf, dataloader_train, dataloader_test, gpu_device, evaluator, ag_features_train, ag_features_test = fetch_train_basic_config()
	bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss = fetch_loss_functions()
	train_forecasting()

# python train_try.py -mode sgcls -ckpt /home/cse/msr/csy227518/scratch/DSG/DSG-DETR/sgcls/model_9.tar -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/


""" python train_obj_mask.py -mode sgdet -save_path forecasting/sgcls_full_context_f5/ -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """
