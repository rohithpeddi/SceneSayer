import sys

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import os
import pandas as pd

from tqdm import tqdm
from constants import Constants as const

from lib.supervised.biased.dsgdetr.track import get_sequence_with_tracking
from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher

from train_base import fetch_train_basic_config, prepare_optimizer, fetch_loss_functions
from lib.supervised.biased.sga.ODE import ODE as ODE


def train_ode():
	max_window = conf.max_window
	ode_ratio = conf.ode_ratio
	bbox_ratio = conf.bbox_ratio
	
	ode = ODE(mode=conf.mode,
	          attention_class_num=len(ag_features_train.attention_relationships),
	          spatial_class_num=len(ag_features_train.spatial_relationships),
	          contact_class_num=len(ag_features_train.contacting_relationships),
	          obj_classes=ag_features_train.object_classes,
	          enc_layer_num=conf.enc_layer,
	          dec_layer_num=conf.dec_layer,
	          max_window=max_window).to(device=gpu_device)
	
	if conf.ckpt:
		ckpt = torch.load(conf.ckpt, map_location=gpu_device)
		ode.load_state_dict(ckpt['ode_state_dict'], strict=False)
		
	optimizer, _ = prepare_optimizer(conf, ode)
	scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.25, verbose=True, threshold=1e-4,  threshold_mode="abs", min_lr=1e-7)
	
	# some parameters
	tr = []
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	
	for epoch in range(3):
		ode.train()
		# diff_func.train()
		num = 0
		start = time.time()
		for entry in tqdm(dataloader_train, position=0, leave=True):
			gt_annotation = entry[const.GT_ANNOTATION]
			frame_size = entry[const.FRAME_SIZE]
			get_sequence_with_tracking(entry, gt_annotation, matcher, frame_size, conf.mode)
			pred = ode(entry)
			global_output = pred["global_output"]
			spatial_distribution = pred["spatial_distribution"]
			contact_distribution = pred["contacting_distribution"]
			attention_distribution = pred["attention_distribution"]
			subject_boxes_rcnn = pred["subject_boxes_rcnn"]
			# object_boxes_rcnn = pred["object_boxes_rcnn"]
			subject_boxes_dsg = pred["subject_boxes_dsg"]
			# object_boxes_dsg = pred["object_boxes_dsg"]
			
			anticipated_global_output = pred["anticipated_vals"]
			anticipated_subject_boxes = pred["anticipated_subject_boxes"]
			# targets = pred["detached_outputs"]
			anticipated_spatial_distribution = pred["anticipated_spatial_distribution"]
			anticipated_contact_distribution = pred["anticipated_contacting_distribution"]
			anticipated_attention_distribution = pred["anticipated_attention_distribution"]
			# anticipated_object_boxes = pred["anticipated_object_boxes"]
			
			attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to \
				(device=attention_distribution.device).squeeze()
			if not conf.bce_loss:
				# multi-label margin loss or adaptive loss
				spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to \
					(device=attention_distribution.device)
				contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to \
					(device=attention_distribution.device)
				for i in range(len(pred["spatial_gt"])):
					spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
					contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])
			
			else:
				# bce loss
				spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to \
					(device=attention_distribution.device)
				contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to \
					(device=attention_distribution.device)
				for i in range(len(pred["spatial_gt"])):
					spatial_label[i, pred["spatial_gt"][i]] = 1
					contact_label[i, pred["contacting_gt"][i]] = 1
			
			vid_no = gt_annotation[0][0]["frame"].split('.')[0]
			# pickle.dump(pred,open('/home/cse/msr/csy227518/Dsg_masked_output/sgdet/train'+'/'+vid_no+'.pkl','wb'))
			
			losses = {}
			if conf.mode == 'sgcls' or conf.mode == 'sgdet':
				losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])
			
			losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
			losses["subject_boxes_loss"] = bbox_ratio * bbox_loss(subject_boxes_dsg, subject_boxes_rcnn)
			# losses["object_boxes_loss"] = bbox_ratio * bbox_loss(object_boxes_dsg, object_boxes_rcnn)
			losses["anticipated_latent_loss"] = 0
			losses["anticipated_subject_boxes_loss"] = 0
			losses["anticipated_spatial_relation_loss"] = 0
			losses["anticipated_contact_relation_loss"] = 0
			losses["anticipated_attention_relation_loss"] = 0
			# losses["anticipated_object_boxes_loss"] = 0
			if not conf.bce_loss:
				losses["spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label)
				losses["contact_relation_loss"] = mlm_loss(contact_distribution, contact_label)
				for i in range(1, max_window + 1):
					mask_curr = entry["mask_curr_" + str(i)]
					mask_gt = entry["mask_gt_" + str(i)]
					losses["anticipated_latent_loss"] += ode_ratio * abs_loss(
						anticipated_global_output[i - 1][mask_curr],
						global_output[mask_gt])
					losses["anticipated_subject_boxes_loss"] += bbox_ratio * bbox_loss \
						(anticipated_subject_boxes[i - 1][mask_curr], subject_boxes_rcnn[mask_gt])
					losses["anticipated_spatial_relation_loss"] += mlm_loss \
						(anticipated_spatial_distribution[i - 1][mask_curr], spatial_label[mask_gt])
					losses["anticipated_contact_relation_loss"] += mlm_loss \
						(anticipated_contact_distribution[i - 1][mask_curr], contact_label[mask_gt])
					losses["anticipated_attention_relation_loss"] += ce_loss \
						(anticipated_attention_distribution[i - 1][mask_curr], attention_label[mask_gt])
				# losses["anticipated_object_boxes_loss"] += bbox_ratio * bbox_loss(anticipated_object_boxes[i - 1][mask_curr], object_boxes_rcnn[mask_gt])
			else:
				losses["spatial_relation_loss"] = bce_loss(spatial_distribution, spatial_label)
				losses["contact_relation_loss"] = bce_loss(contact_distribution, contact_label)
				for i in range(1, max_window + 1):
					mask_curr = entry["mask_curr_" + str(i)]
					mask_gt = entry["mask_gt_" + str(i)]
					losses["anticipated_latent_loss"] += ode_ratio * abs_loss(
						anticipated_global_output[i - 1][mask_curr],
						global_output[mask_gt])
					losses["anticipated_subject_boxes_loss"] += bbox_ratio * bbox_loss \
						(anticipated_subject_boxes[i - 1][mask_curr], subject_boxes_rcnn[mask_gt])
					losses["anticipated_spatial_relation_loss"] += bce_loss \
						(anticipated_spatial_distribution[i - 1][mask_curr], spatial_label[mask_gt])
					losses["anticipated_contact_relation_loss"] += bce_loss \
						(anticipated_contact_distribution[i - 1][mask_curr], contact_label[mask_gt])
					losses["anticipated_attention_relation_loss"] += ce_loss \
						(anticipated_attention_distribution[i - 1][mask_curr], attention_label[mask_gt])
				# losses["anticipated_object_boxes_loss"] += bbox_ratio * bbox_loss(anticipated_object_boxes[i - 1][mask_curr], object_boxes_rcnn[mask_gt])
			# optimizer_diff.zero_grad()
			optimizer.zero_grad()
			loss = sum(losses.values())
			loss.backward()
			# torch.nn.utils.clip_grad_norm_(diff_func.parameters(), max_norm=5, norm_type=2)
			torch.nn.utils.clip_grad_norm_(ode.parameters(), max_norm=5, norm_type=2)
			# optimizer_diff.step()
			optimizer.step()
			tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
			num += 1
			if num % 1000 == 0 and num >= 1000:
				time_per_batch = (time.time() - start) / 1000
				print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, num, len(dataloader_train),
				                                                                    time_per_batch,
				                                                                    len(dataloader_train) * time_per_batch / 60))
				
				mn = pd.concat(tr[-1000:], axis=1).mean(1)
				print(mn)
				start = time.time()
		
		# torch.save({"diff_func_state_dict": diff_func.state_dict()}, os.path.join(conf.save_path, "diff_func_{}.tar".format(epoch)))
		torch.save({"ode_state_dict": ode.state_dict()}, os.path.join(conf.save_path, "ode_{}.tar".format(epoch + 1)))
		print("*" * 40)
		print("save the checkpoint after {} epochs".format(epoch))
		with open(evaluator.save_file, "a") as f:
			f.write("save the checkpoint after {} epochs\n".format(epoch))
		# ckpt = torch.load(os.path.join(conf.save_path, "ode_{}.tar".format(epoch)), map_location=gpu_device)
		# ode.load_state_dict(ckpt['ode_state_dict'], strict=False)
		ode.eval()
		# diff_func.eval()
		with torch.no_grad():
			for entry in tqdm(dataloader_test, position=0, leave=True):
				gt_annotation = entry[const.GT_ANNOTATION]
				frame_size = entry[const.FRAME_SIZE]
				get_sequence_with_tracking(entry, gt_annotation, matcher, frame_size, conf.mode)
				pred = ode(entry, True)
				vid_no = gt_annotation[0][0]["frame"].split('.')[0]
				for i in range(1, max_window + 1):
					pred_anticipated = pred.copy()
					mask_curr = pred["mask_curr_" + str(i)]
					pred_anticipated["spatial_distribution"] = pred["anticipated_spatial_distribution"][i - 1][
						mask_curr]
					pred_anticipated["contacting_distribution"] = pred["anticipated_contacting_distribution"][i - 1][
						mask_curr]
					pred_anticipated["attention_distribution"] = pred["anticipated_attention_distribution"][i - 1][
						mask_curr]
					pred_anticipated["im_idx"] = pred["im_idx_test_" + str(i)]
					pred_anticipated["pair_idx"] = pred["pair_idx_test_" + str(i)]
					if conf.mode == "predcls":
						pred_anticipated["scores"] = pred["scores_test_" + str(i)]
						pred_anticipated["labels"] = pred["labels_test_" + str(i)]
					else:
						pred_anticipated["pred_scores"] = pred["pred_scores_test_" + str(i)]
						pred_anticipated["pred_labels"] = pred["pred_labels_test_" + str(i)]
					pred_anticipated["boxes"] = pred["boxes_test_" + str(i)]
					evaluator.evaluate_scene_graph(gt_annotation[i:], pred_anticipated)
			print('-----------')
			sys.stdout.flush()
		score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
		evaluator.print_stats()
		evaluator.reset_result()
		# scheduler_diff.step(score)
		scheduler.step(score)


if __name__ == '__main__':
	conf, dataloader_train, dataloader_test, gpu_device, evaluator, ag_features_train, ag_features_test = fetch_train_basic_config()
	bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss = fetch_loss_functions()
	train_ode()

# python train_try.py -mode sgcls -ckpt /home/cse/msr/csy227518/scratch/DSG/DSG-DETR/sgcls/model_9.tar -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/


""" python train_DSG_masked.py -mode sgdet -save_path sgdet_masked/  -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """
""" python train_cttran.py -mode sgdet -save_path cttran/1_temporal/ -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ -bce_loss """
