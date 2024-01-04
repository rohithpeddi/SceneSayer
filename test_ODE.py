import os
import csv

import numpy as np
import torch
from time import time
from lib.supervised.biased.sga.ODE import ODE as ODE

from constants import Constants as const
from tqdm import tqdm
from lib.supervised.biased.dsgdetr.track import get_sequence
from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher
from test_base import fetch_diffeq_test_basic_config, write_future_evaluators_stats, write_percentage_evaluators_stats, \
	write_gen_evaluators_stats, evaluate_anticipated_future_frame_scene_graph


def test_ode():
	max_window = conf.max_window
	
	ode = ODE(mode=conf.mode,
	          attention_class_num=len(ag_features_test.attention_relationships),
	          spatial_class_num=len(ag_features_test.spatial_relationships),
	          contact_class_num=len(ag_features_test.contacting_relationships),
	          obj_classes=ag_features_test.object_classes,
	          enc_layer_num=conf.enc_layer,
	          dec_layer_num=conf.dec_layer,
	          max_window=max_window).to(device=gpu_device)
	
	ode.eval()
	
	ckpt = torch.load(conf.model_path, map_location=gpu_device)
	ode.load_state_dict(ckpt['ode_state_dict'], strict=False)
	
	print('*' * 50)
	print('CKPT {} is loaded'.format(conf.model_path))
	
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	all_time = []
	
	# with torch.no_grad():
	# 	for entry in tqdm(dataloader_test, position=0, leave=True):
	# 		start = time()
	# 		gt_annotation = entry[const.GT_ANNOTATION]
	# 		frame_size = entry[const.FRAME_SIZE]
	# 		get_sequence(entry, gt_annotation, matcher, frame_size, conf.mode)
	# 		pred = ode(entry, True)
	# 		global_output = pred["global_output"]
	# 		times = pred["times"]
	# 		global_output_mod = global_output.clone().to(global_output.device)
	# 		denominator = torch.zeros(global_output.size(0)).to(global_output.device) + 1.0
	# 		all_time.append(time() - start)
	# 		w = max_window
	# 		n = int(torch.max(pred["im_idx"]) + 1)
	# 		if max_window == -1:
	# 			w = n - 1
	# 		w = min(w, n - 1)
	# 		for i in range(1, w + 1):
	# 			pred_anticipated = pred.copy()
	# 			mask_curr = pred["mask_curr_" + str(i)]
	# 			mask_gt = pred["mask_gt_" + str(i)]
	# 			last = pred["last_" + str(i)]
	# 			pred_anticipated["spatial_distribution"] = pred["anticipated_spatial_distribution"][i - 1, : last]
	# 			pred_anticipated["contacting_distribution"] = pred["anticipated_contacting_distribution"][i - 1, : last]
	# 			pred_anticipated["attention_distribution"] = pred["anticipated_attention_distribution"][i - 1, : last]
	# 			pred_anticipated["im_idx"] = pred["im_idx_test_" + str(i)]
	# 			pred_anticipated["pair_idx"] = pred["pair_idx_test_" + str(i)]
	# 			if conf.mode == "predcls":
	# 				pred_anticipated["scores"] = pred["scores_test_" + str(i)]
	# 				pred_anticipated["labels"] = pred["labels_test_" + str(i)]
	# 			else:
	# 				pred_anticipated["pred_scores"] = pred["pred_scores_test_" + str(i)]
	# 				pred_anticipated["pred_labels"] = pred["pred_labels_test_" + str(i)]
	# 			pred_anticipated["boxes"] = pred["boxes_test_" + str(i)]
	# 			# evaluate_anticipated_future_frame_scene_graph(
	# 			# 	entry["gt_annotation_" + str(i)][i:],
	# 			# 	pred_anticipated,
	# 			# 	future_frame_count=i,
	# 			# 	is_modified_gt=True,
	# 			# 	future_evaluators=future_evaluators,
	# 			# 	future_evaluators_modified_gt=future_evaluators_modified_gt
	# 			# )
	# 			evaluate_anticipated_future_frame_scene_graph(
	# 				entry["gt_annotation"][i:],
	# 				pred_anticipated,
	# 				future_frame_count=i,
	# 				is_modified_gt=False,
	# 				future_evaluators=future_evaluators,
	# 				future_evaluators_modified_gt=future_evaluators_modified_gt
	# 			)
	# 			global_output_mod[mask_gt] += pred["anticipated_vals"][i - 1][mask_curr] / torch.reshape(
	# 				(times[mask_gt] - times[mask_curr] + 1), (-1, 1))
	# 			denominator[mask_gt] += 1 / (times[mask_gt] - times[mask_curr] + 1)
	# 		global_output_mod = global_output_mod / torch.reshape(denominator, (-1, 1))
	# 		pred["global_output"] = global_output_mod
	# 		pred["attention_distribution"] = ode.dsgdetr.a_rel_compress(global_output)
	# 		pred["spatial_distribution"] = ode.dsgdetr.s_rel_compress(global_output)
	# 		pred["contacting_distribution"] = ode.dsgdetr.c_rel_compress(global_output)
	# 		pred["spatial_distribution"] = torch.sigmoid(pred["spatial_distribution"])
	# 		pred["contacting_distribution"] = torch.sigmoid(pred["contacting_distribution"])
	# 		gen_evaluators[0].evaluate_scene_graph(gt_annotation, pred)
	# 		gen_evaluators[1].evaluate_scene_graph(gt_annotation, pred)
	# 		gen_evaluators[2].evaluate_scene_graph(gt_annotation, pred)
	# print('Average inference time', np.mean(all_time))
	# # Write future and gen evaluators stats
	# write_future_evaluators_stats(mode, future_frame_loss_num, method_name, future_evaluators)
	# write_gen_evaluators_stats(mode, future_frame_loss_num, method_name, gen_evaluators)
	
	# Context Fraction Evaluation
	with torch.no_grad():
		for context_fraction in [0.3, 0.5, 0.7, 0.9]:
			for entry in tqdm(dataloader_test, position=0, leave=True):
				gt_annotation = entry[const.GT_ANNOTATION]
				frame_size = entry[const.FRAME_SIZE]
				get_sequence(entry, gt_annotation, matcher, frame_size, conf.mode)
				ind, pred = ode.forward_single_entry(context_fraction=context_fraction, entry=entry)
				if ind >= len(gt_annotation):
					continue
				percentage_evaluators[context_fraction][0].evaluate_scene_graph(gt_annotation[ind:], pred)
				percentage_evaluators[context_fraction][1].evaluate_scene_graph(gt_annotation[ind:], pred)
				percentage_evaluators[context_fraction][2].evaluate_scene_graph(gt_annotation[ind:], pred)
			# Write percentage evaluation stats
			write_percentage_evaluators_stats(mode, future_frame_loss_num, method_name, percentage_evaluators, context_fraction)


if __name__ == '__main__':
	ag_features_test, dataloader_test, gen_evaluators, future_evaluators, future_evaluators_modified_gt, percentage_evaluators, percentage_evaluators_modified_gt, gpu_device, conf = fetch_diffeq_test_basic_config()
	model_name = os.path.basename(conf.model_path).split('.')[0]
	future_frame_loss_num = model_name.split('_')[-3]
	mode = model_name.split('_')[-5]
	method_name = "NeuralODE"
	test_ode()

#  python test_cttran.py -mode sgdet -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ -model_sttran_path cttran/no_temporal/sttran_9.tar -model_cttran_path cttran/no_temporal/cttran_9.tar
