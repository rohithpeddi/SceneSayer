import numpy as np
import torch

from test_base import fetch_test_basic_config

np.set_printoptions(precision=4)
from time import time
from lib.supervised.biased.sga.SDE import SDE as SDE

from constants import Constants as const
from tqdm import tqdm
from lib.supervised.biased.dsgdetr.track import get_sequence
from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher


def test_sde():
	max_window = conf.max_window
	brownian_size = conf.brownian_size
	
	sde = SDE(mode=conf.mode,
	          attention_class_num=len(ag_features_test.attention_relationships),
	          spatial_class_num=len(ag_features_test.spatial_relationships),
	          contact_class_num=len(ag_features_test.contacting_relationships),
	          obj_classes=ag_features_test.object_classes,
	          enc_layer_num=conf.enc_layer,
	          dec_layer_num=conf.dec_layer,
	          max_window=max_window,
	          brownian_size=brownian_size).to(device=gpu_device)
	
	sde.eval()
	# cttran.eval()
	
	ckpt = torch.load(conf.model_path, map_location=gpu_device)
	sde.load_state_dict(ckpt['sde_state_dict'], strict=False)
	
	# ckpt = torch.load(conf.model_cttran_path, map_location=gpu_device)
	# cttran.load_state_dict(ckpt['cttran_state_dict'], strict=False)
	
	print('*' * 50)
	print('CKPT {} is loaded'.format(conf.model_path))
	
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	all_time = []
	
	with torch.no_grad():
		for entry in tqdm(dataloader_test, position=0, leave=True):
			start = time()
			gt_annotation = entry[const.GT_ANNOTATION]
			frame_size = entry[const.FRAME_SIZE]
			get_sequence(entry, gt_annotation, matcher, frame_size, conf.mode)
			pred = sde(entry, True)
			vid_no = gt_annotation[0][0]["frame"].split('.')[0]
			all_time.append(time() - start)
			w = max_window
			if max_window == -1:
				w = len(pred["gt_annotation"]) - 1
			global_output = pred["global_output"]
			times = pred["times"]
			global_output_mod = global_output.clone().to(global_output.device)
			denominator = torch.zeros(global_output.size(0)).to(global_output.device) + 1.0
			for i in range(1, max_window + 1):
				pred_anticipated = pred.copy()
				mask_curr = pred["mask_curr_" + str(i)]
				mask_gt = pred["mask_gt_" + str(i)]
				pred_anticipated["spatial_distribution"] = pred["anticipated_spatial_distribution"][i - 1][mask_curr]
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
				if conf.modified_gt:
					with_constraint_evaluator.evaluate_scene_graph(entry["gt_annotation_" + str(i)][i:],
					                                               pred_anticipated)
					no_constraint_evaluator.evaluate_scene_graph(entry["gt_annotation_" + str(i)][i:], pred_anticipated)
				else:
					with_constraint_evaluator.evaluate_scene_graph(entry["gt_annotation"][i:], pred_anticipated)
					no_constraint_evaluator.evaluate_scene_graph(entry["gt_annotation"][i:], pred_anticipated)
				global_output_mod[mask_gt] += pred["anticipated_vals"][i - 1][mask_curr] / torch.reshape((times[mask_gt] - times[mask_curr] + 1), (-1, 1))
				denominator[mask_gt] += 1 / (times[mask_gt] - times[mask_curr] + 1)
			global_output_mod = global_output_mod / torch.reshape(denominator, (-1, 1))
			pred["global_output"] = global_output_mod
			pred["attention_distribution"] = sde.dsgdetr.a_rel_compress(global_output)
			pred["spatial_distribution"] = sde.dsgdetr.s_rel_compress(global_output)
			pred["contacting_distribution"] = sde.dsgdetr.c_rel_compress(global_output)
			pred["spatial_distribution"] = torch.sigmoid(pred["spatial_distribution"])
			pred["contacting_distribution"] = torch.sigmoid(pred["contacting_distribution"])
			with_constraint_evaluator_gen.evaluate_scene_graph(gt_annotation, pred)
			no_constraint_evaluator_gen.evaluate_scene_graph(gt_annotation, pred)
	print('Average inference time', np.mean(all_time))
	
	print("anticipation evaluation:")
	print('-------------------------with constraint-------------------------------')
	with_constraint_evaluator.print_stats()
	print('-------------------------no constraint-------------------------------')
	no_constraint_evaluator.print_stats()
	print("generation evaluation")
	print('-------------------------with constraint-------------------------------')
	with_constraint_evaluator_gen.print_stats()
	print('-------------------------no constraint-------------------------------')
	no_constraint_evaluator_gen.print_stats()


if __name__ == '__main__':
	ag_features_test, dataloader_test, with_constraint_evaluator, no_constraint_evaluator, semi_constraint_evaluator, gpu_device, conf = fetch_test_basic_config()
	x, y, with_constraint_evaluator_gen, no_constraint_evaluator_gen, semi_constraint_evaluator_gen, z, w = fetch_test_basic_config() 
	test_sde()

#  python test_cttran.py -mode sgdet -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ -model_sttran_path cttran/no_temporal/sttran_9.tar -model_cttran_path cttran/no_temporal/cttran_9.tar
