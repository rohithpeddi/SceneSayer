import copy
import math
import os

import torch
import pdb

from lib.object_detector import detector
from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher
from lib.supervised.biased.sga.baseline_anticipation_gen_loss import BaselineWithAnticipationGenLoss
from test_base import (fetch_transformer_test_basic_config, get_sequence_no_tracking,
                       send_future_evaluators_stats_to_firebase, write_future_evaluators_stats,
                       write_percentage_evaluators_stats, send_percentage_evaluators_stats_to_firebase)


# Future Context - forward(entry,  context_fraction, entry)
def evaluate_model_context_fraction(model, entry, gt_annotation, conf, context_fraction, percentage_evaluators):
	get_sequence_no_tracking(entry, conf.mode)

	pred = model.forward_single_entry(context_fraction=context_fraction, entry=entry)
	
	count = 0
	total_frames = len(entry["im_idx"].unique())
	context = min(int(math.ceil(context_fraction * total_frames)), total_frames - 1)
	future = total_frames - context
	
	future_frame_start_id = entry["im_idx"].unique()[context]
	prev_con = entry["im_idx"].unique()[context - 1]
	future_frame_end_id = entry["im_idx"].unique()[context + future - 1]
	
	context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
	context_idx = entry["im_idx"][:context_end_idx]
	context_len = context_idx.shape[0]
	
	future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
	future_idx = entry["im_idx"][context_end_idx:future_end_idx]
	future_len = future_idx.shape[0]
	
	gt_future = gt_annotation[context: context + future]

	
	evaluators = percentage_evaluators[context_fraction]

	evaluators[0].evaluate_scene_graph(gt_future, pred)
	evaluators[1].evaluate_scene_graph(gt_future, pred)
	evaluators[2].evaluate_scene_graph(gt_future, pred)

# Future frames - Normal Test Evaluation - forward(entry, context, future)
def evaluate_model_future_frames(model, entry, gt_annotation, conf, num_future_frames, future_evaluators):
	get_sequence_no_tracking(entry, conf.mode)
	pred = model(entry, conf.baseline_context, num_future_frames)
	start = 0
	count = 0
	context = conf.baseline_context
	future = num_future_frames
	total_frames = len(entry["im_idx"].unique())
	
	context = min(context, total_frames - 1)
	future = min(future, total_frames - context)
	
	if (start + context + 1 > total_frames):
		while (start + context + 1 != total_frames and context > 1):
			context -= 1
		future = 1
	
	if (start + context + future > total_frames > start + context):
		future = total_frames - (start + context)
	while start + context + 1 <= total_frames:
		
		future_frame_start_id = entry["im_idx"].unique()[context]
		prev_con = entry["im_idx"].unique()[context - 1]
		
		if (start + context + future > total_frames > start + context):
			future = total_frames - (start + context)
		
		future_frame_end_id = entry["im_idx"].unique()[context + future - 1]
		
		context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
		context_idx = entry["im_idx"][:context_end_idx]
		context_len = context_idx.shape[0]
		
		future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
		future_idx = entry["im_idx"][context_end_idx:future_end_idx]
		future_len = future_idx.shape[0]
		
		gt_future = gt_annotation[start + context:start + context + future]
		
		vid_no = gt_annotation[0][0]["frame"].split('.')[0]

		pred_dict = pred["output"][count]
		
		evaluators = future_evaluators[num_future_frames]
		evaluators[0].evaluate_scene_graph(gt_future, pred_dict)
		evaluators[1].evaluate_scene_graph(gt_future, pred_dict)
		evaluators[2].evaluate_scene_graph(gt_future, pred_dict)
		count += 1
		context += 1


def load_model(conf, dataset, gpu_device, model_name):
	model = BaselineWithAnticipationGenLoss(mode=conf.mode,
	                                        attention_class_num=len(dataset.attention_relationships),
	                                        spatial_class_num=len(dataset.spatial_relationships),
	                                        contact_class_num=len(dataset.contacting_relationships),
	                                        obj_classes=dataset.object_classes,
	                                        enc_layer_num=conf.enc_layer,
	                                        dec_layer_num=conf.dec_layer).to(device=gpu_device)
	
	ckpt = torch.load(conf.ckpt, map_location=gpu_device)
	model.load_state_dict(ckpt[f'{model_name}_state_dict'], strict=False)
	print(f"Loaded model from checkpoint {conf.ckpt}")
	return model


def main():
	(ag_test_data, dataloader_test, gen_evaluators, future_evaluators,
	 future_evaluators_modified_gt, percentage_evaluators,
	 percentage_evaluators_modified_gt, gpu_device, conf) = fetch_transformer_test_basic_config()
	
	model_name = conf.method_name
	checkpoint_name = os.path.basename(conf.ckpt).split('.')[0]
	future_frame_loss_num = checkpoint_name.split('_')[-3]
	mode = conf.mode
	
	print("----------------------------------------------------------")
	print(f"Model name: {model_name}")
	print(f"Checkpoint name: {checkpoint_name}")
	print(f"Future frame loss num: {future_frame_loss_num}")
	print(f"Mode: {mode}")
	print("----------------------------------------------------------")
	
	model = load_model(conf, ag_test_data, gpu_device, model_name)
	model.eval()
	
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	
	object_detector = detector(
		train=False,
		object_classes=ag_test_data.object_classes,
		use_SUPPLY=True,
		mode=conf.mode
	).to(device=gpu_device)
	object_detector.eval()
	object_detector.is_train = False
	
	test_iter = iter(dataloader_test)
	model.eval()
	future_frames_list = [1, 2, 3, 4, 5]
	context_fractions = [0.3, 0.5, 0.7, 0.9]
	with torch.no_grad():
		for b in range(len(dataloader_test)):
			data = next(test_iter)
			im_data = copy.deepcopy(data[0].cuda(0))
			im_info = copy.deepcopy(data[1].cuda(0))
			gt_boxes = copy.deepcopy(data[2].cuda(0))
			num_boxes = copy.deepcopy(data[3].cuda(0))
			gt_annotation = ag_test_data.gt_annotations[data[4]]
			vid_no = gt_annotation[0][0]["frame"].split('.')[0]

			for num_future_frames in future_frames_list:
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
				evaluate_model_future_frames(model, entry, gt_annotation, conf, num_future_frames, future_evaluators)
			
			for context_fraction in context_fractions:
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
				evaluate_model_context_fraction(model, entry, gt_annotation, conf, context_fraction,
											percentage_evaluators)
		
			
			if b % 5 == 0:
				print(f"Finished processing {b} of {len(dataloader_test)} batches")
		
		# Write future and gen evaluators stats
		write_future_evaluators_stats(conf.mode, future_frame_loss_num, method_name=model_name,
		                              future_evaluators=future_evaluators)
		
		# Send future evaluation and generation evaluation stats to firebase
		send_future_evaluators_stats_to_firebase(future_evaluators, conf.mode, method_name=model_name,
		                                         future_frame_loss_num=future_frame_loss_num)
		
		# Write percentage evaluation stats and send to firebase
		for context_fraction in context_fractions:
			write_percentage_evaluators_stats(
				conf.mode,
				future_frame_loss_num,
				model_name,
				percentage_evaluators,
				context_fraction
			)
			send_percentage_evaluators_stats_to_firebase(
				percentage_evaluators,
				conf.mode,
				model_name,
				future_frame_loss_num,
				context_fraction
			)


if __name__ == '__main__':
	main()

""" python test_forecasting.py -mode sgdet -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ -model_path forecasting/sgdet_full_context_f3/DSG_masked_9.tar """
"""  python test_baseline_gen_loss.py -use_raw_data -mode sgdet -baseline_future 3  -ckpt /home/cse/msr/csy227518/scratch/SGG/SGG_task/baseline_gen_loss_corrected/sgdet_f3/baseline_so_gen_loss/baseline_so_gen_loss_sgdet_future_3_epoch_3.tar -nepoch 5 -method_name baseline_so_gen_loss -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """

