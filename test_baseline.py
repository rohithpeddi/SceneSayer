import copy
import math
import os

import torch

from lib.object_detector import detector
from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher
from lib.supervised.biased.sga.baseline_anticipation import BaselineWithAnticipation
from test_base import (fetch_transformer_test_basic_config, get_sequence_no_tracking, prepare_prediction_graph,
                       send_future_evaluators_stats_to_firebase, write_future_evaluators_stats,
                       write_percentage_evaluators_stats, send_percentage_evaluators_stats_to_firebase)


# Future Context - forward(entry,  context_fraction, entry)
def evaluate_model_context_fraction(model, entry, gt_annotation, conf, context_fraction, percentage_evaluators):
	gt_future, pred_dict = fetch_model_context_pred_dict(model, entry, gt_annotation, conf, context_fraction)
	evaluators = percentage_evaluators[context_fraction]
	evaluators[0].evaluate_scene_graph(gt_future, pred_dict)
	evaluators[1].evaluate_scene_graph(gt_future, pred_dict)
	evaluators[2].evaluate_scene_graph(gt_future, pred_dict)


def fetch_model_context_pred_dict(model, entry, gt_annotation, conf, context_fraction):
	get_sequence_no_tracking(entry, conf.mode)
	pred = model.forward_single_entry(context_fraction=context_fraction, entry=entry)
	
	count = 0
	num_tf = len(entry["im_idx"].unique())
	num_cf = min(int(math.ceil(context_fraction * num_tf)), num_tf - 1)
	num_ff = num_tf - num_cf
	
	ff_start_id = entry["im_idx"].unique()[num_cf]
	cf_end_id = entry["im_idx"].unique()[num_cf - 1]
	ff_end_id = entry["im_idx"].unique()[num_cf + num_ff - 1]
	
	objects_ff_start_id = int(torch.where(entry["im_idx"] == ff_start_id)[0][0])
	objects_ff_end_id = int(torch.where(entry["im_idx"] == ff_end_id)[0][-1]) + 1
	objects_ff = entry["im_idx"][objects_ff_start_id:objects_ff_end_id]
	
	gt_future = gt_annotation[num_cf: num_cf + num_ff]
	
	boxes_ff_start_id = torch.where(entry["boxes"][:, 0] == ff_start_id)[0][0]
	boxes_cf_end_id = torch.where(entry["boxes"][:, 0] == cf_end_id)[0][0]
	boxes_ff_end_id = torch.where(entry["boxes"][:, 0] == ff_end_id)[0][-1]
	if conf.mode == 'predcls':
		labels_pred_cf_end = set(pred["labels"][boxes_cf_end_id:boxes_ff_start_id].tolist())
		labels_pred_ff = set(pred["labels"][boxes_ff_start_id:boxes_ff_end_id + 1].tolist())
		labels_pred_cf = set(pred["labels"][:boxes_ff_start_id].tolist())
	else:
		labels_pred_cf_end = set(pred["pred_labels"][boxes_cf_end_id:boxes_ff_start_id].tolist())
		labels_pred_ff = set(pred["pred_labels"][boxes_ff_start_id:boxes_ff_end_id + 1].tolist())
		labels_pred_cf = set(pred["pred_labels"][boxes_cf_end_id:boxes_ff_start_id].tolist())
	
	ff_boxes_mask = torch.ones(pred["boxes"][boxes_ff_start_id:boxes_ff_end_id + 1].shape[0])
	ff_objects_mask = torch.ones(objects_ff.shape[0])
	
	# ---------------------------------------------------------------------------
	# Remove objects and pairs that are not in the future frame
	
	im_idx = pred["im_idx"][objects_ff_start_id:objects_ff_end_id]
	im_idx = im_idx - im_idx.min()
	
	pair_idx = pred["pair_idx"][objects_ff_start_id:objects_ff_end_id]
	reshape_pair = pair_idx.view(-1, 2)
	min_value = reshape_pair.min()
	new_pair = reshape_pair - min_value
	pair_idx = new_pair.view(pair_idx.size())
	
	boxes = pred["boxes"][boxes_ff_start_id:boxes_ff_end_id + 1]
	labels = pred["labels"][boxes_ff_start_id:boxes_ff_end_id + 1]
	pred_labels = pred["pred_labels"][boxes_ff_start_id:boxes_ff_end_id + 1]
	scores = pred["scores"][boxes_ff_start_id:boxes_ff_end_id + 1]
	if conf.mode != 'predcls':
		pred_scores = pred["pred_scores"][boxes_ff_start_id:boxes_ff_end_id + 1]
	
	ob1 = labels_pred_ff - labels_pred_cf_end
	ob2 = labels_pred_cf - labels_pred_cf_end
	objects = ob1.union(ob2)
	objects = list(objects)
	for obj in objects:
		for idx, pair in enumerate(pred["pair_idx"][objects_ff_start_id:objects_ff_end_id]):
			if conf.mode == 'predcls':
				if pred["labels"][pair[1]] == obj:
					ff_objects_mask[idx] = 0
					ff_boxes_mask[pair_idx[idx, 1]] = 0
			else:
				if pred["pred_labels"][pair[1]] == obj:
					ff_objects_mask[idx] = 0
					ff_boxes_mask[pair_idx[idx, 1]] = 0
	
	im_idx = im_idx[ff_objects_mask == 1]
	removed = pair_idx[ff_objects_mask == 0]
	mask_pair_idx = pair_idx[ff_objects_mask == 1]
	
	flat_pair = mask_pair_idx.view(-1)
	flat_pair_copy = mask_pair_idx.view(-1).detach().clone()
	for pair in removed:
		idx = pair[1]
		for i, p in enumerate(flat_pair_copy):
			if p > idx:
				flat_pair[i] -= 1
	new_pair_idx = flat_pair.view(mask_pair_idx.size())
	
	if conf.mode == 'predcls':
		scores = scores[ff_boxes_mask == 1]
		labels = labels[ff_boxes_mask == 1]
	pred_labels = pred_labels[ff_boxes_mask == 1]
	
	boxes = boxes[ff_boxes_mask == 1]
	if conf.mode != 'predcls':
		pred_scores = pred_scores[ff_boxes_mask == 1]
	
	atten = pred["output"][count]['attention_distribution'][ff_objects_mask == 1]
	spatial = pred["output"][count]['spatial_distribution'][ff_objects_mask == 1]
	contact = pred["output"][count]['contacting_distribution'][ff_objects_mask == 1]
	
	if conf.mode == 'predcls':
		pred_dict = {'attention_distribution': atten,
		             'spatial_distribution': spatial,
		             'contacting_distribution': contact,
		             'boxes': boxes,
		             'pair_idx': new_pair_idx,
		             'im_idx': im_idx,
		             'labels': labels,
		             'pred_labels': pred_labels,
		             'scores': scores
		             }
	else:
		pred_dict = {'attention_distribution': atten,
		             'spatial_distribution': spatial,
		             'contacting_distribution': contact,
		             'boxes': boxes,
		             'pair_idx': new_pair_idx,
		             'im_idx': im_idx,
		             # 'labels':labels,
		             'pred_labels': pred_labels,
		             # 'scores':scores,
		             'pred_scores': pred_scores
		             }
	
	return gt_future, pred_dict


def generate_context_qualitative_results(model, entry, gt_annotation, conf, context_fraction, percentage_evaluators,
                                         video_id, ag_test_data):
	gt_future, pred_dict = fetch_model_context_pred_dict(model, entry, gt_annotation, conf, context_fraction)
	
	evaluators = percentage_evaluators[context_fraction]
	with_constraint_predictions_map = evaluators[0].fetch_pred_tuples(gt_future, pred_dict)
	no_constraint_prediction_map = evaluators[1].fetch_pred_tuples(gt_future, pred_dict)
	
	prepare_prediction_graph(
		with_constraint_predictions_map,
		ag_test_data, video_id, "baseline_so",
		"with_constraints", conf.mode, context_fraction
	)
	
	prepare_prediction_graph(
		no_constraint_prediction_map,
		ag_test_data, video_id, "baseline_so",
		"no_constraints", conf.mode, context_fraction
	)


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
		# print(vid_no)
		ind = torch.where(entry["boxes"][:, 0] == future_frame_start_id)[0][0]
		prev_ind = torch.where(entry["boxes"][:, 0] == prev_con)[0][0]
		f_ind = torch.where(entry["boxes"][:, 0] == future_frame_end_id)[0][-1]
		if conf.mode == 'predcls':
			con = set(pred["labels"][prev_ind:ind].tolist())
			fut = set(pred["labels"][ind:f_ind + 1].tolist())
			all_con = set(pred["labels"][:ind].tolist())
		else:
			con = set(pred["pred_labels"][prev_ind:ind].tolist())
			fut = set(pred["pred_labels"][ind:f_ind + 1].tolist())
			all_con = set(pred["pred_labels"][prev_ind:ind].tolist())
		
		box_mask = torch.ones(pred["boxes"][ind:f_ind + 1].shape[0])
		frame_mask = torch.ones(future_idx.shape[0])
		
		im_idx = pred["im_idx"][context_end_idx:future_end_idx]
		im_idx = im_idx - im_idx.min()
		
		pair_idx = pred["pair_idx"][context_end_idx:future_end_idx]
		reshape_pair = pair_idx.view(-1, 2)
		min_value = reshape_pair.min()
		new_pair = reshape_pair - min_value
		pair_idx = new_pair.view(pair_idx.size())
		
		boxes = pred["boxes"][ind:f_ind + 1]
		labels = pred["labels"][ind:f_ind + 1]
		pred_labels = pred["pred_labels"][ind:f_ind + 1]
		scores = pred["scores"][ind:f_ind + 1]
		if conf.mode != 'predcls':
			pred_scores = pred["pred_scores"][ind:f_ind + 1]
		
		ob1 = fut - con
		ob2 = all_con - con
		objects = ob1.union(ob2)
		objects = list(objects)
		for obj in objects:
			for idx, pair in enumerate(pred["pair_idx"][context_end_idx:future_end_idx]):
				if conf.mode == 'predcls':
					if pred["labels"][pair[1]] == obj:
						frame_mask[idx] = 0
						box_mask[pair_idx[idx, 1]] = 0
				else:
					if pred["pred_labels"][pair[1]] == obj:
						frame_mask[idx] = 0
						box_mask[pair_idx[idx, 1]] = 0
		
		im_idx = im_idx[frame_mask == 1]
		removed = pair_idx[frame_mask == 0]
		mask_pair_idx = pair_idx[frame_mask == 1]
		
		flat_pair = mask_pair_idx.view(-1)
		flat_pair_copy = mask_pair_idx.view(-1).detach().clone()
		for pair in removed:
			idx = pair[1]
			for i, p in enumerate(flat_pair_copy):
				if p > idx:
					flat_pair[i] -= 1
		new_pair_idx = flat_pair.view(mask_pair_idx.size())
		
		if conf.mode == 'predcls':
			scores = scores[box_mask == 1]
			labels = labels[box_mask == 1]
		pred_labels = pred_labels[box_mask == 1]
		
		boxes = boxes[box_mask == 1]
		if conf.mode != 'predcls':
			pred_scores = pred_scores[box_mask == 1]
		
		atten = pred["output"][count]['attention_distribution'][frame_mask == 1]
		spatial = pred["output"][count]['spatial_distribution'][frame_mask == 1]
		contact = pred["output"][count]['contacting_distribution'][frame_mask == 1]
		
		if conf.mode == 'predcls':
			pred_dict = {'attention_distribution': atten,
			             'spatial_distribution': spatial,
			             'contacting_distribution': contact,
			             'boxes': boxes,
			             'pair_idx': new_pair_idx,
			             'im_idx': im_idx,
			             'labels': labels,
			             'pred_labels': pred_labels,
			             'scores': scores
			             }
		else:
			pred_dict = {'attention_distribution': atten,
			             'spatial_distribution': spatial,
			             'contacting_distribution': contact,
			             'boxes': boxes,
			             'pair_idx': new_pair_idx,
			             'im_idx': im_idx,
			             # 'labels':labels,
			             'pred_labels': pred_labels,
			             # 'scores':scores,
			             'pred_scores': pred_scores
			             }
		evaluators = future_evaluators[num_future_frames]
		evaluators[0].evaluate_scene_graph(gt_future, pred_dict)
		evaluators[1].evaluate_scene_graph(gt_future, pred_dict)
		evaluators[2].evaluate_scene_graph(gt_future, pred_dict)
		count += 1
		context += 1


def load_model(conf, dataset, gpu_device, model_name):
	model = BaselineWithAnticipation(mode=conf.mode,
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
	mode = checkpoint_name.split('_')[-5]
	
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
			
			for num_future_frames in future_frames_list:
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
				evaluate_model_future_frames(model, entry, gt_annotation, conf, num_future_frames, future_evaluators)
			
			for context_fraction in context_fractions:
				evaluate_model_context_fraction(model, entry, gt_annotation, conf, context_fraction,
				                                percentage_evaluators)
			
			if b % 50 == 0:
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
				mode,
				model_name,
				future_frame_loss_num,
				context_fraction
			)


def generate_qualitative_results():
	(ag_test_data, dataloader_test, gen_evaluators, future_evaluators,
	 future_evaluators_modified_gt, percentage_evaluators,
	 percentage_evaluators_modified_gt, gpu_device, conf) = fetch_transformer_test_basic_config()
	
	model_name = conf.method_name
	checkpoint_name = os.path.basename(conf.ckpt).split('.')[0]
	future_frame_loss_num = checkpoint_name.split('_')[-3]
	mode = checkpoint_name.split('_')[-5]
	
	video_id_index_map = {}
	for index, video_gt_annotation in enumerate(ag_test_data.gt_annotations):
		video_id = video_gt_annotation[0][0]['frame'].split(".")[0]
		video_id_index_map[video_id] = index
	
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
	
	model.eval()
	context_fractions = [0.3, 0.5, 0.7, 0.9]
	video_id_list = ["21F9H", "X95D0", "M18XP", "0A8CF", "LUQWY", "QE4YE", "ENOLD"]
	with torch.no_grad():
		for video_id in video_id_list:
			d_im_data, d_im_info, d_gt_boxes, d_num_boxes, d_index = ag_test_data.fetch_video_data(
				video_id_index_map[video_id])
			im_data = copy.deepcopy(d_im_data.cuda(0))
			im_info = copy.deepcopy(d_im_info.cuda(0))
			gt_boxes = copy.deepcopy(d_gt_boxes.cuda(0))
			num_boxes = copy.deepcopy(d_num_boxes.cuda(0))
			gt_annotation = ag_test_data.gt_annotations[video_id_index_map[video_id]]
			for context_fraction in context_fractions:
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
				generate_context_qualitative_results(model, entry, gt_annotation, conf, context_fraction,
				                                     percentage_evaluators, video_id, ag_test_data)


if __name__ == '__main__':
	generate_qualitative_results()

""" python test_forecasting.py -mode sgdet -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ -model_path forecasting/sgdet_full_context_f3/DSG_masked_9.tar """
