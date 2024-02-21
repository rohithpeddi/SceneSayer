import copy
import math
import os

import torch

from lib.object_detector import detector
from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher
from lib.supervised.biased.sga.baseline_anticipation import BaselineWithAnticipation
from lib.supervised.biased.sga.baseline_anticipation_gen_loss import BaselineWithAnticipationGenLoss
from test_base import (fetch_transformer_test_basic_config, get_sequence_no_tracking, prepare_prediction_graph,
                       send_future_evaluators_stats_to_firebase, write_future_evaluators_stats,
                       write_percentage_evaluators_stats, send_percentage_evaluators_stats_to_firebase)


def fetch_ff_rep_for_eval(entry, pred, gt_annotation, conf, num_cf, num_ff, count):
	ff_start_id = entry["im_idx"].unique()[num_cf]
	cf_end_id = entry["im_idx"].unique()[num_cf - 1]
	ff_end_id = entry["im_idx"].unique()[num_cf + num_ff - 1]
	
	objects_ff_start_id = int(torch.where(entry["im_idx"] == ff_start_id)[0][0])
	boxes_ff_start_id = int(torch.where(entry["boxes"][:, 0] == ff_start_id)[0][0])
	boxes_cf_last_start_id = int(torch.where(entry["boxes"][:, 0] == cf_end_id)[0][0])
	
	# Add +1 for all end ids to include the last item when a range is provided
	objects_ff_end_id = int(torch.where(entry["im_idx"] == ff_end_id)[0][-1]) + 1
	boxes_ff_end_id = int(torch.where(entry["boxes"][:, 0] == ff_end_id)[0][-1]) + 1
	
	objects_ff = entry["im_idx"][objects_ff_start_id:objects_ff_end_id]
	pred_label_dict = pred["labels"] if conf.mode == 'predcls' else pred["pred_labels"]
	
	pred_labels_set_cf_last = set(pred_label_dict[boxes_cf_last_start_id:boxes_ff_start_id].tolist())
	pred_labels_set_ff = set(pred_label_dict[boxes_ff_start_id:boxes_ff_end_id].tolist())
	pred_labels_set_cf = set(pred_label_dict[:boxes_ff_start_id].tolist())
	
	# ---------------------------------------------------------------------------
	# Remove objects and pairs that are in the future frames in order to include
	# only those that are present in the last frames of the given context
	
	boxes_mask_ff = torch.ones(pred["boxes"][boxes_ff_start_id:boxes_ff_end_id].shape[0])
	objects_mask_ff = torch.ones(objects_ff.shape[0])
	
	im_idx_ff = pred["im_idx"][objects_ff_start_id:objects_ff_end_id]
	im_idx_ff = im_idx_ff - im_idx_ff.min()
	
	# TODO: Correct pair_idx logic
	pair_idx_ff = pred["pair_idx"][objects_ff_start_id:objects_ff_end_id]
	reshape_pair = pair_idx_ff.view(-1, 2)
	min_value = reshape_pair.min()
	new_pair = reshape_pair - min_value
	pair_idx_ff = new_pair.view(pair_idx_ff.size())
	
	boxes_ff = pred["boxes"][boxes_ff_start_id:boxes_ff_end_id]
	labels_ff = pred["labels"][boxes_ff_start_id:boxes_ff_end_id]
	pred_labels_ff = pred["pred_labels"][boxes_ff_start_id:boxes_ff_end_id]
	scores_ff = pred["scores"][boxes_ff_start_id:boxes_ff_end_id]
	
	pred_labels_in_ff_not_in_clf = pred_labels_set_ff - pred_labels_set_cf_last
	pred_labels_in_cf_not_in_clf = pred_labels_set_cf - pred_labels_set_cf_last
	pred_labels_not_in_clf = pred_labels_in_ff_not_in_clf.union(pred_labels_in_cf_not_in_clf)
	for idx, pair in enumerate(pair_idx_ff):
		if pred_label_dict[pair[1]] in list(pred_labels_not_in_clf):
			objects_mask_ff[idx] = 0
			boxes_mask_ff[pair_idx_ff[idx, 1]] = 0
	
	im_idx_ff = im_idx_ff[objects_mask_ff == 1]
	removed_pair_idx_ff = pair_idx_ff[objects_mask_ff == 0]
	mask_pair_idx_ff = pair_idx_ff[objects_mask_ff == 1]
	flat_pair = mask_pair_idx_ff.view(-1)
	
	decrement_count = torch.zeros_like(flat_pair)
	# For each removed pair, increment the decrement count for indices greater than the removed index
	for idx in removed_pair_idx_ff[:, 1]:  # Assuming the second element of each pair is the relevant index
		decrement_count += (flat_pair > idx).int()
	new_flat_pair = flat_pair - decrement_count
	updated_pair_idx_ff = new_flat_pair.view(mask_pair_idx_ff.size())
	
	if conf.mode == 'predcls':
		scores_ff = scores_ff[boxes_mask_ff == 1]
		labels_ff = labels_ff[boxes_mask_ff == 1]
	else:
		pred_scores_ff = pred["pred_scores"][boxes_ff_start_id:boxes_ff_end_id]
		pred_scores_ff = pred_scores_ff[boxes_mask_ff == 1]
	
	pred_labels_ff = pred_labels_ff[boxes_mask_ff == 1]
	boxes_ff = boxes_ff[boxes_mask_ff == 1]
	
	attention_distribution_ff = pred["output"][count]['attention_distribution'][objects_mask_ff == 1]
	spatial_distribution_ff = pred["output"][count]['spatial_distribution'][objects_mask_ff == 1]
	contact_distribution_ff = pred["output"][count]['contacting_distribution'][objects_mask_ff == 1]
	
	gt_future = gt_annotation[num_cf: num_cf + num_ff]
	
	pred_dict = {
		'attention_distribution': attention_distribution_ff,
		'spatial_distribution': spatial_distribution_ff,
		'contacting_distribution': contact_distribution_ff,
		'boxes': boxes_ff,
		'pair_idx': updated_pair_idx_ff,
		'im_idx': im_idx_ff,
		'pred_labels': pred_labels_ff
	}
	
	if conf.mode == 'predcls':
		pred_dict['labels'] = labels_ff
		pred_dict['scores'] = scores_ff
	else:
		pred_dict['pred_scores'] = pred_scores_ff
	
	return gt_future, pred_dict


def evaluate_model_context_fraction(model, entry, gt_annotation, conf, context_fraction, percentage_evaluators):
	gt_future, pred_dict = fetch_model_context_pred_dict(model, entry, gt_annotation, conf, context_fraction)
	evaluators = percentage_evaluators[context_fraction]
	evaluators[0].evaluate_scene_graph(gt_future, pred_dict)
	evaluators[1].evaluate_scene_graph(gt_future, pred_dict)
	evaluators[2].evaluate_scene_graph(gt_future, pred_dict)


def evaluate_model_future_frames(model, entry, gt_annotation, conf, num_ff, future_evaluators):
	get_sequence_no_tracking(entry, conf.mode)
	pred = model(entry, conf.baseline_context, num_ff)
	
	count = 0
	num_cf = conf.baseline_context
	num_tf = len(entry["im_idx"].unique())
	num_cf = min(num_cf, num_tf - 1)
	while num_cf + 1 <= num_tf:
		num_ff = min(num_ff, num_tf - num_cf)
		gt_future, pred_dict = fetch_ff_rep_for_eval(entry, pred, gt_annotation, conf, num_cf, num_ff, count)
		
		evaluators = future_evaluators[num_ff]
		evaluators[0].evaluate_scene_graph(gt_future, pred_dict)
		evaluators[1].evaluate_scene_graph(gt_future, pred_dict)
		evaluators[2].evaluate_scene_graph(gt_future, pred_dict)
		count += 1
		num_cf += 1


def fetch_model_context_pred_dict(model, entry, gt_annotation, conf, context_fraction):
	get_sequence_no_tracking(entry, conf.mode)
	pred = model.forward_single_entry(context_fraction=context_fraction, entry=entry)
	num_tf = len(entry["im_idx"].unique())
	num_cf = min(int(math.ceil(context_fraction * num_tf)), num_tf - 1)
	num_ff = num_tf - num_cf
	gt_future, pred_dict = fetch_ff_rep_for_eval(entry, pred, gt_annotation, conf, num_cf, num_ff, count=0)
	return gt_future, pred_dict


def generate_context_qualitative_results(model, entry, gt_annotation, conf, context_fraction,
                                         percentage_evaluators, video_id, ag_test_data):
	gt_future, pred_dict = fetch_model_context_pred_dict(model, entry, gt_annotation, conf, context_fraction)
	
	evaluators = percentage_evaluators[context_fraction]
	with_constraint_predictions_map = evaluators[0].fetch_pred_tuples(gt_future, pred_dict)
	no_constraint_prediction_map = evaluators[1].fetch_pred_tuples(gt_future, pred_dict)
	
	prepare_prediction_graph(
		with_constraint_predictions_map,
		ag_test_data, video_id, conf.method_name,
		"with_constraints", conf.mode, context_fraction
	)
	
	prepare_prediction_graph(
		no_constraint_prediction_map,
		ag_test_data, video_id, conf.method_name,
		"no_constraints", conf.mode, context_fraction
	)


def load_baseline(conf, dataset, gpu_device):
	model = BaselineWithAnticipation(mode=conf.mode,
	                                 attention_class_num=len(dataset.attention_relationships),
	                                 spatial_class_num=len(dataset.spatial_relationships),
	                                 contact_class_num=len(dataset.contacting_relationships),
	                                 obj_classes=dataset.object_classes,
	                                 enc_layer_num=conf.enc_layer,
	                                 dec_layer_num=conf.dec_layer).to(device=gpu_device)
	
	ckpt = torch.load(conf.ckpt, map_location=gpu_device)
	model.load_state_dict(ckpt[f'{conf.method_name}_state_dict'], strict=False)
	print(f"Loaded model from checkpoint {conf.ckpt}")
	return model


def load_sgatformer(conf, dataset, gpu_device):
	model = BaselineWithAnticipationGenLoss(mode=conf.mode,
	                                        attention_class_num=len(dataset.attention_relationships),
	                                        spatial_class_num=len(dataset.spatial_relationships),
	                                        contact_class_num=len(dataset.contacting_relationships),
	                                        obj_classes=dataset.object_classes,
	                                        enc_layer_num=conf.enc_layer,
	                                        dec_layer_num=conf.dec_layer).to(device=gpu_device)
	
	ckpt = torch.load(conf.ckpt, map_location=gpu_device)
	model.load_state_dict(ckpt[f'{conf.method_name}_state_dict'], strict=False)
	print(f"Loaded model from checkpoint {conf.ckpt}")
	return model


def test_model():
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
	
	if conf.method_name == "baseline_so":
		model = load_baseline(conf, ag_test_data, gpu_device)
	else:
		model = load_sgatformer(conf, ag_test_data, gpu_device)
	
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
	
	if conf.method_name == "baseline_so":
		model = load_baseline(conf, ag_test_data, gpu_device)
	else:
		model = load_sgatformer(conf, ag_test_data, gpu_device)
	
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

