import copy
import os

import torch

from lib.object_detector import detector
from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher
from lib.supervised.biased.sga.baseline_gen_loss import BaselineWithAnticipationGenLoss
from test_base import fetch_transformer_test_basic_config


def get_sequence(entry, task="sgcls"):
	if task == "predcls":
		indices = []
		for i in entry["labels"].unique():
			indices.append(torch.where(entry["labels"] == i)[0])
		entry["indices"] = indices
		return
	
	if task == "sgdet" or task == "sgcls":
		# for sgdet, use the predicted object classes, as a special case of
		# the proposed method, comment this out for general coase tracking.
		indices = [[]]
		# indices[0] store single-element sequence, to save memory
		pred_labels = torch.argmax(entry["distribution"], 1)
		for i in pred_labels.unique():
			index = torch.where(pred_labels == i)[0]
			if len(index) == 1:
				indices[0].append(index)
			else:
				indices.append(index)
		if len(indices[0]) > 0:
			indices[0] = torch.cat(indices[0])
		else:
			indices[0] = torch.tensor([])
		entry["indices"] = indices
		return


def evaluate_baseline(model, entry, gt_annotation, context, future_frames):
	pred = model(entry, context, future_frames)
	start = 0
	count = 0
	total_frames = len(entry["im_idx"].unique())
	if start + context + 1 > total_frames:
		while start + context + 1 != total_frames and context > 1:
			context -= 1
		future_frames = 1
	if start + context + future_frames > total_frames > start + context:
		future_frames = total_frames - (start + context)
	while start + context + 1 <= total_frames:
		future_frame_start_id = entry["im_idx"].unique()[context]
		
		if start + context + future_frames > total_frames > start + context:
			future_frames = total_frames - (start + context)
		
		future_frame_end_id = entry["im_idx"].unique()[context + future_frames - 1]
		context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
		future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
		future_idx = entry["im_idx"][context_end_idx:future_end_idx]
		gt_future = gt_annotation[start + context:start + context + future_frames]
		future_evaluators[future_frames].evaluate_scene_graph_forecasting(gt_future, pred, context_end_idx,
		                                                                  future_end_idx, future_idx, count)
		count += 1
		context += 1


def test_baseline_with_gen_loss():
	object_detector = detector(
		train=False,
		object_classes=ag_test_data.object_classes,
		use_SUPPLY=True,
		mode=conf.mode
	).to(device=gpu_device)
	object_detector.eval()
	
	model = BaselineWithAnticipationGenLoss(mode=conf.mode,
	                                        attention_class_num=len(ag_test_data.attention_relationships),
	                                        spatial_class_num=len(ag_test_data.spatial_relationships),
	                                        contact_class_num=len(ag_test_data.contacting_relationships),
	                                        obj_classes=ag_test_data.object_classes,
	                                        enc_layer_num=conf.enc_layer,
	                                        dec_layer_num=conf.dec_layer).to(device=gpu_device)
	model.eval()
	
	ckpt = torch.load(conf.model_path, map_location=gpu_device)
	model.load_state_dict(ckpt['state_dict'], strict=False)
	print('*' * 50)
	print('CKPT {} is loaded'.format(conf.model_path))
	
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	future_frames_list = [1, 2, 3, 4, 5]
	with torch.no_grad():
		for id, data in enumerate(dataloader_test):
			im_data = copy.deepcopy(data[0].cuda(0))
			im_info = copy.deepcopy(data[1].cuda(0))
			gt_boxes = copy.deepcopy(data[2].cuda(0))
			num_boxes = copy.deepcopy(data[3].cuda(0))
			gt_annotation = ag_test_data.gt_annotations[data[4]]
			
			for future_frames in future_frames_list:
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
				get_sequence(entry, conf.mode)
				evaluate_baseline(model, entry, gt_annotation, conf.baseline_context, future_frames)


# TODO: Add code to save the results to a CSV file

# print('Average inference time', np.mean(all_time))
# print(f'------------------------- for future = {future}--------------------------')
# print('-------------------------with constraint-------------------------------')
# with_constraint_evaluator.print_stats()
# print('-------------------------semi constraint-------------------------------')
# semi_constraint_evaluator.print_stats()
# print('-------------------------no constraint-------------------------------')
# no_constraint_evaluator.print_stats()


if __name__ == '__main__':
	ag_test_data, dataloader_test, gen_evaluators, future_evaluators, future_evaluators_modified_gt, percentage_evaluators, percentage_evaluators_modified_gt, gpu_device, conf = fetch_transformer_test_basic_config()
	model_name = os.path.basename(conf.model_path).split('.')[0]
	evaluator_save_file_dir = os.path.join(os.path.abspath('.'), conf.results_path, model_name)
	test_baseline_with_gen_loss()

""" python test_forecasting.py -mode sgdet -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ -model_path forecasting/sgdet_full_context_f3/DSG_masked_9.tar """
