import numpy as np

from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher
from lib.supervised.biased.dsgdetr.track import get_sequence_with_tracking
from test_base import fetch_test_basic_config

import copy
import torch
from time import time
from lib.object_detector import detector
from lib.supervised.biased.sga.forecasting import STTran


def test_forecasting():
	object_detector = detector(
		train=False,
		object_classes=ag_features_test.object_classes,
		use_SUPPLY=True,
		mode=conf.mode
	).to(device=gpu_device)
	object_detector.eval()
	
	model = STTran(mode=conf.mode,
	               attention_class_num=len(ag_features_test.attention_relationships),
	               spatial_class_num=len(ag_features_test.spatial_relationships),
	               contact_class_num=len(ag_features_test.contacting_relationships),
	               obj_classes=ag_features_test.object_classes,
	               enc_layer_num=conf.enc_layer,
	               dec_layer_num=conf.dec_layer).to(device=gpu_device)
	
	model.eval()
	
	ckpt = torch.load(conf.model_path, map_location=gpu_device)
	model.load_state_dict(ckpt['state_dict'], strict=False)
	print('*' * 50)
	print('CKPT {} is loaded'.format(conf.model_path))
	
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	all_time = []
	c = 0
	with torch.no_grad():
		for b, data in enumerate(dataloader_test):
			start_time = time()
			im_data = copy.deepcopy(data[0].cuda(0))
			im_info = copy.deepcopy(data[1].cuda(0))
			gt_boxes = copy.deepcopy(data[2].cuda(0))
			num_boxes = copy.deepcopy(data[3].cuda(0))
			gt_annotation = ag_features_test.gt_annotations[data[4]]
			vid_no = gt_annotation[0][0]["frame"].split('.')[0]
			
			entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
			get_sequence_with_tracking(entry, gt_annotation, matcher, (im_info[0][:2] / im_info[0, 2]).cpu().data, conf.mode)
			
			count = 0
			start = 0
			future = 3
			context = 4
			
			pred = model(entry, context, future)
			all_time.append(time() - start_time)
			
			if start + context + 1 > len(entry["im_idx"].unique()):
				while start + context + 1 != len(entry["im_idx"].unique()) and context > 1:
					context -= 1
				future = 1
			
			if (start + context + future > len(entry["im_idx"].unique()) and start + context < len(
					entry["im_idx"].unique())):
				future = len(entry["im_idx"].unique()) - (start + context)
			
			while start + context + 1 <= len(entry["im_idx"].unique()):
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
				with_constraint_evaluator.evaluate_scene_graph_forecasting(gt_future, pred, context_end_idx,
				                                                           future_end_idx,
				                                                           future_idx, count)
				no_constraint_evaluator.evaluate_scene_graph_forecasting(gt_future, pred, context_end_idx,
				                                                         future_end_idx,
				                                                         future_idx, count)
				semi_constraint_evaluator.evaluate_scene_graph_forecasting(gt_future, pred, context_end_idx,
				                                                           future_end_idx,
				                                                           future_idx, count)
				
				# evaluator.print_stats()
				count += 1
				context += 1
				
				if start + context + future > len(entry["im_idx"].unique()):
					break
	
	print('Average inference time', np.mean(all_time))
	print(f'------------------------- for future = {future}--------------------------')
	print('-------------------------with constraint-------------------------------')
	with_constraint_evaluator.print_stats()
	print('-------------------------semi constraint-------------------------------')
	semi_constraint_evaluator.print_stats()
	print('-------------------------no constraint-------------------------------')
	no_constraint_evaluator.print_stats()


if __name__ == '__main__':
	ag_features_test, dataloader_test, with_constraint_evaluator, no_constraint_evaluator, semi_constraint_evaluator, gpu_device, conf = fetch_test_basic_config()
	test_forecasting()

""" python test_forecasting.py -mode sgdet -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ -model_path forecasting/sgdet_full_context_f3/DSG_masked_9.tar """
