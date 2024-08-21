import copy
import os

import torch

from lib.object_detector import Detector
from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
from lib.supervised.dysgg import DyDsgDETR
from lib.supervised.dysgg import DySTTran
from test_base import fetch_dysgg_test_basic_config, write_evaluators_stats_dysgg, get_sequence_no_tracking, \
	send_results_to_firebase


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
		print(f"Loaded checkpoint from {conf.ckpt}")
	
	return model


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
		print(f"Loaded checkpoint from {conf.ckpt}")
	
	return model


def process_data(model, object_detector, ag_test_data, dataloader_test, gen_evaluators, matcher, conf, model_name,
                 is_tracking_enabled):
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
				get_sequence_with_tracking(entry, gt_annotation, matcher, (im_info[0][:2] / im_info[0, 2]).cpu().data,
				                           conf.mode)
			else:
				get_sequence_no_tracking(entry, task=conf.mode)
			
			pred = model(entry)
			
			for gen_evaluator in gen_evaluators:
				gen_evaluator.evaluate_scene_graph(gt_annotation, pred)
			
			if b % 50 == 0:
				print(f"Finished processing {b} of {len(dataloader_test)} batches")
	
	write_evaluators_stats_dysgg(conf.mode, model_name, gen_evaluators)
	send_results_to_firebase(gen_evaluators, task_name="dysgg", method_name=model_name, mode=conf.mode)


def load_object_detector(conf, gpu_device, dataset):
	object_detector = Detector(
		train=True,
		object_classes=dataset.object_classes,
		use_SUPPLY=True,
		mode=conf.mode
	).to(device=gpu_device)
	object_detector.eval()
	print("Finished loading object detector", flush=True)
	return object_detector


def main():
	ag_test_data, dataloader_test, gen_evaluators, gpu_device, conf = fetch_dysgg_test_basic_config()
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	method_name = conf.method_name
	
	evaluator_save_file_path = os.path.join(os.path.abspath('..'), conf.results_path, method_name,
	                                        f"train_{method_name}_{conf.mode}.txt")
	os.makedirs(os.path.dirname(evaluator_save_file_path), exist_ok=True)
	for gen_evaluator in gen_evaluators:
		gen_evaluator.save_file = evaluator_save_file_path
	
	model, optimizer, scheduler = None, None, None
	model_name = None
	
	is_tracking_enabled = False
	if method_name == "dysttran":
		model = load_DySTTran(conf, ag_test_data, gpu_device)
		model_name = method_name
	elif method_name == "dydsgdetr":
		model = load_DyDsgDETR(conf, ag_test_data, gpu_device)
		model_name = method_name
		is_tracking_enabled = True
	
	assert model is not None and model_name is not None
	
	object_detector = load_object_detector(conf, gpu_device, ag_test_data)
	process_data(model, object_detector, ag_test_data, dataloader_test, gen_evaluators, matcher, conf, model_name,
	             is_tracking_enabled)


if __name__ == '__main__':
	# Should set method_name appropriately
	main()

# python train_dysgg.py -method_name dysttran -use_raw_data -mode sgdet -save_path /home/rxp190007/DATA/ag/checkpoints -datasize large -data_path /home/rxp190007/DATA/ag
# python train_dysgg.py -method_name dydsgdetr -use_raw_data -mode sgdet -save_path /home/rxp190007/DATA/ag/checkpoints -datasize large -data_path /home/rxp190007/DATA/ag
