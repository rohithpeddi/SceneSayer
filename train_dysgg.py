import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from constants import Constants as const
from lib.supervised.biased.dysgg.DyDsgDETR import DyDsgDETR
from lib.supervised.biased.dysgg.DySTTran import DySTTran

from train_base import fetch_train_basic_config, fetch_transformer_loss_functions, save_model, get_sequence_no_tracking, \
	prepare_optimizer
from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher
from lib.supervised.biased.dsgdetr.track import get_sequence_with_tracking


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
	
	optimizer, scheduler = prepare_optimizer(conf, model)
	return model, optimizer, scheduler


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
	
	optimizer, scheduler = prepare_optimizer(conf, model)
	return model, optimizer, scheduler


def train_model(conf, model, matcher, optimizer, dataloader_train, tr, epoch, is_tracking_enabled=False):
	bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss = fetch_transformer_loss_functions()
	
	model.train()
	start = time.time()
	counter = 0
	for train_entry in tqdm(dataloader_train):
		gt_annotation = train_entry[const.GT_ANNOTATION]
		frame_size = train_entry[const.FRAME_SIZE]
		
		if is_tracking_enabled:
			get_sequence_with_tracking(train_entry, gt_annotation, matcher, frame_size, conf.mode)
		else:
			get_sequence_no_tracking(train_entry, task=conf.mode)
		
		pred = model(train_entry)
		
		attention_distribution = pred[const.ATTENTION_DISTRIBUTION]
		spatial_distribution = pred[const.SPATIAL_DISTRIBUTION]
		contact_distribution = pred[const.CONTACTING_DISTRIBUTION]
		
		attention_label = torch.tensor(pred[const.ATTENTION_GT], dtype=torch.long).to(
			device=attention_distribution.device).squeeze()
		if not conf.bce_loss:
			# multi-label margin loss or adaptive loss
			spatial_label = -torch.ones([len(pred[const.SPATIAL_GT]), 6], dtype=torch.long).to(
				device=attention_distribution.device)
			contact_label = -torch.ones([len(pred[const.CONTACTING_GT]), 17], dtype=torch.long).to(
				device=attention_distribution.device)
			for i in range(len(pred[const.SPATIAL_GT])):
				spatial_label[i, : len(pred[const.SPATIAL_GT][i])] = torch.tensor(pred[const.SPATIAL_GT][i])
				contact_label[i, : len(pred[const.CONTACTING_GT][i])] = torch.tensor(pred[const.CONTACTING_GT][i])
		else:
			# bce loss
			spatial_label = torch.zeros([len(pred[const.SPATIAL_GT]), 6], dtype=torch.float32).to(
				device=attention_distribution.device)
			contact_label = torch.zeros([len(pred[const.CONTACTING_GT]), 17], dtype=torch.float32).to(
				device=attention_distribution.device)
			for i in range(len(pred[const.SPATIAL_GT])):
				spatial_label[i, pred[const.SPATIAL_GT][i]] = 1
				contact_label[i, pred[const.CONTACTING_GT][i]] = 1
		
		losses = {}
		if conf.mode == const.SGCLS or conf.mode == const.SGDET:
			losses[const.OBJECT_LOSS] = ce_loss(pred[const.DISTRIBUTION], pred[const.LABELS])
		
		losses[const.ATTENTION_RELATION_LOSS] = ce_loss(attention_distribution, attention_label)
		if not conf.bce_loss:
			losses[const.SPATIAL_RELATION_LOSS] = mlm_loss(spatial_distribution, spatial_label)
			losses[const.CONTACTING_RELATION_LOSS] = mlm_loss(contact_distribution, contact_label)
		else:
			losses[const.SPATIAL_RELATION_LOSS] = bce_loss(spatial_distribution, spatial_label)
			losses[const.CONTACTING_RELATION_LOSS] = bce_loss(contact_distribution, contact_label)
		
		optimizer.zero_grad()
		loss = sum(losses.values())
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
		optimizer.step()
		
		tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
		
		if counter % 1000 == 0 and counter >= 1000:
			time_per_batch = (time.time() - start) / 1000
			print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, counter, len(dataloader_train),
			                                                                    time_per_batch,
			                                                                    len(dataloader_train) * time_per_batch / 60))
			
			mn = pd.concat(tr[-1000:], axis=1).mean(1)
			print(mn)
			start = time.time()
		counter += 1


def test_model(model, dataloader_test, evaluator, conf, matcher, is_tracking_enabled=False):
	model.eval()
	with torch.no_grad():
		for test_entry in tqdm(dataloader_test):
			gt_annotation = test_entry[const.GT_ANNOTATION]
			frame_size = test_entry[const.FRAME_SIZE]
			
			if is_tracking_enabled:
				get_sequence_with_tracking(test_entry, gt_annotation, matcher, frame_size, conf.mode)
			else:
				get_sequence_no_tracking(test_entry, task=conf.mode)
				
			pred = model(test_entry)
			
			evaluator.evaluate_scene_graph(gt_annotation, pred)


def main():
	conf, dataloader_train, dataloader_test, gpu_device, evaluator, ag_train_data, ag_test_data = fetch_train_basic_config()
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	method_name = conf.method_name
	checkpoint_name = f"{method_name}_{conf.mode}"
	checkpoint_save_file_path = os.path.join(conf.save_path, method_name)
	os.makedirs(checkpoint_save_file_path, exist_ok=True)
	evaluator_save_file_path = os.path.join(os.path.abspath('.'), conf.results_path, method_name,
	                                        f"train_{method_name}_{conf.mode}.txt")
	os.makedirs(os.path.dirname(evaluator_save_file_path), exist_ok=True)
	evaluator.save_file = evaluator_save_file_path
	
	model, optimizer, scheduler = None, None, None
	model_name = None
	
	is_tracking_enabled = False
	if method_name == "DySTTran":
		model, optimizer, scheduler = load_DySTTran(conf, ag_train_data, gpu_device)
		model_name = "dysttran"
	elif method_name == "DyDsgDETR":
		model, optimizer, scheduler = load_DyDsgDETR(conf, ag_train_data, gpu_device)
		model_name = "dydsgdetr"
		is_tracking_enabled = True
	
	assert model is not None and optimizer is not None and scheduler is not None
	assert model_name is not None
	
	tr = []
	for epoch in range(conf.num_epochs):
		train_model(conf, model, matcher, optimizer, dataloader_train, tr, epoch,
		            is_tracking_enabled=is_tracking_enabled)
		save_model(model, epoch, checkpoint_save_file_path, checkpoint_name, model_name)
		test_model(model, dataloader_test, evaluator, conf, matcher, is_tracking_enabled=is_tracking_enabled)
		score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
		evaluator.print_stats()
		evaluator.reset_result()
		scheduler.step(score)


if __name__ == '__main__':
	# Should set method_name appropriately
	main()

# python train_try.py -mode sgcls -ckpt /home/cse/msr/csy227518/scratch/DSG/DSG-DETR/sgcls/model_9.tar -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/

""" python train_obj_mask.py -mode sgdet -save_path forecasting/sgcls_full_context_f5/ -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """
