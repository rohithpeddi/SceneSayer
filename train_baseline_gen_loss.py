import copy
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from constants import Constants as const
from lib.object_detector import detector
from lib.supervised.biased.sga.baseline_gen_loss import BaselineWithGenLoss
from train_base import fetch_train_basic_config, prepare_optimizer, fetch_loss_functions


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


def process_train_video(entry, optimizer, model, epoch, num, tr):
	get_sequence(entry, conf.mode)
	pred = model(entry, conf.baseline_context, conf.baseline_future)
	start_time = time.time()
	start = 0
	prev_context_len = 0
	context = conf.baseline_context
	future = conf.baseline_future
	count = 0
	losses = {}
	
	total_frames = len(entry["im_idx"].unique())
	if conf.mode == 'sgcls' or conf.mode == 'sgdet':
		losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])
	
	losses["attention_relation_loss"] = 0
	losses["spatial_relation_loss"] = 0
	losses["contact_relation_loss"] = 0
	
	if start + context + 1 > total_frames:
		while start + context + 1 != total_frames and context > 1:
			context -= 1
		future = 1
	if start + context + future > total_frames > start + context:
		future = total_frames - (start + context)
	
	while start + context + 1 <= total_frames:
		future_frame_start_id = entry["im_idx"].unique()[context]
		
		if start + context + future > total_frames > start + context:
			future = total_frames - (start + context)
		
		future_frame_end_id = entry["im_idx"].unique()[context + future - 1]
		
		context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
		context_idx = entry["im_idx"][:context_end_idx]
		context_len = context_idx.shape[0]
		
		future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
		future_idx = entry["im_idx"][context_end_idx:future_end_idx]
		future_len = future_idx.shape[0]
		
		attention_distribution = pred["output"][count]["attention_distribution"]
		spatial_distribution = pred["output"][count]["spatial_distribution"]
		contact_distribution = pred["output"][count]["contacting_distribution"]
		
		attention_label = torch.tensor(pred["attention_gt"][context_end_idx:future_end_idx], dtype=torch.long).to(
			device=attention_distribution.device).squeeze()
		
		if not conf.bce_loss:
			spatial_label = -torch.ones([len(pred["spatial_gt"][context_end_idx:future_end_idx]), 6],
			                            dtype=torch.long).to(device=attention_distribution.device)
			contact_label = -torch.ones([len(pred["contacting_gt"][context_end_idx:future_end_idx]), 17],
			                            dtype=torch.long).to(device=attention_distribution.device)
			for i in range(len(pred["spatial_gt"][context_end_idx:future_end_idx])):
				spatial_label[i, : len(pred["spatial_gt"][context_end_idx:future_end_idx][i])] = torch.tensor(
					pred["spatial_gt"][context_end_idx:future_end_idx][i])
				contact_label[i,
				: len(pred["contacting_gt"][context_end_idx:future_end_idx][i])] = torch.tensor(
					pred["contacting_gt"][context_end_idx:future_end_idx][i])
		else:
			spatial_label = torch.zeros([len(pred["spatial_gt"][context_end_idx:future_end_idx]), 6],
			                            dtype=torch.float32).to(device=attention_distribution.device)
			contact_label = torch.zeros([len(pred["contacting_gt"][context_end_idx:future_end_idx]), 17],
			                            dtype=torch.float32).to(device=attention_distribution.device)
			for i in range(len(pred["spatial_gt"][context_end_idx:future_end_idx])):
				spatial_label[i, pred["spatial_gt"][context_end_idx:future_end_idx][i]] = 1
				contact_label[i, pred["contacting_gt"][context_end_idx:future_end_idx][i]] = 1
		try:
			losses["attention_relation_loss"] += ce_loss(attention_distribution, attention_label)
		except ValueError:
			attention_label = attention_label.unsqueeze(0)
			losses["attention_relation_loss"] += ce_loss(attention_distribution, attention_label)
		if not conf.bce_loss:
			losses["spatial_relation_loss"] += mlm_loss(spatial_distribution, spatial_label)
			losses["contact_relation_loss"] += mlm_loss(contact_distribution, contact_label)
		else:
			losses["spatial_relation_loss"] += bce_loss(spatial_distribution, spatial_label)
			losses["contact_relation_loss"] += bce_loss(contact_distribution, contact_label)
		
		context += 1
		count += 1
	
	gen_attention_out = pred["gen_attention_distribution"]
	gen_spatial_out = pred["gen_spatial_distribution"]
	gen_contacting_out = pred["gen_contacting_distribution"]
	
	gen_attention_label = torch.tensor(pred["attention_gt"][:-future_len], dtype=torch.long).to(
		device=attention_distribution.device).squeeze()
	if not conf.bce_loss:
		# multi-label margin loss or adaptive loss
		gen_spatial_label = -torch.ones([len(pred["spatial_gt"][:-future_len]), 6], dtype=torch.long).to(
			device=attention_distribution.device)
		gen_contact_label = -torch.ones([len(pred["contacting_gt"][:-future_len]), 17], dtype=torch.long).to(
			device=attention_distribution.device)
		for i in range(len(pred["spatial_gt"][:-future_len])):
			gen_spatial_label[i, : len(pred["spatial_gt"][:-future_len][i])] = torch.tensor(
				pred["spatial_gt"][:-future_len][i])
			gen_contact_label[i, : len(pred["contacting_gt"][:-future_len][i])] = torch.tensor(
				pred["contacting_gt"][:-future_len][i])
	else:
		gen_spatial_label = torch.zeros([len(pred["spatial_gt"][:-future_len]), 6], dtype=torch.float32).to(
			device=attention_distribution.device)
		gen_contact_label = torch.zeros([len(pred["contacting_gt"][:-future_len]), 17], dtype=torch.float32).to(
			device=attention_distribution.device)
		for i in range(len(pred["spatial_gt"][:-future_len])):
			gen_spatial_label[i, pred["spatial_gt"][:-future_len][i]] = 1
			gen_contact_label[i, pred["contacting_gt"][:-future_len][i]] = 1
	
	try:
		losses["gen_attention_relation_loss"] = ce_loss(gen_attention_out, gen_attention_label)
	except ValueError:
		gen_attention_label = attention_label.unsqueeze(0)
		losses["gen_attention_relation_loss"] = ce_loss(gen_attention_out, gen_attention_label)
	
	if not conf.bce_loss:
		losses["gen_spatial_relation_loss"] = mlm_loss(gen_spatial_out, gen_spatial_label)
		losses["gen_contact_relation_loss"] = mlm_loss(gen_contacting_out, gen_contact_label)
	
	else:
		losses["gen_spatial_relation_loss"] = bce_loss(gen_spatial_out, gen_spatial_label)
		losses["gen_contact_relation_loss"] = bce_loss(gen_contacting_out, gen_contact_label)
	
	losses["attention_relation_loss"] = losses["attention_relation_loss"] / count
	losses["spatial_relation_loss"] = losses["spatial_relation_loss"] / count
	losses["contact_relation_loss"] = losses["contact_relation_loss"] / count
	optimizer.zero_grad()
	loss = sum(losses.values())
	loss.backward()
	torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
	optimizer.step()
	
	num += 1
	
	if num % 50 == 0:
		print("epoch {:2d}  batch {:5d}/{:5d}  loss {:.4f}".format(epoch, num, len(dataloader_train),
		                                                           loss.item()))
	
	tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
	if num % 1000 == 0 and num >= 1000:
		time_per_batch = (time.time() - start_time) / 1000
		print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, num, len(dataloader_train),
		                                                                    time_per_batch,
		                                                                    len(dataloader_train) * time_per_batch / 60))
		
		mn = pd.concat(tr[-1000:], axis=1).mean(1)
		print(mn)
	
	return num


def save_model(model, epoch):
	torch.save({"state_dict": model.state_dict()},
	           os.path.join(checkpoint_save_file_path, f"{checkpoint_name}_{epoch}.tar"))
	print("*" * 40)
	print("save the checkpoint after {} epochs".format(epoch))
	with open(evaluator.save_file, "a") as f:
		f.write("save the checkpoint after {} epochs\n".format(epoch))


def process_test_video(entry, model, gt_annotation):
	get_sequence(entry, conf.mode)
	pred = model(entry, conf.baseline_context, conf.baseline_future)
	
	start = 0
	count = 0
	context = conf.baseline_context
	future = conf.baseline_future
	total_frames = len(entry["im_idx"].unique())
	if start + context + 1 > total_frames:
		while start + context + 1 != total_frames and context > 1:
			context -= 1
		future = 1
	if start + context + future > total_frames > start + context:
		future = total_frames - (start + context)
	while start + context + 1 <= total_frames:
		future_frame_start_id = entry["im_idx"].unique()[context]
		
		if start + context + future > total_frames > start + context:
			future = total_frames - (start + context)
		
		future_frame_end_id = entry["im_idx"].unique()[context + future - 1]
		
		context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
		future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
		future_idx = entry["im_idx"][context_end_idx:future_end_idx]
		
		gt_future = gt_annotation[start + context:start + context + future]
		
		evaluator.evaluate_scene_graph_forecasting(gt_future, pred, context_end_idx, future_end_idx, future_idx, count)
		count += 1
		context += 1


def train_baseline_with_gen_loss():
	model = BaselineWithGenLoss(mode=conf.mode,
	                            attention_class_num=len(ag_train_data.attention_relationships),
	                            spatial_class_num=len(ag_train_data.spatial_relationships),
	                            contact_class_num=len(ag_train_data.contacting_relationships),
	                            obj_classes=ag_train_data.object_classes,
	                            enc_layer_num=conf.enc_layer,
	                            dec_layer_num=conf.dec_layer).to(device=gpu_device)
	if conf.ckpt:
		ckpt = torch.load(conf.ckpt, map_location=gpu_device)
		model.load_state_dict(ckpt['state_dict'], strict=False)
	num_epochs = conf.nepoch
	last_epoch = -1
	if conf.ckpt is None:
		# Load latest model from checkpoint directory and train further
		if len(os.listdir(checkpoint_save_file_path)) > 0:
			available_models = [stored_checkpoint for stored_checkpoint in os.listdir(checkpoint_save_file_path) if
			                    checkpoint_name in stored_checkpoint]
			latest_model_name = sorted(available_models, key=lambda x: int(x.split('_')[-1][:-4]))[-1]
			latest_model_path = os.path.join(checkpoint_save_file_path, latest_model_name)
			ckpt = torch.load(latest_model_path, map_location=gpu_device)
			model.load_state_dict(ckpt['state_dict'], strict=False)
			last_epoch = int(latest_model_name.split('_')[-1][:-4])
			num_epochs = 10 - last_epoch - 1
			print(f"Loaded model from {latest_model_path} and training for {num_epochs} more epochs")
		else:
			print("No models found in checkpoint directory, training from scratch")
	
	object_detector = None
	if conf.use_raw_data:
		object_detector = detector(
			train=True,
			object_classes=ag_train_data.object_classes,
			use_SUPPLY=True,
			mode=conf.mode
		).to(device=gpu_device)
		object_detector.eval()
		print("Finished loading object detector", flush=True)
	
	optimizer, scheduler = prepare_optimizer(conf, model)
	
	tr = []
	for epoch in range(last_epoch + 1, num_epochs):
		print("Begin epoch {:d}".format(epoch))
		model.train()
		num = 0
		# Train using only features
		if not conf.use_raw_data:
			print('----------------------------------------------------------', flush=True)
			print('Training using features', flush=True)
			for entry in tqdm(dataloader_train, position=0, leave=True):
				num = process_train_video(entry, optimizer, model, epoch, num, tr)
			save_model(model, epoch)
			if epoch % 3 == 0:
				model.eval()
				with torch.no_grad():
					for entry in tqdm(dataloader_test, position=0, leave=True):
						gt_annotation = entry[const.GT_ANNOTATION]
						process_test_video(entry, model, gt_annotation)
			print('----------------------------------------------------------', flush=True)
			print('epoch: {}'.format(epoch))
			print('----------------------------------------------------------', flush=True)
		else:
			print('----------------------------------------------------------', flush=True)
			print('Training using raw data', flush=True)
			# Train using raw data and object detector instead of features
			train_iter = iter(dataloader_train)
			test_iter = iter(dataloader_test)
			object_detector.is_train = True
			model.train()
			object_detector.train_x = True
			num = 0
			for b in range(len(dataloader_train)):
				data = next(train_iter)
				im_data = copy.deepcopy(data[0].cuda(0))
				im_info = copy.deepcopy(data[1].cuda(0))
				gt_boxes = copy.deepcopy(data[2].cuda(0))
				num_boxes = copy.deepcopy(data[3].cuda(0))
				gt_annotation = ag_train_data.gt_annotations[data[4]]
				with torch.no_grad():
					entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
				num = process_train_video(entry, optimizer, model, epoch, num, tr)
			print(f"Finished training an epoch {epoch}")
			save_model(model, epoch)
			print(f"Saving model after epoch {epoch}")
			if epoch % 3 == 0:
				model.eval()
				object_detector.is_train = False
				with torch.no_grad():
					for b in range(len(dataloader_test)):
						data = next(test_iter)
						im_data = copy.deepcopy(data[0].cuda(0))
						im_info = copy.deepcopy(data[1].cuda(0))
						gt_boxes = copy.deepcopy(data[2].cuda(0))
						num_boxes = copy.deepcopy(data[3].cuda(0))
						gt_annotation = ag_test_data.gt_annotations[data[4]]
						entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
						process_test_video(entry, model, gt_annotation)
						if b % 50 == 0:
							print(f"Finished processing {b} of {len(dataloader_test)} batches")
		score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
		evaluator.print_stats()
		evaluator.reset_result()
		scheduler.step(score)


if __name__ == '__main__':
	conf, dataloader_train, dataloader_test, gpu_device, evaluator, ag_train_data, ag_test_data = fetch_train_basic_config()
	bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss = fetch_loss_functions()
	model_name = "baseline_so_gen_loss"
	checkpoint_name = f"{model_name}_{conf.mode}_future_{conf.baseline_future}"
	checkpoint_save_file_path = os.path.join(conf.save_path, model_name)
	os.makedirs(checkpoint_save_file_path, exist_ok=True)
	evaluator_save_file_path = os.path.join(os.path.abspath('.'), conf.results_path, model_name,
	                                        f"train_{model_name}_{conf.mode}_{conf.baseline_future}.txt")
	os.makedirs(os.path.dirname(evaluator_save_file_path), exist_ok=True)
	evaluator.save_file = evaluator_save_file_path
	train_baseline_with_gen_loss()

# python train_try.py -mode sgcls -ckpt /home/cse/msr/csy227518/scratch/DSG/DSG-DETR/sgcls/model_9.tar -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/

""" python train_obj_mask.py -mode sgdet -save_path forecasting/sgcls_full_context_f5/ -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """
