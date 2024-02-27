import os

import torch

from lib.object_detector import detector
from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher
from lib.supervised.biased.dsgdetr.track import get_sequence_with_tracking
from lib.supervised.biased.sga.rel.rel_dsgdetr_ant import RelDsgDetrAnt
from lib.supervised.biased.sga.rel.rel_dsgdetr_gen_ant import RelDsgDetrGenAnt
from lib.supervised.biased.sga.rel.rel_sttran_ant import RelSTTranAnt
from lib.supervised.biased.sga.rel.rel_sttran_gen_ant import RelSTTranGenAnt
from lib.supervised.biased.sga.obj.obj_dsgdetr_ant import ObjDsgDetrAnt
from lib.supervised.biased.sga.obj.obj_dsgdetr_gen_ant import ObjDsgDetrGenAnt
from lib.supervised.biased.sga.obj.obj_sttran_ant import ObjSTTranAnt
from lib.supervised.biased.sga.obj.obj_sttran_gen_ant import ObjSTTranGenAnt


def get_sequence_no_tracking(entry, task="sgcls"):
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


def fetch_sequences_after_tracking(conf, entry, gt_annotation, matcher, frame_size):
	if conf.method_name in ["rel_sttran_ant", "rel_sttran_gen_ant", "obj_sttran_ant", "obj_sttran_gen_ant"]:
		get_sequence_no_tracking(entry, conf)
	elif conf.method_name in ["rel_dsgdetr_ant", "rel_dsgdetr_gen_ant", "obj_dsgdetr_ant", "obj_dsgdetr_gen_ant"]:
		get_sequence_with_tracking(entry, gt_annotation, matcher, frame_size, conf.mode)
	else:
		raise ValueError(f"Method name {conf.method_name} not recognized")


def load_model_from_checkpoint(model, conf, gpu_device):
	if conf.ckpt is not None:
		ckpt = torch.load(conf.ckpt, map_location=gpu_device)
		model.load_state_dict(ckpt[f'{conf.method_name}_state_dict'], strict=False)
		print(f"Loaded model from checkpoint {conf.ckpt}")
	else:
		print("No checkpoint to load from...... Training from scratch.")
	return model


def load_rel_sttran_ant(conf, dataset, gpu_device):
	model = RelSTTranAnt(mode=conf.mode,
	                     attention_class_num=len(dataset.attention_relationships),
	                     spatial_class_num=len(dataset.spatial_relationships),
	                     contact_class_num=len(dataset.contacting_relationships),
	                     obj_classes=dataset.object_classes,
	                     enc_layer_num=conf.enc_layer,
	                     dec_layer_num=conf.dec_layer).to(device=gpu_device)
	model = load_model_from_checkpoint(model, conf, gpu_device)
	return model


def load_rel_sttran_gen_ant(conf, dataset, gpu_device):
	model = RelSTTranGenAnt(mode=conf.mode,
	                        attention_class_num=len(dataset.attention_relationships),
	                        spatial_class_num=len(dataset.spatial_relationships),
	                        contact_class_num=len(dataset.contacting_relationships),
	                        obj_classes=dataset.object_classes,
	                        enc_layer_num=conf.enc_layer,
	                        dec_layer_num=conf.dec_layer).to(device=gpu_device)
	model = load_model_from_checkpoint(model, conf, gpu_device)
	return model


def load_rel_dsgdetr_ant(conf, dataset, gpu_device):
	model = RelDsgDetrAnt(mode=conf.mode,
	                      attention_class_num=len(dataset.attention_relationships),
	                      spatial_class_num=len(dataset.spatial_relationships),
	                      contact_class_num=len(dataset.contacting_relationships),
	                      obj_classes=dataset.object_classes,
	                      enc_layer_num=conf.enc_layer,
	                      dec_layer_num=conf.dec_layer).to(device=gpu_device)
	model = load_model_from_checkpoint(model, conf, gpu_device)
	return model


def load_rel_dsgdetr_gen_ant(conf, dataset, gpu_device):
	model = RelDsgDetrGenAnt(mode=conf.mode,
	                         attention_class_num=len(dataset.attention_relationships),
	                         spatial_class_num=len(dataset.spatial_relationships),
	                         contact_class_num=len(dataset.contacting_relationships),
	                         obj_classes=dataset.object_classes,
	                         enc_layer_num=conf.enc_layer,
	                         dec_layer_num=conf.dec_layer).to(device=gpu_device)
	model = load_model_from_checkpoint(model, conf, gpu_device)
	return model


def load_obj_sttran_ant(conf, dataset, gpu_device):
	model = ObjSTTranAnt(mode=conf.mode,
	                     attention_class_num=len(dataset.attention_relationships),
	                     spatial_class_num=len(dataset.spatial_relationships),
	                     contact_class_num=len(dataset.contacting_relationships),
	                     obj_classes=dataset.object_classes,
	                     enc_layer_num=conf.enc_layer,
	                     dec_layer_num=conf.dec_layer).to(device=gpu_device)
	model = load_model_from_checkpoint(model, conf, gpu_device)
	return model


def load_obj_sttran_gen_ant(conf, dataset, gpu_device):
	model = ObjSTTranGenAnt(mode=conf.mode,
	                        attention_class_num=len(dataset.attention_relationships),
	                        spatial_class_num=len(dataset.spatial_relationships),
	                        contact_class_num=len(dataset.contacting_relationships),
	                        obj_classes=dataset.object_classes,
	                        enc_layer_num=conf.enc_layer,
	                        dec_layer_num=conf.dec_layer).to(device=gpu_device)
	model = load_model_from_checkpoint(model, conf, gpu_device)
	return model


def load_obj_dsgdetr_ant(conf, dataset, gpu_device):
	model = ObjDsgDetrAnt(mode=conf.mode,
	                      attention_class_num=len(dataset.attention_relationships),
	                      spatial_class_num=len(dataset.spatial_relationships),
	                      contact_class_num=len(dataset.contacting_relationships),
	                      obj_classes=dataset.object_classes,
	                      enc_layer_num=conf.enc_layer,
	                      dec_layer_num=conf.dec_layer).to(device=gpu_device)
	model = load_model_from_checkpoint(model, conf, gpu_device)
	return model


def load_obj_dsgdetr_gen_ant(conf, dataset, gpu_device):
	model = ObjDsgDetrGenAnt(mode=conf.mode,
	                         attention_class_num=len(dataset.attention_relationships),
	                         spatial_class_num=len(dataset.spatial_relationships),
	                         contact_class_num=len(dataset.contacting_relationships),
	                         obj_classes=dataset.object_classes,
	                         enc_layer_num=conf.enc_layer,
	                         dec_layer_num=conf.dec_layer).to(device=gpu_device)
	model = load_model_from_checkpoint(model, conf, gpu_device)
	return model


def load_common_config(conf, ag_train_data, gpu_device):
	method_name = conf.method_name
	
	print("----------------------------------------------------------")
	print(f"Method name: {method_name}")
	
	if conf.ckpt is not None:
		checkpoint_name = os.path.basename(conf.ckpt).split('.')[0]
		future_frame_loss_num = checkpoint_name.split('_')[-3]
		mode = checkpoint_name.split('_')[-5]
		print(f"Checkpoint name: {checkpoint_name}")
	else:
		future_frame_loss_num = conf.baseline_future
		mode = conf.mode
	print(f"Future frame loss num: {future_frame_loss_num}")
	print(f"Mode: {mode}")
	print("----------------------------------------------------------")
	
	if conf.method_name == "obj_sttran_ant":
		model = load_obj_sttran_ant(conf, ag_train_data, gpu_device)
	elif conf.method_name == "obj_sttran_gen_ant":
		model = load_obj_sttran_gen_ant(conf, ag_train_data, gpu_device)
	elif conf.method_name == "obj_dsgdetr_ant":
		model = load_obj_dsgdetr_ant(conf, ag_train_data, gpu_device)
	elif conf.method_name == "obj_dsgdetr_gen_ant":
		model = load_obj_dsgdetr_gen_ant(conf, ag_train_data, gpu_device)
	elif conf.method_name == "rel_sttran_ant":
		model = load_rel_sttran_ant(conf, ag_train_data, gpu_device)
	elif conf.method_name == "rel_sttran_gen_ant":
		model = load_rel_sttran_gen_ant(conf, ag_train_data, gpu_device)
	elif conf.method_name == "rel_dsgdetr_ant":
		model = load_rel_dsgdetr_ant(conf, ag_train_data, gpu_device)
	elif conf.method_name == "rel_dsgdetr_gen_ant":
		model = load_rel_dsgdetr_gen_ant(conf, ag_train_data, gpu_device)
	else:
		raise ValueError(f"Method name {conf.method_name} not recognized")
	
	model.eval()
	
	matcher = HungarianMatcher(0.5, 1, 1, 0.5)
	matcher.eval()
	
	object_detector = detector(
		train=False,
		object_classes=ag_train_data.object_classes,
		use_SUPPLY=True,
		mode=conf.mode
	).to(device=gpu_device)
	object_detector.eval()
	object_detector.is_train = False
	
	return model, object_detector, future_frame_loss_num, mode, method_name, matcher
