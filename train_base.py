import os

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from constants import Constants as const
from dataloader.supervised.generation.action_genome.ag_features import AGFeatures, cuda_collate_fn
from lib.AdamW import AdamW
from lib.supervised.config import Config
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator


def prepare_optimizer(conf, model):
	if conf.optimizer == const.ADAMW:
		optimizer = AdamW(model.parameters(), lr=conf.lr)
	elif conf.optimizer == const.ADAM:
		optimizer = optim.Adam(model.parameters(), lr=conf.lr)
	elif conf.optimizer == const.SGD:
		optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)
	else:
		raise NotImplementedError
	
	scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4,
	                              threshold_mode="abs", min_lr=1e-7)
	return optimizer, scheduler


def fetch_loss_functions():
	bce_loss = nn.BCELoss()
	ce_loss = nn.CrossEntropyLoss()
	mlm_loss = nn.MultiLabelMarginLoss()
	bbox_loss = nn.SmoothL1Loss()
	abs_loss = nn.L1Loss()
	mse_loss = nn.MSELoss()
	return bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss


def fetch_train_basic_config():
	conf = Config()
	print('The CKPT saved here:', conf.save_path)
	if not os.path.exists(conf.save_path):
		os.mkdir(conf.save_path)
	print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
	for i in conf.args:
		print(i, ':', conf.args[i])
	
	# Set the preferred device
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	ag_features_train = AGFeatures(
		mode=conf.mode,
		data_split=const.TRAIN,
		device=device,
		data_path=conf.data_path,
		is_compiled_together=False,
		filter_nonperson_box_frame=True,
		filter_small_box=False if conf.mode == const.PREDCLS else True
	)
	
	dataloader_train = DataLoader(
		ag_features_train,
		shuffle=True,
		collate_fn=cuda_collate_fn,
		pin_memory=False
	)
	
	ag_features_test = AGFeatures(
		mode=conf.mode,
		data_split=const.TEST,
		device=device,
		data_path=conf.data_path,
		is_compiled_together=False,
		filter_nonperson_box_frame=True,
		filter_small_box=False if conf.mode == const.PREDCLS else True
	)
	
	dataloader_test = DataLoader(
		ag_features_test,
		shuffle=False,
		collate_fn=cuda_collate_fn,
		pin_memory=False
	)
	
	gpu_device = torch.device("cuda:0")
	
	evaluator = BasicSceneGraphEvaluator(
		mode=conf.mode,
		AG_object_classes=ag_features_train.object_classes,
		AG_all_predicates=ag_features_train.relationship_classes,
		AG_attention_predicates=ag_features_train.attention_relationships,
		AG_spatial_predicates=ag_features_train.spatial_relationships,
		AG_contacting_predicates=ag_features_train.contacting_relationships,
		iou_threshold=0.5,
		save_file=os.path.join(conf.save_path, const.PROGRESS_TEXT_FILE),
		constraint='with'
	)
	
	return conf, dataloader_train, dataloader_test, gpu_device, evaluator, ag_features_train, ag_features_test
