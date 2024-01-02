import os
import torch
from torch.utils.data import DataLoader

from dataloader.supervised.generation.action_genome.ag_features import AGFeatures
from dataloader.supervised.generation.action_genome.ag_features import cuda_collate_fn as ag_features_cuda_collate_fn
from dataloader.supervised.generation.action_genome.ag_dataset import AG
from dataloader.supervised.generation.action_genome.ag_dataset import cuda_collate_fn as ag_data_cuda_collate_fn
from constants import Constants as const
from lib.supervised.config import Config
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator


def generate_test_config_metadata():
	conf = Config()
	for i in conf.args:
		print(i, ':', conf.args[i])
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	gpu_device = torch.device("cuda:0")
	
	return conf, device, gpu_device


def generate_test_dataset_metadata(conf, device):
	ag_test_data = AGFeatures(
		mode=conf.mode,
		data_split=const.TEST,
		device=device,
		data_path=conf.data_path,
		is_compiled_together=False,
		filter_nonperson_box_frame=True,
		filter_small_box=False if conf.mode == const.PREDCLS else True
	)
	
	dataloader_test = DataLoader(
		ag_test_data,
		shuffle=False,
		collate_fn=ag_features_cuda_collate_fn,
		pin_memory=False
	)
	
	return ag_test_data, dataloader_test


def generate_evaluator_set(conf, dataset):
	# Evaluators order - [With Constraint, No Constraint, Semi Constraint]
	evaluators = []
	
	with_constraint_evaluator = BasicSceneGraphEvaluator(
		mode=conf.mode,
		AG_object_classes=dataset.object_classes,
		AG_all_predicates=dataset.relationship_classes,
		AG_attention_predicates=dataset.attention_relationships,
		AG_spatial_predicates=dataset.spatial_relationships,
		AG_contacting_predicates=dataset.contacting_relationships,
		iou_threshold=0.5,
		constraint='with')
	
	no_constraint_evaluator = BasicSceneGraphEvaluator(
		mode=conf.mode,
		AG_object_classes=dataset.object_classes,
		AG_all_predicates=dataset.relationship_classes,
		AG_attention_predicates=dataset.attention_relationships,
		AG_spatial_predicates=dataset.spatial_relationships,
		AG_contacting_predicates=dataset.contacting_relationships,
		iou_threshold=0.5,
		constraint='no')
	
	semi_constraint_evaluator = BasicSceneGraphEvaluator(
		mode=conf.mode,
		AG_object_classes=dataset.object_classes,
		AG_all_predicates=dataset.relationship_classes,
		AG_attention_predicates=dataset.attention_relationships,
		AG_spatial_predicates=dataset.spatial_relationships,
		AG_contacting_predicates=dataset.contacting_relationships,
		iou_threshold=0.5,
		constraint='semi', semi_threshold=0.9)
	
	evaluators.append(with_constraint_evaluator)
	evaluators.append(no_constraint_evaluator)
	evaluators.append(semi_constraint_evaluator)
	
	return evaluators


def fetch_test_basic_config():
	conf, device, gpu_device = generate_test_config_metadata()
	
	ag_features_test, dataloader_test = generate_test_dataset_metadata(conf, device)
	
	evaluators = generate_evaluator_set(conf, ag_features_test)
	
	return ag_features_test, dataloader_test, evaluators[0], evaluators[1], evaluators[2], gpu_device, conf


def generate_required_test_evaluators(conf, ag_test_data):
	gen_evaluators = generate_evaluator_set(conf, ag_test_data)
	
	future_frame_count_list = [1, 2, 3, 4, 5]
	future_evaluators = {}
	for future_frame_count in future_frame_count_list:
		future_evaluators[future_frame_count] = generate_evaluator_set(conf, ag_test_data)
	
	future_evaluators_modified_gt = {}
	for future_frame_count in future_frame_count_list:
		future_evaluators_modified_gt[future_frame_count] = generate_evaluator_set(conf, ag_test_data)
	
	percentage_count_list = [0.3, 0.5, 0.7, 0.9]
	percentage_evaluators = {}
	for percentage_count in percentage_count_list:
		percentage_evaluators[percentage_count] = generate_evaluator_set(conf, ag_test_data)
	
	percentage_evaluators_modified_gt = {}
	for percentage_count in percentage_count_list:
		percentage_evaluators_modified_gt[percentage_count] = generate_evaluator_set(conf, ag_test_data)
	
	return gen_evaluators, future_evaluators, future_evaluators_modified_gt, percentage_evaluators, percentage_evaluators_modified_gt


def fetch_diffeq_test_basic_config():
	# Evaluators order - [With Constraint, No Constraint, Semi Constraint]
	conf, device, gpu_device = generate_test_config_metadata()
	
	ag_test_data, dataloader_test = generate_test_dataset_metadata(conf, device)
	
	(gen_evaluators, future_evaluators,
	 future_evaluators_modified_gt, percentage_evaluators,
	 percentage_evaluators_modified_gt) = generate_required_test_evaluators(conf, ag_test_data)
	
	return ag_test_data, dataloader_test, gen_evaluators, future_evaluators, future_evaluators_modified_gt, percentage_evaluators, percentage_evaluators_modified_gt, gpu_device, conf


def fetch_transformer_test_basic_config():
	conf, device, gpu_device = generate_test_config_metadata()
	
	if not conf.use_raw_data:
		ag_test_data = AGFeatures(
			mode=conf.mode,
			data_split=const.TEST,
			device=device,
			data_path=conf.data_path,
			is_compiled_together=False,
			filter_nonperson_box_frame=True,
			filter_small_box=False if conf.mode == const.PREDCLS else True
		)
		
		dataloader_test = DataLoader(
			ag_test_data,
			shuffle=False,
			collate_fn=ag_features_cuda_collate_fn,
			pin_memory=False
		)
	else:
		ag_test_data = AG(
			phase="test",
			datasize=conf.datasize,
			data_path=conf.data_path,
			filter_nonperson_box_frame=True,
			filter_small_box=False if conf.mode == 'predcls' else True
		)
		
		dataloader_test = DataLoader(
			ag_test_data,
			shuffle=False,
			collate_fn=ag_data_cuda_collate_fn,
			pin_memory=False
		)
	
	(gen_evaluators, future_evaluators,
	 future_evaluators_modified_gt, percentage_evaluators,
	 percentage_evaluators_modified_gt) = generate_required_test_evaluators(conf, ag_test_data)
	
	return ag_test_data, dataloader_test, gen_evaluators, future_evaluators, future_evaluators_modified_gt, percentage_evaluators, percentage_evaluators_modified_gt, gpu_device, conf
