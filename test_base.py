import torch
from torch.utils.data import DataLoader

from dataloader.supervised.generation.action_genome.ag_features import AGFeatures, cuda_collate_fn
from constants import Constants as const
from lib.supervised.config import Config
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator


def fetch_test_basic_config():
	conf = Config()
	for i in conf.args:
		print(i, ':', conf.args[i])
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	gpu_device = torch.device("cuda:0")
	
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
	
	with_constraint_evaluator = BasicSceneGraphEvaluator(
		mode=conf.mode,
		AG_object_classes=ag_features_test.object_classes,
		AG_all_predicates=ag_features_test.relationship_classes,
		AG_attention_predicates=ag_features_test.attention_relationships,
		AG_spatial_predicates=ag_features_test.spatial_relationships,
		AG_contacting_predicates=ag_features_test.contacting_relationships,
		iou_threshold=0.5,
		constraint='with')
	
	no_constraint_evaluator = BasicSceneGraphEvaluator(
		mode=conf.mode,
		AG_object_classes=ag_features_test.object_classes,
		AG_all_predicates=ag_features_test.relationship_classes,
		AG_attention_predicates=ag_features_test.attention_relationships,
		AG_spatial_predicates=ag_features_test.spatial_relationships,
		AG_contacting_predicates=ag_features_test.contacting_relationships,
		iou_threshold=0.5,
		constraint='no')
	
	semi_constraint_evaluator = BasicSceneGraphEvaluator(
		mode=conf.mode,
		AG_object_classes=AG_dataset.object_classes,
		AG_all_predicates=AG_dataset.relationship_classes,
		AG_attention_predicates=AG_dataset.attention_relationships,
		AG_spatial_predicates=AG_dataset.spatial_relationships,
		AG_contacting_predicates=AG_dataset.contacting_relationships,
		iou_threshold=0.5,
		constraint='semi', semithreshold=0.9)
	
	return ag_features_test, dataloader_test, with_constraint_evaluator, no_constraint_evaluator, semi_constraint_evaluator, gpu_device, conf
