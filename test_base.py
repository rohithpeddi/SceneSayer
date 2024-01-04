import csv
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


def evaluate_anticipated_future_frame_scene_graph(gt, pred, future_frame_count, is_modified_gt, future_evaluators, future_evaluators_modified_gt):
	future_frame_count_list = [1, 2, 3, 4, 5]
	if is_modified_gt:
		for reference_frame_count in future_frame_count_list:
			if reference_frame_count >= future_frame_count:
				evaluators = future_evaluators_modified_gt[reference_frame_count]
				evaluators[0].evaluate_scene_graph(gt, pred)
				evaluators[1].evaluate_scene_graph(gt, pred)
				evaluators[2].evaluate_scene_graph(gt, pred)
	else:
		for reference_frame_count in future_frame_count_list:
			if reference_frame_count >= future_frame_count:
				evaluators = future_evaluators[reference_frame_count]
				evaluators[0].evaluate_scene_graph(gt, pred)
				evaluators[1].evaluate_scene_graph(gt, pred)
				evaluators[2].evaluate_scene_graph(gt, pred)


def collate_evaluation_stats(with_constraint_evaluator_stats, no_constraint_evaluator_stats,
                             semi_constraint_evaluator_stats):
	collated_stats = [
		with_constraint_evaluator_stats["recall"][10],
		with_constraint_evaluator_stats["recall"][20],
		with_constraint_evaluator_stats["recall"][50],
		with_constraint_evaluator_stats["mean_recall"][10],
		with_constraint_evaluator_stats["mean_recall"][20],
		with_constraint_evaluator_stats["mean_recall"][50],
		with_constraint_evaluator_stats["harmonic_mean_recall"][10],
		with_constraint_evaluator_stats["harmonic_mean_recall"][20],
		with_constraint_evaluator_stats["harmonic_mean_recall"][50],
		no_constraint_evaluator_stats["recall"][10],
		no_constraint_evaluator_stats["recall"][20],
		no_constraint_evaluator_stats["recall"][50],
		no_constraint_evaluator_stats["mean_recall"][10],
		no_constraint_evaluator_stats["mean_recall"][20],
		no_constraint_evaluator_stats["mean_recall"][50],
		no_constraint_evaluator_stats["harmonic_mean_recall"][10],
		no_constraint_evaluator_stats["harmonic_mean_recall"][20],
		no_constraint_evaluator_stats["harmonic_mean_recall"][50],
		semi_constraint_evaluator_stats["recall"][10],
		semi_constraint_evaluator_stats["recall"][20],
		semi_constraint_evaluator_stats["recall"][50],
		semi_constraint_evaluator_stats["mean_recall"][10],
		semi_constraint_evaluator_stats["mean_recall"][20],
		semi_constraint_evaluator_stats["mean_recall"][50],
		semi_constraint_evaluator_stats["harmonic_mean_recall"][10],
		semi_constraint_evaluator_stats["harmonic_mean_recall"][20],
		semi_constraint_evaluator_stats["harmonic_mean_recall"][50]
	]
	
	return collated_stats


def write_future_evaluators_stats(mode, future_frame_loss_num, method_name, future_evaluators):
	results_dir = os.path.join(os.getcwd(), 'results')
	mode_results_dir = os.path.join(results_dir, mode)
	os.makedirs(mode_results_dir, exist_ok=True)
	
	num_future_frame_evaluations = [1, 2, 3, 4, 5]
	for future_frame_evaluation in num_future_frame_evaluations:
		results_file_path = os.path.join(mode_results_dir,
		                                 f'{mode}_train_{future_frame_loss_num}_evaluation_{future_frame_evaluation}.csv')
		with_constraint_evaluator_stats = future_evaluators[future_frame_evaluation][0].fetch_stats_json()
		no_constraint_evaluator_stats = future_evaluators[future_frame_evaluation][1].fetch_stats_json()
		semi_constraint_evaluator_stats = future_evaluators[future_frame_evaluation][2].fetch_stats_json()
		collated_stats = [method_name]
		collated_stats.extend(collate_evaluation_stats(with_constraint_evaluator_stats, no_constraint_evaluator_stats,
		                                               semi_constraint_evaluator_stats))
		
		file_exist = os.path.isfile(results_file_path)
		
		with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
			writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
			if not file_exist:
				writer.writerow([
					"Method Name", "R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
					"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
					"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50"])
			writer.writerow(collated_stats)


def write_gen_evaluators_stats(mode, future_frame_loss_num, method_name, gen_evaluators):
	results_dir = os.path.join(os.getcwd(), 'results')
	mode_results_dir = os.path.join(results_dir, mode)
	os.makedirs(mode_results_dir, exist_ok=True)
	
	results_file_path = os.path.join(mode_results_dir, f'{mode}_train_{future_frame_loss_num}_generation_impact.csv')
	with_constraint_evaluator_stats = gen_evaluators[0].fetch_stats_json()
	no_constraint_evaluator_stats = gen_evaluators[1].fetch_stats_json()
	semi_constraint_evaluator_stats = gen_evaluators[2].fetch_stats_json()
	collated_stats = [method_name]
	collated_stats.extend(collate_evaluation_stats(with_constraint_evaluator_stats, no_constraint_evaluator_stats,
	                                               semi_constraint_evaluator_stats))
	
	file_exists = os.path.isfile(results_file_path)
	
	with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
		writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
		if not file_exists:
			writer.writerow([
				"Method Name", "R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
				"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
				"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50"])
		writer.writerow(collated_stats)


def write_percentage_evaluators_stats(mode, future_frame_loss_num, method_name, percentage_evaluators, context_fraction):
	results_dir = os.path.join(os.getcwd(), 'results')
	mode_results_dir = os.path.join(results_dir, mode)
	os.makedirs(mode_results_dir, exist_ok=True)
	
	results_file_path = os.path.join(mode_results_dir, f'{mode}_train_{future_frame_loss_num}_percentage_evaluation_{context_fraction}.csv')
	with_constraint_evaluator_stats = percentage_evaluators[context_fraction][0].fetch_stats_json()
	no_constraint_evaluator_stats = percentage_evaluators[context_fraction][1].fetch_stats_json()
	semi_constraint_evaluator_stats = percentage_evaluators[context_fraction][2].fetch_stats_json()
	collated_stats = [method_name]
	collated_stats.extend(collate_evaluation_stats(with_constraint_evaluator_stats, no_constraint_evaluator_stats,
	                                               semi_constraint_evaluator_stats))
	
	file_exists = os.path.isfile(results_file_path)
	
	with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
		writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
		if not file_exists:
			writer.writerow([
				"Method Name", "R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
				"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
				"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50"])
		writer.writerow(collated_stats)
