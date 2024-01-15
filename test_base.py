import csv
import os
from typing import List

import networkx as nx
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from torch.utils.data import DataLoader

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result, ResultDetails, Metrics
from dataloader.supervised.generation.action_genome.ag_features import AGFeatures
from dataloader.supervised.generation.action_genome.ag_features import cuda_collate_fn as ag_features_cuda_collate_fn
from dataloader.supervised.generation.action_genome.ag_dataset import AG
from dataloader.supervised.generation.action_genome.ag_dataset import cuda_collate_fn as ag_data_cuda_collate_fn
from constants import Constants as const
from lib.supervised.config import Config
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator


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


def generate_evaluator_set_unlocalized(conf, dataset):
	# Evaluators order - [With Constraint, No Constraint, Semi Constraint]
	evaluators = []
	iou_threshold = 0.0
	with_constraint_evaluator = BasicSceneGraphEvaluator(
		mode=conf.mode,
		AG_object_classes=dataset.object_classes,
		AG_all_predicates=dataset.relationship_classes,
		AG_attention_predicates=dataset.attention_relationships,
		AG_spatial_predicates=dataset.spatial_relationships,
		AG_contacting_predicates=dataset.contacting_relationships,
		iou_threshold=iou_threshold,
		constraint='with')
	
	no_constraint_evaluator = BasicSceneGraphEvaluator(
		mode=conf.mode,
		AG_object_classes=dataset.object_classes,
		AG_all_predicates=dataset.relationship_classes,
		AG_attention_predicates=dataset.attention_relationships,
		AG_spatial_predicates=dataset.spatial_relationships,
		AG_contacting_predicates=dataset.contacting_relationships,
		iou_threshold=iou_threshold,
		constraint='no')
	
	semi_constraint_evaluator = BasicSceneGraphEvaluator(
		mode=conf.mode,
		AG_object_classes=dataset.object_classes,
		AG_all_predicates=dataset.relationship_classes,
		AG_attention_predicates=dataset.attention_relationships,
		AG_spatial_predicates=dataset.spatial_relationships,
		AG_contacting_predicates=dataset.contacting_relationships,
		iou_threshold=iou_threshold,
		constraint='semi', semi_threshold=0.9)
	
	evaluators.append(with_constraint_evaluator)
	evaluators.append(no_constraint_evaluator)
	evaluators.append(semi_constraint_evaluator)
	
	return evaluators


def fetch_test_basic_config():
	conf, device, gpu_device = generate_test_config_metadata()
	
	ag_features_test, dataloader_test = generate_test_dataset_metadata(conf, device)
	
	evaluators = generate_evaluator_set_unlocalized(conf, ag_features_test)
	
	return ag_features_test, dataloader_test, evaluators[0], evaluators[1], evaluators[2], gpu_device, conf


def generate_required_test_evaluators(conf, ag_test_data):
	gen_evaluators = generate_evaluator_set_unlocalized(conf, ag_test_data)
	
	future_frame_count_list = [1, 2, 3, 4, 5]
	future_evaluators = {}
	for future_frame_count in future_frame_count_list:
		future_evaluators[future_frame_count] = generate_evaluator_set_unlocalized(conf, ag_test_data)
	
	future_evaluators_modified_gt = {}
	for future_frame_count in future_frame_count_list:
		future_evaluators_modified_gt[future_frame_count] = generate_evaluator_set_unlocalized(conf, ag_test_data)
	
	percentage_count_list = [0.3, 0.5, 0.7, 0.9]
	percentage_evaluators = {}
	for percentage_count in percentage_count_list:
		percentage_evaluators[percentage_count] = generate_evaluator_set_unlocalized(conf, ag_test_data)
	
	percentage_evaluators_modified_gt = {}
	for percentage_count in percentage_count_list:
		percentage_evaluators_modified_gt[percentage_count] = generate_evaluator_set_unlocalized(conf, ag_test_data)
	
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


def evaluate_anticipated_future_frame_scene_graph(gt, pred, future_frame_count, is_modified_gt, future_evaluators,
                                                  future_evaluators_modified_gt):
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
		with_constraint_evaluator_stats["recall"][100],
		with_constraint_evaluator_stats["mean_recall"][10],
		with_constraint_evaluator_stats["mean_recall"][20],
		with_constraint_evaluator_stats["mean_recall"][50],
		with_constraint_evaluator_stats["mean_recall"][100],
		with_constraint_evaluator_stats["harmonic_mean_recall"][10],
		with_constraint_evaluator_stats["harmonic_mean_recall"][20],
		with_constraint_evaluator_stats["harmonic_mean_recall"][50],
		with_constraint_evaluator_stats["harmonic_mean_recall"][100],
		no_constraint_evaluator_stats["recall"][10],
		no_constraint_evaluator_stats["recall"][20],
		no_constraint_evaluator_stats["recall"][50],
		no_constraint_evaluator_stats["recall"][100],
		no_constraint_evaluator_stats["mean_recall"][10],
		no_constraint_evaluator_stats["mean_recall"][20],
		no_constraint_evaluator_stats["mean_recall"][50],
		no_constraint_evaluator_stats["mean_recall"][100],
		no_constraint_evaluator_stats["harmonic_mean_recall"][10],
		no_constraint_evaluator_stats["harmonic_mean_recall"][20],
		no_constraint_evaluator_stats["harmonic_mean_recall"][50],
		no_constraint_evaluator_stats["harmonic_mean_recall"][100],
		semi_constraint_evaluator_stats["recall"][10],
		semi_constraint_evaluator_stats["recall"][20],
		semi_constraint_evaluator_stats["recall"][50],
		semi_constraint_evaluator_stats["recall"][100],
		semi_constraint_evaluator_stats["mean_recall"][10],
		semi_constraint_evaluator_stats["mean_recall"][20],
		semi_constraint_evaluator_stats["mean_recall"][50],
		semi_constraint_evaluator_stats["mean_recall"][100],
		semi_constraint_evaluator_stats["harmonic_mean_recall"][10],
		semi_constraint_evaluator_stats["harmonic_mean_recall"][20],
		semi_constraint_evaluator_stats["harmonic_mean_recall"][50],
		semi_constraint_evaluator_stats["harmonic_mean_recall"][100]
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
					"Method Name",
					"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
					"hR@100"
					"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
					"hR@100",
					"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
					"hR@100"
				])
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
				"Method Name",
				"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
				"hR@100"
				"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
				"hR@100",
				"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
				"hR@100"
			])
		writer.writerow(collated_stats)


def write_evaluators_stats_dysgg(mode, method_name, gen_evaluators):
	results_dir = os.path.join(os.getcwd(), 'results')
	mode_results_dir = os.path.join(results_dir, mode)
	os.makedirs(mode_results_dir, exist_ok=True)
	
	results_file_path = os.path.join(mode_results_dir, f'{mode}_{method_name}_dysgg.csv')
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
				"Method Name",
				"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
				"hR@100"
				"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
				"hR@100",
				"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
				"hR@100"
			])
		writer.writerow(collated_stats)


def write_percentage_evaluators_stats(mode, future_frame_loss_num, method_name, percentage_evaluators,
                                      context_fraction):
	results_dir = os.path.join(os.getcwd(), 'results')
	mode_results_dir = os.path.join(results_dir, mode)
	os.makedirs(mode_results_dir, exist_ok=True)
	
	results_file_path = os.path.join(mode_results_dir,
	                                 f'{mode}_train_{future_frame_loss_num}_percentage_evaluation_{context_fraction}.csv')
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
				"Method Name",
				"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
				"hR@100"
				"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
				"hR@100",
				"R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
				"hR@100"
			])
		writer.writerow(collated_stats)


def fetch_dysgg_test_basic_config():
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
	
	gen_evaluators = generate_evaluator_set_unlocalized(conf, ag_test_data)
	return ag_test_data, dataloader_test, gen_evaluators, gpu_device, conf


def prepare_metrics_from_stats(evaluator_stats):
	metrics = Metrics(
		evaluator_stats["recall"][10],
		evaluator_stats["recall"][20],
		evaluator_stats["recall"][50],
		evaluator_stats["recall"][100],
		evaluator_stats["mean_recall"][10],
		evaluator_stats["mean_recall"][20],
		evaluator_stats["mean_recall"][50],
		evaluator_stats["mean_recall"][100],
		evaluator_stats["harmonic_mean_recall"][10],
		evaluator_stats["harmonic_mean_recall"][20],
		evaluator_stats["harmonic_mean_recall"][50],
		evaluator_stats["harmonic_mean_recall"][100]
	)
	
	return metrics


def send_results_to_firebase(evaluators: List[BasicSceneGraphEvaluator], task_name: str, method_name: str, mode: str,
                             train_num_future_frames=None, context_fraction=None, test_future_frame_count=None):
	assert task_name is not None and method_name is not None and mode is not None
	db_service = FirebaseService()
	
	result = Result(
		task_name=task_name,
		method_name=method_name,
		mode=mode,
	)
	
	if train_num_future_frames is not None:
		result.train_num_future_frames = train_num_future_frames
	
	if context_fraction is not None:
		result.context_fraction = context_fraction
	
	if test_future_frame_count is not None:
		result.test_num_future_frames = test_future_frame_count
	
	result_details = ResultDetails()
	with_constraint_metrics = prepare_metrics_from_stats(evaluators[0].fetch_stats_json())
	no_constraint_metrics = prepare_metrics_from_stats(evaluators[1].fetch_stats_json())
	semi_constraint_metrics = prepare_metrics_from_stats(evaluators[2].fetch_stats_json())
	
	result_details.add_with_constraint_metrics(with_constraint_metrics)
	result_details.add_no_constraint_metrics(no_constraint_metrics)
	result_details.add_semi_constraint_metrics(semi_constraint_metrics)
	
	result.add_result_details(result_details)
	
	print("Saving result: ", result.result_id)
	db_service.update_result(result.result_id, result.to_dict())
	print("Saved result: ", result.result_id)
	
	return result


def send_future_evaluators_stats_to_firebase(future_evaluators, mode, method_name, future_frame_loss_num):
	try:
		future_frame_count_list = [1, 2, 3, 4, 5]
		for reference_frame_count in future_frame_count_list:
			reference_future_evaluators = future_evaluators[reference_frame_count]
			send_results_to_firebase(
				evaluators=reference_future_evaluators,
				task_name="sga",
				method_name=method_name,
				mode=mode,
				train_num_future_frames=future_frame_loss_num,
				context_fraction=None,
				test_future_frame_count=reference_frame_count
			)
	except Exception as e:
		print(f"Error in sending future evaluator results to firebase {e}")


def send_gen_evaluators_stats_to_firebase(gen_evaluators, mode, method_name, future_frame_loss_num):
	try:
		send_results_to_firebase(
			evaluators=gen_evaluators,
			task_name="sga",
			method_name=method_name,
			mode=mode,
			train_num_future_frames=future_frame_loss_num,
			context_fraction=None,
			test_future_frame_count=None
		)
	except Exception as e:
		print(f"Error in sending generation evaluator results to firebase {e}")


def send_percentage_evaluators_stats_to_firebase(percentage_evaluators, mode, method_name, future_frame_loss_num,
                                                 context_fraction):
	try:
		send_results_to_firebase(
			evaluators=percentage_evaluators[context_fraction],
			task_name="sga",
			method_name=method_name,
			mode=mode,
			train_num_future_frames=future_frame_loss_num,
			context_fraction=context_fraction,
			test_future_frame_count=None
		)
	except Exception as e:
		print(f"Error in sending percentage evaluator results to firebase {e}")


def prepare_prediction_graph(predictions_map, dataset, video_id, model_name, constraint_type, mode):
	# Loop through each frame in the video
	for frame_idx, pred_tuple_array in predictions_map.items():
		pred_set = set()
		for idx in range(len(pred_tuple_array)):
			pred_set.add(list(pred_tuple_array[idx]))
		
		graph = nx.MultiGraph()
		
		subject_classes = set()
		object_classes = set()
		for pred_list in pred_set:
			subject_class = dataset.object_classes[pred_list[0]]
			object_class = dataset.object_classes[pred_list[1]]
			subject_classes.add(subject_class)
			object_classes.add(object_class)
		
		for subject_class in subject_classes:
			graph.add_node(subject_class, label=subject_class)
		
		for object_class in object_classes:
			graph.add_node(object_class, label=object_class)
		
		for pred_list in pred_set:
			subject_class = dataset.object_classes[pred_list[0]]
			object_class = dataset.object_classes[pred_list[1]]
			predicate_class = dataset.relationships[pred_list[2]]
			graph.add_edge(subject_class, object_class, label=predicate_class)
			
		draw_and_save_graph(graph, video_id, frame_idx, model_name, constraint_type, mode)


def draw_and_save_graph(graph, video_id, frame_idx, model_name, constraint_type, mode):
	plt.figure(figsize=(12, 12))
	
	pos = nx.spring_layout(graph, seed=42)  # positions for all nodes, with a fixed layout
	
	# Draw nodes and labels
	nx.draw_networkx_nodes(graph, pos, node_size=700)
	nx.draw_networkx_labels(graph, pos)
	
	# Custom drawing of the edges using FancyArrowPatch
	for u, v, key, data in graph.edges(keys=True, data=True):
		# Determine if there are multiple edges and calculate offset
		num_edges = graph.number_of_edges(u, v)
		edge_count = sum(1 for _ in graph[u][v])
		offset = 0.13 * (key - edge_count // 2)  # Offset for curvature
		
		# Parameters for the FancyArrowPatch
		arrow_options = {
			'arrowstyle': '-',
			'connectionstyle': f"arc3,rad={offset}",
			'color': 'black',
			'linewidth': 1
		}
		
		# Draw the edge with curvature
		edge = FancyArrowPatch(pos[u], pos[v], **arrow_options)
		plt.gca().add_patch(edge)
		
		# Improved calculation for the position of the edge label
		label_pos_x = (pos[u][0] + pos[v][0]) / 2 + offset * 0.75 * (pos[v][1] - pos[u][1])
		label_pos_y = (pos[u][1] + pos[v][1]) / 2 - offset * 0.75 * (pos[v][0] - pos[u][0])
		plt.text(label_pos_x, label_pos_y, str(data['label']), color='blue', fontsize=10, ha='center', va='center')
	
	# Save graph
	file_name = "{}_{}.png".format(video_id, frame_idx)
	file_directory_path = os.path.join(os.path.dirname(__file__), "analysis", "docs",
	                                   "qualitative_results", model_name, video_id, mode, constraint_type)
	os.makedirs(file_directory_path, exist_ok=True)
	file_path = os.path.join(file_directory_path, file_name)
	plt.savefig(file_path)
