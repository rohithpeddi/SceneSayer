import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from torch.utils.data import DataLoader

from constants import Constants as const
from dataloader.supervised.generation.action_genome.ag_dataset import AG
from dataloader.supervised.generation.action_genome.ag_dataset import cuda_collate_fn as ag_data_cuda_collate_fn
from dataloader.supervised.generation.action_genome.ag_features import AGFeatures
from dataloader.supervised.generation.action_genome.ag_features import cuda_collate_fn as ag_features_cuda_collate_fn
# from lib.supervised.biased.sga.ODE import ODE
# from lib.supervised.biased.sga.SDE import SDE
# from lib.supervised.biased.sga.baseline_anticipation import BaselineWithAnticipation
# from lib.supervised.biased.sga.baseline_anticipation_gen_loss import BaselineWithAnticipationGenLoss
from lib.supervised.config import Config
import networkx as nx

from test_base import generate_test_config_metadata


def load_features_data(dataset, video_id):
	entry = dataset.fetch_video_data(video_id)
	return entry


def load_raw_data(dataset, video_id):
	img_tensor, im_info, gt_boxes, num_boxes, index = dataset.fetch_video_data(video_id)
	return img_tensor, im_info, gt_boxes, num_boxes, index


# def load_model(model_name, conf, dataset, gpu_device):
# 	if model_name == "baseline_so":
# 		model = BaselineWithAnticipation(mode=conf.mode,
# 		                                 attention_class_num=len(dataset.attention_relationships),
# 		                                 spatial_class_num=len(dataset.spatial_relationships),
# 		                                 contact_class_num=len(dataset.contacting_relationships),
# 		                                 obj_classes=dataset.object_classes,
# 		                                 enc_layer_num=conf.enc_layer,
# 		                                 dec_layer_num=conf.dec_layer).to(device=gpu_device)
# 	elif model_name == "baseline_so_gen_loss":
# 		model = BaselineWithAnticipationGenLoss(mode=conf.mode,
# 		                                        attention_class_num=len(dataset.attention_relationships),
# 		                                        spatial_class_num=len(dataset.spatial_relationships),
# 		                                        contact_class_num=len(dataset.contacting_relationships),
# 		                                        obj_classes=dataset.object_classes,
# 		                                        enc_layer_num=conf.enc_layer,
# 		                                        dec_layer_num=conf.dec_layer).to(device=gpu_device)
# 	elif model_name == "ode":
# 		model = ODE(mode=conf.mode,
# 		            attention_class_num=len(dataset.attention_relationships),
# 		            spatial_class_num=len(dataset.spatial_relationships),
# 		            contact_class_num=len(dataset.contacting_relationships),
# 		            obj_classes=dataset.object_classes,
# 		            enc_layer_num=conf.enc_layer,
# 		            dec_layer_num=conf.dec_layer,
# 		            max_window=conf.max_window).to(device=gpu_device)
# 	elif model_name == "sde":
# 		brownian_size = conf.brownian_size
# 		model = SDE(mode=conf.mode,
# 		            attention_class_num=len(dataset.attention_relationships),
# 		            spatial_class_num=len(dataset.spatial_relationships),
# 		            contact_class_num=len(dataset.contacting_relationships),
# 		            obj_classes=dataset.object_classes,
# 		            enc_layer_num=conf.enc_layer,
# 		            dec_layer_num=conf.dec_layer,
# 		            max_window=conf.max_window,
# 		            brownian_size=brownian_size).to(device=gpu_device)
#
# 	ckpt = torch.load(conf.ckpt, map_location=gpu_device)
# 	model.load_state_dict(ckpt[f'{model_name}_state_dict'], strict=False)
# 	print(f"Loaded model from checkpoint {conf.ckpt}")
# 	model.eval()
# 	return model
#
#
# def main():
# 	conf, device, gpu_device = generate_test_config_metadata()
# 	if conf.use_raw_data:
# 		dataset, video_id_index_map, dataloader_test = load_action_genome_dataset(conf.data_path, conf)
# 	else:
# 		dataset, video_id_index_map, dataloader_test = load_action_genome_features_dataset(conf.data_path, conf, device)
#
# 	model_name = conf.method_name
# 	mode = conf.mode
# 	model = load_model(model_name, conf, dataset, gpu_device)
#
# 	for video_id in video_id_list:
# 		if conf.use_raw_data:
# 			img_tensor, im_info, gt_boxes, num_boxes, index = load_raw_data(dataset, video_id_index_map[video_id])
# 		else:
# 			entry = load_features_data(dataset, video_id_index_map[video_id])


def load_action_genome_features_dataset(data_path, conf, device):
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
	
	video_id_index_map = {}
	for index, video_gt_annotation in enumerate(ag_test_data.gt_annotations):
		video_id = video_gt_annotation[0][0]['frame'].split(".")[0]
		video_id_index_map[video_id] = index
	
	return ag_test_data, video_id_index_map, dataloader_test


def load_action_genome_dataset(data_path, conf):
	ag_test_data = AG(
		phase=const.TEST,
		datasize=conf.datasize,
		data_path=data_path,
		filter_nonperson_box_frame=True,
		filter_small_box=False if conf.mode == 'predcls' else True
	)
	
	video_id_index_map = {}
	for index, video_gt_annotation in enumerate(ag_test_data.gt_annotations):
		video_id = video_gt_annotation[0][0]['frame'].split(".")[0]
		video_id_index_map[video_id] = index
	
	return ag_test_data, video_id_index_map


def draw_and_save_graph(graph, video_id, frame_idx):
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
	                                   "qualitative_results", "ground_truth_graphs", video_id)
	os.makedirs(file_directory_path, exist_ok=True)
	file_path = os.path.join(file_directory_path, file_name)
	plt.savefig(file_path)


def generate_ground_truth_graphs():
	conf = Config()
	conf.data_path = action_genome_data_path
	dataset, video_id_index_map = load_action_genome_dataset(action_genome_data_path, conf)
	
	for video_id in video_id_list:
		video_gt_annotation = dataset.gt_annotations[video_id_index_map[video_id]]
		print("Loaded data for video {}".format(video_id))
		for frame_gt_annotation in video_gt_annotation:
			print("Loaded data for frame {}".format(frame_gt_annotation[0]['frame']))
			frame_idx = frame_gt_annotation[0]['frame'].split("/")[1][:-4]
			graph = nx.MultiGraph()
			subject_node_class = "Person"
			graph.add_node(subject_node_class, label=const.PERSON)
			
			for object_node_idx in range(1, len(frame_gt_annotation)):
				object_node_class = dataset.object_classes[frame_gt_annotation[object_node_idx]['class']]
				graph.add_node(object_node_class, label=object_node_class)
				
				attention_relationships = frame_gt_annotation[object_node_idx][
					const.ATTENTION_RELATIONSHIP].cpu().numpy()
				spatial_relationships = frame_gt_annotation[object_node_idx][const.SPATIAL_RELATIONSHIP].cpu().numpy()
				contacting_relationships = frame_gt_annotation[object_node_idx][
					const.CONTACTING_RELATIONSHIP].cpu().numpy()
				
				print("Attention Relationships: {}".format(len(attention_relationships)))
				print("Spatial Relationships: {}".format(len(spatial_relationships)))
				print("Contacting Relationships: {}".format(len(contacting_relationships)))
				
				for attention_relationship_idx in range(len(attention_relationships)):
					if attention_relationships[attention_relationship_idx] in [0, 1]:
						graph.add_edge(subject_node_class, object_node_class,
						               label=dataset.attention_relationships[attention_relationship_idx])
				for spatial_relationship_idx in range(len(spatial_relationships)):
					graph.add_edge(subject_node_class, object_node_class,
					               label=dataset.spatial_relationships[spatial_relationship_idx])
				for contacting_relationship_idx in range(len(contacting_relationships)):
					graph.add_edge(subject_node_class, object_node_class,
					               label=dataset.contacting_relationships[contacting_relationship_idx])
			
			draw_and_save_graph(graph, video_id, frame_idx, dataset)


if __name__ == '__main__':
	modes = ["sgdet", "sgcls", "predcls"]
	models = ["baseline_so", "baseline_so_gen_loss", "ode", "sde"]
	context_list = ["0.3", "0.5", "0.7", "0.9"]
	
	action_genome_data_path = r"D:\DATA\OPEN\action_genome"
	video_id_list = ["21F9H", "X95D0", "M18XP", "0A8CF", "LUQWY", "QE4YE", "ENOLD"]
	
	# main()
	generate_ground_truth_graphs()
