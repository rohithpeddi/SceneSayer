import os

import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from constants import Constants as const
from dataloader.supervised.generation.action_genome.ag_dataset import AG
from dataloader.supervised.generation.action_genome.ag_dataset import cuda_collate_fn as ag_data_cuda_collate_fn
from dataloader.supervised.generation.action_genome.ag_features import AGFeatures
from dataloader.supervised.generation.action_genome.ag_features import cuda_collate_fn as ag_features_cuda_collate_fn
from lib.supervised.biased.sga.ODE import ODE
from lib.supervised.biased.sga.SDE import SDE
from lib.supervised.biased.sga.baseline_anticipation import BaselineWithAnticipation
from lib.supervised.biased.sga.baseline_anticipation_gen_loss import BaselineWithAnticipationGenLoss
from lib.supervised.config import Config
import networkx as nx

from test_base import generate_test_config_metadata


def load_features_data(dataset, video_id):
	entry = dataset.fetch_video_data(video_id)
	return entry


def load_raw_data(dataset, video_id):
	img_tensor, im_info, gt_boxes, num_boxes, index = dataset.fetch_video_data(video_id)
	return img_tensor, im_info, gt_boxes, num_boxes, index


def load_model(model_name, conf, dataset, gpu_device):
	if model_name == "baseline_so":
		model = BaselineWithAnticipation(mode=conf.mode,
		                                 attention_class_num=len(dataset.attention_relationships),
		                                 spatial_class_num=len(dataset.spatial_relationships),
		                                 contact_class_num=len(dataset.contacting_relationships),
		                                 obj_classes=dataset.object_classes,
		                                 enc_layer_num=conf.enc_layer,
		                                 dec_layer_num=conf.dec_layer).to(device=gpu_device)
	elif model_name == "baseline_so_gen_loss":
		model = BaselineWithAnticipationGenLoss(mode=conf.mode,
		                                        attention_class_num=len(dataset.attention_relationships),
		                                        spatial_class_num=len(dataset.spatial_relationships),
		                                        contact_class_num=len(dataset.contacting_relationships),
		                                        obj_classes=dataset.object_classes,
		                                        enc_layer_num=conf.enc_layer,
		                                        dec_layer_num=conf.dec_layer).to(device=gpu_device)
	elif model_name == "ode":
		model = ODE(mode=conf.mode,
		            attention_class_num=len(dataset.attention_relationships),
		            spatial_class_num=len(dataset.spatial_relationships),
		            contact_class_num=len(dataset.contacting_relationships),
		            obj_classes=dataset.object_classes,
		            enc_layer_num=conf.enc_layer,
		            dec_layer_num=conf.dec_layer,
		            max_window=conf.max_window).to(device=gpu_device)
	elif model_name == "sde":
		brownian_size = conf.brownian_size
		model = SDE(mode=conf.mode,
		            attention_class_num=len(dataset.attention_relationships),
		            spatial_class_num=len(dataset.spatial_relationships),
		            contact_class_num=len(dataset.contacting_relationships),
		            obj_classes=dataset.object_classes,
		            enc_layer_num=conf.enc_layer,
		            dec_layer_num=conf.dec_layer,
		            max_window=conf.max_window,
		            brownian_size=brownian_size).to(device=gpu_device)
	
	ckpt = torch.load(conf.ckpt, map_location=gpu_device)
	model.load_state_dict(ckpt[f'{model_name}_state_dict'], strict=False)
	print(f"Loaded model from checkpoint {conf.ckpt}")
	model.eval()
	return model


def main():
	conf, device, gpu_device = generate_test_config_metadata()
	if conf.use_raw_data:
		dataset, video_id_index_map, dataloader_test = load_action_genome_dataset(conf.data_path, conf)
	else:
		dataset, video_id_index_map, dataloader_test = load_action_genome_features_dataset(conf.data_path, conf, device)
	
	model_name = conf.method_name
	mode = conf.mode
	model = load_model(model_name, conf, dataset, gpu_device)
	
	for video_id in video_id_list:
		if conf.use_raw_data:
			img_tensor, im_info, gt_boxes, num_boxes, index = load_raw_data(dataset, video_id_index_map[video_id])
		else:
			entry = load_features_data(dataset, video_id_index_map[video_id])


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
	
	dataloader_test = DataLoader(
		ag_test_data,
		shuffle=False,
		collate_fn=ag_data_cuda_collate_fn,
		pin_memory=False
	)
	
	video_id_index_map = {}
	for index, video_gt_annotation in enumerate(ag_test_data.gt_annotations):
		video_id = video_gt_annotation[0][0]['frame'].split(".")[0]
		video_id_index_map[video_id] = index
	
	return ag_test_data, video_id_index_map, dataloader_test


def draw_and_save_graph(graph, video_id, frame_idx, dataset):
	plt.figure(figsize=(8, 6))
	
	# Assuming 'graph' is your NetworkX MultiGraph object
	
	# Positions for all nodes
	pos = nx.spring_layout(graph)
	
	# Draw nodes
	nx.draw_networkx_nodes(graph, pos)
	
	# Extract different types of relationships
	attention_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['label'] in dataset.attention_relationships]
	spatial_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['label'] in dataset.spatial_relationships]
	contacting_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['label'] in dataset.contacting_relationships]
	
	# Draw edges for different types of relationships
	nx.draw_networkx_edges(graph, pos, edgelist=attention_edges, edge_color='red', label='Attention Relationships')
	nx.draw_networkx_edges(graph, pos, edgelist=spatial_edges, edge_color='blue', label='Spatial Relationships')
	nx.draw_networkx_edges(graph, pos, edgelist=contacting_edges, edge_color='green', label='Contacting Relationships')
	
	# Draw node labels
	nx.draw_networkx_labels(graph, pos)
	
	# Add legend
	plt.legend()
	
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
			graph.add_node(0, label=const.PERSON)
			subject_node_idx = 0
			for object_node_idx in range(1, len(frame_gt_annotation)):
				object_node_class = dataset.object_classes[frame_gt_annotation[object_node_idx]['class']]
				graph.add_node(object_node_idx, label=object_node_class)
				
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
						graph.add_edge(subject_node_idx, object_node_idx,
						               label=dataset.attention_relationships[attention_relationship_idx])
				for spatial_relationship_idx in range(len(spatial_relationships)):
					graph.add_edge(subject_node_idx, object_node_idx,
					               label=dataset.spatial_relationships[spatial_relationship_idx])
				for contacting_relationship_idx in range(len(contacting_relationships)):
					graph.add_edge(subject_node_idx, object_node_idx,
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
