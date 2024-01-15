import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import DataloaderConstants as const
from logger_config import get_logger

logger = get_logger(__name__)


class AGFeatures(Dataset):
	
	def __init__(
			self,
			mode,
			data_split,
			device,
			data_path=None,
			is_compiled_together=True,
			filter_nonperson_box_frame=True,
			filter_small_box=False,
			features_path=None,
			additional_data_path=None
	):
		self.root_path = data_path
		self.data_split = data_split
		self.mode = mode
		self.device = device
		self.change_device = self.device != torch.device("cuda:0")
		self.is_compiled_together = is_compiled_together
		self.filter_nonperson_box_frame = filter_nonperson_box_frame
		self.filter_small_box = filter_small_box

		self.is_train = True if self.data_split == const.TRAIN else False
		self.entry_mode = self.mode  # Same attribute dictionary for SGCLS and PREDCLS
		
		if features_path is not None:
			self.features_path = features_path
		else:
			self.features_path = os.path.join(self.root_path, const.FEATURES, const.SUPERVISED, self.data_split)
			
		if additional_data_path is not None:
			self.additional_data_path = additional_data_path
		else:
			self.additional_data_path = os.path.join(self.root_path, const.FEATURES, const.SUPERVISED, const.ADDITIONAL, self.data_split)

		logger.info(f"Initializing static data from dataset in {self.mode}")
		self._init_dataset_static_data()
		logger.info(f"Finished processing static data from dataset in {self.mode}")
		
		logger.info(f"Initializing ground truth information from dataset in {self.mode}")
		self._init_gt_info()
		logger.info(f"Finished processing ground truth information from dataset in {self.mode}")
		
		self.video_list = []
		for video_feature_file in os.listdir(self.features_path):
			video_feature_file_path = os.path.join(self.features_path, video_feature_file)
			if self.is_compiled_together:
				if os.path.isfile(video_feature_file_path):
					self.video_list.append(video_feature_file_path)
			else:
				if self.entry_mode in video_feature_file and os.path.isfile(video_feature_file_path):
					self.video_list.append(video_feature_file_path)

		logger.info(f"Finished initializing dataset in {self.mode}")
	
	def _init_dataset_static_data(self):
		self.object_classes = [const.BACKGROUND]
		
		with open(os.path.join(self.root_path, const.ANNOTATIONS, const.OBJECT_CLASSES_FILE), 'r',
		          encoding='utf-8') as f:
			for line in f.readlines():
				line = line.strip('\n')
				self.object_classes.append(line)
		f.close()
		self.object_classes[9] = 'closet/cabinet'
		self.object_classes[11] = 'cup/glass/bottle'
		self.object_classes[23] = 'paper/notebook'
		self.object_classes[24] = 'phone/camera'
		self.object_classes[31] = 'sofa/couch'
		
		# collect relationship classes
		self.relationship_classes = []
		with open(os.path.join(self.root_path, const.ANNOTATIONS, const.RELATIONSHIP_CLASSES_FILE), 'r') as f:
			for line in f.readlines():
				line = line.strip('\n')
				self.relationship_classes.append(line)
		f.close()
		self.relationship_classes[0] = 'looking_at'
		self.relationship_classes[1] = 'not_looking_at'
		self.relationship_classes[5] = 'in_front_of'
		self.relationship_classes[7] = 'on_the_side_of'
		self.relationship_classes[10] = 'covered_by'
		self.relationship_classes[11] = 'drinking_from'
		self.relationship_classes[13] = 'have_it_on_the_back'
		self.relationship_classes[15] = 'leaning_on'
		self.relationship_classes[16] = 'lying_on'
		self.relationship_classes[17] = 'not_contacting'
		self.relationship_classes[18] = 'other_relationship'
		self.relationship_classes[19] = 'sitting_on'
		self.relationship_classes[20] = 'standing_on'
		self.relationship_classes[25] = 'writing_on'
		
		self.attention_relationships = self.relationship_classes[0:3]
		self.spatial_relationships = self.relationship_classes[3:9]
		self.contacting_relationships = self.relationship_classes[9:]
	
	def _init_gt_info(self):
		logger.info('-------loading annotations---------slowly-----------')
		
		annotations_path = os.path.join(self.root_path, const.ANNOTATIONS)
		if self.filter_small_box:
			with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
				person_bbox = pickle.load(f)
			f.close()
			with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_FILTERSMALL_PKL), 'rb') as f:
				object_bbox = pickle.load(f)
		else:
			with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
				person_bbox = pickle.load(f)
			f.close()
			with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL), 'rb') as f:
				object_bbox = pickle.load(f)
			f.close()
		logger.info('--------------------finish!-------------------------')
		
		# collect valid frames
		video_dict = {}
		q = []
		for i in person_bbox.keys():
			if object_bbox[i][0][const.METADATA][const.SET] == self.data_split:  # train or testing?
				video_name, frame_num = i.split('/')
				q.append(video_name)
				frame_valid = False
				for j in object_bbox[i]:  # the frame is valid if there is visible bbox
					if j[const.VISIBLE]:
						frame_valid = True
				if frame_valid:
					video_name, frame_num = i.split('/')
					if video_name in video_dict.keys():
						video_dict[video_name].append(i)
					else:
						video_dict[video_name] = [i]
		
		all_video_names = np.unique(q)
		self.valid_video_names = []
		self.video_list = []
		self.video_size = {}  # (w,h)
		self.gt_annotations = {}
		self.non_gt_human_nums = 0
		self.non_heatmap_nums = 0
		self.non_person_video = 0
		self.one_frame_video = 0
		self.valid_nums = 0
		self.invalid_videos = []
		
		'''
		filter_nonperson_box_frame = True (default): according to the stanford method, remove the frames without person box both for training and testing
		filter_nonperson_box_frame = False: still use the frames without person box, FasterRCNN may find the person
		'''
		for i in video_dict.keys():
			video = []
			gt_annotation_video = []
			for j in video_dict[i]:
				if self.filter_nonperson_box_frame:
					if person_bbox[j][const.BOUNDING_BOX].shape[0] == 0:
						self.non_gt_human_nums += 1
						continue
					else:
						video.append(j)
						self.valid_nums += 1
				
				gt_annotation_frame = [
					{
						const.PERSON_BOUNDING_BOX: person_bbox[j][const.BOUNDING_BOX],
						const.FRAME: j
					}
				]
				# each frame's objects and human
				for k in object_bbox[j]:
					if k[const.VISIBLE]:
						assert k[const.BOUNDING_BOX] is not None, 'warning! The object is visible without bbox'
						k[const.CLASS] = self.object_classes.index(k[const.CLASS])
						# from xywh to xyxy
						k[const.BOUNDING_BOX] = np.array([
							k[const.BOUNDING_BOX][0], k[const.BOUNDING_BOX][1],
							k[const.BOUNDING_BOX][0] + k[const.BOUNDING_BOX][2],
							k[const.BOUNDING_BOX][1] + k[const.BOUNDING_BOX][3]
						])
						
						k[const.ATTENTION_RELATIONSHIP] = torch.tensor(
							[self.attention_relationships.index(r) for r in k[const.ATTENTION_RELATIONSHIP]],
							dtype=torch.long)
						k[const.SPATIAL_RELATIONSHIP] = torch.tensor(
							[self.spatial_relationships.index(r) for r in k[const.SPATIAL_RELATIONSHIP]],
							dtype=torch.long)
						k[const.CONTACTING_RELATIONSHIP] = torch.tensor(
							[self.contacting_relationships.index(r) for r in k[const.CONTACTING_RELATIONSHIP]],
							dtype=torch.long)
						gt_annotation_frame.append(k)
				gt_annotation_video.append(gt_annotation_frame)
			
			if len(video) > 2:
				self.video_list.append(video)
				self.video_size[i] = person_bbox[j][const.BOUNDING_BOX_SIZE]
				self.gt_annotations[i] = gt_annotation_video
			elif len(video) == 1:
				self.one_frame_video += 1
			else:
				self.non_person_video += 1
		
		print('x' * 60)
		if self.filter_nonperson_box_frame:
			print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
			print('{} videos are invalid (no person), remove them'.format(self.non_person_video))
			print('{} videos are invalid (only one frame), remove them'.format(self.one_frame_video))
			print('{} frames have no human bbox in GT, remove them!'.format(self.non_gt_human_nums))
		else:
			print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
			print('{} frames have no human bbox in GT'.format(self.non_gt_human_nums))
			print('Removed {} of them without joint heatmaps which means FasterRCNN also cannot find the human'.format(
				self.non_heatmap_nums))
		print('x' * 60)
		
		self.invalid_video_names = np.setdiff1d(all_video_names, self.valid_video_names, assume_unique=False)
	
	def _construct_entry(self, attribute_dictionary):
		entry = {}
		if self.mode == const.SGDET:
			if self.is_train:
				entry = {
					const.BOXES: attribute_dictionary[const.FINAL_BBOXES_X],
					const.LABELS: attribute_dictionary[const.FINAL_LABELS_X],
					const.SCORES: attribute_dictionary[const.FINAL_SCORES_X],
					const.DISTRIBUTION: attribute_dictionary[const.FINAL_DISTRIBUTIONS],
					const.IM_IDX: attribute_dictionary[const.IM_IDX],
					const.PAIR_IDX: attribute_dictionary[const.PAIR],
					const.FEATURES: attribute_dictionary[const.FINAL_FEATURES_X],
					const.UNION_FEAT: attribute_dictionary[const.UNION_FEAT],
					const.SPATIAL_MASKS: attribute_dictionary[const.SPATIAL_MASKS],
					const.ATTENTION_GT: attribute_dictionary[const.ATTENTION_REL],
					const.SPATIAL_GT: attribute_dictionary[const.SPATIAL_REL],
					const.CONTACTING_GT: attribute_dictionary[const.CONTACTING_REL]
				}
			else:
				entry = {
					const.BOXES: attribute_dictionary[const.FINAL_BBOXES],
					const.SCORES: attribute_dictionary[const.FINAL_SCORES],
					const.DISTRIBUTION: attribute_dictionary[const.FINAL_DISTRIBUTIONS],
					const.PRED_LABELS: attribute_dictionary[const.PRED_LABELS],
					const.FEATURES: attribute_dictionary[const.FINAL_FEATURES],
					const.FMAPS: attribute_dictionary[const.FINAL_BASE_FEATURES],
					const.IM_INFO: attribute_dictionary[const.IM_INFO],
					const.LABELS: attribute_dictionary[const.ASSIGNED_LABELS]
				}
		elif self.mode == const.SGCLS:
			entry = {
				const.BOXES: attribute_dictionary[const.FINAL_BBOXES],
				const.LABELS: attribute_dictionary[const.FINAL_LABELS],  # labels are gt!
				const.SCORES: attribute_dictionary[const.FINAL_PRED_SCORES],
				const.IMAGE_IDX: attribute_dictionary[const.IMAGE_IDX],
				const.PAIR_IDX: attribute_dictionary[const.PAIR],
				const.HUMAN_IDX: attribute_dictionary[const.HUMAN_IDX],
				const.FEATURES: attribute_dictionary[const.FINAL_FEATURES],
				const.ATTENTION_GT: attribute_dictionary[const.ATTENTION_REL],
				const.SPATIAL_GT: attribute_dictionary[const.SPATIAL_REL],
				const.CONTACTING_GT: attribute_dictionary[const.CONTACTING_REL],
				const.DISTRIBUTION: attribute_dictionary[const.FINAL_DISTRIBUTIONS],
				const.PRED_LABELS: attribute_dictionary[const.PRED_LABELS]
			}
			if self.is_train:
				entry[const.UNION_FEAT] = attribute_dictionary[const.UNION_FEAT]
				entry[const.UNION_BOX] = attribute_dictionary[const.UNION_BOX]
				entry[const.SPATIAL_MASKS] = attribute_dictionary[const.SPATIAL_MASKS]
			else:
				entry[const.FMAPS] = attribute_dictionary[const.FINAL_BASE_FEATURES]
				entry[const.IM_INFO] = attribute_dictionary[const.IM_INFO]
		elif self.mode == const.PREDCLS:
			entry = {
				const.BOXES: attribute_dictionary[const.FINAL_BBOXES],
				const.LABELS: attribute_dictionary[const.FINAL_LABELS],  # labels are gt!
				const.SCORES: attribute_dictionary[const.FINAL_SCORES],
				const.IMAGE_IDX: attribute_dictionary[const.IMAGE_IDX],
				const.PAIR_IDX: attribute_dictionary[const.PAIR],
				const.HUMAN_IDX: attribute_dictionary[const.HUMAN_IDX],
				const.FEATURES: attribute_dictionary[const.FINAL_FEATURES],
				const.UNION_FEAT: attribute_dictionary[const.UNION_FEAT],
				const.UNION_BOX: attribute_dictionary[const.UNION_BOX],
				const.SPATIAL_MASKS: attribute_dictionary[const.SPATIAL_MASKS],
				const.ATTENTION_GT: attribute_dictionary[const.ATTENTION_REL],
				const.SPATIAL_GT: attribute_dictionary[const.SPATIAL_REL],
				const.CONTACTING_GT: attribute_dictionary[const.CONTACTING_REL]
			}
		return entry
	
	def _add_additional_info(self, entry, video_name):
		additional_info_path = os.path.join(self.additional_data_path, video_name + "_frame_idx.pkl")
		with open(additional_info_path, 'rb') as pkl_file:
			additional_info = pickle.load(pkl_file)
		entry[const.FRAME_IDX] = additional_info[const.FRAME_IDX]
		entry[const.GT_ANNOTATION] = self.gt_annotations[video_name]
		entry[const.FRAME_SIZE] = self.video_size[video_name]
		return entry
	
	def _load_dictionary_tensors_to_device(self, attribute_dictionary):
		for key in attribute_dictionary.keys():
			if type(attribute_dictionary[key]) == torch.Tensor:
				attribute_dictionary[key] = attribute_dictionary[key].to(self.device)
			elif type(attribute_dictionary[key]) == np.ndarray:
				attribute_dictionary[key] = torch.from_numpy(attribute_dictionary[key]).to(self.device)
		return attribute_dictionary
	
	def fetch_video_data(self, video_name):
		video_feature_file_path = os.path.join(self.features_path, f"{video_name}_{self.mode}.pkl")
		print(video_feature_file_path)
		with open(os.path.join(video_feature_file_path), 'rb') as pkl_file:
			data_dictionary = pickle.load(pkl_file)
			if self.is_compiled_together:
				attribute_dictionary = data_dictionary[self.entry_mode]
			else:
				attribute_dictionary = data_dictionary
		pkl_file.close()
		entry = self._construct_entry(attribute_dictionary)
		
		if self.is_compiled_together:
			video_name = video_feature_file_path.split('/')[-1][:-4]
		else:
			video_feature_filename = video_feature_file_path.split('/')[-1]
			video_name = video_feature_filename.split('.mp4')[0] + ".mp4"
		
		entry = self._add_additional_info(entry, video_name)
		if self.change_device:
			entry = self._load_dictionary_tensors_to_device(entry)
		return entry
	
	def __getitem__(self, index):
		video_feature_file_path = self.video_list[index]
		print(video_feature_file_path)
		with open(os.path.join(video_feature_file_path), 'rb') as pkl_file:
			data_dictionary = pickle.load(pkl_file)
			if self.is_compiled_together:
				attribute_dictionary = data_dictionary[self.entry_mode]
			else:
				attribute_dictionary = data_dictionary
		pkl_file.close()
		entry = self._construct_entry(attribute_dictionary)
		
		if self.is_compiled_together:
			video_name = video_feature_file_path.split('/')[-1][:-4]
		else:
			video_feature_filename = video_feature_file_path.split('/')[-1]
			video_name = video_feature_filename.split('.mp4')[0] + ".mp4"
		
		entry = self._add_additional_info(entry, video_name)
		if self.change_device:
			entry = self._load_dictionary_tensors_to_device(entry)
		return entry
	
	def __len__(self):
		return len(self.video_list)


def cuda_collate_fn(batch):
	"""
    don't need to zip the tensor

    """
	return batch[0]