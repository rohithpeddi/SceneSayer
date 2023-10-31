from torch.utils.data import Dataset

import pickle
import os
from constants import DataloaderConstants as const


class AG(Dataset):
	
	def __init__(
			self,
			mode,
			data_split,
			data_path=None
	):
		root_path = data_path
		self.data_split = data_split
		self.mode = mode
		entry_mode = const.SGDET if self.mode == const.SGDET else const.SGCLS  # Same attribute dictionary for SGCLS and PREDCLS
		self.features_path = os.path.join(root_path, const.SUPERVISED, self.data_split)
		self.video_list = []
		for video_feature_file in os.listdir(self.features_path):
			video_feature_file_path = os.path.join(self.features_path, video_feature_file)
			if entry_mode in video_feature_file and os.path.isdir(video_feature_file_path):
				self.video_list.append(video_feature_file_path)
	
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
	
	def __getitem__(self, index):
		video_feature_file_path = self.video_list[index]
		with open(os.path.join(video_feature_file_path), 'rb') as pkl_file:
			attribute_dictionary = pickle.load(pkl_file)
		pkl_file.close()
		entry = self._construct_entry(attribute_dictionary)
		return entry
	
	def __len__(self):
		return len(self.video_list)


def cuda_collate_fn(batch):
	"""
    don't need to zip the tensor

    """
	return batch[0]
