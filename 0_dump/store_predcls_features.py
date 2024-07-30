import copy
import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import Constants as const
from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import Detector
from lib.supervised.config import Config
from logger_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class SupervisedFeatureExtractor:
	
	def __init__(self):
		self.config = Config()
		self._print_config()
		self._initialize_data_loaders()
		self.device = torch.device("cuda:0")
		self._initialize_object_detector()
	
	def _initialize_object_detector(self):
		self.predcls_object_detector = Detector(
			train=True,
			object_classes=self.train_dataset.object_classes,
			use_SUPPLY=True,
			mode=const.PREDCLS
		).to(device=self.device)
		
		self.predcls_object_detector.eval()
	
	def _print_config(self):
		logger.info("---------------------------------------------")
		logger.info('Configurations:')
		for i in self.config.args:
			logger.info(f"{i} : {self.config.args[i]}")
		logger.info("---------------------------------------------")
	
	def _initialize_data_loaders(self):
		self.train_dataset = AG(
			phase=const.TRAIN,
			datasize=self.config.datasize,
			data_path=self.config.data_path,
			filter_nonperson_box_frame=True,
			filter_small_box=False
		)
		
		self.train_dataloader = DataLoader(
			self.train_dataset,
			shuffle=True,
			num_workers=0,
			collate_fn=cuda_collate_fn,
			pin_memory=False
		)
		
		self.test_dataset = AG(
			phase=const.TEST,
			datasize=self.config.datasize,
			data_path=self.config.data_path,
			filter_nonperson_box_frame=True,
			filter_small_box=False
		)
		
		self.test_dataloader = DataLoader(
			self.test_dataset,
			shuffle=False,
			num_workers=0,
			collate_fn=cuda_collate_fn,
			pin_memory=False
		)
	
	def _make_numpy_array(self, entry):
		numpy_dict = {}
		for key in entry:
			if isinstance(entry[key], torch.Tensor):
				numpy_dict[key] = entry[key].cpu().numpy()
			elif isinstance(entry[key], dict):
				numpy_dict[key] = self._make_numpy_array(entry[key])
			else:
				numpy_dict[key] = entry[key]
		return numpy_dict
	
	def _generate_features(self, video_data, output_directory, dataset, mode):
		im_data = copy.deepcopy(video_data[0].cuda(0))
		im_info = copy.deepcopy(video_data[1].cuda(0))
		gt_boxes = copy.deepcopy(video_data[2].cuda(0))
		num_boxes = copy.deepcopy(video_data[3].cuda(0))
		gt_annotation = dataset.gt_annotations[video_data[4]]
		
		video_name = gt_annotation[0][0][const.FRAME].split('/')[0]
		
		with torch.no_grad():
			if mode == const.TRAIN:
				self.predcls_object_detector.is_train = True
			else:
				self.predcls_object_detector.is_train = False
			
			predcls_entry = self.predcls_object_detector(
				im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None, is_feature_extraction=True
			)
		
		predcls_pkl_path = os.path.join(output_directory, mode, f"{video_name}_predcls.pkl")
		os.makedirs(os.path.dirname(predcls_pkl_path), exist_ok=True)
		try:
			with open(predcls_pkl_path, 'wb') as pkl_file:
				pickle.dump(predcls_entry, pkl_file)
				logger.info("Dumped sgdet features for video: {}".format(video_name))
		except Exception as e:
			logger.error("Error in dumping features for video: {}".format(video_name))
			logger.error("Error: {}".format(e))
	
	def generate_supervised_features(self, output_directory):
		os.makedirs(output_directory, exist_ok=True)
		# logger.info("Generating features for train data")
		# for video in tqdm(self.train_dataloader):
		# 	self._generate_features(video, output_directory, self.train_dataset, mode=const.TRAIN)
		logger.info("Generating features for test data")
		for video in tqdm(self.test_dataloader):
			self._generate_features(video, output_directory, self.test_dataset, mode=const.TEST)
	
	def _generate_video_frame_idx_pkl(self, video_data, output_directory, dataset, mode):
		gt_annotation = dataset.gt_annotations[video_data[4]]
		
		video_name = gt_annotation[0][0][const.FRAME].split('/')[0]
		
		frame_idx_list = []
		for frame_gt_annotation in gt_annotation:
			frame_id = int(frame_gt_annotation[0][const.FRAME].split('/')[1][:-4])
			frame_idx_list.append(frame_id)
		
		entry = {
			const.FRAME_IDX: frame_idx_list
		}
		
		pkl_path = os.path.join(output_directory, mode, video_name + '_frame_idx.pkl')
		os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
		try:
			with open(pkl_path, 'wb') as pkl_file:
				pickle.dump(entry, pkl_file)
				logger.info("Dumped features for video: {}".format(video_name))
		except Exception as e:
			logger.error("Error in dumping features for video: {}".format(video_name))
			logger.error("Error: {}".format(e))
	
	def generate_frame_idx(self, output_directory):
		os.makedirs(output_directory, exist_ok=True)
		# logger.info("Generating frame idx pkl for train data")
		# for video in tqdm(self.train_dataloader):
		# 	self._generate_video_frame_idx_pkl(video, output_directory, self.train_dataset, mode=const.TRAIN)
		logger.info("Generating frame idx pkl for test data")
		for video in tqdm(self.test_dataloader):
			self._generate_video_frame_idx_pkl(video, output_directory, self.test_dataset, mode=const.TEST)


def load_pickle(pkl_path):
	with open(pkl_path, 'rb') as pkl_file:
		entry = pickle.load(pkl_file)
	return entry


if __name__ == "__main__":
	supervised_feature_extractor = SupervisedFeatureExtractor()
	supervised_feature_extractor.generate_supervised_features(output_directory="/data/rohith/ag/features_mod/supervised")
	supervised_feature_extractor.generate_frame_idx(
		output_directory="/data/rohith/ag/features_mod/supervised/additional")
