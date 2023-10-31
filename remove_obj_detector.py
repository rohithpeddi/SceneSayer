import copy
import gzip
import os
import pickle

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.supervised.generation.action_genome.ag_dataset import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.supervised.config import Config
from constants import Constants as const
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
		self.sgcls_object_detector = detector(
			train=True,
			object_classes=self.train_dataset.object_classes,
			use_SUPPLY=True,
			mode=const.SGCLS
		).to(device=self.device)
		
		self.sgcls_object_detector.eval()
		
		self.sgdet_object_detector = detector(
			train=True,
			object_classes=self.train_dataset.object_classes,
			use_SUPPLY=True,
			mode=const.SGDET
		).to(device=self.device)
		
		self.sgdet_object_detector.eval()
	
	def _print_config(self):
		logger.info("---------------------------------------------")
		logger.info('Configurations:')
		for i in self.config.args:
			logger.info(f"{i} : {self.config.args[i]}")
		logger.info("---------------------------------------------")
	
	def _initialize_data_loaders(self):
		self.train_dataset = AG(
			mode=const.TRAIN,
			datasize=self.config.datasize,
			data_path=self.config.data_path,
			filter_nonperson_box_frame=True,
			filter_small_box=False if self.config.mode == const.PREDCLS else True
		)
		
		self.train_dataloader = DataLoader(
			self.train_dataset,
			shuffle=True,
			num_workers=1,
			collate_fn=cuda_collate_fn,
			pin_memory=False
		)
		
		self.test_dataset = AG(
			mode=const.TEST,
			datasize=self.config.datasize,
			data_path=self.config.data_path,
			filter_nonperson_box_frame=True,
			filter_small_box=False if self.config.mode == const.PREDCLS else True
		)
		
		self.test_dataloader = DataLoader(
			self.test_dataset,
			shuffle=False,
			num_workers=1,
			collate_fn=cuda_collate_fn,
			pin_memory=False
		)
	
	def _generate_features(self, video_data, output_directory, dataset, mode):
		im_data = copy.deepcopy(video_data[0].cuda(0))
		im_info = copy.deepcopy(video_data[1].cuda(0))
		gt_boxes = copy.deepcopy(video_data[2].cuda(0))
		num_boxes = copy.deepcopy(video_data[3].cuda(0))
		gt_annotation = dataset.gt_annotations[video_data[4]]
		
		video_name = gt_annotation[0][0][const.FRAME].split('/')[0]
		
		with torch.no_grad():
			if mode == const.TRAIN:
				self.sgdet_object_detector.is_train = True
				self.sgcls_object_detector.is_train = True
			else:
				self.sgdet_object_detector.is_train = False
				self.sgcls_object_detector.is_train = False
			
			sgdet_entry = self.sgdet_object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
			sgcls_entry = self.sgcls_object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
			
			entry = {
				const.SGDET: sgdet_entry,
				const.SGCLS: sgcls_entry
			}
		
		pkl_path = os.path.join(output_directory, mode, video_name + '.pkl')
		os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
		try:
			with open(pkl_path, 'wb') as pkl_file:
				pickle.dump(entry, pkl_file)
				logger.info("Dumped features for video: {}".format(video_name))
		except Exception as e:
			logger.error("Error in dumping features for video: {}".format(video_name))
			logger.error("Error: {}".format(e))
	
	def generate_supervised_features(self, output_directory):
		os.makedirs(output_directory, exist_ok=True)
		logger.info("Generating features for train data")
		for video in tqdm(self.train_dataloader):
			self._generate_features(video, output_directory, self.train_dataset, mode=const.TRAIN)
		logger.info("Generating features for test data")
		for video in tqdm(self.test_dataloader):
			self._generate_features(video, output_directory, self.test_dataset, mode=const.TEST)
	
	def _generate_video_frame_idx_pkl(self, video_data, output_directory, dataset, mode):
		im_data = copy.deepcopy(video_data[0].cuda(0))
		im_info = copy.deepcopy(video_data[1].cuda(0))
		gt_boxes = copy.deepcopy(video_data[2].cuda(0))
		num_boxes = copy.deepcopy(video_data[3].cuda(0))
		gt_annotation = dataset.gt_annotations[video_data[4]]
		
		video_name = gt_annotation[0][0][const.FRAME].split('/')[0]
		
		entry = {}
		
		pkl_path = os.path.join(output_directory, mode, video_name + '.pkl')
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
		logger.info("Generating frame idx pkl for train data")
		for video in tqdm(self.train_dataloader):
			self._generate_video_frame_idx_pkl(video, output_directory, self.train_dataset, mode=const.TRAIN)
		logger.info("Generating frame idx pkl for test data")
		for video in tqdm(self.test_dataloader):
			self._generate_video_frame_idx_pkl(video, output_directory, self.test_dataset, mode=const.TEST)


def load_pickle(pkl_path):
	with open(pkl_path, 'rb') as pkl_file:
		entry = pickle.load(pkl_file)
	return entry


if __name__ == "__main__":
	supervised_feature_extractor = SupervisedFeatureExtractor()
	# supervised_feature_extractor.generate_supervised_features(output_directory="/data/rohith/ag/features/supervised")
	supervised_feature_extractor.generate_frame_idx(output_directory="/data/rohith/ag/features/supervised")
# load_pickle("/data/rohith/ag/features/supervised/train/1BVUA.mp4.pkl")
