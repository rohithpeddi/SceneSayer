import cv2
import torch
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os
import numpy as np
from logger_config import get_logger

logger = get_logger(__name__)


class DepthEstimator:
	def __init__(self, input_path, output_path, model_type="DPT_Large", batch_size=1):
		self.batch_size = batch_size
		self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
		
		self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
		self.midas.to(self.device)
		self.midas.eval()
		
		self.input_path = input_path
		self.output_path = output_path
		os.makedirs(self.output_path, exist_ok=True)
		
		midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
		if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
			self.transform = midas_transforms.dpt_transform
		else:
			self.transform = midas_transforms.small_transform
	
	def run(self):
		logger.info("----------------------------------------------------")
		logger.info("Processing images files in input path: {}".format(self.input_path))
		for image_name in os.listdir(self.input_path):
			image_path = os.path.join(self.input_path, image_name)
			img = cv2.imread(image_path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			
			with torch.no_grad():
				prediction = self.midas(img)
				prediction = torch.nn.functional.interpolate(
					prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic",
					align_corners=False).squeeze()
			
			output = prediction.cpu().numpy()
			plt.imsave(os.path.join(self.output_path, image_name), output)
		logger.info("Finished processing images files in input path: {}".format(self.input_path))


if __name__ == "__main__":
	depth_estimator = DepthEstimator(input_path="/data/rohith/ag/frames_annotated/0BX9N.mp4",
	                                 output_path="/data/rohith/ag/depth/frames_annotated/0BX9N.mp4")
	depth_estimator.run()
