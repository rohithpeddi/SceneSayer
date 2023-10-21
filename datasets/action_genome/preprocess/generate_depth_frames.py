import cv2
import torch
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os
import numpy as np


class DepthEstimator:
	def __init__(self, modelType="DPT_Large", imgs_dir="./test/", depth_dir="./output/", batch_size=1):
		self.TEST_PATH = imgs_dir
		self.IMAGE_SIZE = 384
		self.BATCH_SIZE = batch_size
		self.DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
		self.OUTPUT_PATH = depth_dir
		os.makedirs(self.OUTPUT_PATH, exist_ok=True)
		self.testTransform = Compose([
			Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)), ToTensor()])
		self.midas = torch.hub.load("intel-isl/MiDaS", modelType)
		self.midas.to(self.DEVICE)
		self.midas.eval()
		# self.testDataset = ImageFolder(self.TEST_PATH)
		# self.testLoader = DataLoader(self.testDataset, batch_size=self.BATCH_SIZE, shuffle=False)
		if not os.path.exists(self.OUTPUT_PATH):
			os.makedirs(self.OUTPUT_PATH)
		self.idx = 0
		midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
		if modelType == "DPT_Large" or modelType == "DPT_Hybrid":
			self.transform = midas_transforms.dpt_transform
		else:
			self.transform = midas_transforms.small_transform
	
	def run(self):
		for image_name in os.listdir(self.TEST_PATH):
			print("Processing: ", image_name)
			image_path = os.path.join(self.TEST_PATH, image_name)
			img = cv2.imread(image_path)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			self.process_images(img)
	
	def process_images(self, img):
		images = self.transform(img).to(self.DEVICE)
		with torch.no_grad():
			prediction = self.midas(images)
			prediction = torch.nn.functional.interpolate(
				prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic",
				align_corners=False).squeeze()
		# convert to a numpy array
		if len(prediction.shape) == 2:
			output = np.array([prediction.cpu().numpy()])
		else:
			output = prediction.cpu().numpy()
		
		axes = []
		fig = plt.figure(figsize=(10, 5))
		# loop over the rows and columns
		for i in range(self.BATCH_SIZE):
			axes.append(fig.add_subplot(1, 2, 1))
			plt.imshow(images[i].permute((1, 2, 0)).cpu().detach().numpy())
			axes.append(fig.add_subplot(1, 2, 2))
			plt.imshow(output[i])
			fig.tight_layout()
			outputFileName = os.path.join(self.OUTPUT_PATH, f'{self.idx}.jpg')
			plt.savefig(outputFileName)
			self.idx += 1


if __name__ == "__main__":
	depth_estimator = DepthEstimator(imgs_dir="/data/rohith/ag/frames_annotated/0BX9N.mp4", depth_dir="/data/rohith/ag/depth/frames_annotated/0BX9N.mp4")
	depth_estimator.run()
