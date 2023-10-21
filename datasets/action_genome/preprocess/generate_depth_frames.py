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
		self.testTransform = Compose([
			Resize((self.IMAGE_SIZE, self.IMAGE_SIZE)), ToTensor()])
		self.midas = torch.hub.load("intel-isl/MiDaS", modelType)
		self.midas.to(self.DEVICE)
		self.midas.eval()
		self.testDataset = ImageFolder(self.TEST_PATH, self.testTransform)
		self.testLoader = DataLoader(self.testDataset, batch_size=self.BATCH_SIZE, shuffle=False)
		if not os.path.exists(self.OUTPUT_PATH):
			os.makedirs(self.OUTPUT_PATH)
		self.idx = 0
	
	def run(self):
		for images, _ in self.testLoader:
			self.process_images(images)
	
	def process_images(self, images):
		images = images.to(self.DEVICE)
		with torch.no_grad():
			prediction = self.midas(images)
			prediction = torch.nn.functional.interpolate(
				prediction.unsqueeze(1), size=[self.IMAGE_SIZE, self.IMAGE_SIZE], mode="bicubic",
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
			# self.testLoader.
			outputFileName = os.path.join(self.OUTPUT_PATH, f'{self.idx}.jpg')
			plt.savefig(outputFileName)
			self.idx += 1


if __name__ == "__main__":
	depth_estimator = DepthEstimator(imgs_dir="./test/", depth_dir="./output/")
	depth_estimator.run()
