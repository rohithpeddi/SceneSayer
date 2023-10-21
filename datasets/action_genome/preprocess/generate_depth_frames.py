import concurrent

import cv2
import matplotlib.pyplot as plt
import torch
import os
import logging

from tqdm import tqdm

log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
	os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, f"std.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

logger = logging.getLogger(__name__)


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
			
			input_batch = self.transform(img).to(self.device)
			with torch.no_grad():
				prediction = self.midas(input_batch)
				prediction = torch.nn.functional.interpolate(
					prediction.unsqueeze(1), size=img.shape[:2], mode="bicubic",
					align_corners=False).squeeze()
			
			output = prediction.cpu().numpy()
			plt.imsave(os.path.join(self.output_path, image_name), output)
		logger.info("Finished processing images files in input path: {}".format(self.input_path))


def process_frame_directory(input_path, output_path):
	depth_estimator = DepthEstimator(input_path=input_path, output_path=output_path)
	depth_estimator.run()


def estimate_monocular_depth(input_path, output_path, num_workers=4):
	frame_directories = os.listdir(input_path)
	with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
		futures = []
		for frame_directory_name in tqdm(frame_directories, desc="Processing"):
			input_dir = os.path.join(input_path, frame_directory_name)
			output_dir = os.path.join(output_path, frame_directory_name)
			future = executor.submit(process_frame_directory, input_dir, output_dir)
			futures.append(future)
		for future in concurrent.futures.as_completed(futures):
			try:
				future.result()
			except Exception as e:
				logger.error(f"Error: {e}")


if __name__ == "__main__":
	estimate_monocular_depth("/data/rohith/ag/frames", "/data/rohith/ag/depth/frames", num_workers=4)
