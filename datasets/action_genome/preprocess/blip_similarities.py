import os
import cv2
import torch
import numpy as np
import logging
from PIL import Image

from lavis.models import load_model_and_preprocess


from tqdm import tqdm
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, CenterCrop
log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
	os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, f"std.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
					format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

logger = logging.getLogger(__name__)


def transform_center():
	interp_mode = Image.BICUBIC
	tfm_test = [
		Resize(224, interpolation=interp_mode),
		CenterCrop((224, 224)),
		ToTensor(),
		Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
	]
	return Compose(tfm_test)


def get_videos(video_file_name, video_directory_path):
	all_frames = []
	video_input_path = os.path.join(video_directory_path, video_file_name)
	video_capture_file = cv2.VideoCapture(video_input_path)
	if not video_capture_file.isOpened():
		print('Video is not opened! {}'.format(video_input_path))
	else:
		total_frame_numbers = video_capture_file.get(cv2.CAP_PROP_FRAME_COUNT)
		if total_frame_numbers != 0:
			for _ in range(int(total_frame_numbers)):
				rval, frame = video_capture_file.read()
				if frame is not None:
					img = Image.fromarray(frame.astype('uint8')).convert('RGB')
					transformed_img = center_transform(img).numpy()
					all_frames.append(transformed_img)
	return np.array(all_frames)


def load_blip_model():
	blip_model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base",
	                                                                  is_eval=True, device=device)
	for blip_parameter in blip_model.parameters():
		blip_parameter.requires_grad = False
	return blip_model


def process_video_frames(video_frames, blip_model):
	video_features = []
	for k in range(int(len(video_frames) / max_length) + 1):
		segment_frames = torch.from_numpy(video_frames[k * max_length:(k + 1) * max_length]).to(device)
		segment_features = blip_model.extract_features(segment_frames, mode="image").image_embeds[0, 0, :]
		video_features.extend(segment_features.cpu().numpy().tolist())
	return np.array(video_features)


if __name__ == "__main__":
	device = 'cuda'
	# Number of frames to process at once
	max_length = 2000
	features_output_path = "/data/rohith/ag/blip/visual_features/"
	data_path = "/data/rohith/ag/videos/"
	center_transform = transform_center()
	model = load_blip_model()
	
	for video_name in tqdm(os.listdir(data_path), desc="Processing videos"):
		logger.info("---------------------------------------------------------------------")
		logger.info(f'Processing video {video_name}')
		frames = get_videos(video_name, data_path)
		logger.info(f'Video {video_name} has {len(frames)} frames')
		features = process_video_frames(frames, model)
		assert len(features) == len(frames)
		
		if not os.path.exists(features_output_path):
			os.makedirs(features_output_path, exist_ok=True)
		
		np.save(os.path.join(features_output_path, f'{video_name[:-4]}.npy'), features)
		logger.info(f'Visual features of {video_name} have been saved!')
	
	logger.info(f'All visual features have been saved!')
