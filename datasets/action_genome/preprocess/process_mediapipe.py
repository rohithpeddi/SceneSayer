import os
import cv2
import numpy as np
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from tqdm import tqdm

log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
	os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, f"std.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

logger = logging.getLogger(__name__)


def draw_landmarks_on_image(rgb_image, detection_result):
	pose_landmarks_list = detection_result.pose_landmarks
	annotated_image = np.copy(rgb_image)
	
	# Loop through the detected poses to visualize.
	for idx in range(len(pose_landmarks_list)):
		pose_landmarks = pose_landmarks_list[idx]
		
		# Draw the pose landmarks.
		pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
		pose_landmarks_proto.landmark.extend([
			landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
		])
		solutions.drawing_utils.draw_landmarks(
			annotated_image,
			pose_landmarks_proto,
			solutions.pose.POSE_CONNECTIONS,
			solutions.drawing_styles.get_default_pose_landmarks_style())
	return annotated_image


class PoseLandMarkDetection:
	
	def __init__(self):
		self.model_path = '/home/rxp190007/CODE/NeSysVideoPrediction/checkpoints/pose_landmarker_heavy.task'
		BaseOptions = mp.tasks.BaseOptions
		PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
		VisionRunningMode = mp.tasks.vision.RunningMode
		
		options = PoseLandmarkerOptions(
			base_options=BaseOptions(model_asset_path=self.model_path),
			running_mode=VisionRunningMode.IMAGE,
			output_segmentation_masks=False)
		self.mp_detector = vision.PoseLandmarker.create_from_options(options)
	
	def process_frame_directory(self, frame_directory, output_directory):
		"""
		:param output_directory:
		:param frame_directory: path to directory containing frames
		:return: list of landmarks for each frame
		"""
		logger.info("Processing frame directory: {}".format(frame_directory))
		os.makedirs(output_directory, exist_ok=True)
		if len(os.listdir(output_directory)) > 0:
			return
		for image_name in os.listdir(frame_directory):
			image_path = os.path.join(frame_directory, image_name)
			mp_image = mp.Image.create_from_file(image_path)
			
			detection_result = self.mp_detector.detect(mp_image)
			processed_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
			cv2.imwrite(os.path.join(output_directory, image_name), processed_image)
		logger.info("Finished processing frame directory: {}".format(frame_directory))
	
	def process_videos(self, input_path, output_path):
		for video_directory_name in tqdm(os.listdir(input_path), desc="Processing videos"):
			video_input_path = os.path.join(input_path, video_directory_name)
			video_output_path = os.path.join(output_path, video_directory_name)
			self.process_frame_directory(video_input_path, video_output_path)


if __name__ == '__main__':
	pose_landmark_detector = PoseLandMarkDetection()
	pose_landmark_detector.process_videos("/data/rohith/ag/frames", "/data/rohith/ag/pose/frames")
