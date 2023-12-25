import cv2
import matplotlib.pyplot as plt
import torch
import logging
import os
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000

from tqdm import tqdm

log_directory = os.path.join(os.getcwd(), 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file_path = os.path.join(log_directory, f"std.log")
logging.basicConfig(filename=log_file_path, filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')

logger = logging.getLogger(__name__)


class DepthEstimator:
    def __init__(self, model_type="DPT_Large", batch_size=1):
        self.batch_size = batch_size
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def process_image_batch(self, image_batch, image_size):
        with torch.no_grad():
            prediction = self.midas(image_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=image_size, mode="bicubic",
                align_corners=False).squeeze()
        return prediction

    def process_video_directory(self, input_path, output_path):
        if os.path.exists(output_path):
            logger.info("Output path already exists: {}".format(output_path))
            if len(os.listdir(output_path)) == len(os.listdir(input_path)):
                logger.info("Output path already has all the files. Skipping.")
                return
            else:
                logger.info(f"Output: {len(os.listdir(output_path))}, Input: {len(os.listdir(input_path))} Output "
                            f"path does not have all the files. Deleting and recreating.")
                os.system("rm -rf {}".format(output_path))

        os.makedirs(output_path, exist_ok=True)
        logger.info("Processing images files in input path: {}".format(input_path))

        # Set the batch size
        batch_size = 40

        # Loop through image files in the input_path
        for i in range(0, len(os.listdir(input_path)), batch_size):
            batch_images = []
            batch_names = []
            original_image_size = None
            # Load and preprocess images for the batch
            for image_name in os.listdir(input_path)[i:i + batch_size]:
                image_path = os.path.join(input_path, image_name)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if original_image_size is None:
                    original_image_size = img.shape[:2]
                input_tensor = self.transform(img).to(self.device)
                batch_images.append(input_tensor)
                batch_names.append(image_name)

            # Stack the batch of images
            batch_images = torch.stack(batch_images).squeeze(dim=1)

            # Perform predictions for the batch
            batch_prediction = self.process_image_batch(batch_images, original_image_size)

            # Save the batch of predictions
            for j in range(len(batch_names)):
                if len(batch_names) == 1:
                    output = batch_prediction.cpu().numpy()
                else:
                    output = batch_prediction[j].cpu().numpy()
                plt.imsave(os.path.join(output_path, batch_names[j]), output)

        logger.info("Finished processing images files in input path: {}".format(input_path))

    def process_videos(self, input_path, output_path):
        for video_directory_name in tqdm(os.listdir(input_path), desc="Processing videos"):
            video_input_path = os.path.join(input_path, video_directory_name)
            video_output_path = os.path.join(output_path, video_directory_name)
            self.process_video_directory(video_input_path, video_output_path)


if __name__ == "__main__":
    depth_estimator = DepthEstimator()
    depth_estimator.process_videos("/data/rohith/ag/frames", "/data/rohith/ag/depth/frames")
