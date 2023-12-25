import os

from matplotlib import pyplot as plt


class Video:
	
	def __init__(self, video_id):
		self.video_id = video_id
		self.annotated_frames = []
		self.frame_duration_differences = []
		self.max_frame_duration_difference = 0
	
	def add_annotated_frame(self, frame):
		self.annotated_frames.append(frame)
	
	def compute_frame_duration_differences(self):
		for idx in range(1, len(self.annotated_frames)):
			self.frame_duration_differences.append((self.annotated_frames[idx] - self.annotated_frames[idx - 1]) / 24)
	
	def update_max_frame_duration_difference(self):
		self.max_frame_duration_difference = max(self.frame_duration_differences)


def plot_histogram(binned_data, dataset_name):
	bin_labels = list(binned_data.keys())
	bin_counts = [len(binned_data[bin_label]) for bin_label in bin_labels]
	
	plt.bar(bin_labels, bin_counts, color='blue', alpha=0.7)
	plt.xlabel('Seconds')
	plt.ylabel('Count')
	plt.title('{}'.format(dataset_name))
	plt.xticks(rotation=45)
	plt.tight_layout()
	# plt.show()
	
	assets_path = os.path.join(os.path.dirname(__file__), "assets")
	os.makedirs(assets_path, exist_ok=True)
	figure_path = os.path.join(assets_path, f"{dataset_name}.png")
	plt.savefig(figure_path)


def prepare_videos(directory, filename):
	frame_list_file_path = os.path.join(directory, filename)
	
	with open(frame_list_file_path, "r") as frame_list_file:
		for line in frame_list_file.readlines():
			line = line.strip()
			line = line.split(".mp4/")
			video_id = line[0]
			frame_number = int(line[1][:-4])
			if video_id not in video_id_to_video_map:
				video_id_to_video_map[video_id] = Video(video_id)
			video_id_to_video_map[video_id].add_annotated_frame(frame_number)


def analyse_videos():
	combined_frame_duration_differences = []
	max_frame_duration_difference = 0
	for video_id in video_id_to_video_map:
		video_id_to_video_map[video_id].compute_frame_duration_differences()
		video_id_to_video_map[video_id].update_max_frame_duration_difference()
		combined_frame_duration_differences += video_id_to_video_map[video_id].frame_duration_differences
		max_frame_duration_difference = max(max_frame_duration_difference,
		                                    video_id_to_video_map[video_id].max_frame_duration_difference)
		
	max_range = int(max_frame_duration_difference) + 1
	
	bins = {f"{i}-{i + 1}": [] for i in range(0, max_range, 1)}
	
	for num in combined_frame_duration_differences:
		for i in range(0, max_range, 1):
			if i <= num < i + 1:
				bins[f"{i}-{i + 1}"].append(num)
				break
	
	plot_histogram(bins, "Action Genome Annotated Frame Duration Differences")
	print(f"Max frame duration difference: {max_frame_duration_difference}")


if __name__ == "__main__":
	ag_annotation_directory = r"D:\DATA\OPEN\action_genome\annotations"
	ag_frame_list_filename = "frame_list.txt"
	video_id_to_video_map = {}
	prepare_videos(ag_annotation_directory, ag_frame_list_filename)
	analyse_videos()
