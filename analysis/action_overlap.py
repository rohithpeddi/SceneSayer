import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Action:
	
	def __init__(self, action_class, start_time, end_time):
		self.action_class = action_class
		self.start_time = start_time
		self.end_time = end_time
		self.duration = end_time - start_time
	
	def overlap(self, other_action):
		return min(self.end_time, other_action.end_time) - max(self.start_time, other_action.start_time)
	
	def overlap_percentage(self, other_action):
		overlap = self.overlap(other_action)
		return (overlap / self.duration) * 100


class Video:
	
	def __init__(self, video_id, duration):
		self.video_id = video_id
		self.duration = duration
		self.actions = []
		self.overlaps = []
	
	def add_action(self, action):
		self.actions.append(action)


class Dataset:
	
	def __init__(self):
		self.videos = []
	
	def add_video(self, video):
		self.videos.append(video)


def prepare_dataset(data_directory, csv_file_name):
	file_path = os.path.join(data_directory, csv_file_name)
	dataset_df = pd.read_csv(file_path)
	dataset = Dataset()
	for index, row in dataset_df.iterrows():
		actions = row["actions"]
		if pd.isna(actions):
			continue
		
		video_id = row["id"]
		video_duration = row["length"]
		video = Video(video_id, video_duration)
		
		actions = actions.split(";")
		for action in actions:
			action = action.split(" ")
			action_class = action[0]
			start_time = float(action[1])
			end_time = float(action[2])
			video.add_action(Action(action_class, start_time, end_time))
		dataset.add_video(video)
	return dataset


def plot_histogram(binned_data, dataset_name):
	bin_labels = list(binned_data.keys())
	bin_counts = [len(binned_data[bin_label]) for bin_label in bin_labels]
	
	plt.bar(bin_labels, bin_counts, color='blue', alpha=0.7)
	plt.xlabel('Overlap Percentage')
	plt.ylabel('Count')
	plt.title('Histogram from {}'.format(dataset_name))
	plt.xticks(rotation=45)
	plt.tight_layout()
	
	assets_path = os.path.join(os.path.dirname(__file__), "assets")
	os.makedirs(assets_path, exist_ok=True)
	figure_path = os.path.join(assets_path, f"{dataset_name}.png")
	plt.savefig(figure_path)
	plt.show()


def analyse_overlaps(dataset, dataset_name):
	overlap_percentage_list = []
	
	for video in dataset.videos:
		actions = video.actions
		for i in range(len(actions)):
			for j in range(i + 1, len(actions)):
				overlap_percentage = actions[i].overlap_percentage(actions[j])
				inverse_overlap_percentage = actions[j].overlap_percentage(actions[i])
				
				if overlap_percentage <= 1:
					continue
				else:
					absolute_overlap_percentage = max(overlap_percentage, inverse_overlap_percentage)
					overlap_percentage_list.append(absolute_overlap_percentage)
					video.overlaps.append(absolute_overlap_percentage)
	
	sorted_overlap_percentage_list = sorted(overlap_percentage_list)
	
	bins = {f"{i}-{i + 10}": [] for i in range(0, 100, 10)}
	
	for num in sorted_overlap_percentage_list:
		for i in range(0, 100, 10):
			if i <= num < i + 10:
				bins[f"{i}-{i + 10}"].append(num)
				break
	plot_histogram(bins, dataset_name)


def analyse_video_duration_statistics(dataset, dataset_name):
	video_duration_list = []
	max_duration = 0
	
	for video in dataset.videos:
		video_duration_list.append(video.duration)
		max_duration = max(max_duration, video.duration)
	
	video_duration_list = sorted(video_duration_list)
	
	max_range = int(max_duration / 10) * 10 + 10
	
	bins = {f"{i}-{i + 10}": [] for i in range(0, max_range, 10)}
	
	for num in video_duration_list:
		for i in range(0, max_range, 10):
			if i <= num < i + 10:
				bins[f"{i}-{i + 10}"].append(num)
				break
	
	plot_histogram(bins, "{} Video Duration".format(dataset_name))


if __name__ == "__main__":
	charades_annotation_directory = r"D:\DATA\OPEN\Charades\Charades\Charades"
	charades_train_file_name = "Charades_v1_train.csv"
	charades_test_file_name = "Charades_v1_test.csv"
	
	train_dataset = prepare_dataset(charades_annotation_directory, charades_train_file_name)
	# analyse_overlaps(train_dataset, "Charades Train")
	analyse_video_duration_statistics(train_dataset, "Charades Train")
	
	test_dataset = prepare_dataset(charades_annotation_directory, charades_test_file_name)
	# analyse_overlaps(test_dataset, "Charades Test")
	analyse_video_duration_statistics(test_dataset, "Charades Test")
	
	combined_dataset = Dataset()
	combined_dataset.videos = train_dataset.videos + test_dataset.videos
	# analyse_overlaps(combined_dataset, "Charades Combined")
	analyse_video_duration_statistics(combined_dataset, "Charades Combined")
