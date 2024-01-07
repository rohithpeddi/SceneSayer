import argparse
import csv
import os

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Metrics, ResultDetails, Result
from constants import ResultConstants as const


def process_and_save_results_from_file(result_file_path):
	file_name = os.path.basename(result_file_path)
	mode = file_name.split('_')[0]
	future_frame_loss_num = file_name.split('_')[2]
	
	task_name = None
	context_fraction = None
	test_future_frame_count = None
	
	if const.GENERATION_IMPACT in file_name:
		task_name = const.DYSGG
	elif const.PERCENTAGE_EVALUATION in file_name:
		task_name = const.DYSGA
		context_fraction = file_name.split('_')[5]
	elif const.EVALUATION in file_name:
		task_name = const.DYSGA
		test_future_frame_count = file_name.split('_')[4]
	
	assert task_name is not None
	
	# Read CSV file using csv reader method
	with open(result_file_path, 'r') as read_obj:
		# pass the file object to reader() to get the reader object
		csv_reader = csv.reader(read_obj)
		# Iterate over each row in the csv using reader object
		for row in csv_reader:
			method_name = row[0]
			with_constraint_metrics = Metrics(
				row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8],
				row[9], row[10], row[11], row[12]
			)
			no_constraint_metrics = Metrics(
				row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20],
				row[21], row[22], row[23], row[24]
			)
			semi_constraint_metrics = Metrics(
				row[25], row[26], row[27], row[28], row[29], row[30], row[31], row[32],
				row[33], row[34], row[35], row[36]
			)
			
			result_details = ResultDetails()
			result_details.add_with_constraint_metrics(with_constraint_metrics)
			result_details.add_no_constraint_metrics(no_constraint_metrics)
			result_details.add_semi_constraint_metrics(semi_constraint_metrics)
			
			result = Result(
				task_name=task_name,
				method_name=method_name,
				mode=mode,
			)
			
			result.train_num_future_frames = future_frame_loss_num
			result.add_result_details(result_details)
			
			if context_fraction is not None:
				result.context_fraction = context_fraction
			
			if test_future_frame_count is not None:
				result.test_num_future_frames = test_future_frame_count
				
			db_service.update_result(result.result_id, result.to_dict())


def process_and_save_results_from_folder(folder_path):
	files = os.listdir(folder_path)
	for file_name in files:
		process_and_save_results_from_file(os.path.join(folder_path, file_name))


def main():
	if args.folder_path is not None:
		process_and_save_results_from_folder(args.folder_path)
	elif args.result_file_path is not None:
		process_and_save_results_from_file(args.result_file_path)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--folder_path', type=str)
	parser.add_argument('--result_file_path', type=str)
	
	args = parser.parse_args()
	
	db_service = FirebaseService()
	
	main()
