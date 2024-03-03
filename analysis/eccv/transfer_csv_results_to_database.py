import csv
import os

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Metrics, ResultDetails, Result
from constants import ResultConstants as const


def transfer_csv_from_file_to_database(result_file_path):
	base_file_name = os.path.basename(result_file_path)
	context_fraction = None
	test_future_frame_count = None
	mode = base_file_name.split('_')[0]
	train_num_future_frames = base_file_name.split('_')[2]
	
	task_name = const.SGA
	if const.PERCENTAGE_EVALUATION in base_file_name:
		context_fraction = base_file_name.split('_')[-1][:-4]
	elif const.EVALUATION in base_file_name:
		test_future_frame_count = base_file_name.split('_')[4][:-4]
	elif const.GENERATION_IMPACT in base_file_name:
		print("Skipping generation impact file: ", base_file_name)
		return
	
	assert task_name is not None and (test_future_frame_count is not None or context_fraction is not None)
	
	# Read CSV file using csv reader method
	with open(result_file_path, 'r') as read_obj:
		# pass the file object to reader() to get the reader object
		csv_reader = csv.reader(read_obj)
		# Iterate over each row in the csv using reader object
		# Ignore the first row as it is the header
		next(csv_reader)
		for row in csv_reader:
			method_name = row[0]
			print("Processing method: ", method_name)
			
			with_constraint_metrics = Metrics(
				row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12]
			)
			no_constraint_metrics = Metrics(
				row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23],
				row[24]
			)
			semi_constraint_metrics = Metrics(
				row[25], row[26], row[27], row[28], row[29], row[30], row[31], row[32], row[33], row[34], row[35],
				row[36]
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
			result.train_num_future_frames = train_num_future_frames
			result.add_result_details(result_details)
			if context_fraction is not None:
				result.context_fraction = context_fraction
			if test_future_frame_count is not None:
				result.test_num_future_frames = test_future_frame_count
			print("-----------------------------------------------------------------------------------")
			print("Saving result: ", result.result_id)
			db_service.update_result_to_db("results_eccv", result.result_id, result.to_dict())
			print("Saved result: ", result.result_id)
			print("-----------------------------------------------------------------------------------")


def transfer_results_from_directories():
	results_parent_directory = os.path.join(os.path.dirname(__file__), "result_directories")
	for result_directory in os.listdir(results_parent_directory):
		result_directory_path = os.path.join(results_parent_directory, result_directory)
		for file_name in os.listdir(result_directory_path):
			if file_name.endswith(".csv"):
				transfer_csv_from_file_to_database(os.path.join(result_directory_path, file_name))


if __name__ == '__main__':
	db_service = FirebaseService()
	transfer_results_from_directories()
