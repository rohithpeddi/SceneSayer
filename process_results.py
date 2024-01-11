import argparse
import csv
import json
import os

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Metrics, ResultDetails, Result
from constants import ResultConstants as const


def print_results():
	results_dict = db_service.fetch_results()
	results_file_path = os.path.join(os.path.dirname(__file__), "firebase_data.csv")
	with open(results_file_path, "a", newline='') as firebase_data_csv_file:
		writer = csv.writer(firebase_data_csv_file, quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow([
			"Task Name", "Method Name", "Mode", "Train Num Future Frames", "Test Num Future Frames", "Context Fraction"
		])
		for result_id, result_dict in results_dict.items():
			result = Result.from_dict(result_dict)
			writer.writerow([
				result.task_name, result.method_name, result.mode, result.train_num_future_frames,
				result.test_num_future_frames, result.context_fraction
			])


def remove_de_results():
	results_dict = db_service.fetch_results()
	for result_id, result_dict in results_dict.items():
		result = Result.from_dict(result_dict)
		if result.method_name == "NeuralODE" or result.method_name == "NeuralSDE":
			db_service.db.child(const.RESULTS).child(result_id).remove()


def process_and_save_results_from_file(result_file_path):
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
				row[1], row[2], row[3], None, row[4], row[5], row[6], None, row[7], row[8], row[9], None
			)
			no_constraint_metrics = Metrics(
				row[10], row[11], row[12], None, row[13], row[14], row[15], None, row[16], row[17], row[18], None
			)
			semi_constraint_metrics = Metrics(
				row[19], row[20], row[21], None, row[22], row[23], row[24], None, row[25], row[26], row[27], None
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
			
			print("Saving result: ", result.result_id)
			db_service.update_result(result.result_id, result.to_dict())
			print("Saved result: ", result.result_id)


def process_and_save_results_from_folder(folder_path):
	files = os.listdir(folder_path)
	for file_name in files:
		print(f"Processing file: {file_name}")
		process_and_save_results_from_file(os.path.join(folder_path, file_name))
	
	print(f"Processed a total of {len(files)} files")


def process_results():
	if args.folder_path is not None:
		process_and_save_results_from_folder(args.folder_path)
	elif args.result_file_path is not None:
		process_and_save_results_from_file(args.result_file_path)


# Checks what evaluations are done/missing
def complete_evaluation_check_anticipation():
	models = ["baseline_so", "baseline_so_gen_loss", "ode", "sde"]
	modes = ["sgdet", "sgcls", "predcls"]
	train_future_frame_loss_list = ["1", "3", "5"]
	evaluation_future_frame_loss_list = ["1", "2", "3", "4", "5"]
	context_fraction_list = ["0.3", "0.5", "0.7", "0.9"]
	
	evaluation_check_json = {}
	for mode in modes:
		evaluation_check_json[mode] = {}
		for model_name in models:
			if model_name == "NeuralODE" or model_name == "ode":
				model_name = "ode"
			elif model_name == "NeuralSDE" or model_name == "sde":
				model_name = "sde"
			evaluation_check_json[mode][model_name] = {}
			for train_future_frame in train_future_frame_loss_list:
				evaluation_check_json[mode][model_name][train_future_frame] = {}
				for evaluation_future_frame in evaluation_future_frame_loss_list:
					evaluation_check_json[mode][model_name][train_future_frame][evaluation_future_frame] = False
				for context_fraction in context_fraction_list:
					evaluation_check_json[mode][model_name][train_future_frame][context_fraction] = False
	
	results_dict = db_service.fetch_results()
	for result_id, result_dict in results_dict.items():
		result = Result.from_dict(result_dict)
		if result.task_name == const.DYSGG:
			continue
		method_name = result.method_name
		train_future_frame = str(result.train_num_future_frames)
		context_fraction = str(result.context_fraction) if result.context_fraction is not None else None
		evaluation_future_frame = str(
			result.test_num_future_frames) if result.test_num_future_frames is not None else None
		mode = result.mode
		
		if method_name == "NeuralODE" or method_name == "ode":
			method_name = "ode"
		elif method_name == "NeuralSDE" or method_name == "sde":
			method_name = "sde"
		
		if context_fraction is not None:
			evaluation_check_json[mode][method_name][train_future_frame][context_fraction] = True
		elif evaluation_future_frame is not None:
			evaluation_check_json[mode][method_name][train_future_frame][evaluation_future_frame] = True
	
	results_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs", "complete_evaluation_check.csv")
	with open(results_file_path, "a", newline='') as firebase_data_csv_file:
		writer = csv.writer(firebase_data_csv_file, quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow([
			"Mode", "Method Name", "Train Num Future Frames", "Test Num Future Frames", "Context Fraction", "Status"
		])
		for mode in modes:
			for model_name in models:
				for train_future_frame in train_future_frame_loss_list:
					for evaluation_future_frame in evaluation_future_frame_loss_list:
						writer.writerow([
							mode, model_name, train_future_frame, evaluation_future_frame, None,
							evaluation_check_json[mode][model_name][train_future_frame][evaluation_future_frame]
						])
					for context_fraction in context_fraction_list:
						writer.writerow([
							mode, model_name, train_future_frame, None, context_fraction,
							evaluation_check_json[mode][model_name][train_future_frame][context_fraction]
						])


def model_evaluation_check_anticipation():
	models = ["baseline_so", "baseline_so_gen_loss", "ode", "sde"]
	modes = ["sgdet", "sgcls", "predcls"]
	train_future_frame_loss_list = ["1", "3", "5"]
	
	evaluation_check_json = {}
	for mode in modes:
		evaluation_check_json[mode] = {}
		for model_name in models:
			if model_name == "NeuralODE" or model_name == "ode":
				model_name = "ode"
			elif model_name == "NeuralSDE" or model_name == "sde":
				model_name = "sde"
			evaluation_check_json[mode][model_name] = {}
			for train_future_frame in train_future_frame_loss_list:
				evaluation_check_json[mode][model_name][train_future_frame] = False
	
	results_dict = db_service.fetch_results()
	for result_id, result_dict in results_dict.items():
		result = Result.from_dict(result_dict)
		if result.task_name == const.DYSGG:
			continue
		method_name = result.method_name
		train_future_frame = str(result.train_num_future_frames)
		mode = result.mode
		
		if method_name == "NeuralODE" or method_name == "ode":
			method_name = "ode"
		elif method_name == "NeuralSDE" or method_name == "sde":
			method_name = "sde"
		
		evaluation_check_json[mode][method_name][train_future_frame] = True
	
	results_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs", "model_evaluation_check.csv")
	with open(results_file_path, "a", newline='') as firebase_data_csv_file:
		writer = csv.writer(firebase_data_csv_file, quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow([
			"Mode", "Method Name", "Train Num Future Frames", "Status"
		])
		for mode in modes:
			for model_name in models:
				for train_future_frame in train_future_frame_loss_list:
					writer.writerow([
						mode, model_name, train_future_frame,
						evaluation_check_json[mode][model_name][train_future_frame]
					])


def store_versioned_results(version):
	results_dict = db_service.fetch_results()
	for result_id, result_dict in results_dict.items():
		result = Result.from_dict(result_dict)
		if result.result_id is None:
			result.result_id = result_id
			print("Saving result: ", result.result_id)
			db_service.update_result(result_id, result.to_dict())
		json_file_path = os.path.join(os.path.dirname(__file__), "analysis", f"v{version}", result_id + ".json")
		with open(json_file_path, "w") as json_file:
			json.dump(result.to_dict(), json_file)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder_path', type=str)
	parser.add_argument('-result_file_path', type=str)
	
	args = parser.parse_args()
	db_service = FirebaseService()
	
	# process_results()
	# compile_results()
	# print_results()
	# model_evaluation_check_anticipation()
	
	store_versioned_results(1)
