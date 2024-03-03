import argparse
import csv
import os

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result
from constants import ResultConstants as const


def fetch_actual_method_name(method_name):
	if method_name == "NeuralODE" or method_name == "ode":
		return "ode"
	elif method_name == "NeuralSDE" or method_name == "sde":
		return "sde"
	return method_name


def model_evaluation_check_anticipation():
	methods = ["sttran_ant", "sttran_gen_ant", "dsgdetr_ant", "dsgdetr_gen_ant", "ode", "sde"]
	modes = ["sgdet", "sgcls", "predcls"]
	train_future_frame_loss_list = ["1", "3", "5"]
	
	evaluation_check_json = {}
	for mode in modes:
		evaluation_check_json[mode] = {}
		for method_name in methods:
			method_name = fetch_actual_method_name(method_name)
			evaluation_check_json[mode][method_name] = {}
			for train_future_frame in train_future_frame_loss_list:
				evaluation_check_json[mode][method_name][train_future_frame] = False
	
	results_dict = db_service.fetch_results()
	for result_id, result_dict in results_dict.items():
		result = Result.from_dict(result_dict)
		if result.task_name == const.DYSGG:
			continue
		method_name = result.method_name
		train_future_frame = str(result.train_num_future_frames)
		mode = result.mode
		
		method_name = fetch_actual_method_name(method_name)
		
		evaluation_check_json[mode][method_name][train_future_frame] = True
	
	results_directory = os.path.join(os.path.dirname(__file__), "docs")
	os.makedirs(results_directory, exist_ok=True)
	results_file_path = os.path.join(results_directory, "model_evaluation_check.csv")
	with open(results_file_path, "a", newline='') as firebase_data_csv_file:
		writer = csv.writer(firebase_data_csv_file, quoting=csv.QUOTE_NONNUMERIC)
		writer.writerow([
			"Train Num Future Frames", "Mode", "Method Name", "Status"
		])
		for train_future_frame in train_future_frame_loss_list:
			for mode in modes:
				for method_name in methods:
					writer.writerow([
						train_future_frame, mode, method_name,
						evaluation_check_json[mode][method_name][train_future_frame]
					])


if __name__ == '__main__':
	db_service = FirebaseService()
	model_evaluation_check_anticipation()
