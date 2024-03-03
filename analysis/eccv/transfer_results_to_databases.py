import json
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


def transfer_results_from_version_to_db(version, mode, method_name, train_num_future_frames, database_name):
	results_directory = os.path.join(os.path.dirname(__file__), "docs", "firebase", f"v{version}")
	for file_name in os.listdir(results_directory):
		if file_name.endswith(".json"):
			with open(os.path.join(results_directory, file_name), "r") as json_file:
				result_dict = json.load(json_file)
				result = Result.from_dict(result_dict)
				if result.mode == mode and result.method_name == method_name and result.train_num_future_frames == train_num_future_frames:
					print("-----------------------------------------------------------------------------------------")
					print(
						f"Mode: {result.mode}, Method Name: {result.method_name}, Train Num Future Frames: {result.train_num_future_frames}")
					print("-----------------------------------------------------------------------------------------")
					db_service.update_result_to_db(database_name, result.result_id, result_dict)
				else:
					print(
						f"Skipping result: {result.result_id}, Mode: {result.mode}, Method Name: {result.method_name}, Train Num Future Frames: {result.train_num_future_frames}")


def transfer_method_results_from_version_to_db(version, method_name_list, database_name):
	results_directory = os.path.join(os.path.dirname(__file__),  "docs", "firebase", f"v{version}")
	for file_name in os.listdir(results_directory):
		if file_name.endswith(".json"):
			with open(os.path.join(results_directory, file_name), "r") as json_file:
				result_dict = json.load(json_file)
				result = Result.from_dict(result_dict)
				if result.method_name in method_name_list:
					
					method_name = fetch_actual_method_name(result.method_name)
					result.method_name = method_name
					result_dict[const.METHOD_NAME] = method_name
					
					print("-----------------------------------------------------------------------------------------")
					print(
						f"Mode: {result.mode}, Method Name: {result.method_name}, Train Num Future Frames: {result.train_num_future_frames}")
					print("-----------------------------------------------------------------------------------------")
					
					db_service.update_result_to_db(database_name, result.result_id, result_dict)
				else:
					print(
						f"Skipping result: {result.result_id}, Mode: {result.mode}, Method Name: {result.method_name}, Train Num Future Frames: {result.train_num_future_frames}")


if __name__ == '__main__':
	db_service = FirebaseService()
	transfer_method_results_from_version_to_db(1, ["NeuralODE", "NeuralSDE", "ode", "sde"], "results_eccv")
