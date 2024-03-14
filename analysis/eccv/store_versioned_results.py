import json
import os

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result


def store_versioned_results(version):
	results_dict = db_service.fetch_results()
	for result_id, result_dict in results_dict.items():
		result = Result.from_dict(result_dict)
		if result.result_id is None:
			result.result_id = result_id
			print("Saving result: ", result.result_id)
			db_service.update_result(result_id, result.to_dict())
		results_directory = os.path.join(os.path.dirname(__file__), "docs")
		os.makedirs(results_directory, exist_ok=True)
		json_file_path = os.path.join(results_directory, "firebase", f"v{version}", result_id + ".json")
		os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
		with open(json_file_path, "w") as json_file:
			json.dump(result.to_dict(), json_file)
			

def store_filtered_results(version):
	results_dict = db_service.fetch_results_from_db("results")
	for result_id, result_dict in results_dict.items():
		result = Result.from_dict(result_dict)
		if result.method_name in ["NeuralSDE", "NeuralODE", "ode", "sde"]:
			print("------------------------------------------------------------------------------------------")
			print(f"Saving result: {result.method_name}, {result.mode}, {result.train_num_future_frames}")
			print("------------------------------------------------------------------------------------------")
			results_directory = os.path.join(os.path.dirname(__file__), "docs")
			os.makedirs(results_directory, exist_ok=True)
			json_file_path = os.path.join(results_directory, "firebase", f"v{version}", result_id + ".json")
			os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
			with open(json_file_path, "w") as json_file:
				json.dump(result.to_dict(), json_file)
		else:
			print(f"Skipping result: {result.method_name}, {result.mode}, {result.train_num_future_frames}")


if __name__ == '__main__':
	db_service = FirebaseService()
	store_versioned_results(12)
