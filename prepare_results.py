import argparse
import csv
import os

import pandas as pd

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result
from constants import ResultConstants as const


def fetch_sga_results():
	results_dict = db_service.fetch_results()
	sga_result_list = []
	for result_id, result_dict in results_dict.items():
		result = Result.from_dict(result_dict)
		if result.task_name == const.DYSGG:
			continue
		if result.task_name == const.SGA:
			sga_result_list.append(result)
	
	return sga_result_list


# 0 - With constraint evaluator
# 1 - No constraint evaluator
# 2 - Semi Constraint evaluator
def fetch_empty_metrics_json():
	metrics_json = {}
	for i in range(3):
		metrics_json[i] = {
			"R@10": "-",
			"R@20": "-",
			"R@50": "-",
			"mR@10": "-",
			"mR@20": "-",
			"mR@50": "-",
			"hR@10": "-",
			"hR@20": "-",
			"hR@50": "-",
		}
	
	return metrics_json


def formatted_metric_num(metric_num):
	if metric_num == "-":
		return metric_num
	else:
		return round(float(metric_num) * 100, 2)


def fetch_completed_metrics_json(
		with_constraint_metrics,
		no_constraint_metrics,
		semi_constraint_metrics
):
	metrics_json = {
		0: {
			"R@10": formatted_metric_num(with_constraint_metrics.recall_10),
			"R@20": formatted_metric_num(with_constraint_metrics.recall_20),
			"R@50": formatted_metric_num(with_constraint_metrics.recall_50),
			"mR@10": formatted_metric_num(with_constraint_metrics.mean_recall_10),
			"mR@20": formatted_metric_num(with_constraint_metrics.mean_recall_20),
			"mR@50": formatted_metric_num(with_constraint_metrics.mean_recall_50),
			"hR@10": formatted_metric_num(with_constraint_metrics.harmonic_recall_10),
			"hR@20": formatted_metric_num(with_constraint_metrics.harmonic_recall_20),
			"hR@50": formatted_metric_num(with_constraint_metrics.harmonic_recall_50)
		},
		1: {
			"R@10": formatted_metric_num(no_constraint_metrics.recall_10),
			"R@20": formatted_metric_num(no_constraint_metrics.recall_20),
			"R@50": formatted_metric_num(no_constraint_metrics.recall_50),
			"mR@10": formatted_metric_num(no_constraint_metrics.mean_recall_10),
			"mR@20": formatted_metric_num(no_constraint_metrics.mean_recall_20),
			"mR@50": formatted_metric_num(no_constraint_metrics.mean_recall_50),
			"hR@10": formatted_metric_num(no_constraint_metrics.harmonic_recall_10),
			"hR@20": formatted_metric_num(no_constraint_metrics.harmonic_recall_20),
			"hR@50": formatted_metric_num(no_constraint_metrics.harmonic_recall_50)
		},
		2: {
			"R@10": formatted_metric_num(semi_constraint_metrics.recall_10),
			"R@20": formatted_metric_num(semi_constraint_metrics.recall_20),
			"R@50": formatted_metric_num(semi_constraint_metrics.recall_50),
			"mR@10": formatted_metric_num(semi_constraint_metrics.mean_recall_10),
			"mR@20": formatted_metric_num(semi_constraint_metrics.mean_recall_20),
			"mR@50": formatted_metric_num(semi_constraint_metrics.mean_recall_50),
			"hR@10": formatted_metric_num(semi_constraint_metrics.harmonic_recall_10),
			"hR@20": formatted_metric_num(semi_constraint_metrics.harmonic_recall_20),
			"hR@50": formatted_metric_num(semi_constraint_metrics.harmonic_recall_50)
		}
	}
	
	return metrics_json


def fetch_model_name(model_name):
	if model_name == "NeuralODE" or model_name == "ode":
		model_name = "SgatODE"
	elif model_name == "NeuralSDE" or model_name == "sde":
		model_name = "SgatSDE"
	elif model_name == "baseline_so":
		model_name = "Baseline"
	elif model_name == "baseline_so_gen_loss":
		model_name = "Sgatformer"
	
	return model_name


def compile_context_results():
	sga_result_list = fetch_sga_results()
	context_results_json = {}
	for context_fraction in context_fraction_list:
		context_results_json[context_fraction] = {}
		for mode in modes:
			context_results_json[context_fraction][mode] = {}
			for train_future_frame in train_future_frame_loss_list:
				context_results_json[context_fraction][mode][train_future_frame] = {}
				for model_name in models:
					model_name = fetch_model_name(model_name)
					context_results_json[context_fraction][mode][train_future_frame][
						model_name] = fetch_empty_metrics_json()
	
	for context_fraction in context_fraction_list:
		context_sga_list = []
		for result in sga_result_list:
			if result.context_fraction is not None and str(result.context_fraction) == context_fraction:
				context_sga_list.append(result)
		
		for result in context_sga_list:
			mode = result.mode
			model_name = result.method_name
			model_name = fetch_model_name(model_name)
			
			train_future_frame = str(result.train_num_future_frames)
			
			with_constraint_metrics = result.result_details.with_constraint_metrics
			no_constraint_metrics = result.result_details.no_constraint_metrics
			semi_constraint_metrics = result.result_details.semi_constraint_metrics
			
			completed_metrics_json = fetch_completed_metrics_json(
				with_constraint_metrics,
				no_constraint_metrics,
				semi_constraint_metrics
			)
			
			context_results_json[context_fraction][mode][train_future_frame][model_name] = completed_metrics_json
	
	return context_results_json


def generate_context_results_csvs(context_results_json):
	for context_fraction in context_fraction_list:
		for mode in modes:
			csv_file_name = f"{mode}_{context_fraction}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs", "context_results_csvs",
			                             csv_file_name)
			os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
			with open(csv_file_path, "a", newline='') as csv_file:
				writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
				writer.writerow([
					"Anticipation Loss", "Method Name",
					"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
					"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50",
					"R@10", "R@20", "R@50", "mR@10", "mR@20", "mR@50", "hR@10", "hR@20", "hR@50"
				])
				for train_future_frame in train_future_frame_loss_list:
					for model_name in models:
						model_name = fetch_model_name(model_name)
						writer.writerow([
							train_future_frame,
							model_name,
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["R@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["mR@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["hR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["hR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["hR@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["R@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["mR@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["hR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["hR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["hR@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["R@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["mR@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["hR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["hR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["hR@50"]
						])


def prepare_context_results():
	context_results_json = compile_context_results()
	# Generate Context Results CSVs
	generate_context_results_csvs(context_results_json)
	# Generate Context Results Latex Tables
	
	return


def compile_future_frame_results():
	pass


def combine_csv_to_excel(folder_path, output_file):
	# Create a Pandas Excel writer using openpyxl as the engine
	writer = pd.ExcelWriter(output_file, engine='openpyxl')
	
	# Iterate over all CSV files in the folder
	for csv_file in os.listdir(folder_path):
		if csv_file.endswith('.csv'):
			# Read the CSV file
			df = pd.read_csv(os.path.join(folder_path, csv_file))
			
			# Write the data frame to a sheet named after the CSV file
			sheet_name = os.path.splitext(csv_file)[0]
			df.to_excel(writer, sheet_name=sheet_name, index=False)
	
	# Save the Excel file
	writer.save()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder_path', type=str)
	parser.add_argument('-result_file_path', type=str)
	
	modes = ["sgdet", "sgcls", "predcls"]
	models = ["baseline_so", "baseline_so_gen_loss", "ode", "sde"]
	train_future_frame_loss_list = ["1", "3", "5"]
	context_fraction_list = ["0.3", "0.5", "0.7", "0.9"]
	
	args = parser.parse_args()
	db_service = FirebaseService()
	
	prepare_context_results()
	combine_csv_to_excel(os.path.join(os.path.dirname(__file__), "analysis", "docs", "context_results_csvs"), os.path.join(os.path.dirname(__file__), "analysis", "docs", "combined_context_results.xlsx"))
