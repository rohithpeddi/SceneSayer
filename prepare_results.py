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


def compile_complete_future_frame_results():
	sga_result_list = fetch_sga_results()
	context_results_json = {}
	for test_num_future_frames in test_future_frame_list:
		context_results_json[test_num_future_frames] = {}
		for mode in modes:
			context_results_json[test_num_future_frames][mode] = {}
			for train_future_frame in train_future_frame_loss_list:
				context_results_json[test_num_future_frames][mode][train_future_frame] = {}
				for model_name in models:
					model_name = fetch_model_name(model_name)
					context_results_json[test_num_future_frames][mode][train_future_frame][
						model_name] = fetch_empty_metrics_json()
	
	for test_num_future_frames in test_future_frame_list:
		test_future_num_sga_list = []
		for result in sga_result_list:
			if result.test_num_future_frames is not None and str(
					result.test_num_future_frames) == test_num_future_frames:
				test_future_num_sga_list.append(result)
		
		for result in test_future_num_sga_list:
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
			
			context_results_json[test_num_future_frames][mode][train_future_frame][model_name] = completed_metrics_json
	
	return context_results_json


def generate_complete_future_frame_results_csvs(future_frame_results_json):
	for test_num_future_frames in test_future_frame_list:
		for mode in modes:
			csv_file_name = f"{mode}_{test_num_future_frames}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs",
			                             "complete_test_future_results_csvs",
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
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"R@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"mR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"hR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"hR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"hR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"R@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"mR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"hR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"hR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"hR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"R@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"mR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"hR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"hR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"hR@50"]
						])


def prepare_complete_future_frame_results():
	complete_future_frame_json = compile_complete_future_frame_results()
	# Generate Context Results CSVs
	generate_complete_future_frame_results_csvs(complete_future_frame_json)
	# Generate Context Results Latex Tables
	
	return


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


# --------------------------------------------------------------------------------------------
# RESULTS IN PAPER
# --------------------------------------------------------------------------------------------


def generate_paper_combined_context_recall_results_csvs(context_results_json):
	for mode in modes:
		csv_file_name = f"recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs",
		                             "paper_combined_context_results_csvs", csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Anticipation Loss", "Context Fraction", "Method Name",
				"R@10", "R@20", "R@50",
				"R@10", "R@20", "R@50",
				"R@10", "R@20", "R@50",
			])
			for train_future_frame in train_future_frame_loss_list:
				for context_fraction in context_fraction_list:
					for model_name in models:
						model_name = fetch_model_name(model_name)
						writer.writerow([
							train_future_frame,
							context_fraction,
							model_name,
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["R@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["R@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["R@50"]
						])


def generate_paper_combined_context_mean_recall_results_csvs(context_results_json):
	for mode in modes:
		csv_file_name = f"mean_recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs",
		                             "paper_combined_context_results_csvs", csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Anticipation Loss", "Context Fraction", "Method Name",
				"mR@10", "mR@20", "mR@50",
				"mR@10", "mR@20", "mR@50",
				"mR@10", "mR@20", "mR@50",
			])
			for train_future_frame in train_future_frame_loss_list:
				for context_fraction in context_fraction_list:
					for model_name in models:
						model_name = fetch_model_name(model_name)
						writer.writerow([
							train_future_frame,
							context_fraction,
							model_name,
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["mR@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["mR@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["mR@50"]
						])


def generate_paper_combined_context_harmonic_recall_results_csvs(context_results_json):
	for mode in modes:
		csv_file_name = f"harmonic_recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs",
		                             "paper_combined_context_results_csvs", csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Anticipation Loss", "Context Fraction", "Method Name",
				"hR@10", "hR@20", "hR@50",
				"hR@10", "hR@20", "hR@50",
				"hR@10", "hR@20", "hR@50",
			])
			for train_future_frame in train_future_frame_loss_list:
				for context_fraction in context_fraction_list:
					for model_name in models:
						model_name = fetch_model_name(model_name)
						writer.writerow([
							train_future_frame,
							context_fraction,
							model_name,
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["hR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["hR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][0]["hR@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["hR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["hR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][1]["hR@50"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["hR@10"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["hR@20"],
							context_results_json[context_fraction][mode][train_future_frame][model_name][2]["hR@50"]
						])


def generate_paper_combined_future_frame_recall_results_csvs(future_frame_results_json):
	for mode in modes:
		csv_file_name = f"recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs",
		                             "paper_combined_future_frame_results_csvs", csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Anticipation Loss", "Context Fraction", "Method Name",
				"R@10", "R@20", "R@50",
				"R@10", "R@20", "R@50",
				"R@10", "R@20", "R@50",
			])
			for train_future_frame in train_future_frame_loss_list:
				for test_num_future_frames in test_future_frame_list:
					for model_name in models:
						model_name = fetch_model_name(model_name)
						writer.writerow([
							train_future_frame,
							test_num_future_frames,
							model_name,
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"R@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"R@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"R@50"]
						])


def generate_paper_combined_future_frame_mean_recall_results_csvs(future_frame_results_json):
	for mode in modes:
		csv_file_name = f"mean_recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs",
		                             "paper_combined_future_frame_results_csvs", csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Anticipation Loss", "Context Fraction", "Method Name",
				"mR@10", "mR@20", "mR@50",
				"mR@10", "mR@20", "mR@50",
				"mR@10", "mR@20", "mR@50",
			])
			for train_future_frame in train_future_frame_loss_list:
				for test_num_future_frames in test_future_frame_list:
					for model_name in models:
						model_name = fetch_model_name(model_name)
						writer.writerow([
							train_future_frame,
							test_num_future_frames,
							model_name,
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"mR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"mR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"mR@50"]
						])


def generate_paper_combined_future_frame_harmonic_recall_results_csvs(future_frame_results_json):
	for mode in modes:
		csv_file_name = f"harmonic_recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs",
		                             "paper_combined_future_frame_results_csvs", csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Anticipation Loss", "Context Fraction", "Method Name",
				"hR@10", "hR@20", "hR@50",
				"hR@10", "hR@20", "hR@50",
				"hR@10", "hR@20", "hR@50",
			])
			for train_future_frame in train_future_frame_loss_list:
				for test_num_future_frames in test_future_frame_list:
					for model_name in models:
						model_name = fetch_model_name(model_name)
						writer.writerow([
							train_future_frame,
							test_num_future_frames,
							model_name,
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"hR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"hR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][0][
								"hR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"hR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"hR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][1][
								"hR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"hR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"hR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][model_name][2][
								"hR@50"]
						])


def prepare_paper_combined_future_frame_results():
	complete_future_frame_json = compile_complete_future_frame_results()
	# Generate Context Results CSVs
	generate_paper_combined_future_frame_recall_results_csvs(complete_future_frame_json)
	generate_paper_combined_future_frame_mean_recall_results_csvs(complete_future_frame_json)
	generate_paper_combined_future_frame_harmonic_recall_results_csvs(complete_future_frame_json)
	# Generate Context Results Latex Tables
	return


def fetch_setting_name(mode):
	if mode == "sgdet":
		setting_name = "$\\mathcal{C}_{f}$"
	elif mode == "sgcls":
		setting_name = "$\\mathcal{C}_{lf}$"
	elif mode == "predcls":
		setting_name = "$\\mathcal{C}_{llf}$"
	
	return setting_name


def generate_latex_header(setting_name, metric):
	latex_header = "\\begin{table}[!ht]\n"
	latex_header += "    \\centering\n"
	latex_header += "    \\resizebox{0.8\\textwidth}{!}{\n"
	latex_header += "    \\begin{tabular}{|l|l|ccc|ccc|}\n"
	latex_header += "    \\hline\n"
	latex_header += "        \\rowcolor{gray!25} \n"
	latex_header += "        \\multicolumn{2}{|c|}{\\textbf{" + setting_name + "}} & \\multicolumn{3}{c|}{\\textbf{With Constraint}} & \\multicolumn{3}{c|}{\\textbf{No Constraint}} \\\\ \\hline \n"
	latex_header += "        \\rowcolor{gray!25}\n"
	
	if metric == "recall":
		latex_header += "        $\\mathcal{F}$ & \\textbf{Method} & \\textbf{R@10} & \\textbf{R@20} & \\textbf{R@50} & \\textbf{R@10} & \\textbf{R@20} & \\textbf{R@50}  \\\\ \\hline\n"
	elif metric == "mean_recall":
		latex_header += "        $\\mathcal{F}$ & \\textbf{Method} & \\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50} & \\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  \\\\ \\hline\n"
	elif metric == "harmonic_recall":
		latex_header += "        $\\mathcal{F}$ & \\textbf{Method} & \\textbf{hR@10} & \\textbf{hR@20} & \\textbf{hR@50} & \\textbf{hR@10} & \\textbf{hR@20} & \\textbf{hR@50}  \\\\ \\hline\n"
	
	return latex_header


def generate_latex_footer():
	latex_footer = "    \\end{tabular}\n"
	latex_footer += "    }\n"
	latex_footer += "\\end{table}\n"
	
	return latex_footer


def generate_paper_combined_context_recall_vertical_latex_tables(context_results_json):
	for mode in modes:
		for train_num_future_frame in train_future_frame_loss_list:
			latex_file_name = f"recall_{mode}_{train_num_future_frame}.txt"
			latex_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs",
			                               "paper_combined_context_vertical_latex_tables", latex_file_name)
			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
			
			setting_name = fetch_setting_name(mode)
			
			latex_table = generate_latex_header(setting_name, "recall")
			
			for context_fraction in context_fraction_list:
				for idx, model_name in enumerate(models):
					model_name = fetch_model_name(model_name)
					if idx == 0:
						latex_table += f"        \\multirow{{4}}{{*}}{{{context_fraction}}} & {model_name} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['R@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['R@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['R@50']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['R@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['R@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['R@50']}  \\\\ \n"
					elif idx in [1, 2]:
						latex_table += f"        & {model_name} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['R@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['R@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['R@50']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['R@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['R@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['R@50']}  \\\\ \n"
					else:
						latex_table += f"        & {model_name} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['R@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['R@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['R@50']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['R@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['R@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['R@50']}  \\\\ \\hline\n"
			
			latex_footer = generate_latex_footer()
			latex_table += latex_footer
			
			with open(latex_file_path, "a", newline='') as latex_file:
				latex_file.write(latex_table)


def generate_paper_combined_context_mean_recall_vertical_latex_tables(context_results_json):
	for mode in modes:
		for train_num_future_frame in train_future_frame_loss_list:
			latex_file_name = f"mean_recall_{mode}_{train_num_future_frame}.txt"
			latex_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs",
			                               "paper_combined_context_vertical_latex_tables", latex_file_name)
			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
			
			setting_name = fetch_setting_name(mode)
			
			latex_table = generate_latex_header(setting_name, "mean_recall")
			
			for context_fraction in context_fraction_list:
				for idx, model_name in enumerate(models):
					model_name = fetch_model_name(model_name)
					if idx == 0:
						latex_table += f"        \\multirow{{4}}{{*}}{{{context_fraction}}} & {model_name} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['mR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['mR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['mR@50']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['mR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['mR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['mR@50']}  \\\\ \n"
					elif idx in [1, 2]:
						latex_table += f"        & {model_name} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['mR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['mR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['mR@50']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['mR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['mR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['mR@50']}  \\\\ \n"
					else:
						latex_table += f"        & {model_name} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['mR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['mR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['mR@50']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['mR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['mR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['mR@50']}  \\\\ \\hline\n"
			
			latex_footer = generate_latex_footer()
			latex_table += latex_footer
			
			with open(latex_file_path, "a", newline='') as latex_file:
				latex_file.write(latex_table)


def generate_paper_combined_context_harmonic_recall_vertical_latex_tables(context_results_json):
	for mode in modes:
		for train_num_future_frame in train_future_frame_loss_list:
			latex_file_name = f"harmonic_recall_{mode}_{train_num_future_frame}.txt"
			latex_file_path = os.path.join(os.path.dirname(__file__), "analysis", "docs",
			                               "paper_combined_context_vertical_latex_tables", latex_file_name)
			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
			
			setting_name = fetch_setting_name(mode)
			
			latex_table = generate_latex_header(setting_name, "harmonic_recall")
			
			for context_fraction in context_fraction_list:
				for idx, model_name in enumerate(models):
					model_name = fetch_model_name(model_name)
					if idx == 0:
						latex_table += f"        \\multirow{{4}}{{*}}{{{context_fraction}}} & {model_name} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['hR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['hR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['hR@50']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['hR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['hR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['hR@50']}  \\\\ \n"
					elif idx in [1, 2]:
						latex_table += f"        & {model_name} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['hR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['hR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['hR@50']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['hR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['hR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['hR@50']}  \\\\ \n"
					else:
						latex_table += f"        & {model_name} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['hR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['hR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]['hR@50']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['hR@10']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['hR@20']} & {context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]['hR@50']}  \\\\ \\hline\n"
			
			latex_footer = generate_latex_footer()
			latex_table += latex_footer
			
			with open(latex_file_path, "a", newline='') as latex_file:
				latex_file.write(latex_table)


def prepare_paper_combined_context_results():
	context_results_json = compile_context_results()
	# Generate Context Results CSVs
	generate_paper_combined_context_recall_results_csvs(context_results_json)
	generate_paper_combined_context_mean_recall_results_csvs(context_results_json)
	generate_paper_combined_context_harmonic_recall_results_csvs(context_results_json)
	# Generate Context LateX Tables
	generate_paper_combined_context_recall_vertical_latex_tables(context_results_json)
	generate_paper_combined_context_mean_recall_vertical_latex_tables(context_results_json)
	generate_paper_combined_context_harmonic_recall_vertical_latex_tables(context_results_json)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder_path', type=str)
	parser.add_argument('-result_file_path', type=str)
	
	modes = ["sgdet", "sgcls", "predcls"]
	models = ["baseline_so", "baseline_so_gen_loss", "ode", "sde"]
	train_future_frame_loss_list = ["1", "3", "5"]
	context_fraction_list = ["0.3", "0.5", "0.7", "0.9"]
	
	test_future_frame_list = ["1", "2", "3", "4", "5"]
	
	args = parser.parse_args()
	db_service = FirebaseService()
	
	prepare_context_results()
	combine_csv_to_excel(os.path.join(os.path.dirname(__file__), "analysis", "docs", "context_results_csvs"),
	                     os.path.join(os.path.dirname(__file__), "analysis", "docs", "combined_context_results.xlsx"))
	
	prepare_complete_future_frame_results()
	combine_csv_to_excel(
		os.path.join(os.path.dirname(__file__), "analysis", "docs", "complete_test_future_results_csvs"),
		os.path.join(os.path.dirname(__file__), "analysis", "docs", "complete_test_future_results.xlsx"))
	
	prepare_paper_combined_context_results()
	combine_csv_to_excel(
		os.path.join(os.path.dirname(__file__), "analysis", "docs", "paper_combined_context_results_csvs"),
		os.path.join(os.path.dirname(__file__), "analysis", "docs", "paper_combined_context_results.xlsx"))
	
	prepare_paper_combined_future_frame_results()
	combine_csv_to_excel(
		os.path.join(os.path.dirname(__file__), "analysis", "docs", "paper_combined_future_frame_results_csvs"),
		os.path.join(os.path.dirname(__file__), "analysis", "docs", "paper_combined_future_frame_results.xlsx"))
