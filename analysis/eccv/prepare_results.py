import argparse
import csv
import os

import numpy as np
import pandas as pd



from prepare_results_base import *


def compile_context_results():
	sga_result_list = fetch_sga_results()
	context_results_json = {}
	for context_fraction in context_fraction_list:
		context_results_json[context_fraction] = {}
		for mode in modes:
			context_results_json[context_fraction][mode] = {}
			for train_future_frame in train_future_frame_loss_list:
				context_results_json[context_fraction][mode][train_future_frame] = {}
				for method_name in methods:
					method_name = fetch_method_name_json(method_name)
					context_results_json[context_fraction][mode][train_future_frame][
						method_name] = fetch_empty_metrics_json()
	
	for context_fraction in context_fraction_list:
		context_sga_list = []
		for result in sga_result_list:
			if result.context_fraction is not None and str(result.context_fraction) == context_fraction:
				context_sga_list.append(result)
		
		for result in context_sga_list:
			mode = result.mode
			method_name = result.method_name
			method_name = fetch_method_name_json(method_name)
			
			train_future_frame = str(result.train_num_future_frames)
			
			with_constraint_metrics = result.result_details.with_constraint_metrics
			no_constraint_metrics = result.result_details.no_constraint_metrics
			semi_constraint_metrics = result.result_details.semi_constraint_metrics
			
			completed_metrics_json = fetch_completed_metrics_json(
				with_constraint_metrics,
				no_constraint_metrics,
				semi_constraint_metrics
			)
			
			context_results_json[context_fraction][mode][train_future_frame][method_name] = completed_metrics_json
	return context_results_json


def compile_complete_future_frame_results():
	sga_result_list = fetch_sga_results()
	context_results_json = {}
	for test_num_future_frames in test_future_frame_list:
		context_results_json[test_num_future_frames] = {}
		for mode in modes:
			context_results_json[test_num_future_frames][mode] = {}
			for train_future_frame in train_future_frame_loss_list:
				context_results_json[test_num_future_frames][mode][train_future_frame] = {}
				for method_name in methods:
					method_name = fetch_method_name_json(method_name)
					context_results_json[test_num_future_frames][mode][train_future_frame][
						method_name] = fetch_empty_metrics_json()
	
	for test_num_future_frames in test_future_frame_list:
		test_future_num_sga_list = []
		for result in sga_result_list:
			if result.test_num_future_frames is not None and str(
					result.test_num_future_frames) == test_num_future_frames:
				test_future_num_sga_list.append(result)
		
		for result in test_future_num_sga_list:
			mode = result.mode
			method_name = result.method_name
			method_name = fetch_method_name_json(method_name)
			
			train_future_frame = str(result.train_num_future_frames)
			
			with_constraint_metrics = result.result_details.with_constraint_metrics
			no_constraint_metrics = result.result_details.no_constraint_metrics
			semi_constraint_metrics = result.result_details.semi_constraint_metrics
			
			completed_metrics_json = fetch_completed_metrics_json(
				with_constraint_metrics,
				no_constraint_metrics,
				semi_constraint_metrics
			)
			
			context_results_json[test_num_future_frames][mode][train_future_frame][method_name] = completed_metrics_json
	
	return context_results_json


def prepare_complete_future_frame_results():
	complete_future_frame_json = compile_complete_future_frame_results()
	# Generate Context Results CSVs
	generate_complete_future_frame_results_csvs(complete_future_frame_json, test_future_frame_list,
	                                            train_future_frame_loss_list, modes, methods)
	# Generate Context Results Latex Tables
	return


def prepare_context_results():
	context_results_json = compile_context_results()
	# Generate Context Results CSVs
	generate_context_results_csvs(context_results_json, context_fraction_list, train_future_frame_loss_list, modes,
	                              methods)
	# Generate Context Results Latex Tables
	return


# --------------------------------------------------------------------------------------------
# RESULTS IN PAPER
# --------------------------------------------------------------------------------------------


def generate_paper_combined_context_recall_results_csvs(context_results_json):
	for mode in modes:
		csv_file_name = f"recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
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
					for method_name in methods:
						method_name = fetch_method_name_json(method_name)
						writer.writerow([
							train_future_frame,
							context_fraction,
							method_name,
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["R@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["R@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["R@50"]
						])


def generate_paper_combined_context_mean_recall_results_csvs(context_results_json):
	for mode in modes:
		csv_file_name = f"mean_recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
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
					for method_name in methods:
						method_name = fetch_method_name_json(method_name)
						writer.writerow([
							train_future_frame,
							context_fraction,
							method_name,
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["mR@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["mR@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["mR@50"]
						])


def generate_paper_combined_future_frame_recall_results_csvs(future_frame_results_json):
	for mode in modes:
		csv_file_name = f"recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
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
					for method_name in methods:
						method_name = fetch_method_name_json(method_name)
						writer.writerow([
							train_future_frame,
							test_num_future_frames,
							method_name,
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"R@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"R@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"R@50"]
						])


def generate_paper_combined_future_frame_mean_recall_results_csvs(future_frame_results_json):
	for mode in modes:
		csv_file_name = f"mean_recall_{mode}.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
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
					for method_name in methods:
						method_name = fetch_method_name_json(method_name)
						writer.writerow([
							train_future_frame,
							test_num_future_frames,
							method_name,
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"mR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"mR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"mR@50"]
						])


def generate_paper_combined_context_recalls_latex_tables(context_results_json):
	for mode in modes:
		for train_num_future_frame in train_future_frame_loss_list:
			latex_file_name = f"com_recalls_{mode}_{train_num_future_frame}.txt"
			latex_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
			                               "paper_combined_context_recalls_latex_tables", latex_file_name)
			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
			
			setting_name = fetch_setting_name(mode)
			
			latex_table = generate_combined_recalls_latex_header(
				setting_name,
				"com_recalls",
				train_horizon=train_num_future_frame,
				mode=mode,
				eval_horizon="context"
			)
			
			for context_fraction in context_fraction_list:
				values_matrix = np.zeros((6, 12), dtype=np.float32)
				
				for idx, method_name in enumerate(methods):
					values_matrix = fill_combined_context_fraction_values_matrix(values_matrix, idx, method_name,
					                                                             context_results_json,
					                                                             context_fraction, mode,
					                                                             train_num_future_frame)
				
				max_boolean_matrix = values_matrix == np.max(values_matrix, axis=0)
				
				for idx, method_name in enumerate(methods):
					method_name = fetch_method_name_latex(method_name)
					
					initial_string = ""
					if idx == 0:
						initial_string = f"        \\multirow{{4}}{{*}}{{{context_fraction}}} & {method_name}"
					elif idx in [1, 2, 3, 4, 5]:
						initial_string = f"        & {method_name}"
					
					latex_row = initial_string
					for col_idx in range(12):
						if max_boolean_matrix[idx, col_idx]:
							latex_row += f" & \\cellcolor{{highlightColor}} \\textbf{{{fetch_rounded_value(values_matrix[idx, col_idx])}}}"
						else:
							latex_row += f" & {fetch_rounded_value(values_matrix[idx, col_idx])}"
					
					if idx == 5:
						latex_row += "  \\\\ \\hline\n"
					else:
						latex_row += "  \\\\ \n"
					
					latex_table += latex_row
			latex_footer = generate_latex_footer()
			latex_table += latex_footer
			
			with open(latex_file_path, "a", newline='') as latex_file:
				latex_file.write(latex_table)


def generate_paper_combined_wn_context_recalls_latex_tables(context_results_json):
	for mode in modes:
		for train_num_future_frame in train_future_frame_loss_list:
			latex_file_name = f"com_wn_recalls_{mode}_{train_num_future_frame}.txt"
			latex_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
			                               "paper_combined_wn_context_recalls_latex_tables", latex_file_name)
			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
			
			setting_name = fetch_setting_name(mode)
			
			latex_table = generate_combined_wn_recalls_latex_header(
				setting_name,
				"com_recalls",
				train_horizon=train_num_future_frame,
				mode=mode,
				eval_horizon="context"
			)
			
			for context_fraction in context_fraction_list:
				values_matrix = np.zeros((6, 12), dtype=np.float32)
				
				for idx, method_name in enumerate(methods):
					values_matrix = fill_combined_wn_context_fraction_values_matrix(values_matrix, idx, method_name,
					                                                                context_results_json,
					                                                                context_fraction, mode,
					                                                                train_num_future_frame)
				
				max_boolean_matrix = values_matrix == np.max(values_matrix, axis=0)
				
				for idx, method_name in enumerate(methods):
					method_name = fetch_method_name_latex(method_name)
					
					initial_string = ""
					if idx == 0:
						initial_string = f"        \\multirow{{4}}{{*}}{{{context_fraction}}} & {method_name}"
					elif idx in [1, 2, 3, 4, 5]:
						initial_string = f"        & {method_name}"
					
					latex_row = initial_string
					for col_idx in range(12):
						if max_boolean_matrix[idx, col_idx]:
							latex_row += f" & \\cellcolor{{highlightColor}}  \\textbf{{{fetch_rounded_value(values_matrix[idx, col_idx])}}}"
						else:
							latex_row += f" & {fetch_rounded_value(values_matrix[idx, col_idx])}"
					
					if idx == 5:
						latex_row += "  \\\\ \\hline\n"
					else:
						latex_row += "  \\\\ \n"
					
					latex_table += latex_row
			latex_footer = generate_latex_footer()
			latex_table += latex_footer
			
			with open(latex_file_path, "a", newline='') as latex_file:
				latex_file.write(latex_table)


def generate_paper_combined_future_frame_recalls_latex_tables(context_results_json):
	for mode in modes:
		for train_num_future_frame in train_future_frame_loss_list:
			latex_file_name = f"com_recalls_{mode}_{train_num_future_frame}.txt"
			latex_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
			                               "paper_combined_future_frame_recalls_latex_tables", latex_file_name)
			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
			
			setting_name = fetch_setting_name(mode)
			
			latex_table = generate_combined_wn_recalls_latex_header(
				setting_name,
				"com_recalls",
				train_horizon=train_num_future_frame,
				mode=mode,
				eval_horizon="future_frame"
			)
			
			for test_future_frame in test_future_frame_list:
				values_matrix = np.zeros((6, 12), dtype=np.float32)
				
				for idx, method_name in enumerate(methods):
					values_matrix = fill_combined_future_frame_values_matrix(values_matrix, idx, method_name,
					                                                         context_results_json,
					                                                         test_future_frame, mode,
					                                                         train_num_future_frame)
				
				max_boolean_matrix = values_matrix == np.max(values_matrix, axis=0)
				
				for idx, method_name in enumerate(methods):
					method_name = fetch_method_name_latex(method_name)
					
					initial_string = ""
					if idx == 0:
						initial_string = f"        \\multirow{{4}}{{*}}{{{test_future_frame}}} & {method_name}"
					elif idx in [1, 2, 3, 4, 5]:
						initial_string = f"        & {method_name}"
					
					latex_row = initial_string
					for col_idx in range(12):
						if max_boolean_matrix[idx, col_idx]:
							latex_row += f" & \\cellcolor{{highlightColor}} \\textbf{{{fetch_rounded_value(values_matrix[idx, col_idx])}}}"
						else:
							latex_row += f" & {fetch_rounded_value(values_matrix[idx, col_idx])}"
					
					if idx == 5:
						latex_row += "  \\\\ \\hline\n"
					else:
						latex_row += "  \\\\ \n"
					
					latex_table += latex_row
			latex_footer = generate_latex_footer()
			latex_table += latex_footer
			
			with open(latex_file_path, "a", newline='') as latex_file:
				latex_file.write(latex_table)


def generate_paper_combined_wn_future_frame_recalls_latex_tables(context_results_json):
	for mode in modes:
		for train_num_future_frame in train_future_frame_loss_list:
			latex_file_name = f"com_wn_recalls_{mode}_{train_num_future_frame}.txt"
			latex_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
			                               "paper_combined_wn_future_frame_recalls_latex_tables", latex_file_name)
			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
			
			setting_name = fetch_setting_name(mode)
			
			latex_table = generate_combined_wn_recalls_latex_header(
				setting_name,
				"com_recalls",
				train_horizon=train_num_future_frame,
				mode=mode,
				eval_horizon="future_frame"
			)
			
			for test_future_frame in test_future_frame_list:
				values_matrix = np.zeros((6, 12), dtype=np.float32)
				
				for idx, method_name in enumerate(methods):
					values_matrix = fill_combined_wn_future_frame_values_matrix(values_matrix, idx, method_name,
					                                                            context_results_json,
					                                                            test_future_frame, mode,
					                                                            train_num_future_frame)
				
				max_boolean_matrix = values_matrix == np.max(values_matrix, axis=0)
				
				for idx, method_name in enumerate(methods):
					method_name = fetch_method_name_latex(method_name)
					
					initial_string = ""
					if idx == 0:
						initial_string = f"        \\multirow{{4}}{{*}}{{{test_future_frame}}} & {method_name}"
					elif idx in [1, 2, 3, 4, 5]:
						initial_string = f"        & {method_name}"
					
					latex_row = initial_string
					for col_idx in range(12):
						if max_boolean_matrix[idx, col_idx]:
							latex_row += f" & \\cellcolor{{highlightColor}} \\textbf{{{fetch_rounded_value(values_matrix[idx, col_idx])}}}"
						else:
							latex_row += f" & {fetch_rounded_value(values_matrix[idx, col_idx])}"
					
					if idx == 5:
						latex_row += "  \\\\ \\hline\n"
					else:
						latex_row += "  \\\\ \n"
					
					latex_table += latex_row
			latex_footer = generate_latex_footer()
			latex_table += latex_footer
			
			with open(latex_file_path, "a", newline='') as latex_file:
				latex_file.write(latex_table)


def prepare_paper_combined_context_results():
	context_results_json = compile_context_results()
	# Generate Context Results CSVs
	generate_paper_combined_context_recall_results_csvs(context_results_json)
	generate_paper_combined_context_mean_recall_results_csvs(context_results_json)
	
	# Generate Context LateX Tables
	generate_paper_combined_context_recalls_latex_tables(context_results_json)
	generate_paper_combined_wn_context_recalls_latex_tables(context_results_json)


def prepare_paper_combined_future_frame_results():
	future_frame_results_json = compile_complete_future_frame_results()
	# Generate Future Frame Results CSVs
	generate_paper_combined_future_frame_recall_results_csvs(future_frame_results_json)
	generate_paper_combined_future_frame_mean_recall_results_csvs(future_frame_results_json)
	
	# Generate Future Frame Results Latex Tables
	generate_paper_combined_future_frame_recalls_latex_tables(future_frame_results_json)
	generate_paper_combined_wn_future_frame_recalls_latex_tables(future_frame_results_json)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-folder_path', type=str)
	parser.add_argument('-result_file_path', type=str)
	
	modes = ["sgdet", "sgcls", "predcls"]
	methods = ["sttran_ant", "dsgdetr_ant", "sttran_gen_ant", "dsgdetr_gen_ant", "ode", "sde"]
	train_future_frame_loss_list = ["1", "3", "5"]
	context_fraction_list = ["0.3", "0.5", "0.7", "0.9"]
	test_future_frame_list = ["1", "2", "3", "4", "5"]
	
	args = parser.parse_args()
	db_service = FirebaseService()
	
	prepare_context_results()
	combine_csv_to_excel(os.path.join(os.path.dirname(__file__), "results_docs", "context_results_csvs"),
	                     os.path.join(os.path.dirname(__file__), "results_docs", "combined_context_results.xlsx"))
	
	prepare_complete_future_frame_results()
	combine_csv_to_excel(
		os.path.join(os.path.dirname(__file__), "results_docs", "complete_test_future_results_csvs"),
		os.path.join(os.path.dirname(__file__), "results_docs", "complete_test_future_results.xlsx"))
	
	prepare_paper_combined_context_results()
	combine_csv_to_excel(
		os.path.join(os.path.dirname(__file__), "results_docs", "paper_combined_context_results_csvs"),
		os.path.join(os.path.dirname(__file__), "results_docs", "paper_combined_context_results.xlsx"))
	
	prepare_paper_combined_future_frame_results()
	combine_csv_to_excel(
		os.path.join(os.path.dirname(__file__), "results_docs", "paper_combined_future_frame_results_csvs"),
		os.path.join(os.path.dirname(__file__), "results_docs", "paper_combined_future_frame_results.xlsx"))
