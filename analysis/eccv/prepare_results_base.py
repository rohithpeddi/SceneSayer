import csv
import os

import pandas as pd

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result
from constants import ResultConstants as const

db_service = FirebaseService()


def fetch_sga_results():
	results_dict = db_service.fetch_results_from_db("results_eccv")
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


def fetch_setting_name(mode):
	if mode == "sgdet":
		setting_name = "\\textbf{SGA of AGS}"
	elif mode == "sgcls":
		setting_name = "\\textbf{SGA of PGAGS}"
	elif mode == "predcls":
		setting_name = "\\textbf{SGA of GAGS}"
	return setting_name


def fetch_ref_tab_name(mode, eval_horizon, train_future_frame):
	if mode == "sgdet":
		setting_name = f"sga_ags_{eval_horizon}_{train_future_frame}"
	elif mode == "sgcls":
		setting_name = f"sga_pgags_{eval_horizon}_{train_future_frame}"
	elif mode == "predcls":
		setting_name = f"sga_gags_{eval_horizon}_{train_future_frame}"
	return setting_name


def fetch_method_name_latex(method_name):
	if method_name == "NeuralODE" or method_name == "ode":
		method_name = "\\textbf{SceneSayerODE (Ours)}"
	elif method_name == "NeuralSDE" or method_name == "sde":
		method_name = "\\textbf{SceneSayerSDE (Ours)}"
	elif method_name == "sttran_ant":
		method_name = "STTran+ \cite{cong_et_al_sttran_2021}"
	elif method_name == "sttran_gen_ant":
		method_name = "STTran++ \cite{cong_et_al_sttran_2021}"
	elif method_name == "dsgdetr_ant":
		method_name = "DSGDetr+ \cite{Feng_2021}"
	elif method_name == "dsgdetr_gen_ant":
		method_name = "DSGDetr++ \cite{Feng_2021}"
	elif method_name == "sde_wo_bb":
		method_name = "\\textbf{SceneSayerSDE (w/o BB)}"
	elif method_name == "sde_wo_recon":
		method_name = "\\textbf{SceneSayerSDE (w/o Recon)}"
	elif method_name == "sde_wo_gen":
		method_name = "\\textbf{SceneSayerSDE (w/o GenLoss)}"
	return method_name


def fetch_method_name_csv(method_name):
	if method_name == "NeuralODE" or method_name == "ode":
		method_name = "SceneSayerODE"
	elif method_name == "NeuralSDE" or method_name == "sde":
		method_name = "SceneSayerSDE"
	elif method_name == "sttran_ant":
		method_name = "STTran+"
	elif method_name == "sttran_gen_ant":
		method_name = "STTran++"
	elif method_name == "dsgdetr_ant":
		method_name = "DSGDetr+"
	elif method_name == "dsgdetr_gen_ant":
		method_name = "DSGDetr++"
	elif method_name == "sde_wo_bb":
		method_name = "SceneSayerSDE(w/oBB)"
	elif method_name == "sde_wo_recon":
		method_name = "SceneSayerSDE(w/oRecon)"
	elif method_name == "sde_wo_gen":
		method_name = "SceneSayerSDE(w/oGenLoss)"
	return method_name


def fetch_method_name_json(method_name):
	if method_name == "NeuralODE" or method_name == "ode":
		method_name = "SceneSayerODE"
	elif method_name == "NeuralSDE" or method_name == "sde":
		method_name = "SceneSayerSDE"
	elif method_name == "sttran_ant":
		method_name = "STTran+"
	elif method_name == "sttran_gen_ant":
		method_name = "STTran++"
	elif method_name == "dsgdetr_ant":
		method_name = "DSGDetr+"
	elif method_name == "dsgdetr_gen_ant":
		method_name = "DSGDetr++"
	elif method_name == "sde_wo_bb":
		method_name = "SceneSayerSDE(w/oBB)"
	elif method_name == "sde_wo_recon":
		method_name = "SceneSayerSDE(w/oRecon)"
	elif method_name == "sde_wo_gen":
		method_name = "SceneSayerSDE(w/oGenLoss)"
	return method_name


def generate_context_results_csvs(context_results_json, context_fraction_list, train_future_frame_loss_list, modes,
                                  methods):
	for context_fraction in context_fraction_list:
		for mode in modes:
			csv_file_name = f"{mode}_{context_fraction}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "results_docs", "context_results_csvs",
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
					for method_name in methods:
						method_name = fetch_method_name_json(method_name)
						method_name_csv = fetch_method_name_csv(method_name)
						writer.writerow([
							train_future_frame,
							method_name_csv,
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["R@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["mR@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["hR@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["hR@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][0]["hR@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["R@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["mR@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["hR@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["hR@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][1]["hR@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["R@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["R@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["R@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["mR@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["mR@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["mR@50"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["hR@10"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["hR@20"],
							context_results_json[context_fraction][mode][train_future_frame][method_name][2]["hR@50"]
						])


def generate_complete_future_frame_results_csvs(future_frame_results_json, test_future_frame_list,
                                                train_future_frame_loss_list, modes, methods):
	for test_num_future_frames in test_future_frame_list:
		for mode in modes:
			csv_file_name = f"{mode}_{test_num_future_frames}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
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
					for method_name in methods:
						method_name = fetch_method_name_json(method_name)
						method_name_csv = fetch_method_name_csv(method_name)
						writer.writerow([
							train_future_frame,
							method_name_csv,
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"R@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"mR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"hR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"hR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][0][
								"hR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"R@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"mR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"hR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"hR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][1][
								"hR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"R@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"R@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"R@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"mR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"mR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"mR@50"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"hR@10"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"hR@20"],
							future_frame_results_json[test_num_future_frames][mode][train_future_frame][method_name][2][
								"hR@50"]
						])


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


def generate_combined_recalls_latex_header(setting_name, metric, train_horizon, mode, eval_horizon):
	tab_name = fetch_ref_tab_name(mode, eval_horizon, train_horizon)
	latex_header = "\\begin{table}[!h]\n"
	latex_header += "    \\centering\n"
	latex_header += "    \\captionsetup{font=small}\n"
	latex_header += "    \\caption{Results for " + setting_name + ", when trained using anticipatory horizon of " + train_horizon + " future frames.}\n"
	latex_header += "    \\label{tab:anticipation_results_" + tab_name + "_" + metric + "}\n"
	latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
	latex_header += "    \\resizebox{\\textwidth}{!}{\n"
	latex_header += "    \\begin{tabular}{ll|cccccc|cccccc}\n"
	latex_header += "    \\hline\n"
	latex_header += "        \\multicolumn{2}{c}{\\textbf{" + setting_name + "}} & \\multicolumn{6}{c}{\\textbf{With Constraint}} & \\multicolumn{6}{c}{\\textbf{No Constraint}} \\\\ \n"
	latex_header += "        \\cmidrule(lr){1-2}\\cmidrule(lr){3-8} \\cmidrule(lr){9-14} \n "
	
	latex_header += ("        $\\mathcal{F}$ & \\textbf{Method} & \\textbf{R@10} & \\textbf{R@20} & \\textbf{R@50} & "
	                 "\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
	                 "\\textbf{R@10} & \\textbf{R@20} & \\textbf{R@50} & "
	                 "\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}   \\\\ \\hline\n")
	return latex_header


def generate_combined_wn_recalls_latex_header(setting_name, metric, train_horizon, mode, eval_horizon):
	tab_name = fetch_ref_tab_name(mode, eval_horizon, train_horizon)
	latex_header = "\\begin{table}[!h]\n"
	latex_header += "    \\centering\n"
	latex_header += "    \\captionsetup{font=small}\n"
	latex_header += "    \\caption{Results for " + setting_name + ", when trained using anticipatory horizon of " + train_horizon + " future frames.}\n"
	latex_header += "    \\label{tab:anticipation_results_" + tab_name + "_" + metric + "}\n"
	latex_header += "    \\setlength{\\tabcolsep}{5pt} \n"
	latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
	latex_header += "    \\resizebox{\\textwidth}{!}{\n"
	latex_header += "    \\begin{tabular}{ll|cccccc|cccccc}\n"
	latex_header += "    \\hline\n"
	latex_header += "         & & \\multicolumn{6}{c|}{\\textbf{Recall (R)}} & \\multicolumn{6}{c}{\\textbf{Mean Recall (mR)}} \\\\ \n"
	latex_header += "        \\cmidrule(lr){3-8} \\cmidrule(lr){9-14} \n "
	latex_header += "        \\multicolumn{2}{c|}{\\textbf{" + setting_name + "}} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c|}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}}\\\\ \n"
	latex_header += "        \\cmidrule(lr){1-2}\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \n "
	latex_header += ("        $\\mathcal{F}$ & \\textbf{Method} & \\textbf{10} & \\textbf{20} & \\textbf{50} & "
	                 "\\textbf{10} & \\textbf{20} & \\textbf{50} & "
	                 "\\textbf{10} & \\textbf{20} & \\textbf{50}  & "
	                 "\\textbf{10} & \\textbf{20} & \\textbf{50}   \\\\ \\hline\n")
	return latex_header


def generate_latex_footer():
	latex_footer = "    \\end{tabular}\n"
	latex_footer += "    }\n"
	latex_footer += "\\end{table}\n"
	return latex_footer


def fetch_value(value_string):
	if value_string == "-":
		return 0.0
	else:
		return round(float(value_string), 1)


def fetch_rounded_value(value):
	return round(float(value), 1)


def fill_combined_context_fraction_values_matrix(values_matrix, idx, method_name, context_results_json,
                                                 context_fraction, mode,
                                                 train_num_future_frame):
	method_name = fetch_method_name_json(method_name)
	values_matrix[idx, 0] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@10"])
	values_matrix[idx, 1] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@20"])
	values_matrix[idx, 2] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@50"])
	values_matrix[idx, 3] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@10"])
	values_matrix[idx, 4] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@20"])
	values_matrix[idx, 5] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@50"])
	values_matrix[idx, 6] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@10"])
	values_matrix[idx, 7] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@20"])
	values_matrix[idx, 8] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@50"])
	values_matrix[idx, 9] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@10"])
	values_matrix[idx, 10] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@20"])
	values_matrix[idx, 11] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@50"])
	return values_matrix


def fill_combined_wn_context_fraction_values_matrix(values_matrix, idx, method_name, context_results_json,
                                                    context_fraction, mode,
                                                    train_num_future_frame):
	method_name = fetch_method_name_json(method_name)
	values_matrix[idx, 0] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@10"])
	values_matrix[idx, 1] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@20"])
	values_matrix[idx, 2] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["R@50"])
	values_matrix[idx, 3] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@10"])
	values_matrix[idx, 4] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@20"])
	values_matrix[idx, 5] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["R@50"])
	values_matrix[idx, 6] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@10"])
	values_matrix[idx, 7] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@20"])
	values_matrix[idx, 8] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][0]["mR@50"])
	values_matrix[idx, 9] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@10"])
	values_matrix[idx, 10] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@20"])
	values_matrix[idx, 11] = fetch_value(
		context_results_json[context_fraction][mode][train_num_future_frame][method_name][1]["mR@50"])
	return values_matrix


def fill_combined_future_frame_values_matrix(values_matrix, idx, method_name, context_results_json, test_future_frame,
                                             mode, train_num_future_frame):
	method_name = fetch_method_name_json(method_name)
	values_matrix[idx, 0] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@10"])
	values_matrix[idx, 1] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@20"])
	values_matrix[idx, 2] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@50"])
	values_matrix[idx, 3] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@10"])
	values_matrix[idx, 4] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@20"])
	values_matrix[idx, 5] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@50"])
	values_matrix[idx, 6] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@10"])
	values_matrix[idx, 7] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@20"])
	values_matrix[idx, 8] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@50"])
	values_matrix[idx, 9] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@10"])
	values_matrix[idx, 10] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@20"])
	values_matrix[idx, 11] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@50"])
	return values_matrix


def fill_combined_wn_future_frame_values_matrix(values_matrix, idx, method_name, context_results_json,
                                                test_future_frame,
                                                mode, train_num_future_frame):
	method_name = fetch_method_name_json(method_name)
	values_matrix[idx, 0] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@10"])
	values_matrix[idx, 1] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@20"])
	values_matrix[idx, 2] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["R@50"])
	values_matrix[idx, 3] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@10"])
	values_matrix[idx, 4] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@20"])
	values_matrix[idx, 5] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["R@50"])
	values_matrix[idx, 6] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@10"])
	values_matrix[idx, 7] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@20"])
	values_matrix[idx, 8] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][0]["mR@50"])
	values_matrix[idx, 9] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@10"])
	values_matrix[idx, 10] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@20"])
	values_matrix[idx, 11] = fetch_value(
		context_results_json[test_future_frame][mode][train_num_future_frame][method_name][1]["mR@50"])
	return values_matrix
