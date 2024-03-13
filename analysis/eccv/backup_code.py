# def generate_paper_combined_context_mean_recall_vertical_latex_tables(context_results_json):
# 	for mode in modes:
# 		for train_num_future_frame in train_future_frame_loss_list:
# 			latex_file_name = f"mean_recall_{mode}_{train_num_future_frame}.txt"
# 			latex_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
# 			                               "paper_combined_context_vertical_latex_tables", latex_file_name)
# 			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
#
# 			setting_name = fetch_setting_name(mode)
#
# 			latex_table = generate_latex_header(setting_name, "mean_recall", train_horizon=train_num_future_frame)
#
# 			for context_fraction in context_fraction_list:
# 				values_matrix = np.zeros((6, 6), dtype=np.float32)
#
# 				for idx, model_name in enumerate(methods):
# 					model_name = fetch_method_name(model_name)
# 					values_matrix[idx, 0] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]["mR@10"])
# 					values_matrix[idx, 1] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]["mR@20"])
# 					values_matrix[idx, 2] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]["mR@50"])
# 					values_matrix[idx, 3] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]["mR@10"])
# 					values_matrix[idx, 4] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]["mR@20"])
# 					values_matrix[idx, 5] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]["mR@50"])
#
# 				max_boolean_matrix = values_matrix == np.max(values_matrix, axis=0)
#
# 				for idx, model_name in enumerate(methods):
# 					model_name = fetch_method_name(model_name)
#
# 					initial_string = ""
# 					if idx == 0:
# 						initial_string = f"        \\multirow{{4}}{{*}}{{{context_fraction}}} & {model_name}"
# 					elif idx in [1, 2, 3, 4, 5]:
# 						initial_string = f"        & {model_name}"
#
# 					latex_row = initial_string
# 					for col_idx in range(6):
# 						if max_boolean_matrix[idx, col_idx]:
# 							latex_row += f" & \\textbf{{{fetch_rounded_value(values_matrix[idx, col_idx])}}}"
# 						else:
# 							latex_row += f" & {fetch_rounded_value(values_matrix[idx, col_idx])}"
#
# 					if idx == 5:
# 						latex_row += "  \\\\ \\hline\n"
# 					else:
# 						latex_row += "  \\\\ \n"
#
# 					latex_table += latex_row
# 			latex_footer = generate_latex_footer()
# 			latex_table += latex_footer
#
# 			with open(latex_file_path, "a", newline='') as latex_file:
# 				latex_file.write(latex_table)
#
#
# def generate_paper_combined_future_frame_recall_vertical_latex_tables(future_frame_results_json):
# 	for mode in modes:
# 		for train_num_future_frame in train_future_frame_loss_list:
# 			latex_file_name = f"recall_{mode}_{train_num_future_frame}.txt"
# 			latex_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
# 			                               "paper_combined_future_frame_vertical_latex_tables", latex_file_name)
# 			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
#
# 			setting_name = fetch_setting_name(mode)
#
# 			latex_table = generate_latex_header(setting_name, "recall", train_horizon=train_num_future_frame)
#
# 			for test_future_frame in test_future_frame_list:
# 				values_matrix = np.zeros((6, 6), dtype=np.float32)
#
# 				for idx, model_name in enumerate(methods):
# 					model_name = fetch_method_name(model_name)
# 					values_matrix[idx, 0] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][0][
# 							"R@10"])
# 					values_matrix[idx, 1] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][0][
# 							"R@20"])
# 					values_matrix[idx, 2] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][0][
# 							"R@50"])
# 					values_matrix[idx, 3] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][1][
# 							"R@10"])
# 					values_matrix[idx, 4] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][1][
# 							"R@20"])
# 					values_matrix[idx, 5] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][1][
# 							"R@50"])
#
# 				max_boolean_matrix = values_matrix == np.max(values_matrix, axis=0)
#
# 				for idx, model_name in enumerate(methods):
# 					model_name = fetch_method_name(model_name)
#
# 					initial_string = ""
# 					if idx == 0:
# 						initial_string = f"        \\multirow{{4}}{{*}}{{{test_future_frame}}} & {model_name}"
# 					elif idx in [1, 2, 3, 4, 5]:
# 						initial_string = f"        & {model_name}"
#
# 					latex_row = initial_string
# 					for col_idx in range(6):
# 						if max_boolean_matrix[idx, col_idx]:
# 							latex_row += f" & \\textbf{{{fetch_rounded_value(values_matrix[idx, col_idx])}}}"
# 						else:
# 							latex_row += f" & {fetch_rounded_value(values_matrix[idx, col_idx])}"
#
# 					if idx == 5:
# 						latex_row += "  \\\\ \\hline\n"
# 					else:
# 						latex_row += "  \\\\ \n"
#
# 					latex_table += latex_row
# 			latex_footer = generate_latex_footer()
# 			latex_table += latex_footer
#
# 			with open(latex_file_path, "a", newline='') as latex_file:
# 				latex_file.write(latex_table)
#
#
# def generate_paper_combined_future_frame_mean_recall_vertical_latex_tables(future_frame_results_json):
# 	for mode in modes:
# 		for train_num_future_frame in train_future_frame_loss_list:
# 			latex_file_name = f"mean_recall_{mode}_{train_num_future_frame}.txt"
# 			latex_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
# 			                               "paper_combined_future_frame_vertical_latex_tables", latex_file_name)
# 			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
#
# 			setting_name = fetch_setting_name(mode)
#
# 			latex_table = generate_latex_header(setting_name, "mean_recall", train_horizon=train_num_future_frame)
#
# 			for test_future_frame in test_future_frame_list:
# 				values_matrix = np.zeros((6, 6), dtype=np.float32)
#
# 				for idx, model_name in enumerate(methods):
# 					model_name = fetch_method_name(model_name)
# 					values_matrix[idx, 0] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][0][
# 							"mR@10"])
# 					values_matrix[idx, 1] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][0][
# 							"mR@20"])
# 					values_matrix[idx, 2] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][0][
# 							"mR@50"])
# 					values_matrix[idx, 3] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][1][
# 							"mR@10"])
# 					values_matrix[idx, 4] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][1][
# 							"mR@20"])
# 					values_matrix[idx, 5] = fetch_value(
# 						future_frame_results_json[test_future_frame][mode][train_num_future_frame][model_name][1][
# 							"mR@50"])
#
# 				max_boolean_matrix = values_matrix == np.max(values_matrix, axis=0)
#
# 				for idx, model_name in enumerate(methods):
# 					model_name = fetch_method_name(model_name)
#
# 					initial_string = ""
# 					if idx == 0:
# 						initial_string = f"        \\multirow{{4}}{{*}}{{{test_future_frame}}} & {model_name}"
# 					elif idx in [1, 2, 3, 4, 5]:
# 						initial_string = f"        & {model_name}"
#
# 					latex_row = initial_string
# 					for col_idx in range(6):
# 						if max_boolean_matrix[idx, col_idx]:
# 							latex_row += f" & \\textbf{{{fetch_rounded_value(values_matrix[idx, col_idx])}}}"
# 						else:
# 							latex_row += f" & {fetch_rounded_value(values_matrix[idx, col_idx])}"
#
# 					if idx == 5:
# 						latex_row += "  \\\\ \\hline\n"
# 					else:
# 						latex_row += "  \\\\ \n"
#
# 					latex_table += latex_row
# 			latex_footer = generate_latex_footer()
# 			latex_table += latex_footer
#
# 			with open(latex_file_path, "a", newline='') as latex_file:
# 				latex_file.write(latex_table)


# def generate_paper_combined_context_recall_vertical_latex_tables(context_results_json):
# 	for mode in modes:
# 		for train_num_future_frame in train_future_frame_loss_list:
# 			latex_file_name = f"recall_{mode}_{train_num_future_frame}.txt"
# 			latex_file_path = os.path.join(os.path.dirname(__file__), "results_docs",
# 			                               "paper_combined_context_vertical_latex_tables", latex_file_name)
# 			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
#
# 			setting_name = fetch_setting_name(mode)
#
# 			latex_table = generate_latex_header(setting_name, "recall", train_horizon=train_num_future_frame)
#
# 			for context_fraction in context_fraction_list:
# 				values_matrix = np.zeros((6, 6), dtype=np.float32)
#
# 				for idx, model_name in enumerate(methods):
# 					model_name = fetch_method_name(model_name)
# 					values_matrix[idx, 0] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]["R@10"])
# 					values_matrix[idx, 1] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]["R@20"])
# 					values_matrix[idx, 2] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][0]["R@50"])
# 					values_matrix[idx, 3] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]["R@10"])
# 					values_matrix[idx, 4] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]["R@20"])
# 					values_matrix[idx, 5] = fetch_value(
# 						context_results_json[context_fraction][mode][train_num_future_frame][model_name][1]["R@50"])
#
# 				max_boolean_matrix = values_matrix == np.max(values_matrix, axis=0)
#
# 				for idx, model_name in enumerate(methods):
# 					model_name = fetch_method_name(model_name)
#
# 					initial_string = ""
# 					if idx == 0:
# 						initial_string = f"        \\multirow{{4}}{{*}}{{{context_fraction}}} & {model_name}"
# 					elif idx in [1, 2, 3, 4, 5]:
# 						initial_string = f"        & {model_name}"
#
# 					latex_row = initial_string
# 					for col_idx in range(6):
# 						if max_boolean_matrix[idx, col_idx]:
# 							latex_row += f" & \\textbf{{{fetch_rounded_value(values_matrix[idx, col_idx])}}}"
# 						else:
# 							latex_row += f" & {fetch_rounded_value(values_matrix[idx, col_idx])}"
#
# 					if idx == 5:
# 						latex_row += "  \\\\ \\hline\n"
# 					else:
# 						latex_row += "  \\\\ \n"
#
# 					latex_table += latex_row
# 			latex_footer = generate_latex_footer()
# 			latex_table += latex_footer
#
# 			with open(latex_file_path, "a", newline='') as latex_file:
# 				latex_file.write(latex_table)
#
