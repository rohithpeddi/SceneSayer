import logging
import os

import logging.config


def setup_logging(filename='std.log', default_level=logging.INFO):
	log_directory = os.path.join(os.getcwd(), 'logs')
	if not os.path.exists(log_directory):
		os.makedirs(log_directory)
	
	log_file_path = os.path.join(log_directory, f"{filename}")
	logging.basicConfig(filename=log_file_path, filemode='a', level=default_level,
	                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s')


def get_logger(name):
	return logging.getLogger(name)

