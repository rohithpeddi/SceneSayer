# This file contains all files related to firebase
import pyrebase
from logger_config import get_logger
from constants import ResultConstants as const

logger = get_logger(__name__)

firebaseProdConfig = {
	"apiKey": "AIzaSyBDL886IT8dIY-AFrTNRwNtxn3RocNKEZM",
	"authDomain": "dsga-48082.firebaseapp.com",
	"databaseURL": "https://dsga-48082-default-rtdb.firebaseio.com",
	"projectId": "dsga-48082",
	"storageBucket": "dsga-48082.appspot.com",
	"messagingSenderId": "132679038200",
	"appId": "1:132679038200:web:b53e1a3ee8315049d0770d"
}

firebase = pyrebase.initialize_app(firebaseProdConfig)

logger.info("----------------------------------------------------------------------")
logger.info("Setting up Firebase Service...in ")
logger.info("----------------------------------------------------------------------")


class FirebaseService:
	
	# 1. Have this db connection constant
	def __init__(self):
		self.db = firebase.database()
	
	# ---------------------- BEGIN RESULTS ----------------------
	def fetch_results(self):
		return self.db.child(const.RESULTS).get().val()
	
	# ---------------------- END RESULTS ----------------------
	
	# ---------------------- BEGIN RESULT ----------------------
	
	def fetch_result(self, result_id: str):
		return self.db.child(const.RESULTS).child(result_id).get().val()
	
	def remove_all_results(self):
		self.db.child(const.RESULTS).remove()
	
	def update_result(self, result_id: str, result: dict):
		self.db.child(const.RESULTS).child(result_id).set(result)
		logger.info(f"Updated result in the firebase - {result.__str__()}")
	
	def remove_result(self, result_id: str):
		self.db.child(const.RESULTS).child(result_id).remove()
	
	# ---------------------- END RESULT ----------------------


if __name__ == "__main__":
	db_service = FirebaseService()
