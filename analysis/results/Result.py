import uuid
from datetime import datetime

from constants import ResultConstants as const


class Metrics:
	
	def __init__(
			self,
			recall_10=None,
			recall_20=None,
			recall_50=None,
			recall_100=None,
			mean_recall_10=None,
			mean_recall_20=None,
			mean_recall_50=None,
			mean_recall_100=None,
			harmonic_recall_10=None,
			harmonic_recall_20=None,
			harmonic_recall_50=None,
			harmonic_recall_100=None,
	):
		self.recall_10 = recall_10
		self.recall_20 = recall_20
		self.recall_50 = recall_50
		self.recall_100 = recall_100
		self.mean_recall_10 = mean_recall_10
		self.mean_recall_20 = mean_recall_20
		self.mean_recall_50 = mean_recall_50
		self.mean_recall_100 = mean_recall_100
		self.harmonic_recall_10 = harmonic_recall_10
		self.harmonic_recall_20 = harmonic_recall_20
		self.harmonic_recall_50 = harmonic_recall_50
		self.harmonic_recall_100 = harmonic_recall_100
	
	def to_dict(self):
		return {
			const.RECALL_10: self.recall_10,
			const.RECALL_20: self.recall_20,
			const.RECALL_50: self.recall_50,
			const.RECALL_100: self.recall_100,
			const.MEAN_RECALL_10: self.mean_recall_10,
			const.MEAN_RECALL_20: self.mean_recall_20,
			const.MEAN_RECALL_50: self.mean_recall_50,
			const.MEAN_RECALL_100: self.mean_recall_100,
			const.HARMONIC_RECALL_10: self.harmonic_recall_10,
			const.HARMONIC_RECALL_20: self.harmonic_recall_20,
			const.HARMONIC_RECALL_50: self.harmonic_recall_50,
			const.HARMONIC_RECALL_100: self.harmonic_recall_100,
		}
	
	@classmethod
	def from_dict(cls, metrics_dict):
		return cls(
			recall_10=metrics_dict[const.RECALL_10],
			recall_20=metrics_dict[const.RECALL_20],
			recall_50=metrics_dict[const.RECALL_50],
			recall_100=metrics_dict[const.RECALL_100] if const.RECALL_100 in metrics_dict else None,
			mean_recall_10=metrics_dict[const.MEAN_RECALL_10],
			mean_recall_20=metrics_dict[const.MEAN_RECALL_20],
			mean_recall_50=metrics_dict[const.MEAN_RECALL_50],
			mean_recall_100=metrics_dict[const.MEAN_RECALL_100] if const.MEAN_RECALL_100 in metrics_dict else None,
			harmonic_recall_10=metrics_dict[const.HARMONIC_RECALL_10],
			harmonic_recall_20=metrics_dict[const.HARMONIC_RECALL_20],
			harmonic_recall_50=metrics_dict[const.HARMONIC_RECALL_50],
			harmonic_recall_100=metrics_dict[const.HARMONIC_RECALL_100] if const.HARMONIC_RECALL_100 in metrics_dict else None,
		)


class ResultDetails:
	
	def __init__(
			self
	):
		self.with_constraint_metrics = None
		self.no_constraint_metrics = None
		self.semi_constraint_metrics = None
	
	def add_with_constraint_metrics(self, metrics):
		self.with_constraint_metrics = metrics
	
	def add_no_constraint_metrics(self, metrics):
		self.no_constraint_metrics = metrics
	
	def add_semi_constraint_metrics(self, metrics):
		self.semi_constraint_metrics = metrics
	
	def to_dict(self):
		result_details_dict = {}
		if self.with_constraint_metrics is not None:
			result_details_dict['with_constraint_metrics'] = self.with_constraint_metrics.to_dict()
		if self.no_constraint_metrics is not None:
			result_details_dict['no_constraint_metrics'] = self.no_constraint_metrics.to_dict()
		if self.semi_constraint_metrics is not None:
			result_details_dict['semi_constraint_metrics'] = self.semi_constraint_metrics.to_dict()
		
		return result_details_dict
	
	@classmethod
	def from_dict(cls, result_details_dict):
		result_details = cls()
		if 'with_constraint_metrics' in result_details_dict:
			result_details.add_with_constraint_metrics(
				Metrics.from_dict(result_details_dict['with_constraint_metrics']))
		if 'no_constraint_metrics' in result_details_dict:
			result_details.add_no_constraint_metrics(Metrics.from_dict(result_details_dict['no_constraint_metrics']))
		if 'semi_constraint_metrics' in result_details_dict:
			result_details.add_semi_constraint_metrics(
				Metrics.from_dict(result_details_dict['semi_constraint_metrics']))
		
		return result_details


class Result:
	
	def __init__(
			self,
			task_name,
			method_name,
			mode,
			result_id=None,
			train_future_frames=None,
			test_future_frames=None,
			context_fraction=None
	):
		self.task_name = task_name
		self.method_name = method_name
		self.mode = mode
		
		# Specific Attributes
		self.train_num_future_frames = train_future_frames
		self.test_num_future_frames = test_future_frames
		self.context_fraction = context_fraction
		
		# Common Attributes
		self.result_details = None
		
		if result_id is None:
			self.result_id = str(uuid.uuid4())
		else:
			self.result_id = result_id
			
		self.result_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	
	def add_result_details(self, result_details):
		self.result_details = result_details
	
	def to_dict(self):
		result_dict = {
			const.TASK_NAME: self.task_name,
			const.METHOD_NAME: self.method_name,
			const.MODE: self.mode,
			const.RESULT_ID: self.result_id,
			const.DATE: self.result_date
		}
		
		if self.train_num_future_frames is not None:
			result_dict[const.TRAIN_NUM_FUTURE_FRAMES] = self.train_num_future_frames
		
		if self.test_num_future_frames is not None:
			result_dict[const.TEST_NUM_FUTURE_FRAMES] = self.test_num_future_frames
			
		if self.context_fraction is not None:
			result_dict[const.CONTEXT_FRACTION] = self.context_fraction
			
		if self.result_details is not None:
			result_dict[const.RESULT_DETAILS] = self.result_details.to_dict()
		
		return result_dict
	
	@classmethod
	def from_dict(cls, result_dict):
		result = cls(
			task_name=result_dict[const.TASK_NAME],
			method_name=result_dict[const.METHOD_NAME],
			mode=result_dict[const.MODE],
			result_id=result_dict[const.RESULT_ID],
		)
		
		if const.TRAIN_NUM_FUTURE_FRAMES in result_dict:
			result.train_num_future_frames = result_dict[const.TRAIN_NUM_FUTURE_FRAMES]
		
		if const.TEST_NUM_FUTURE_FRAMES in result_dict:
			result.test_num_future_frames = result_dict[const.TEST_NUM_FUTURE_FRAMES]
		
		if const.CONTEXT_FRACTION in result_dict:
			result.context_fraction = result_dict[const.CONTEXT_FRACTION]
		
		if const.RESULT_DETAILS in result_dict:
			result.add_result_details(ResultDetails.from_dict(result_dict[const.RESULT_DETAILS]))
			
		if const.DATE in result_dict:
			result.result_date = result_dict[const.DATE]
		
		return result
