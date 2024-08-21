class Constants:
	DISTRIBUTION = "distribution"
	INDICES = "indices"
	LABELS = "labels"
	FRAME = "frame"
	FRAMES = "frames"
	CLASS = "class"
	BACKGROUND = "__background__"
	ANNOTATIONS = "annotations"
	MINI = "mini"
	METADATA = "metadata"
	SET = "set"
	VISIBLE = "visible"
	BOUNDING_BOX = "bbox"
	UNION_BOX = "union_box"
	PERSON_BOUNDING_BOX = "person_bbox"
	BOUNDING_BOX_SIZE = "bbox_size"
	STATE_DICT = "state_dict"
	FEATURES = "features"
	PAIR_IDX = "pair_idx"
	IM_IDX = "im_idx"
	HUMAN_IDX = "human_idx"
	IM_INFO = "im_info"
	UNION_FEAT = "union_feat"
	SPATIAL_MASKS = "spatial_masks"
	PRED_LABELS = "pred_labels"
	BOXES = "boxes"
	FMAPS = "fmaps"
	PRED_SCORES = "pred_scores"
	ATTENTION_DISTRIBUTION = "attention_distribution"
	SPATIAL_DISTRIBUTION = "spatial_distribution"
	CONTACTING_DISTRIBUTION = "contacting_distribution"
	ADAM = "adam"
	ADAMW = "adamw"
	SGD = "sgd"
	OBJECT_LOSS = "object_loss"
	FRAMES_PREDCLS = "frames_predcls"
	FRAMES_SGCLS = "frames_sgcls"
	FRAMES_SGDET = "frames_sgdet"
	
	GLOVE_6B = "glove.6B"
	
	SGDET = "sgdet"
	SGCLS = "sgcls"
	PREDCLS = "predcls"
	
	TRAIN = "train"
	TEST = "test"

	FRAME_IDX = "frame_idx"
	
	OBJECT_CLASSES_FILE = "object_classes.txt"
	RELATIONSHIP_CLASSES_FILE = "relationship_classes.txt"
	PERSON_BOUNDING_BOX_PKL = "person_bbox.pkl"
	OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL = "object_bbox_and_relationship.pkl"
	PROGRESS_TEXT_FILE = "progress.txt"
	
	ATTENTION_RELATIONSHIP = "attention_relationship"
	SPATIAL_RELATIONSHIP = "spatial_relationship"
	CONTACTING_RELATIONSHIP = "contacting_relationship"
	
	ATTENTION_GT = "attention_gt"
	SPATIAL_GT = "spatial_gt"
	CONTACTING_GT = "contacting_gt"
	GLOBAL_OUTPUT = "global_output"
	
	ATTENTION_RELATION_LOSS = "attention_relation_loss"
	SPATIAL_RELATION_LOSS = "spatial_relation_loss"
	CONTACTING_RELATION_LOSS = "contacting_relation_loss"
	
	CUDA_DEVICE = 0
	NMS_THRESHOLD = 0.4
	SCORE_THRESHOLD = 0.1
	FASTER_RCNN_BATCH_SIZE = 10
	
	FINAL_BBOXES = "FINAL_BBOXES"
	FINAL_SCORES = "FINAL_SCORES"
	FINAL_LABELS = "FINAL_LABELS"
	IMAGE_IDX = "im_idx"
	PAIR_IDX = "pair_idx"
	HUMAN_IDX = "human_idx"
	FINAL_FEATURES = "FINAL_FEATURES"
	UNION_FEATURES = "union_features"
	UNION_BOX = "union_box"
	SPATIAL_MASKS = "spatial_masks"
	SCORES = "scores"
	
	ATTENTION_REL = "a_rel"
	SPATIAL_REL = "s_rel"
	CONTACTING_REL = "c_rel"
	
	FINAL_DISTRIBUTIONS = "FINAL_DISTRIBUTIONS"
	FINAL_BASE_FEATURES = "FINAL_BASE_FEATURES"
	
	DETECTOR_FOUND_IDX = "DETECTOR_FOUND_IDX"
	GT_RELATIONS = "GT_RELATIONS"
	SUPPLY_RELATIONS = "SUPPLY_RELATIONS"
	ASSIGNED_LABELS = "ASSIGNED_LABELS"
	
	FINAL_BBOXES_X = "FINAL_BBOXES_X"
	FINAL_LABELS_X = "FINAL_LABELS_X"
	FINAL_SCORES_X = "FINAL_SCORES_X"
	FINAL_FEATURES_X = "FINAL_FEATURES_X"
	PAIR = "pair"
	
	GT_ANNOTATION = "gt_annotation"
	FRAME_SIZE = "frame_size"
	
	PERSON = "person"
	OBJECT = "object"


class DetectorConstants:
	BOXES = "boxes"
	LABELS = "labels"
	SCORES = "scores"
	DISTRIBUTION = "distribution"
	PRED_LABELS = "pred_labels"
	IM_IDX = "im_idx"
	PAIR_IDX = "pair_idx"
	HUMAN_IDX = "human_idx"
	FEATURES = "features"
	UNION_FEAT = "union_feat"
	UNION_BOX = "union_box"
	SPATIAL_MASKS = "spatial_masks"
	ATTENTION_GT = "attention_gt"
	SPATIAL_GT = "spatial_gt"
	CONTACTING_GT = "contacting_gt"
	FMAPS = "fmaps"
	IM_INFO = "im_info"
	FINAL_PRED_SCORES = "FINAL_PRED_SCORES"

	FRAME = "frame"
	FRAME_IDX = "frame_idx"
	PERSON_BBOX = "person_bbox"
	BBOX = "bbox"
	CLASS = "class"
	ATTENTION_RELATIONSHIP = "attention_relationship"
	SPATIAL_RELATIONSHIP = "spatial_relationship"
	CONTACTING_RELATIONSHIP = "contacting_relationship"
	
	FINAL_BBOXES = "FINAL_BBOXES"
	FINAL_SCORES = "FINAL_SCORES"
	FINAL_LABELS = "FINAL_LABELS"
	IMAGE_IDX = "im_idx"
	PAIR = "pair"
	FINAL_FEATURES = "FINAL_FEATURES"
	ATTENTION_REL = "a_rel"
	SPATIAL_REL = "s_rel"
	CONTACTING_REL = "c_rel"
	FINAL_DISTRIBUTIONS = "FINAL_DISTRIBUTIONS"
	FINAL_BASE_FEATURES = "FINAL_BASE_FEATURES"
	
	FASTER_RCNN_BATCH_SIZE = 10
	
	DETECTOR_FOUND_IDX = "DETECTOR_FOUND_IDX"
	GT_RELATIONS = "GT_RELATIONS"
	SUPPLY_RELATIONS = "SUPPLY_RELATIONS"
	ASSIGNED_LABELS = "ASSIGNED_LABELS"
	
	FINAL_BBOXES_X = "FINAL_BBOXES_X"
	FINAL_LABELS_X = "FINAL_LABELS_X"
	FINAL_SCORES_X = "FINAL_SCORES_X"
	FINAL_FEATURES_X = "FINAL_FEATURES_X"


class DataloaderConstants:
	TRAIN = "train"
	SGDET = "sgdet"
	SGCLS = "sgcls"
	PREDCLS = "predcls"
	
	FEATURES = "features"
	
	SUPERVISED = "supervised"
	
	ADDITIONAL = "additional"
	FRAME_IDX = "frame_idx"
	
	BOXES = "boxes"
	LABELS = "labels"
	SCORES = "scores"
	DISTRIBUTION = "distribution"
	PRED_LABELS = "pred_labels"
	IM_IDX = "im_idx"
	PAIR_IDX = "pair_idx"
	HUMAN_IDX = "human_idx"
	FEATURES = "features"
	UNION_FEAT = "union_feat"
	UNION_BOX = "union_box"
	SPATIAL_MASKS = "spatial_masks"
	ATTENTION_GT = "attention_gt"
	SPATIAL_GT = "spatial_gt"
	CONTACTING_GT = "contacting_gt"
	FMAPS = "fmaps"
	IM_INFO = "im_info"
	FINAL_PRED_SCORES = "FINAL_PRED_SCORES"
	
	PERSON_BBOX = "person_bbox"
	BBOX = "bbox"
	CLASS = "class"
	ATTENTION_RELATIONSHIP = "attention_relationship"
	SPATIAL_RELATIONSHIP = "spatial_relationship"
	CONTACTING_RELATIONSHIP = "contacting_relationship"
	
	FINAL_BBOXES = "FINAL_BBOXES"
	FINAL_SCORES = "FINAL_SCORES"
	FINAL_LABELS = "FINAL_LABELS"
	IMAGE_IDX = "im_idx"
	PAIR = "pair"
	FINAL_FEATURES = "FINAL_FEATURES"
	ATTENTION_REL = "a_rel"
	SPATIAL_REL = "s_rel"
	CONTACTING_REL = "c_rel"
	FINAL_DISTRIBUTIONS = "FINAL_DISTRIBUTIONS"
	FINAL_BASE_FEATURES = "FINAL_BASE_FEATURES"
	
	FASTER_RCNN_BATCH_SIZE = 10
	
	DETECTOR_FOUND_IDX = "DETECTOR_FOUND_IDX"
	GT_RELATIONS = "GT_RELATIONS"
	SUPPLY_RELATIONS = "SUPPLY_RELATIONS"
	ASSIGNED_LABELS = "ASSIGNED_LABELS"
	
	FINAL_BBOXES_X = "FINAL_BBOXES_X"
	FINAL_LABELS_X = "FINAL_LABELS_X"
	FINAL_SCORES_X = "FINAL_SCORES_X"
	FINAL_FEATURES_X = "FINAL_FEATURES_X"
	
	BACKGROUND = "__background__"
	ANNOTATIONS = "annotations"
	OBJECT_CLASSES_FILE = "object_classes.txt"
	RELATIONSHIP_CLASSES_FILE = "relationship_classes.txt"
	PERSON_BOUNDING_BOX_PKL = "person_bbox.pkl"
	OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL = "object_bbox_and_relationship.pkl"
	PROGRESS_TEXT_FILE = "progress.txt"
	OBJECT_BOUNDING_BOX_RELATIONSHIP_FILTERSMALL_PKL = "object_bbox_and_relationship_filtersmall.pkl"
	
	METADATA = "metadata"
	VISIBLE = "visible"
	SET = "set"
	
	BOUNDING_BOX = "bbox"
	PERSON_BOUNDING_BOX = "person_bbox"
	FRAME = "frame"
	BOUNDING_BOX_SIZE = "bbox_size"
	GT_ANNOTATION = "gt_annotation"
	
	FRAME_SIZE = "frame_size"


class ResultConstants:
	RECALL_10 = "recall_10"
	RECALL_20 = "recall_20"
	RECALL_50 = "recall_50"
	RECALL_100 = "recall_100"
	MEAN_RECALL_10 = "mean_recall_10"
	MEAN_RECALL_20 = "mean_recall_20"
	MEAN_RECALL_50 = "mean_recall_50"
	MEAN_RECALL_100 = "mean_recall_100"
	HARMONIC_RECALL_10 = "harmonic_recall_10"
	HARMONIC_RECALL_20 = "harmonic_recall_20"
	HARMONIC_RECALL_50 = "harmonic_recall_50"
	HARMONIC_RECALL_100 = "harmonic_recall_100"
	
	WITH_CONSTRAINT_METRICS = "with_constraint_metrics"
	NO_CONSTRAINT_METRICS = "no_constraint_metrics"
	SEMI_CONSTRAINT_METRICS = "semi_constraint_metrics"
	
	TASK_NAME = "task_name"
	METHOD_NAME = "method_name"
	MODE = "mode"
	TRAIN_NUM_FUTURE_FRAMES = "train_num_future_frames"
	TEST_NUM_FUTURE_FRAMES = "test_num_future_frames"
	CONTEXT_FRACTION = "context_fraction"
	RESULT_DETAILS = "result_details"
	
	RESULTS = "results_eccv"
	# RESULTS = "results"
	RESULT_ID = "result_id"
	DATE = "date"
	
	# Task Names
	DYSGG = "dysgg"
	SGA = "sga"
	
	# Method Names
	DYSTTRAN = "dysttran"
	DYDSGDETR = "dydsgdetr"
	BASELINE_SO_ANT = "baseline_so_ant"
	BASELINE_SO_GEN_LOSS = "baseline_so_gen_loss"
	NEURALODE = "NeuralODE"
	NEURALSDE = "NeuralSDE"
	
	# Modes
	SGDET = "sgdet"
	SGCLS = "sgcls"
	PREDCLS = "predcls"
	
	EVALUATION = "evaluation"
	PERCENTAGE_EVALUATION = "percentage_evaluation"
	GENERATION_IMPACT = "generation_impact"
