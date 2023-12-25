from collections import Counter
import torch

from lib.iou import intersection_over_union


def calculate_average_precision(detections, ground_truths, iou_threshold, box_format):
	"""
	Calculates the Average Precision (AP) for a specific class.

	Parameters:
		detections (list): List of predicted bounding boxes for a specific class.
		ground_truths (list): List of ground truth bounding boxes for a specific class.
		iou_threshold (float): IoU threshold to consider a detection as a true positive.
		box_format (str): Format of bounding boxes, "midpoint" or "corners".

	Returns:
		float: AP for the specific class.
	"""
	# Initialize true positive and false positive tensors
	TP = torch.zeros((len(detections)))
	FP = torch.zeros((len(detections)))
	total_true_bboxes = len(ground_truths)
	
	# find the amount of bboxes for each training example
	# Counter here finds how many ground truth bboxes we get
	# for each training example, so let's say img 0 has 3,
	# img 1 has 5 then we will obtain a dictionary with:
	# amount_bboxes = {0:3, 1:5}
	amount_bboxes = Counter([gt[0] for gt in ground_truths])
	for key, val in amount_bboxes.items():
		amount_bboxes[key] = torch.zeros(val)
	
	# Loop through each detection
	for detection_idx, detection in enumerate(detections):
		# Filter ground_truths for the current training example
		ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
		best_iou = 0
		
		# Loop through each ground truth box to find the best IoU
		for idx, gt in enumerate(ground_truth_img):
			iou = intersection_over_union(
				torch.tensor(detection[3:]),
				torch.tensor(gt[3:]),
				box_format=box_format,
			)
			if iou > best_iou:
				best_iou = iou
				best_gt_idx = idx
		
		# Check if the detection is a true positive or false positive
		if best_iou > iou_threshold:
			if amount_bboxes[detection[0]][best_gt_idx] == 0:
				TP[detection_idx] = 1
				amount_bboxes[detection[0]][best_gt_idx] = 1
			else:
				FP[detection_idx] = 1
		else:
			FP[detection_idx] = 1
	
	# Calculate cumulative sum of true positives and false positives
	TP_cumsum = torch.cumsum(TP, dim=0)
	FP_cumsum = torch.cumsum(FP, dim=0)
	recalls = TP_cumsum / (total_true_bboxes + 1e-6)
	precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
	
	# Concatenate 1 to precisions and 0 to recalls
	precisions = torch.cat((torch.tensor([1]), precisions))
	recalls = torch.cat((torch.tensor([0]), recalls))
	
	# Calculate AP using numerical integration
	average_precision = torch.trapz(precisions, recalls)
	return average_precision


def mean_average_precision(
		pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20
):
	"""
	Calculates the mean Average Precision (mAP) across all classes.

	Parameters:
		pred_boxes (list): List of lists containing all bboxes with each bboxes
	        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
		true_boxes (list): Similar as pred_boxes except all the correct ones
		iou_threshold (float): IoU threshold to consider a detection as a true positive.
		box_format (str): Format of bounding boxes, "midpoint" or "corners".
		num_classes (int): Number of classes in the dataset.

	Returns:
		float: mAP value across all classes given a specific IoU threshold
	"""
	
	average_precisions = []
	
	# Loop through each class
	for class_idx in range(1, num_classes + 1):
		# Filter detections and ground truths for the current class
		detections = [box for box in pred_boxes if box[1] == class_idx]
		ground_truths = [box for box in true_boxes if box[1] == class_idx]
		
		if len(ground_truths) == 0:
			continue
		
		# Calculate AP for the current class
		average_precision = calculate_average_precision(
			detections, ground_truths, iou_threshold, box_format
		)
		average_precisions.append(average_precision)
	
	# Calculate mAP as the mean of average precisions across all classes
	mean_avg_precision = sum(average_precisions) / len(average_precisions)
	return mean_avg_precision
