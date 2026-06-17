from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist


def _box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(0.0, boxes1[:, 3] - boxes1[:, 1])
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(0.0, boxes2[:, 3] - boxes2[:, 1])

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h
    union = area1[:, None] + area2[None, :] - inter
    union = np.maximum(union, 1e-8)
    return inter / union


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])

    change_points = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change_points + 1] - mrec[change_points]) * mpre[change_points + 1]))


def _collect_class_data(predictions, targets, class_id: int):
    gt_by_image: Dict[int, List[np.ndarray]] = {}
    preds: List[Tuple[int, float, np.ndarray]] = []

    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        target_labels = target["labels"].detach().cpu().numpy()
        target_boxes = target["boxes"].detach().cpu().numpy()
        gt_boxes = target_boxes[target_labels == class_id]
        gt_by_image[image_index] = [box for box in gt_boxes]

        pred_labels = prediction["labels"].detach().cpu().numpy()
        pred_boxes = prediction["boxes"].detach().cpu().numpy()
        pred_scores = prediction["scores"].detach().cpu().numpy()
        class_mask = pred_labels == class_id
        for box, score in zip(pred_boxes[class_mask], pred_scores[class_mask]):
            preds.append((image_index, float(score), box))

    preds.sort(key=lambda item: item[1], reverse=True)
    return gt_by_image, preds


def _evaluate_class_at_iou(predictions, targets, class_id: int, iou_threshold: float) -> float:
    gt_by_image, preds = _collect_class_data(predictions, targets, class_id)
    total_gt = sum(len(boxes) for boxes in gt_by_image.values())
    if total_gt == 0:
        return float("nan")

    matched = {
        image_index: np.zeros(len(boxes), dtype=bool)
        for image_index, boxes in gt_by_image.items()
    }
    true_positives = np.zeros(len(preds), dtype=np.float32)
    false_positives = np.zeros(len(preds), dtype=np.float32)

    for pred_index, (image_index, _, pred_box) in enumerate(preds):
        gt_boxes = np.asarray(gt_by_image[image_index], dtype=np.float32)
        if gt_boxes.size == 0:
            false_positives[pred_index] = 1.0
            continue

        ious = _box_iou(np.asarray([pred_box], dtype=np.float32), gt_boxes)[0]
        best_gt_index = int(np.argmax(ious))
        best_iou = float(ious[best_gt_index])

        if best_iou >= iou_threshold and not matched[image_index][best_gt_index]:
            true_positives[pred_index] = 1.0
            matched[image_index][best_gt_index] = True
        else:
            false_positives[pred_index] = 1.0

    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    recalls = tp_cumsum / max(float(total_gt), 1e-8)
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-8)
    return _compute_ap(recalls, precisions)


def evaluate_detection_map(
        model,
        dataloader,
        device,
        num_classes: int,
        iou_thresholds=None,
):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            predictions = model(images)
            cpu_predictions = [{key: value.detach().cpu() for key, value in pred.items()} for pred in predictions]
            cpu_targets = [{key: value.detach().cpu() for key, value in target.items()} for target in targets]
            all_predictions.extend(cpu_predictions)
            all_targets.extend(cpu_targets)

    per_threshold_class_ap = {float(threshold): [] for threshold in iou_thresholds}
    for class_id in range(1, num_classes):
        for threshold in iou_thresholds:
            ap = _evaluate_class_at_iou(all_predictions, all_targets, class_id, float(threshold))
            if not np.isnan(ap):
                per_threshold_class_ap[float(threshold)].append(ap)

    threshold_means = {}
    for threshold, ap_values in per_threshold_class_ap.items():
        threshold_means[threshold] = float(np.mean(ap_values)) if ap_values else 0.0

    per_class_map = {}
    for class_id in range(1, num_classes):
        class_threshold_aps = []
        for threshold in iou_thresholds:
            ap = _evaluate_class_at_iou(all_predictions, all_targets, class_id, float(threshold))
            if not np.isnan(ap):
                class_threshold_aps.append(ap)
        if class_threshold_aps:
            per_class_map[class_id] = float(np.mean(class_threshold_aps))

    metrics = {
        "map": float(np.mean(list(threshold_means.values()))) if threshold_means else 0.0,
        "map_50": threshold_means.get(0.5, 0.0),
        "map_75": threshold_means.get(0.75, 0.0),
        "per_threshold_ap": threshold_means,
        "map_per_class": per_class_map,
        "num_eval_images": len(all_targets),
    }
    return metrics


def evaluate_detection_map_distributed(
        model,
        dataloader,
        device,
        num_classes: int,
        iou_thresholds=None,
):
    if not dist.is_available() or not dist.is_initialized():
        return evaluate_detection_map(
            model=model,
            dataloader=dataloader,
            device=device,
            num_classes=num_classes,
            iou_thresholds=iou_thresholds,
        )

    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    model.eval()
    local_predictions = []
    local_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [image.to(device) for image in images]
            predictions = model(images)
            cpu_predictions = [{key: value.detach().cpu() for key, value in pred.items()} for pred in predictions]
            cpu_targets = [{key: value.detach().cpu() for key, value in target.items()} for target in targets]
            local_predictions.extend(cpu_predictions)
            local_targets.extend(cpu_targets)

    world_size = dist.get_world_size()
    gathered_predictions = [None for _ in range(world_size)]
    gathered_targets = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_predictions, local_predictions)
    dist.all_gather_object(gathered_targets, local_targets)

    if dist.get_rank() != 0:
        return None

    all_predictions = []
    all_targets = []
    for rank_predictions in gathered_predictions:
        all_predictions.extend(rank_predictions)
    for rank_targets in gathered_targets:
        all_targets.extend(rank_targets)

    per_threshold_class_ap = {float(threshold): [] for threshold in iou_thresholds}
    for class_id in range(1, num_classes):
        for threshold in iou_thresholds:
            ap = _evaluate_class_at_iou(all_predictions, all_targets, class_id, float(threshold))
            if not np.isnan(ap):
                per_threshold_class_ap[float(threshold)].append(ap)

    threshold_means = {}
    for threshold, ap_values in per_threshold_class_ap.items():
        threshold_means[threshold] = float(np.mean(ap_values)) if ap_values else 0.0

    per_class_map = {}
    for class_id in range(1, num_classes):
        class_threshold_aps = []
        for threshold in iou_thresholds:
            ap = _evaluate_class_at_iou(all_predictions, all_targets, class_id, float(threshold))
            if not np.isnan(ap):
                class_threshold_aps.append(ap)
        if class_threshold_aps:
            per_class_map[class_id] = float(np.mean(class_threshold_aps))

    return {
        "map": float(np.mean(list(threshold_means.values()))) if threshold_means else 0.0,
        "map_50": threshold_means.get(0.5, 0.0),
        "map_75": threshold_means.get(0.75, 0.0),
        "per_threshold_ap": threshold_means,
        "map_per_class": per_class_map,
        "num_eval_images": len(all_targets),
    }
