import copy

import numpy as np
import torch
import torch.nn as nn
from lib.draw_rectangles.draw_rectangles import draw_union_boxes

from constants import DetectorConstants as const
from fasterRCNN.lib.model.faster_rcnn.resnet import resnet
from fasterRCNN.lib.model.roi_layers import nms
from fasterRCNN.lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from lib.supervised.funcs import assign_relations


class Detector(nn.Module):
    """first part: object detection (image/video)"""

    def __init__(self, train, object_classes, use_SUPPLY, mode='predcls'):
        super(Detector, self).__init__()

        self.is_train = train
        self.use_SUPPLY = use_SUPPLY
        self.object_classes = object_classes
        self.mode = mode

        self.device = torch.device('cuda:0')

        self.fasterRCNN = resnet(classes=self.object_classes, num_layers=101, pretrained=False, class_agnostic=False)
        self.fasterRCNN.create_architecture()
        checkpoint = torch.load('fasterRCNN/models/faster_rcnn_ag.pth')
        self.fasterRCNN.load_state_dict(checkpoint['model'])

        self.ROI_Align = copy.deepcopy(self.fasterRCNN.RCNN_roi_align)
        self.RCNN_Head = copy.deepcopy(self.fasterRCNN._head_to_tail)

        self.NMS_THRESHOLD = 0.4
        self.SCORE_THRESHOLD = 0.1

    def _batch_processing(self, counter, data_list):
        if counter + const.FASTER_RCNN_BATCH_SIZE < data_list[0].shape[0]:
            return [data[counter:counter + const.FASTER_RCNN_BATCH_SIZE] for data in data_list]
        return [data[counter:] for data in data_list]

    def _box_regression(self, bbox_pred, rois):
        box_deltas = bbox_pred.data
        box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor([0.1, 0.1, 0.2, 0.2]).to(self.device) \
                     + torch.FloatTensor([0.0, 0.0, 0.0, 0.0]).to(self.device)
        box_deltas = box_deltas.view(-1, rois.shape[1], 4 * len(self.object_classes))  # post_NMS_NTOP: 30
        return box_deltas

    def _nms_for_class(self, region_scores, region_pred_boxes, j, roi_features):
        indices = torch.nonzero(region_scores[:, j] > self.SCORE_THRESHOLD).view(-1)
        if indices.numel() == 0:
            return []

        cls_scores = region_scores[:, j][indices]
        _, order = torch.sort(cls_scores, 0, True)
        cls_boxes = region_pred_boxes[indices][:, j * 4:(j + 1) * 4]
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        cls_dets = cls_dets[order]

        keep = nms(cls_boxes[order, :], cls_scores[order], self.NMS_THRESHOLD)
        cls_dets = cls_dets[keep.view(-1).long()]

        return [
            cls_boxes[order][keep],
            cls_scores[order][keep],
            roi_features[indices[order[keep]]]
        ]

    def _process_results_for_class(self, roi_idx, class_idx, rois, scores, pred_boxes, roi_features, counter_image):
        indices = torch.nonzero(scores[roi_idx][:, class_idx] > self.SCORE_THRESHOLD).view(-1)
        if indices.numel() == 0:
            return []

        cls_scores = scores[roi_idx][:, class_idx][indices]
        _, order = torch.sort(cls_scores, 0, True)
        cls_boxes = pred_boxes[roi_idx][indices][:, class_idx * 4:(class_idx + 1) * 4]
        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
        cls_dets = cls_dets[order]

        keep = nms(cls_boxes[order, :], cls_scores[order], self.NMS_THRESHOLD)
        cls_dets = cls_dets[keep.view(-1).long()]

        if class_idx == 1:
            # for person we only keep the highest score for person!
            final_bbox = cls_dets[0, 0:4].unsqueeze(0)
            final_score = cls_dets[0, 4].unsqueeze(0)
            final_labels = torch.tensor([class_idx]).cuda(0)
            final_features = roi_features[roi_idx, indices[order[keep][0]]].unsqueeze(0)
        else:
            final_bbox = cls_dets[:, 0:4]
            final_score = cls_dets[:, 4]
            final_labels = torch.tensor([class_idx]).repeat(keep.shape[0]).cuda(0)
            final_features = roi_features[roi_idx, indices[order[keep]]]

        final_bbox = torch.cat((torch.tensor([[counter_image]], dtype=torch.float).repeat(
            final_bbox.shape[0], 1).cuda(0), final_bbox), 1)

        return [final_bbox, final_score, final_labels, final_features]

    def _nms_and_collect_results(self, rois, scores, pred_boxes, base_features, roi_features, counter_image,
                                 attribute_list):
        FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, FINAL_FEATURES, FINAL_BASE_FEATURES = attribute_list[0], \
            attribute_list[1], attribute_list[2], attribute_list[3], attribute_list[4]

        for roi_idx in range(rois.shape[0]):
            for class_idx in range(1, len(self.object_classes)):
                process_class_data = self._process_results_for_class(roi_idx, class_idx, rois, scores,
                                                                     pred_boxes, roi_features, counter_image)
                if len(process_class_data) == 0:
                    continue
                final_bbox, final_score, final_labels, final_features = process_class_data[0], process_class_data[1], \
                    process_class_data[2], process_class_data[3]
                FINAL_BBOXES = torch.cat((FINAL_BBOXES, final_bbox), 0)
                FINAL_LABELS = torch.cat((FINAL_LABELS, final_labels), 0)
                FINAL_SCORES = torch.cat((FINAL_SCORES, final_score), 0)
                FINAL_FEATURES = torch.cat((FINAL_FEATURES, final_features), 0)
            FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_features[roi_idx].unsqueeze(0)), 0)
            counter_image += 1

        return FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, FINAL_FEATURES, FINAL_BASE_FEATURES, counter_image

    def _get_unfound_gt_boxes(self, unfound_gt_info, im_info, idx):
        unfound_gt_bboxes = torch.zeros([len(unfound_gt_info), 5]).to(self.device)
        unfound_gt_classes = torch.zeros([len(unfound_gt_info)], dtype=torch.int64).to(self.device)
        one_scores = torch.ones([len(unfound_gt_info)], dtype=torch.float32).to(self.device)
        for m, n in enumerate(unfound_gt_info):
            if 'bbox' in n.keys():
                unfound_gt_bboxes[m, 1:] = torch.tensor(n['bbox']).to(self.device) * im_info[idx, 2]
                unfound_gt_classes[m] = n['class']
            else:
                unfound_gt_bboxes[m, 1:] = torch.tensor(n['person_bbox']).to(self.device) * im_info[idx, 2]
                unfound_gt_classes[m] = 1  # person class index
        return unfound_gt_bboxes, unfound_gt_classes, one_scores

    def _compute_pooled_feat(self, FINAL_BASE_FEATURES, unfound_gt_bboxes):
        pooled_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES.unsqueeze(0),
                                                     unfound_gt_bboxes.to(self.device))
        return self.fasterRCNN._head_to_tail(pooled_feat)

    def _construct_relations(self, DETECTOR_FOUND_IDX, GT_RELATIONS, FINAL_BBOXES_X, global_idx):
        im_idx = []
        pair = []
        a_rel = []
        s_rel = []
        c_rel = []
        for frame_idx, detector_bboxes_idx in enumerate(DETECTOR_FOUND_IDX):
            for frame_bbox_idx, frame_bbox_info in enumerate(GT_RELATIONS[frame_idx]):
                if 'person_bbox' in frame_bbox_info.keys():
                    frame_human_bbox_idx = frame_bbox_idx
                    break
            local_human = int(global_idx[FINAL_BBOXES_X[:, 0] == frame_idx][frame_human_bbox_idx])
            for m, n in enumerate(detector_bboxes_idx):
                if 'class' in GT_RELATIONS[frame_idx][m].keys():
                    im_idx.append(frame_idx)
                    pair.append([local_human, int(global_idx[FINAL_BBOXES_X[:, 0] == frame_idx][int(n)])])
                    a_rel.append(GT_RELATIONS[frame_idx][m]['attention_relationship'].tolist())
                    s_rel.append(GT_RELATIONS[frame_idx][m]['spatial_relationship'].tolist())
                    c_rel.append(GT_RELATIONS[frame_idx][m]['contacting_relationship'].tolist())
        return im_idx, pair, a_rel, s_rel, c_rel

    def _compute_union_boxes(self, FINAL_BBOXES_X, pair, im_info, im_idx):
        im_idx = torch.tensor(im_idx, dtype=torch.float).to(self.device)
        union_boxes = torch.cat((im_idx[:, None],
                                 torch.min(FINAL_BBOXES_X[:, 1:3][pair[:, 0]],
                                           FINAL_BBOXES_X[:, 1:3][pair[:, 1]]),
                                 torch.max(FINAL_BBOXES_X[:, 3:5][pair[:, 0]],
                                           FINAL_BBOXES_X[:, 3:5][pair[:, 1]])), 1)
        union_boxes[:, 1:] = union_boxes[:, 1:] * im_info[0, 2]
        return union_boxes

    def _augment_gt_annotation(self, prediction, gt_annotation, im_info):
        # Extract values from prediction
        DETECTOR_FOUND_IDX = prediction[const.DETECTOR_FOUND_IDX]
        GT_RELATIONS = prediction[const.GT_RELATIONS]
        SUPPLY_RELATIONS = prediction[const.SUPPLY_RELATIONS]
        assigned_labels = prediction[const.ASSIGNED_LABELS]
        FINAL_BASE_FEATURES = prediction[const.FINAL_BASE_FEATURES]
        FINAL_BBOXES = prediction[const.FINAL_BBOXES]
        FINAL_SCORES = prediction[const.FINAL_SCORES]
        FINAL_FEATURES = prediction[const.FINAL_FEATURES]

        # Initialize variables
        FINAL_BBOXES_X = torch.tensor([]).to(self.device)
        FINAL_LABELS_X = torch.tensor([], dtype=torch.int64).to(self.device)
        FINAL_SCORES_X = torch.tensor([]).to(self.device)
        FINAL_FEATURES_X = torch.tensor([]).to(self.device)
        assigned_labels = torch.tensor(assigned_labels, dtype=torch.long).to(self.device)

        for i, supply_relations in enumerate(SUPPLY_RELATIONS):
            if len(supply_relations) > 0 and self.use_SUPPLY:
                unfound_gt_bboxes, unfound_gt_classes, one_scores = self._get_unfound_gt_boxes(
                    supply_relations, im_info, i
                )
                DETECTOR_FOUND_IDX[i] = list(
                    np.concatenate(
                        (
                            DETECTOR_FOUND_IDX[i],
                            np.arange(
                                start=int(sum(FINAL_BBOXES[:, 0] == i)),
                                stop=int(sum(FINAL_BBOXES[:, 0] == i)) + len(supply_relations)
                            )
                        ), axis=0).astype('int64')
                )
                GT_RELATIONS[i].extend(supply_relations)
                pooled_feat = self._compute_pooled_feat(FINAL_BASE_FEATURES[i], unfound_gt_bboxes)
                unfound_gt_bboxes[:, 0] = i
                unfound_gt_bboxes[:, 1:] = unfound_gt_bboxes[:, 1:] / im_info[i, 2]
                FINAL_BBOXES_X = torch.cat(
                    (FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i], unfound_gt_bboxes))
                FINAL_LABELS_X = torch.cat(
                    (FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i], unfound_gt_classes))
                FINAL_SCORES_X = torch.cat((FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i], one_scores))
                FINAL_FEATURES_X = torch.cat(
                    (FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i], pooled_feat))
            else:
                FINAL_BBOXES_X = torch.cat((FINAL_BBOXES_X, FINAL_BBOXES[FINAL_BBOXES[:, 0] == i]))
                FINAL_LABELS_X = torch.cat((FINAL_LABELS_X, assigned_labels[FINAL_BBOXES[:, 0] == i]))
                FINAL_SCORES_X = torch.cat((FINAL_SCORES_X, FINAL_SCORES[FINAL_BBOXES[:, 0] == i]))
                FINAL_FEATURES_X = torch.cat((FINAL_FEATURES_X, FINAL_FEATURES[FINAL_BBOXES[:, 0] == i]))

        FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES_X)[:, 1:], dim=1)
        global_idx = torch.arange(start=0, end=FINAL_BBOXES_X.shape[0])  # all bbox indices

        im_idx, pair, a_rel, s_rel, c_rel = self._construct_relations(
            DETECTOR_FOUND_IDX, GT_RELATIONS, FINAL_BBOXES_X, global_idx
        )
        pair = torch.tensor(pair).to(self.device)
        im_idx = torch.tensor(im_idx, dtype=torch.float).to(self.device)

        # Compute union boxes
        union_boxes = self._compute_union_boxes(FINAL_BBOXES_X, pair, im_info, im_idx)
        union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)

        pair_rois = torch.cat((FINAL_BBOXES_X[pair[:, 0], 1:], FINAL_BBOXES_X[pair[:, 1], 1:]),
                              1).data.cpu().numpy()
        spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

        prediction[const.FINAL_BBOXES_X] = FINAL_BBOXES_X
        prediction[const.FINAL_LABELS_X] = FINAL_LABELS_X
        prediction[const.FINAL_SCORES_X] = FINAL_SCORES_X
        prediction[const.FINAL_FEATURES_X] = FINAL_FEATURES_X
        prediction[const.FINAL_DISTRIBUTIONS] = FINAL_DISTRIBUTIONS
        prediction[const.PAIR] = pair
        prediction[const.IM_IDX] = im_idx
        prediction[const.UNION_FEAT] = union_feat  # Overriding with supplied ground truth data
        prediction[const.SPATIAL_MASKS] = spatial_masks  # Overriding with supplied ground truth data
        prediction[const.ATTENTION_REL] = a_rel
        prediction[const.SPATIAL_REL] = s_rel
        prediction[const.CONTACTING_REL] = c_rel
        return prediction

    def _init_sgdet_tensors(self):
        FINAL_BBOXES = torch.tensor([]).to(self.device)
        FINAL_LABELS = torch.tensor([], dtype=torch.int64).to(self.device)
        FINAL_SCORES = torch.tensor([]).to(self.device)
        FINAL_FEATURES = torch.tensor([]).to(self.device)
        FINAL_BASE_FEATURES = torch.tensor([]).to(self.device)
        return FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, FINAL_FEATURES, FINAL_BASE_FEATURES

    def _pack_attribute_dictionary(self, FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, FINAL_FEATURES, FINAL_BASE_FEATURES):
        attribute_dictionary = {
            const.FINAL_BBOXES: FINAL_BBOXES,
            const.FINAL_LABELS: FINAL_LABELS,
            const.FINAL_SCORES: FINAL_SCORES,
            const.FINAL_FEATURES: FINAL_FEATURES,
            const.FINAL_BASE_FEATURES: FINAL_BASE_FEATURES
        }
        return attribute_dictionary

    def _forward_sgdet(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all):
        counter = 0
        counter_image = 0
        FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, FINAL_FEATURES, FINAL_BASE_FEATURES = self._init_sgdet_tensors()
        while counter < im_data.shape[0]:
            inputs_data, inputs_info, inputs_gt_boxes, inputs_num_boxes = self._batch_processing(
                counter, [im_data, im_info, gt_boxes, num_boxes])

            rois, cls_prob, bbox_pred, base_feat, roi_features = self.fasterRCNN(
                inputs_data, inputs_info, inputs_gt_boxes, inputs_num_boxes)

            SCORES = cls_prob.data
            boxes = rois.data[:, :, 1:5]
            box_deltas = self._box_regression(bbox_pred, rois)
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)

            transformed_pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            transformed_pred_boxes /= inputs_info[0, 2]

            FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, FINAL_FEATURES, FINAL_BASE_FEATURES, counter_image = self._nms_and_collect_results(
                rois, SCORES, transformed_pred_boxes, base_feat, roi_features, counter_image,
                [FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, FINAL_FEATURES, FINAL_BASE_FEATURES]
            )
            counter += const.FASTER_RCNN_BATCH_SIZE
        FINAL_BBOXES = torch.clamp(FINAL_BBOXES, 0)
        prediction = self._pack_attribute_dictionary(FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, FINAL_FEATURES,
                                                     FINAL_BASE_FEATURES)
        if self.is_train:
            DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = assign_relations(prediction,
                                                                                                   gt_annotation,
                                                                                                   assign_IOU_threshold=0.5)
            prediction[const.DETECTOR_FOUND_IDX] = DETECTOR_FOUND_IDX
            prediction[const.GT_RELATIONS] = GT_RELATIONS
            prediction[const.SUPPLY_RELATIONS] = SUPPLY_RELATIONS
            prediction[const.ASSIGNED_LABELS] = assigned_labels
            return self._augment_gt_annotation(prediction, gt_annotation, im_info)
        else:
            DETECTOR_FOUND_IDX, GT_RELATIONS, SUPPLY_RELATIONS, assigned_labels = assign_relations(prediction,
                                                                                                   gt_annotation,
                                                                                                   assign_IOU_threshold=0.3)
            FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
            FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
            PRED_LABELS = PRED_LABELS + 1

            attribute_dictionary = {
                const.FINAL_BBOXES: FINAL_BBOXES,
                const.ASSIGNED_LABELS: torch.LongTensor(assigned_labels).cuda(),
                const.FINAL_SCORES: FINAL_SCORES,
                const.FINAL_DISTRIBUTIONS: FINAL_DISTRIBUTIONS,
                const.PRED_LABELS: PRED_LABELS,
                const.FINAL_FEATURES: FINAL_FEATURES,
                const.FINAL_BASE_FEATURES: FINAL_BASE_FEATURES,
                const.IM_INFO: im_info[0, 2],
            }
            return attribute_dictionary

    ################################################################################
    # -------------------------- PREDCLS and SGCLS ENTRY ------------------------ #
    ################################################################################

    def _forward_and_fetch_features(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all):
        bbox_num, bbox_idx, bbox_info = self._count_bbox(gt_annotation)
        FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, HUMAN_IDX = self._init_final_tensors(bbox_num, len(gt_annotation))
        FINAL_BBOXES, FINAL_LABELS, HUMAN_IDX, im_idx, pair, a_rel, s_rel, c_rel = self._populate_final_tensors(
            FINAL_BBOXES, FINAL_LABELS, HUMAN_IDX, gt_annotation
        )
        pair, im_idx = map(lambda x: torch.tensor(x).to(self.device), [pair, im_idx])
        FINAL_BASE_FEATURES = self._compute_base_features(im_data)
        FINAL_BBOXES[:, 1:] *= im_info[0, 2]
        FINAL_FEATURES = self._compute_final_features(FINAL_BASE_FEATURES, FINAL_BBOXES)
        union_boxes, spatial_masks = self._compute_union_boxes_and_masks(FINAL_BBOXES, pair, im_idx)
        union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)

        FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
        pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]), 1).data.cpu().numpy()
        spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

        FINAL_DISTRIBUTIONS, FINAL_PRED_SCORES, PRED_LABELS = self._compute_final_distributions_and_labels(
            FINAL_FEATURES)
        attribute_dictionary = self._construct_attribute_dictionary(
            FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, im_idx, pair, HUMAN_IDX, FINAL_FEATURES,
            union_boxes, union_feat, spatial_masks, a_rel, s_rel, c_rel, FINAL_DISTRIBUTIONS, PRED_LABELS,
            FINAL_BASE_FEATURES, im_info, FINAL_PRED_SCORES
        )
        return attribute_dictionary

    def _count_bbox(self, gt_annotation):
        bbox_num = 0
        for gt_frame_bboxes in gt_annotation:
            bbox_num += len(gt_frame_bboxes)
        return bbox_num, 0, []

    def _init_final_tensors(self, bbox_num, ann_len):
        dtype_float = torch.float32
        dtype_int = torch.int64
        FINAL_BBOXES = torch.zeros([bbox_num, 5], dtype=dtype_float).to(self.device)
        FINAL_LABELS = torch.zeros([bbox_num], dtype=dtype_int).to(self.device)
        FINAL_SCORES = torch.ones([bbox_num], dtype=dtype_float).to(self.device)
        HUMAN_IDX = torch.zeros([ann_len, 1], dtype=dtype_int).to(self.device)
        return FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, HUMAN_IDX

    def _populate_final_tensors(self, FINAL_BBOXES, FINAL_LABELS, HUMAN_IDX, gt_annotation):
        bbox_idx = 0
        im_idx, pair, a_rel, s_rel, c_rel = [], [], [], [], []
        for frame_idx, gt_frame_bboxes in enumerate(gt_annotation):
            for frame_bbox in gt_frame_bboxes:
                if const.PERSON_BBOX in frame_bbox.keys():
                    FINAL_BBOXES[bbox_idx, 1:] = torch.from_numpy(frame_bbox[const.PERSON_BBOX][0])
                    FINAL_BBOXES[bbox_idx, 0] = frame_idx
                    FINAL_LABELS[bbox_idx] = 1
                    HUMAN_IDX[frame_idx] = bbox_idx
                    bbox_idx += 1
                else:
                    FINAL_BBOXES[bbox_idx, 1:] = torch.from_numpy(frame_bbox[const.BBOX])
                    FINAL_BBOXES[bbox_idx, 0] = frame_idx
                    FINAL_LABELS[bbox_idx] = frame_bbox[const.CLASS]
                    im_idx.append(frame_idx)
                    pair.append([int(HUMAN_IDX[frame_idx]), bbox_idx])
                    a_rel.append(frame_bbox[const.ATTENTION_RELATIONSHIP].tolist())
                    s_rel.append(frame_bbox[const.SPATIAL_RELATIONSHIP].tolist())
                    c_rel.append(frame_bbox[const.CONTACTING_RELATIONSHIP].tolist())
                    bbox_idx += 1
        return FINAL_BBOXES, FINAL_LABELS, HUMAN_IDX, im_idx, pair, a_rel, s_rel, c_rel

    def _compute_base_features(self, im_data):
        FINAL_BASE_FEATURES = torch.tensor([]).to(self.device)
        counter = 0
        while counter < im_data.shape[0]:
            if counter + 10 < im_data.shape[0]:
                inputs_data = im_data[counter:counter + 10]
            else:
                inputs_data = im_data[counter:]
            base_feat = self.fasterRCNN.RCNN_base(inputs_data)
            FINAL_BASE_FEATURES = torch.cat((FINAL_BASE_FEATURES, base_feat), 0)
            counter += 10
        return FINAL_BASE_FEATURES

    def _compute_final_features(self, FINAL_BASE_FEATURES, FINAL_BBOXES):
        return self.fasterRCNN._head_to_tail(
            self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, FINAL_BBOXES)
        )

    def _compute_union_boxes_and_masks(self, FINAL_BBOXES, pair, im_idx):
        union_boxes = torch.cat(
            (
                im_idx[:, None],
                torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
                torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]]),
            ),
            1,
        )
        pair_rois = torch.cat(
            (FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
            1,
        ).data.cpu().numpy()
        spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(self.device)
        return union_boxes, spatial_masks

    def _compute_final_distributions_and_labels(self, FINAL_FEATURES):
        FINAL_DISTRIBUTIONS = torch.softmax(self.fasterRCNN.RCNN_cls_score(FINAL_FEATURES)[:, 1:], dim=1)
        FINAL_SCORES, PRED_LABELS = torch.max(FINAL_DISTRIBUTIONS, dim=1)
        PRED_LABELS += 1
        return FINAL_DISTRIBUTIONS, FINAL_SCORES, PRED_LABELS

    def _construct_attribute_dictionary(
            self, FINAL_BBOXES, FINAL_LABELS, FINAL_SCORES, im_idx, pair, HUMAN_IDX, FINAL_FEATURES,
            union_boxes, union_feat, spatial_masks, a_rel, s_rel, c_rel, FINAL_DISTRIBUTIONS, PRED_LABELS,
            FINAL_BASE_FEATURES, im_info, FINAL_PRED_SCORES
    ):
        attribute_dictionary = {
            const.FINAL_BBOXES: FINAL_BBOXES,
            const.FINAL_LABELS: FINAL_LABELS,
            const.FINAL_SCORES: FINAL_SCORES,
            const.IMAGE_IDX: im_idx,
            const.PAIR: pair,
            const.HUMAN_IDX: HUMAN_IDX,
            const.FINAL_FEATURES: FINAL_FEATURES,
            const.UNION_FEAT: union_feat,
            const.UNION_BOX: union_boxes,
            const.SPATIAL_MASKS: spatial_masks,
            const.ATTENTION_REL: a_rel,
            const.SPATIAL_REL: s_rel,
            const.CONTACTING_REL: c_rel,
            const.FINAL_DISTRIBUTIONS: FINAL_DISTRIBUTIONS,
            const.PRED_LABELS: PRED_LABELS,
            const.FINAL_BASE_FEATURES: FINAL_BASE_FEATURES,
            const.IM_INFO: im_info[0, 2],
            const.FINAL_PRED_SCORES: FINAL_PRED_SCORES
        }
        return attribute_dictionary

    def _construct_entry(self, attribute_dictionary):
        entry = {}
        if self.mode == 'sgdet':
            if self.is_train:
                entry = {
                    const.BOXES: attribute_dictionary[const.FINAL_BBOXES_X],
                    const.LABELS: attribute_dictionary[const.FINAL_LABELS_X],
                    const.SCORES: attribute_dictionary[const.FINAL_SCORES_X],
                    const.DISTRIBUTION: attribute_dictionary[const.FINAL_DISTRIBUTIONS],
                    const.IM_IDX: attribute_dictionary[const.IM_IDX],
                    const.PAIR_IDX: attribute_dictionary[const.PAIR],
                    const.FEATURES: attribute_dictionary[const.FINAL_FEATURES_X],
                    const.UNION_FEAT: attribute_dictionary[const.UNION_FEAT],
                    const.SPATIAL_MASKS: attribute_dictionary[const.SPATIAL_MASKS],
                    const.ATTENTION_GT: attribute_dictionary[const.ATTENTION_REL],
                    const.SPATIAL_GT: attribute_dictionary[const.SPATIAL_REL],
                    const.CONTACTING_GT: attribute_dictionary[const.CONTACTING_REL]
                }
            else:
                entry = {
                    const.BOXES: attribute_dictionary[const.FINAL_BBOXES],
                    const.SCORES: attribute_dictionary[const.FINAL_SCORES],
                    const.DISTRIBUTION: attribute_dictionary[const.FINAL_DISTRIBUTIONS],
                    const.PRED_LABELS: attribute_dictionary[const.PRED_LABELS],
                    const.FEATURES: attribute_dictionary[const.FINAL_FEATURES],
                    const.FMAPS: attribute_dictionary[const.FINAL_BASE_FEATURES],
                    const.IM_INFO: attribute_dictionary[const.IM_INFO],
                    const.LABELS: attribute_dictionary[const.ASSIGNED_LABELS]
                }
        elif self.mode == 'sgcls':
            entry = {
                const.BOXES: attribute_dictionary[const.FINAL_BBOXES],
                const.LABELS: attribute_dictionary[const.FINAL_LABELS],  # labels are gt!
                const.SCORES: attribute_dictionary[const.FINAL_PRED_SCORES],
                const.IMAGE_IDX: attribute_dictionary[const.IMAGE_IDX],
                const.PAIR_IDX: attribute_dictionary[const.PAIR],
                const.HUMAN_IDX: attribute_dictionary[const.HUMAN_IDX],
                const.FEATURES: attribute_dictionary[const.FINAL_FEATURES],
                const.ATTENTION_GT: attribute_dictionary[const.ATTENTION_REL],
                const.SPATIAL_GT: attribute_dictionary[const.SPATIAL_REL],
                const.CONTACTING_GT: attribute_dictionary[const.CONTACTING_REL],
                const.DISTRIBUTION: attribute_dictionary[const.FINAL_DISTRIBUTIONS],
                const.PRED_LABELS: attribute_dictionary[const.PRED_LABELS]
            }
            if self.is_train:
                entry[const.UNION_FEAT] = attribute_dictionary[const.UNION_FEAT]
                entry[const.UNION_BOX] = attribute_dictionary[const.UNION_BOX]
                entry[const.SPATIAL_MASKS] = attribute_dictionary[const.SPATIAL_MASKS]
            else:
                entry[const.FMAPS] = attribute_dictionary[const.FINAL_BASE_FEATURES]
                entry[const.IM_INFO] = attribute_dictionary[const.IM_INFO]
        elif self.mode == 'predcls':
            entry = {
                const.BOXES: attribute_dictionary[const.FINAL_BBOXES],
                const.LABELS: attribute_dictionary[const.FINAL_LABELS],  # labels are gt!
                const.SCORES: attribute_dictionary[const.FINAL_SCORES],
                const.IMAGE_IDX: attribute_dictionary[const.IMAGE_IDX],
                const.PAIR_IDX: attribute_dictionary[const.PAIR],
                const.HUMAN_IDX: attribute_dictionary[const.HUMAN_IDX],
                const.FEATURES: attribute_dictionary[const.FINAL_FEATURES],
                const.UNION_FEAT: attribute_dictionary[const.UNION_FEAT],
                const.UNION_BOX: attribute_dictionary[const.UNION_BOX],
                const.SPATIAL_MASKS: attribute_dictionary[const.SPATIAL_MASKS],
                const.ATTENTION_GT: attribute_dictionary[const.ATTENTION_REL],
                const.SPATIAL_GT: attribute_dictionary[const.SPATIAL_REL],
                const.CONTACTING_GT: attribute_dictionary[const.CONTACTING_REL]
            }
        return entry

    def forward(self, im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all):
        if self.mode == 'sgdet':
            attribute_dictionary = self._forward_sgdet(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all)
        else:
            attribute_dictionary = self._forward_and_fetch_features(
                im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all
            )

        entry = self._construct_entry(attribute_dictionary)

        # Adding frame information to train the Neural Differential Equation Models.
        frame_idx_list = []
        for frame_gt_annotation in gt_annotation:
            frame_id = int(frame_gt_annotation[0][const.FRAME].split('/')[1][:-4])
            frame_idx_list.append(frame_id)
        entry[const.FRAME_IDX] = frame_idx_list

        return entry
