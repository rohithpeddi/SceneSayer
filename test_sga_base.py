import copy
import csv
import os
from abc import abstractmethod

import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result, ResultDetails, Metrics
from dataloader.action_genome.ag_dataset import AG
from dataloader.action_genome.ag_dataset import cuda_collate_fn
from lib.object_detector import Detector
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator
from constants import DataloaderConstants as const
from sga_base import SGABase


class TestSGABase(SGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._model = None

        # Load while initializing the object detector
        self._object_detector = None

        # Load while initializing the dataset
        self._dataloader_test = None
        self._test_dataset = None
        self._object_classes = None

        # Load checkpoint name
        self._checkpoint_name = None

        # Evaluation paradigm constants
        self._future_frame_windows = [1, 2, 3, 4, 5]
        self._context_fractions = [0.3, 0.5, 0.7, 0.9]

        # Load the evaluators
        self._combined_future_frames_evaluators = []
        self._combined_context_fractions_evaluators = []

    def _init_evaluators(self):
        print("Replaced default single evaluator with multiple evaluators")
        # Evaluators order - [With Constraint, No Constraint, Semi Constraint]
        iou_threshold = 0.5 if self._conf.task_name == 'sgg' else 0.0

        for i in range(len(self._future_frame_windows)):
            future_frame_evaluators = []
            with_constraint_evaluator = BasicSceneGraphEvaluator(
                mode=self._conf.mode,
                AG_object_classes=self._test_dataset.object_classes,
                AG_all_predicates=self._test_dataset.relationship_classes,
                AG_attention_predicates=self._test_dataset.attention_relationships,
                AG_spatial_predicates=self._test_dataset.spatial_relationships,
                AG_contacting_predicates=self._test_dataset.contacting_relationships,
                iou_threshold=iou_threshold,
                constraint='with')

            no_constraint_evaluator = BasicSceneGraphEvaluator(
                mode=self._conf.mode,
                AG_object_classes=self._test_dataset.object_classes,
                AG_all_predicates=self._test_dataset.relationship_classes,
                AG_attention_predicates=self._test_dataset.attention_relationships,
                AG_spatial_predicates=self._test_dataset.spatial_relationships,
                AG_contacting_predicates=self._test_dataset.contacting_relationships,
                iou_threshold=iou_threshold,
                constraint='no')

            semi_constraint_evaluator = BasicSceneGraphEvaluator(
                mode=self._conf.mode,
                AG_object_classes=self._test_dataset.object_classes,
                AG_all_predicates=self._test_dataset.relationship_classes,
                AG_attention_predicates=self._test_dataset.attention_relationships,
                AG_spatial_predicates=self._test_dataset.spatial_relationships,
                AG_contacting_predicates=self._test_dataset.contacting_relationships,
                iou_threshold=iou_threshold,
                constraint='semi', semi_threshold=0.9)

            future_frame_evaluators.append(with_constraint_evaluator)
            future_frame_evaluators.append(no_constraint_evaluator)
            future_frame_evaluators.append(semi_constraint_evaluator)
            self._combined_future_frames_evaluators.append(future_frame_evaluators)

        for i in range(len(self._context_fractions)):
            context_fraction_evaluators = []
            with_constraint_evaluator = BasicSceneGraphEvaluator(
                mode=self._conf.mode,
                AG_object_classes=self._test_dataset.object_classes,
                AG_all_predicates=self._test_dataset.relationship_classes,
                AG_attention_predicates=self._test_dataset.attention_relationships,
                AG_spatial_predicates=self._test_dataset.spatial_relationships,
                AG_contacting_predicates=self._test_dataset.contacting_relationships,
                iou_threshold=iou_threshold,
                constraint='with')

            no_constraint_evaluator = BasicSceneGraphEvaluator(
                mode=self._conf.mode,
                AG_object_classes=self._test_dataset.object_classes,
                AG_all_predicates=self._test_dataset.relationship_classes,
                AG_attention_predicates=self._test_dataset.attention_relationships,
                AG_spatial_predicates=self._test_dataset.spatial_relationships,
                AG_contacting_predicates=self._test_dataset.contacting_relationships,
                iou_threshold=iou_threshold,
                constraint='no')

            semi_constraint_evaluator = BasicSceneGraphEvaluator(
                mode=self._conf.mode,
                AG_object_classes=self._test_dataset.object_classes,
                AG_all_predicates=self._test_dataset.relationship_classes,
                AG_attention_predicates=self._test_dataset.attention_relationships,
                AG_spatial_predicates=self._test_dataset.spatial_relationships,
                AG_contacting_predicates=self._test_dataset.contacting_relationships,
                iou_threshold=iou_threshold,
                constraint='semi', semi_threshold=0.9)

            context_fraction_evaluators.append(with_constraint_evaluator)
            context_fraction_evaluators.append(no_constraint_evaluator)
            context_fraction_evaluators.append(semi_constraint_evaluator)
            self._combined_context_fractions_evaluators.append(context_fraction_evaluators)

    def _init_object_detector(self):
        self._object_detector = Detector(
            train=True,
            object_classes=self._object_classes,
            use_SUPPLY=True,
            mode=self._conf.mode
        ).to(device=self._device)
        self._object_detector.eval()

    def _collate_evaluation_stats(self, index, future_frame):
        if future_frame:
            with_constraint_evaluator_stats = self._combined_future_frames_evaluators[index][0].fetch_stats_json()
            no_constraint_evaluator_stats = self._combined_future_frames_evaluators[index][1].fetch_stats_json()
            semi_constraint_evaluator_stats = self._combined_future_frames_evaluators[index][2].fetch_stats_json()
        else:
            with_constraint_evaluator_stats = self._combined_context_fractions_evaluators[index][0].fetch_stats_json()
            no_constraint_evaluator_stats = self._combined_context_fractions_evaluators[index][1].fetch_stats_json()
            semi_constraint_evaluator_stats = self._combined_context_fractions_evaluators[index][2].fetch_stats_json()

        collated_stats = [
            with_constraint_evaluator_stats["recall"][10],
            with_constraint_evaluator_stats["recall"][20],
            with_constraint_evaluator_stats["recall"][50],
            with_constraint_evaluator_stats["recall"][100],
            with_constraint_evaluator_stats["mean_recall"][10],
            with_constraint_evaluator_stats["mean_recall"][20],
            with_constraint_evaluator_stats["mean_recall"][50],
            with_constraint_evaluator_stats["mean_recall"][100],
            with_constraint_evaluator_stats["harmonic_mean_recall"][10],
            with_constraint_evaluator_stats["harmonic_mean_recall"][20],
            with_constraint_evaluator_stats["harmonic_mean_recall"][50],
            with_constraint_evaluator_stats["harmonic_mean_recall"][100],
            no_constraint_evaluator_stats["recall"][10],
            no_constraint_evaluator_stats["recall"][20],
            no_constraint_evaluator_stats["recall"][50],
            no_constraint_evaluator_stats["recall"][100],
            no_constraint_evaluator_stats["mean_recall"][10],
            no_constraint_evaluator_stats["mean_recall"][20],
            no_constraint_evaluator_stats["mean_recall"][50],
            no_constraint_evaluator_stats["mean_recall"][100],
            no_constraint_evaluator_stats["harmonic_mean_recall"][10],
            no_constraint_evaluator_stats["harmonic_mean_recall"][20],
            no_constraint_evaluator_stats["harmonic_mean_recall"][50],
            no_constraint_evaluator_stats["harmonic_mean_recall"][100],
            semi_constraint_evaluator_stats["recall"][10],
            semi_constraint_evaluator_stats["recall"][20],
            semi_constraint_evaluator_stats["recall"][50],
            semi_constraint_evaluator_stats["recall"][100],
            semi_constraint_evaluator_stats["mean_recall"][10],
            semi_constraint_evaluator_stats["mean_recall"][20],
            semi_constraint_evaluator_stats["mean_recall"][50],
            semi_constraint_evaluator_stats["mean_recall"][100],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][10],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][20],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][50],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][100]
        ]
        return collated_stats

    def _evaluate_predictions(self, index, gt_annotation, prediction):
        for evaluator in self._combined_context_fractions_evaluators[index]:
            evaluator.evaluate_scene_graph(gt_annotation, prediction)

    def _evaluate_anticipated_future_frame_scene_graph(self, gt, pred, future_frame_count, future_evaluators):
        for index, reference_frame_count in enumerate(self._future_frame_windows):
            if reference_frame_count >= future_frame_count:
                evaluators = future_evaluators[index]
                evaluators[0].evaluate_scene_graph(gt, pred)
                evaluators[1].evaluate_scene_graph(gt, pred)
                evaluators[2].evaluate_scene_graph(gt, pred)

    def _publish_evaluation_results(self):
        # 1. Collate the evaluation statistics
        # self._collated_stats = self._collate_evaluation_stats()
        # 2. Write to the CSV File
        self._write_evaluation_statistics()
        # 3. Publish the results to Firebase
        # self._publish_results_to_firebase()

    def _write_evaluation_statistics(self):
        # Create the results directory
        results_dir = os.path.join(os.getcwd(), 'results')
        mode_results_dir = os.path.join(results_dir, self._conf.mode)
        os.makedirs(mode_results_dir, exist_ok=True)

        # TODO: Have train and test future frame windows for the filenames
        # Create the results file
        for i, future_frame_window in enumerate(self._future_frame_windows):
            results_file_path = os.path.join(mode_results_dir,
                                             f'{self._conf.method_name}_{self._conf.mode}_{future_frame_window}.csv')

            with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
                writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
                collated_stats = self._collate_evaluation_stats(i, True)
                # Write the header if the file is empty
                if not os.path.isfile(results_file_path):
                    writer.writerow([
                        "Method Name",
                        "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                        "hR@100"
                        "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                        "hR@100",
                        "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                        "hR@100"
                    ])
                    # Write the results row
                writer.writerow(collated_stats)

        for i, context_fraction in enumerate(self._context_fractions):
            results_file_path = os.path.join(mode_results_dir,
                                             f'{self._conf.method_name}_{self._conf.mode}_{context_fraction}.csv')

            with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
                writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
                collated_stats = self._collate_evaluation_stats(i, False)
                # Write the header if the file is empty
                if not os.path.isfile(results_file_path):
                    writer.writerow([
                        "Method Name",
                        "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                        "hR@100"
                        "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                        "hR@100",
                        "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                        "hR@100"
                    ])
                    # Write the results row
                writer.writerow(collated_stats)

    @staticmethod
    def _prepare_metrics_from_stats(evaluator_stats):
        metrics = Metrics(
            evaluator_stats["recall"][10],
            evaluator_stats["recall"][20],
            evaluator_stats["recall"][50],
            evaluator_stats["recall"][100],
            evaluator_stats["mean_recall"][10],
            evaluator_stats["mean_recall"][20],
            evaluator_stats["mean_recall"][50],
            evaluator_stats["mean_recall"][100],
            evaluator_stats["harmonic_mean_recall"][10],
            evaluator_stats["harmonic_mean_recall"][20],
            evaluator_stats["harmonic_mean_recall"][50],
            evaluator_stats["harmonic_mean_recall"][100]
        )

        return metrics

    # TODO: Use this function inside a loop to evaluate all the future frame windows
    def _publish_results_to_firebase(self, evaluators):
        db_service = FirebaseService()

        result = Result(
            task_name=self._conf.task_name,
            method_name=self._conf.method_name,
            mode=self._conf.mode,
        )

        result_details = ResultDetails()
        with_constraint_metrics = self._prepare_metrics_from_stats(evaluators[0].fetch_stats_json())
        no_constraint_metrics = self._prepare_metrics_from_stats(evaluators[1].fetch_stats_json())
        semi_constraint_metrics = self._prepare_metrics_from_stats(evaluators[2].fetch_stats_json())

        result_details.add_with_constraint_metrics(with_constraint_metrics)
        result_details.add_no_constraint_metrics(no_constraint_metrics)
        result_details.add_semi_constraint_metrics(semi_constraint_metrics)

        result.add_result_details(result_details)

        print("Saving result: ", result.result_id)
        db_service.update_result(result.result_id, result.to_dict())
        print("Saved result: ", result.result_id)

        return result

    def _test_model_diffeq(self):
        test_iter = iter(self._dataloader_test)
        self._model.eval()
        self._object_detector.is_train = False
        with torch.no_grad():
            for num_video_id in tqdm(range(len(self._dataloader_test)), desc="Testing Progress", ascii=True):
                data = next(test_iter)
                im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                gt_annotation = self._test_dataset.gt_annotations[data[4]]
                frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data
                entry = self._object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                entry["gt_annotation"] = gt_annotation
                # To do inside the Dataloader.
                frame_idx_list = []
                for frame_gt_annotation in gt_annotation:
                    frame_id = int(frame_gt_annotation[0][const.FRAME].split('/')[1][:-4])
                    frame_idx_list.append(frame_id)
                entry[const.FRAME_IDX] = frame_idx_list
                # ----------------- Process the video (Method Specific) -----------------
                pred = self.process_test_video_future_frame(entry, frame_size, gt_annotation)
                w = self._conf.max_window
                n = len(gt_annotation)
                w = min(w, n - 1)
                for i in range(1, w + 1):
                    pred_anticipated = pred.copy()
                    last = pred["last_" + str(i)]
                    pred_anticipated["spatial_distribution"] = pred["anticipated_spatial_distribution"][i - 1, : last]
                    pred_anticipated["contacting_distribution"] = pred["anticipated_contacting_distribution"][i - 1,
                                                                  : last]
                    pred_anticipated["attention_distribution"] = pred["anticipated_attention_distribution"][i - 1,
                                                                 : last]
                    pred_anticipated["im_idx"] = pred["im_idx_test_" + str(i)]
                    pred_anticipated["pair_idx"] = pred["pair_idx_test_" + str(i)]
                    if self._conf.mode == "predcls":
                        pred_anticipated["scores"] = pred["scores_test_" + str(i)]
                        pred_anticipated["labels"] = pred["labels_test_" + str(i)]
                    else:
                        pred_anticipated["pred_scores"] = pred["pred_scores_test_" + str(i)]
                        pred_anticipated["pred_labels"] = pred["pred_labels_test_" + str(i)]
                    pred_anticipated["boxes"] = pred["boxes_test_" + str(i)]
                    self._evaluate_anticipated_future_frame_scene_graph(
                        entry["gt_annotation"][i:],
                        pred_anticipated,
                        future_frame_count=i,
                        future_evaluators=self._combined_future_frames_evaluators
                    )
                # ----------------------------------------------------------------------
            print('-----------------------------------------------------------------------------------', flush=True)
            for i, context_fraction in enumerate(self._context_fractions):
                test_iter = iter(self._dataloader_test)
                for num_video_id in tqdm(range(len(self._dataloader_test)), desc="Testing Progress", ascii=True):
                    data = next(test_iter)
                    im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                    gt_annotation = self._test_dataset.gt_annotations[data[4]]
                    frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data
                    entry = self._object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                    entry["gt_annotation"] = gt_annotation
                    frame_idx_list = []
                    for frame_gt_annotation in gt_annotation:
                        frame_id = int(frame_gt_annotation[0][const.FRAME].split('/')[1][:-4])
                        frame_idx_list.append(frame_id)
                    entry[const.FRAME_IDX] = frame_idx_list
                    # ----------------- Process the video (Method Specific) -----------------
                    skip, ind, pred = self.process_test_video_context(entry, frame_size, gt_annotation,
                                                                      context_fraction)
                    if skip:
                        continue
                    self._evaluate_predictions(i, gt_annotation[ind:], pred)
                    # ----------------------------------------------------------------------
                print('-----------------------------------------------------------------------------------', flush=True)

    def _test_model_transformer(self):
        test_iter = iter(self._dataloader_test)
        self._model.eval()
        self._object_detector.is_train = False
        with torch.no_grad():
            for i, context_fraction in enumerate(self._context_fractions):
                test_iter = iter(self._dataloader_test)
                for num_video_id in tqdm(range(len(self._dataloader_test)), desc="Testing Progress", ascii=True):
                    data = next(test_iter)
                    im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                    gt_annotation = self._test_dataset.gt_annotations[data[4]]
                    frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data
                    entry = self._object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                    entry["gt_annotation"] = gt_annotation
                    # To do inside the Dataloader.
                    frame_idx_list = []
                    for frame_gt_annotation in gt_annotation:
                        frame_id = int(frame_gt_annotation[0][const.FRAME].split('/')[1][:-4])
                        frame_idx_list.append(frame_id)
                    entry[const.FRAME_IDX] = frame_idx_list
                    # ----------------- Process the video (Method Specific) -----------------
                    gt_future, pred_dict = self.process_test_video_context(entry, frame_size, gt_annotation,
                                                                           context_fraction)
                    self._evaluate_predictions(i, gt_future, pred_dict)
                    # ----------------------------------------------------------------------
                print('-----------------------------------------------------------------------------------', flush=True)

    def _init_dataset(self):
        # Using the parameters set in the configuration file, initialize the corrupted dataset
        self._test_dataset = AG(
            phase='test',
            datasize=self._conf.datasize,
            data_path=self._conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if self._conf.mode == 'predcls' else True
        )

        self._object_classes = self._test_dataset.object_classes

        self._dataloader_test = DataLoader(
            self._test_dataset,
            shuffle=False,
            collate_fn=cuda_collate_fn,
            pin_memory=False
        )

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def process_test_video_future_frame(self, video_entry, frame_size, gt_annotation):
        pass

    @abstractmethod
    def process_test_video_context(self, video_entry, frame_size, gt_annotation, context_fraction):
        pass

    def init_method_evaluation(self):
        diffeq = ["ode", "sde"]
        # 0. Init config
        self._init_config()

        # 1. Initialize the dataset
        self._init_dataset()

        # 2. Initialize evaluators
        self._init_evaluators()

        # 3. Initialize and load pretrained model
        self.init_model()
        self._load_checkpoint()
        self._init_object_detector()

        # 4. Test the model
        if self._conf.method_name in diffeq:
            self._test_model_diffeq()
        else:
            self._test_model_transformer()

        # 5. Publish the evaluation results
        self._publish_evaluation_results()
