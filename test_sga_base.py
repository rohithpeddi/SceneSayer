import copy
import csv
import os
import pickle
from abc import abstractmethod

import numpy as np
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
        self._combined_ff_evaluators_dict = {}
        self._combined_cf_evaluators_dict = {}

    def _init_evaluators(self):
        print("Replaced default single evaluator with multiple evaluators")
        # Evaluators order - [With Constraint, No Constraint, Semi Constraint]
        iou_threshold = 0.5 if self._conf.task_name == 'sgg' else 0.0

        for i, future_frame_window in enumerate(self._future_frame_windows):
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
            self._combined_ff_evaluators_dict[future_frame_window] = future_frame_evaluators

        for i, context_fraction in enumerate(self._context_fractions):
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
            self._combined_cf_evaluators_dict[context_fraction] = context_fraction_evaluators

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
            with_constraint_evaluator_stats = self._combined_ff_evaluators_dict[index][0].fetch_stats_json()
            no_constraint_evaluator_stats = self._combined_ff_evaluators_dict[index][1].fetch_stats_json()
            semi_constraint_evaluator_stats = self._combined_ff_evaluators_dict[index][2].fetch_stats_json()
        else:
            with_constraint_evaluator_stats = self._combined_cf_evaluators_dict[index][0].fetch_stats_json()
            no_constraint_evaluator_stats = self._combined_cf_evaluators_dict[index][1].fetch_stats_json()
            semi_constraint_evaluator_stats = self._combined_cf_evaluators_dict[index][2].fetch_stats_json()

        collated_stats = [
            self._conf.method_name,
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

    def _evaluate_predictions(self, context_fraction, gt_annotation, prediction):
        for evaluator in self._combined_cf_evaluators_dict[context_fraction]:
            evaluator.evaluate_scene_graph(gt_annotation, prediction)

    def _evaluate_anticipated_future_frame_scene_graph(self, gt, pred, future_frame_count, future_evaluators):
        for index, reference_frame_count in enumerate(self._future_frame_windows):
            if reference_frame_count >= future_frame_count:
                evaluators = future_evaluators[reference_frame_count]
                evaluators[0].evaluate_scene_graph(gt, pred)
                evaluators[1].evaluate_scene_graph(gt, pred)
                evaluators[2].evaluate_scene_graph(gt, pred)

    def _publish_evaluation_results(self, is_future_frame=True):
        # 1. Collate the evaluation statistics
        # self._collated_stats = self._collate_evaluation_stats()
        # 2. Write to the CSV File
        self._write_evaluation_statistics(is_future_frame=is_future_frame)
        # 3. Publish the results to Firebase
        # self._publish_results_to_firebase()

    def _write_evaluation_statistics(self, is_future_frame=True):
        # Create the results directory
        results_dir = os.path.join(os.getcwd(), 'results')
        mode_results_dir = os.path.join(results_dir, self._conf.mode)
        os.makedirs(mode_results_dir, exist_ok=True)

        if is_future_frame:
            # Create the results file
            for i, future_frame_window in enumerate(self._future_frame_windows):
                file_name = f'{self._conf.method_name}_{self._conf.mode}_train_{self._conf.max_window}_test_{future_frame_window}.csv'
                results_file_path = os.path.join(mode_results_dir, file_name)

                with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
                    writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
                    collated_stats = self._collate_evaluation_stats(future_frame_window, True)
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

        if not is_future_frame:
            for i, context_fraction in enumerate(self._context_fractions):
                file_name = f'{self._conf.method_name}_{self._conf.mode}_train_{self._conf.max_window}_test_{context_fraction}.csv'
                results_file_path = os.path.join(mode_results_dir, file_name)

                with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
                    writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
                    collated_stats = self._collate_evaluation_stats(context_fraction, False)
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

    def _publish_results_to_firebase(self, evaluators):
        db_service = FirebaseService()

        for i, future_frame_window in enumerate(self._future_frame_windows):
            result = Result(
                task_name=self._conf.task_name,
                method_name=self._conf.method_name,
                mode=self._conf.mode,
                train_future_frames=self._conf.max_window,
                test_future_frames=future_frame_window
            )

            result_details = ResultDetails()
            with_constraint_metrics = self._prepare_metrics_from_stats(evaluators[0].fetch_stats_json())
            no_constraint_metrics = self._prepare_metrics_from_stats(evaluators[1].fetch_stats_json())
            semi_constraint_metrics = self._prepare_metrics_from_stats(evaluators[2].fetch_stats_json())

            result_details.add_with_constraint_metrics(with_constraint_metrics)
            result_details.add_no_constraint_metrics(no_constraint_metrics)
            result_details.add_semi_constraint_metrics(semi_constraint_metrics)

            result.add_result_details(result_details)

            print(
                f"Saving result for train {self._conf.max_window} and test {future_frame_window} with result id {result.result_id}")
            db_service.update_result(result.result_id, result.to_dict())
            print(
                f"Saved result for train {self._conf.max_window} and test {future_frame_window} with result id {result.result_id}")

        for i, context_fraction in enumerate(self._context_fractions):
            result = Result(
                task_name=self._conf.task_name,
                method_name=self._conf.method_name,
                mode=self._conf.mode,
                train_future_frames=self._conf.max_window,
                test_future_frames=context_fraction
            )

            result_details = ResultDetails()
            with_constraint_metrics = self._prepare_metrics_from_stats(evaluators[0].fetch_stats_json())
            no_constraint_metrics = self._prepare_metrics_from_stats(evaluators[1].fetch_stats_json())
            semi_constraint_metrics = self._prepare_metrics_from_stats(evaluators[2].fetch_stats_json())

            result_details.add_with_constraint_metrics(with_constraint_metrics)
            result_details.add_no_constraint_metrics(no_constraint_metrics)
            result_details.add_semi_constraint_metrics(semi_constraint_metrics)

            result.add_result_details(result_details)

            print(
                f"Saving result for train {self._conf.max_window} and test {context_fraction} with result id {result.result_id}")
            db_service.update_result(result.result_id, result.to_dict())
            print(
                f"Saved result for test {self._conf.max_window} and test {context_fraction} with result id {result.result_id}")

        return

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
    def process_test_video_future_frame(self, entry, frame_size, gt_annotation):
        pass

    @abstractmethod
    def process_test_video_context(self, entry, frame_size, gt_annotation, context_fraction):
        pass

    @abstractmethod
    def compute_test_video_future_frame_score(self, entry, frame_size, gt_annotation):
        pass

    @abstractmethod
    def compute_test_video_context_score(self, result, gt_annotation, i_cf):
        pass

    def compute_scene_sayer_ff_score(self, pred_entry, gt_annotation):
        w = self._conf.max_window
        n = len(gt_annotation)
        w = min(w, n - 1)
        for i in range(1, w + 1):
            pred_anticipated = pred_entry.copy()
            last = pred_entry["last_" + str(i)]
            pred_anticipated["spatial_distribution"] = pred_entry["anticipated_spatial_distribution"][i - 1, : last]
            pred_anticipated["contacting_distribution"] = pred_entry["anticipated_contacting_distribution"][i - 1,
                                                          : last]
            pred_anticipated["attention_distribution"] = pred_entry["anticipated_attention_distribution"][i - 1,
                                                         : last]
            pred_anticipated["im_idx"] = pred_entry["im_idx_test_" + str(i)]
            pred_anticipated["pair_idx"] = pred_entry["pair_idx_test_" + str(i)]

            if self._conf.mode == "predcls":
                pred_anticipated["scores"] = pred_entry["scores_test_" + str(i)]
                pred_anticipated["labels"] = pred_entry["labels_test_" + str(i)]
            else:
                pred_anticipated["pred_scores"] = pred_entry["pred_scores_test_" + str(i)]
                pred_anticipated["pred_labels"] = pred_entry["pred_labels_test_" + str(i)]
            pred_anticipated["boxes"] = pred_entry["boxes_test_" + str(i)]

            self._evaluate_anticipated_future_frame_scene_graph(
                pred_entry["gt_annotation"][i:],
                pred_anticipated,
                future_frame_count=i,
                future_evaluators=self._combined_ff_evaluators_dict
            )

    def compute_scene_sayer_context_score(self, result, gt_annotation, context_fraction):
        pred = result["pred"]
        ind = result["ind"]
        self._evaluate_predictions(context_fraction, gt_annotation[ind:], pred)

    def compute_transformer_ff_score(self, pred, gt_annotation):
        # ----------------- Evaluate future scene graphs -----------------
        count = 0
        num_ff = int(self._conf.max_future)
        num_cf = int(self._conf.baseline_context)
        num_tf = len(pred["im_idx"].unique())
        num_cf = min(num_cf, num_tf - 1)
        while num_cf + 1 <= num_tf:
            num_ff = min(num_ff, num_tf - num_cf)
            gt_future = gt_annotation[num_cf: num_cf + num_ff]
            pred_dict = pred["output"][count]

            evaluators = self._combined_ff_evaluators_dict[num_ff]
            evaluators[0].evaluate_scene_graph(gt_future, pred_dict)
            evaluators[1].evaluate_scene_graph(gt_future, pred_dict)
            evaluators[2].evaluate_scene_graph(gt_future, pred_dict)
            count += 1
            num_cf += 1

    def compute_transformer_context_score(self, result, gt_annotation, context_fraction):
        gt_future = result["gt_future"]
        pred_dict = result["pred_dict"]

        evaluators = self._combined_cf_evaluators_dict[context_fraction]
        evaluators[0].evaluate_scene_graph(gt_future, pred_dict)
        evaluators[1].evaluate_scene_graph(gt_future, pred_dict)
        evaluators[2].evaluate_scene_graph(gt_future, pred_dict)

    @abstractmethod
    def init_model(self):
        pass

    def _test_model(self):
        test_iter = iter(self._dataloader_test)
        self._model.eval()
        self._object_detector.is_train = False
        with torch.no_grad():
            # for num_video_id in tqdm(range(len(self._dataloader_test)), desc="Testing Progress (Future Frames)",
            #                          ascii=True):
            #     data = next(test_iter)
            #     im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
            #     gt_annotation = self._test_dataset.gt_annotations[data[4]]
            #     frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data
            #
            #     entry = self._object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            #     entry["gt_annotation"] = gt_annotation
            #
            #     # Load corresponding extracted feature file for comparison
            #     # video_id = gt_annotation[0][0]["frame"].split("/")[0]
            #     # video_feature_file_path = os.path.join("/data/rohith/ag/features/supervised/test",
            #     #                                        f"{video_id}_{self._conf.mode}.pkl")
            #     # with open(os.path.join(video_feature_file_path), 'rb') as pkl_file:
            #     #     data_dictionary = pickle.load(pkl_file)
            #     #
            #     # assert torch.equal(entry["features"], data_dictionary["FINAL_FEATURES"])
            #
            #     # ----------------- Process the video (Method Specific) ---------------------
            #     pred = self.process_test_video_future_frame(entry, frame_size, gt_annotation)
            #     # ---------------------------------------------------------------------------
            #
            #     # ----------------- Process evaluation score (Method Specific)-----------------
            #     self.compute_test_video_future_frame_score(pred, frame_size, gt_annotation)
            #     # ----------------------------------------------------------------------------
            #
            # print('-----------------------------------------------------------------------------------', flush=True)
            #
            # # 5. Publish the evaluation results
            # self._publish_evaluation_results(is_future_frame=True)

            for i, context_fraction in enumerate(self._context_fractions):
                test_iter = iter(self._dataloader_test)
                for num_video_id in tqdm(range(len(self._dataloader_test)), desc=f"Testing Progress (Context Fraction: {context_fraction})",
                                         ascii=True):
                    data = next(test_iter)
                    im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                    gt_annotation = self._test_dataset.gt_annotations[data[4]]
                    frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data

                    entry = self._object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                    entry["gt_annotation"] = gt_annotation

                    # ----------------- Process the video (Method Specific) ---------------------
                    result = self.process_test_video_context(entry, frame_size, gt_annotation, context_fraction)
                    # ---------------------------------------------------------------------------

                    if "skip" in result and result["skip"]:
                        continue

                    # ----------------- Process evaluation score (Method Specific)---------------
                    self.compute_test_video_context_score(result, gt_annotation, context_fraction)
                    # ---------------------------------------------------------------------------


                print('-----------------------------------------------------------------------------------', flush=True)

            # 5. Publish the evaluation results
            self._publish_evaluation_results(is_future_frame=False)

    def init_method_evaluation(self):
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
        self._test_model()
