import copy
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataloader.action_genome.ag_dataset import AG
from dataloader.action_genome.ag_dataset import cuda_collate_fn
from lib.object_detector import Detector
from sga_base import SGABase


class TrainSGABase(SGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._model = None

        # Load while initializing the object detector
        self._object_detector = None

        # Load while initializing the dataset
        self._train_dataset = None
        self._dataloader_train = None
        self._dataloader_test = None
        self._test_dataset = None
        self._object_classes = None

    def _init_diffeq_loss_functions(self):
        self._bce_loss = nn.BCELoss()
        self._ce_loss = nn.CrossEntropyLoss()
        self._mlm_loss = nn.MultiLabelMarginLoss()
        self._bbox_loss = nn.SmoothL1Loss()
        self._abs_loss = nn.L1Loss()
        self._mse_loss = nn.MSELoss()

    def _init_transformer_loss_functions(self):
        self._bce_loss = nn.BCELoss(reduction='none')
        self._ce_loss = nn.CrossEntropyLoss(reduction='none')
        self._mlm_loss = nn.MultiLabelMarginLoss(reduction='none')
        self._bbox_loss = nn.SmoothL1Loss()
        self._abs_loss = nn.L1Loss()
        self._mse_loss = nn.MSELoss()

    def _init_object_detector(self):
        self._object_detector = Detector(
            train=True,
            object_classes=self._object_classes,
            use_SUPPLY=True,
            mode=self._conf.mode
        ).to(device=self._device)
        self._object_detector.eval()

    def _train_model(self):
        tr = []
        for epoch in range(self._conf.nepoch):
            self._model.train()
            train_iter = iter(self._dataloader_train)

            counter = 0
            start_time = time.time()
            self._object_detector.is_train = True
            for train_idx in range(len(self._dataloader_train)):
                data = next(train_iter)
                im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                gt_annotation = self._train_dataset.gt_annotations[data[4]]
                frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data
                with torch.no_grad():
                    entry = self._object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

                # ----------------- Process the video (Method Specific)-----------------
                pred = self.process_train_video(entry, gt_annotation, frame_size)
                # ----------------------------------------------------------------------

                # ----------------- Compute the loss (Method Specific)-----------------
                losses = self.compute_loss(pred, gt_annotation)
                # ----------------------------------------------------------------------

                self._optimizer.zero_grad()
                loss = sum(losses.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5, norm_type=2)
                self._optimizer.step()
                tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
                counter += 1

                if counter % 1000 == 0 and counter >= 1000:
                    time_per_batch = (time.time() - start_time) / 1000
                    print(
                        "\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, counter,
                                                                                      len(self._dataloader_train),
                                                                                      time_per_batch,
                                                                                      len(self._dataloader_train) * time_per_batch / 60))

                    mn = pd.concat(tr[-1000:], axis=1).mean(1)
                    print(mn)
                    start_time = time.time()
                counter += 1

            self._save_model(
                model=self._model,
                epoch=epoch,
                checkpoint_save_file_path=self._checkpoint_save_file_path,
                checkpoint_name=self._checkpoint_name,
                method_name=self._conf.method_name
            )

            test_iter = iter(self._dataloader_test)
            self._model.eval()
            self._object_detector.is_train = False
            with torch.no_grad():
                for b in range(len(self._dataloader_test)):
                    data = next(test_iter)
                    im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                    gt_annotation = self._test_dataset.gt_annotations[data[4]]
                    frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data

                    entry = self._object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

                    # ----------------- Process the video (Method Specific)-----------------
                    pred = self.process_test_video(entry, gt_annotation, frame_size)
                    # ----------------------------------------------------------------------

                    # ----------------- Process evaluation score (Method Specific)-----------------
                    self.process_evaluation_score(pred, gt_annotation)
                    # ----------------------------------------------------------------------

                print('-----------------------------------------------------------------------------------', flush=True)
            score = np.mean(self._evaluator.result_dict[self._conf.mode + "_recall"][20])
            self._evaluator.print_stats()
            self._evaluator.reset_result()
            self._scheduler.step(score)

    def init_dataset(self):
        self._train_dataset = AG(
            phase="train",
            datasize=self._conf.datasize,
            data_path=self._conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if self._conf.mode == 'predcls' else True
        )

        self._test_dataset = AG(
            phase="test",
            datasize=self._conf.datasize,
            data_path=self._conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if self._conf.mode == 'predcls' else True
        )

        self._dataloader_train = DataLoader(
            self._train_dataset,
            shuffle=True,
            collate_fn=cuda_collate_fn,
            pin_memory=True,
            num_workers=0
        )

        self._dataloader_test = DataLoader(
            self._test_dataset,
            shuffle=False,
            collate_fn=cuda_collate_fn,
            pin_memory=False
        )

    def compute_scene_sayer_evaluation_score(self, pred, gt_annotation):
        w = self._conf.max_window
        n = len(gt_annotation)
        w = min(w, n - 1)
        for i in range(1, w + 1):
            pred_anticipated = pred.copy()
            mask_curr = pred["mask_curr_" + str(i)]
            pred_anticipated["spatial_distribution"] = pred["anticipated_spatial_distribution"][i - 1][
                mask_curr]
            pred_anticipated["contacting_distribution"] = pred["anticipated_contacting_distribution"][i - 1][
                mask_curr]
            pred_anticipated["attention_distribution"] = pred["anticipated_attention_distribution"][i - 1][
                mask_curr]
            pred_anticipated["im_idx"] = pred["im_idx_test_" + str(i)]
            pred_anticipated["pair_idx"] = pred["pair_idx_test_" + str(i)]
            if self._conf.mode == "predcls":
                pred_anticipated["scores"] = pred["scores_test_" + str(i)]
                pred_anticipated["labels"] = pred["labels_test_" + str(i)]
            else:
                pred_anticipated["pred_scores"] = pred["pred_scores_test_" + str(i)]
                pred_anticipated["pred_labels"] = pred["pred_labels_test_" + str(i)]
            pred_anticipated["boxes"] = pred["boxes_test_" + str(i)]
            self._evaluator.evaluate_scene_graph(gt_annotation[i:], pred_anticipated)

    def compute_baseline_evaluation_score(self, pred, gt_annotation):
        count = 0
        num_ff = self._conf.baseline_future
        num_cf = self._conf.baseline_context
        num_tf = len(pred["im_idx"].unique())
        num_cf = min(num_cf, num_tf - 1)
        while num_cf + 1 <= num_tf:
            num_ff = min(num_ff, num_tf - num_cf)
            gt_future = gt_annotation[num_cf: num_cf + num_ff]
            pred_dict = pred["output"][count]
            self._evaluator.evaluate_scene_graph(gt_future, pred_dict)
            count += 1
            num_cf += 1

    def compute_scene_sayer_loss(self, pred, model_ratio):
        """
        Use this method to compute the loss for the scene sayer models
        """
        global_output = pred["global_output"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contacting_distribution"]
        attention_distribution = pred["attention_distribution"]
        subject_boxes_rcnn = pred["subject_boxes_rcnn"]
        # object_boxes_rcnn = pred["object_boxes_rcnn"]
        subject_boxes_dsg = pred["subject_boxes_dsg"]
        # object_boxes_dsg = pred["object_boxes_dsg"]

        anticipated_global_output = pred["anticipated_vals"]
        anticipated_subject_boxes = pred["anticipated_subject_boxes"]
        # targets = pred["detached_outputs"]
        anticipated_spatial_distribution = pred["anticipated_spatial_distribution"]
        anticipated_contact_distribution = pred["anticipated_contacting_distribution"]
        anticipated_attention_distribution = pred["anticipated_attention_distribution"]
        # anticipated_object_boxes = pred["anticipated_object_boxes"]

        attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to \
            (device=attention_distribution.device).squeeze()
        if not self._conf.bce_loss:
            # multi-label margin loss or adaptive loss
            spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to \
                (device=attention_distribution.device)
            contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to \
                (device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])

        else:
            # bce loss
            spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to \
                (device=attention_distribution.device)
            contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to \
                (device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1
                contact_label[i, pred["contacting_gt"][i]] = 1
        losses = {}
        if self._conf.mode == 'sgcls' or self._conf.mode == 'sgdet':
            losses['object_loss'] = self._ce_loss(pred['distribution'], pred['labels'])

        losses["attention_relation_loss"] = self._ce_loss(attention_distribution, attention_label)
        losses["subject_boxes_loss"] = self._conf.bbox_ratio * self._bbox_loss(subject_boxes_dsg, subject_boxes_rcnn)
        # losses["object_boxes_loss"] = bbox_ratio * bbox_loss(object_boxes_dsg, object_boxes_rcnn)
        losses["anticipated_latent_loss"] = 0
        losses["anticipated_subject_boxes_loss"] = 0
        losses["anticipated_spatial_relation_loss"] = 0
        losses["anticipated_contact_relation_loss"] = 0
        losses["anticipated_attention_relation_loss"] = 0
        # losses["anticipated_object_boxes_loss"] = 0
        if not self._conf.bce_loss:
            losses["spatial_relation_loss"] = self._mlm_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = self._mlm_loss(contact_distribution, contact_label)
            for i in range(1, self._conf.max_window + 1):
                mask_curr = pred["mask_curr_" + str(i)]
                mask_gt = pred["mask_gt_" + str(i)]
                losses["anticipated_latent_loss"] += model_ratio * self._abs_loss(
                    anticipated_global_output[i - 1][mask_curr],
                    global_output[mask_gt])
                losses["anticipated_subject_boxes_loss"] += self._conf.bbox_ratio * self._bbox_loss \
                    (anticipated_subject_boxes[i - 1][mask_curr], subject_boxes_rcnn[mask_gt])
                losses["anticipated_spatial_relation_loss"] += self._mlm_loss \
                    (anticipated_spatial_distribution[i - 1][mask_curr], spatial_label[mask_gt])
                losses["anticipated_contact_relation_loss"] += self._mlm_loss \
                    (anticipated_contact_distribution[i - 1][mask_curr], contact_label[mask_gt])
                losses["anticipated_attention_relation_loss"] += self._ce_loss \
                    (anticipated_attention_distribution[i - 1][mask_curr], attention_label[mask_gt])
        # losses["anticipated_object_boxes_loss"] += bbox_ratio * bbox_loss(anticipated_object_boxes[i - 1][mask_curr], object_boxes_rcnn[mask_gt])
        else:
            losses["spatial_relation_loss"] = self._bce_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = self._bce_loss(contact_distribution, contact_label)
            for i in range(1, self._conf.max_window + 1):
                mask_curr = pred["mask_curr_" + str(i)]
                mask_gt = pred["mask_gt_" + str(i)]
                losses["anticipated_latent_loss"] += model_ratio * self._abs_loss(
                    anticipated_global_output[i - 1][mask_curr],
                    global_output[mask_gt])
                losses["anticipated_subject_boxes_loss"] += self._conf.bbox_ratio * self._bbox_loss \
                    (anticipated_subject_boxes[i - 1][mask_curr], subject_boxes_rcnn[mask_gt])
                losses["anticipated_spatial_relation_loss"] += self._bce_loss \
                    (anticipated_spatial_distribution[i - 1][mask_curr], spatial_label[mask_gt])
                losses["anticipated_contact_relation_loss"] += self._bce_loss \
                    (anticipated_contact_distribution[i - 1][mask_curr], contact_label[mask_gt])
                losses["anticipated_attention_relation_loss"] += self._ce_loss \
                    (anticipated_attention_distribution[i - 1][mask_curr], attention_label[mask_gt])

        return losses

    def compute_ff_ant_loss(self, start, context, future, total_frames, pred, count, losses):
        while start + context + 1 <= total_frames:
            future_frame_start_id = pred["im_idx"].unique()[context]

            if start + context + future > total_frames > start + context:
                future = total_frames - (start + context)

            future_frame_end_id = pred["im_idx"].unique()[context + future - 1]

            context_end_idx = int(torch.where(pred["im_idx"] == future_frame_start_id)[0][0])
            future_end_idx = int(torch.where(pred["im_idx"] == future_frame_end_id)[0][-1]) + 1
            if pred["output"][count]["scatter_flag"] == 0:
                attention_distribution = pred["output"][count]["attention_distribution"]
                spatial_distribution = pred["output"][count]["spatial_distribution"]
                contact_distribution = pred["output"][count]["contacting_distribution"]

                attention_label = torch.tensor(pred["attention_gt"][context_end_idx:future_end_idx],
                                               dtype=torch.long).to(
                    device=attention_distribution.device).squeeze()

                if not self._conf.bce_loss:
                    spatial_label = -torch.ones([len(pred["spatial_gt"][context_end_idx:future_end_idx]), 6],
                                                dtype=torch.long).to(device=attention_distribution.device)
                    contact_label = -torch.ones([len(pred["contacting_gt"][context_end_idx:future_end_idx]), 17],
                                                dtype=torch.long).to(device=attention_distribution.device)
                    for i in range(len(pred["spatial_gt"][context_end_idx:future_end_idx])):
                        spatial_label[i, : len(pred["spatial_gt"][context_end_idx:future_end_idx][i])] = torch.tensor(
                            pred["spatial_gt"][context_end_idx:future_end_idx][i])
                        contact_label[i,
                        : len(pred["contacting_gt"][context_end_idx:future_end_idx][i])] = torch.tensor(
                            pred["contacting_gt"][context_end_idx:future_end_idx][i])
                else:
                    spatial_label = torch.zeros([len(pred["spatial_gt"][context_end_idx:future_end_idx]), 6],
                                                dtype=torch.float32).to(device=attention_distribution.device)
                    contact_label = torch.zeros([len(pred["contacting_gt"][context_end_idx:future_end_idx]), 17],
                                                dtype=torch.float32).to(device=attention_distribution.device)
                    for i in range(len(pred["spatial_gt"][context_end_idx:future_end_idx])):
                        spatial_label[i, pred["spatial_gt"][context_end_idx:future_end_idx][i]] = 1
                        contact_label[i, pred["contacting_gt"][context_end_idx:future_end_idx][i]] = 1

                context_boxes_idx = torch.where(pred["boxes"][:, 0] == context)[0][0]
                context_excluding_last_frame_boxes_idx = torch.where(pred["boxes"][:, 0] == context - 1)[0][0]
                future_boxes_idx = torch.where(pred["boxes"][:, 0] == context + future - 1)[0][-1]

                if self._conf.mode == 'predcls':
                    context_last_frame_labels = set(
                        pred["labels"][context_excluding_last_frame_boxes_idx:context_boxes_idx].tolist())
                    future_labels = set(pred["labels"][context_boxes_idx:future_boxes_idx + 1].tolist())
                    context_labels = set(pred["labels"][:context_boxes_idx].tolist())
                else:
                    context_last_frame_labels = set(
                        pred["pred_labels"][context_excluding_last_frame_boxes_idx:context_boxes_idx].tolist())
                    future_labels = set(pred["pred_labels"][context_boxes_idx:future_boxes_idx + 1].tolist())
                    context_labels = set(pred["pred_labels"][:context_boxes_idx].tolist())

                appearing_object_labels = future_labels - context_last_frame_labels
                disappearing_object_labels = context_labels - context_last_frame_labels
                ignored_object_labels = appearing_object_labels.union(disappearing_object_labels)
                ignored_object_labels = list(ignored_object_labels)

                # Weighting loss based on appearance or disappearance of objects
                # We only consider loss on objects that are present in the last frame of the context
                weight = torch.ones(pred["output"][count]["global_output"].shape[0]).cuda()
                for object_label in ignored_object_labels:
                    for idx, pair in enumerate(pred["pair_idx"][context_end_idx:future_end_idx]):
                        if self._conf.mode == 'predcls':
                            if pred["labels"][pair[1]] == object_label:
                                weight[idx] = 0
                        else:
                            if pred["pred_labels"][pair[1]] == object_label:
                                weight[idx] = 0
                try:
                    at_loss = self._ce_loss(attention_distribution, attention_label)
                    losses["attention_relation_loss"] += (at_loss * weight).mean()
                except ValueError:
                    # If there is only one object in the last frame of the context, we need to unsqueeze the label
                    attention_label = attention_label.unsqueeze(0)
                    at_loss = self._ce_loss(attention_distribution, attention_label)
                    losses["attention_relation_loss"] += (at_loss * weight).mean()

                if not self._conf.bce_loss:
                    sp_loss = self._mlm_loss(spatial_distribution, spatial_label)
                    losses["spatial_relation_loss"] += (sp_loss * weight).mean()
                    con_loss = self._mlm_loss(contact_distribution, contact_label)
                    losses["contact_relation_loss"] += (con_loss * weight).mean()
                else:
                    sp_loss = self._bce_loss(spatial_distribution, spatial_label)
                    losses["spatial_relation_loss"] += (sp_loss * weight).mean()
                    con_loss = self._bce_loss(contact_distribution, contact_label)
                    losses["contact_relation_loss"] += (con_loss * weight).mean()

                latent_loss = self._abs_loss(pred["output"][count]["global_output"],
                                             pred["output"][count]["spatial_latents"])
                losses["anticipated_latent_loss"] += (latent_loss * weight).mean()

                context += 1
                count += 1
            else:
                context += 1
                count += 1

        losses["attention_relation_loss"] = losses["attention_relation_loss"] / count
        losses["spatial_relation_loss"] = losses["spatial_relation_loss"] / count
        losses["contact_relation_loss"] = losses["contact_relation_loss"] / count
        losses["anticipated_latent_loss"] = losses["anticipated_latent_loss"] / count

        return losses

    def compute_gen_loss(self, pred, losses):
        gen_attention_out = pred["gen_attention_distribution"]
        gen_spatial_out = pred["gen_spatial_distribution"]
        gen_contacting_out = pred["gen_contacting_distribution"]

        gen_attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=self._device).squeeze()
        if not self._conf.bce_loss:
            # multi-label margin loss or adaptive loss
            gen_spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=self._device)
            gen_contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=self._device)
            for i in range(len(pred["spatial_gt"])):
                gen_spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                gen_contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])
        else:
            gen_spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=self._device)
            gen_contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(
                device=self._device)
            for i in range(len(pred["spatial_gt"])):
                gen_spatial_label[i, pred["spatial_gt"][i]] = 1
                gen_contact_label[i, pred["contacting_gt"][i]] = 1

        try:
            losses["gen_attention_relation_loss"] = self._ce_loss(gen_attention_out, gen_attention_label).mean()
        except ValueError:
            gen_attention_label = gen_attention_label.unsqueeze(0)
            losses["gen_attention_relation_loss"] = self._ce_loss(gen_attention_out, gen_attention_label).mean()

        if not self._conf.bce_loss:
            losses["gen_spatial_relation_loss"] = self._mlm_loss(gen_spatial_out, gen_spatial_label).mean()
            losses["gen_contact_relation_loss"] = self._mlm_loss(gen_contacting_out, gen_contact_label).mean()
        else:
            losses["gen_spatial_relation_loss"] = self._bce_loss(gen_spatial_out, gen_spatial_label).mean()
            losses["gen_contact_relation_loss"] = self._bce_loss(gen_contacting_out, gen_contact_label).mean()

        return losses

    def compute_baseline_ant_loss(self, pred):
        context = self._conf.baseline_context
        future = self._conf.baseline_future
        count = 0
        start = 0
        total_frames = len(pred["im_idx"].unique())

        losses = {}
        if self._conf.mode == 'sgcls' or self._conf.mode == 'sgdet':
            losses['object_loss'] = self._ce_loss(pred['distribution'], pred['labels']).mean()

        losses["attention_relation_loss"] = 0
        losses["spatial_relation_loss"] = 0
        losses["contact_relation_loss"] = 0
        losses["anticipated_latent_loss"] = 0

        context = min(context, total_frames - 1)
        future = min(future, total_frames - context)

        losses = self.compute_ff_ant_loss(start, context, future, total_frames, pred, count, losses)

        return losses

    def compute_baseline_gen_ant_loss(self, pred):
        count = 0
        start = 0
        context = self._conf.baseline_context
        future = self._conf.baseline_future
        total_frames = len(pred["im_idx"].unique())

        losses = {}
        if self._conf.mode == 'sgcls' or self._conf.mode == 'sgdet':
            losses['object_loss'] = self._ce_loss(pred['distribution'], pred['labels']).mean()

        losses["attention_relation_loss"] = 0
        losses["spatial_relation_loss"] = 0
        losses["contact_relation_loss"] = 0

        context = min(context, total_frames - 1)
        future = min(future, total_frames - context)

        losses = self.compute_ff_ant_loss(start, context, future, total_frames, pred, count, losses)

        losses = self.compute_gen_loss(pred, losses)

        return losses

    @abstractmethod
    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        pass

    @abstractmethod
    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        pass

    @abstractmethod
    def compute_loss(self, pred, gt) -> dict:
        pass

    @abstractmethod
    def process_evaluation_score(self, pred, gt_annotation):
        pass

    def init_method_training(self):
        # 0. Initialize the config
        self._init_config()

        # 1. Initialize the dataset
        self.init_dataset()

        # 2. Initialize evaluators
        self._init_evaluators()

        # 3. Initialize and load pre-trained models
        self.init_model()
        self._load_checkpoint()
        self._init_object_detector()

        # 4. Initialize model training
        self._train_model()


