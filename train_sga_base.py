import copy
import time
from abc import abstractmethod

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
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

        # Observed Representations Loss
        self._enable_obj_class_loss = False
        self._enable_gen_pred_class_loss = False

        # Anticipated Representations Loss
        self._enable_ant_pred_loss = False
        self._enable_ant_bb_subject_loss = False
        self._enable_ant_bb_object_loss = False
        self._enable_ant_recon_loss = False

    def _init_diffeq_loss_function_heads(self):
        self._bce_loss = nn.BCELoss()
        self._ce_loss = nn.CrossEntropyLoss()
        self._mlm_loss = nn.MultiLabelMarginLoss()
        self._bbox_loss = nn.SmoothL1Loss()
        self._abs_loss = nn.L1Loss()
        self._mse_loss = nn.MSELoss()

    def _init_transformer_loss_function_heads(self):
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
        for epoch in range(self._conf.nepoch):
            self._model.train()
            train_iter = iter(self._dataloader_train)

            start_time = time.time()
            self._object_detector.is_train = True
            for train_id in tqdm(range(len(self._dataloader_train))):
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

                if self._enable_wandb:
                    wandb.log(losses)

                if train_id % 100 == 0 and train_id >= 100:
                    time_per_batch = (time.time() - start_time) / 1000
                    print(
                        "\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, train_id,
                                                                                      len(self._dataloader_train),
                                                                                      time_per_batch,
                                                                                      len(self._dataloader_train) * time_per_batch / 60))

                    start_time = time.time()

            self._save_model(
                model=self._model,
                epoch=epoch,
                checkpoint_save_file_path=self._checkpoint_save_dir_path,
                checkpoint_name=self._checkpoint_name,
                method_name=self._conf.method_name
            )

            test_iter = iter(self._dataloader_test)
            self._model.eval()
            self._object_detector.is_train = False
            with torch.no_grad():
                for b in tqdm(range(len(self._dataloader_test))):
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

        self._object_classes = self._train_dataset.object_classes

    def compute_scene_sayer_evaluation_score(self, pred, gt_annotation):
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
            self._evaluator.evaluate_scene_graph(gt_annotation[i:], pred_anticipated)

    def compute_baseline_evaluation_score(self, pred, gt_annotation):
        count = 0
        num_ff = self._conf.max_window
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

    def compute_gt_relationship_labels(self, pred):
        attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=self._device).squeeze()
        if not self._conf.bce_loss:
            spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=self._device)
            contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=self._device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])
        else:
            spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=self._device)
            contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=self._device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1
                contact_label[i, pred["contacting_gt"][i]] = 1

        return attention_label, spatial_label, contact_label

    # ------------------------------------------------------------------------------------------------------
    # ----------------------------------- SCENE SAYER LOSS FUNCTIONS ---------------------------------------
    # ------------------------------------------------------------------------------------------------------

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

        attention_label, spatial_label, contact_label = self.compute_gt_relationship_labels(pred)

        losses = {}
        if self._conf.mode == 'sgcls' or self._conf.mode == 'sgdet':
            if self._enable_obj_class_loss:
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
                if "mask_gt_" + str(i) not in pred:
                    print("mask_gt_" + str(i) + " not in pred")
                    continue

                mask_curr = pred["mask_curr_" + str(i)]
                mask_gt = pred["mask_gt_" + str(i)]

                if self._enable_ant_recon_loss:
                    losses["anticipated_latent_loss"] += model_ratio * self._abs_loss(
                        anticipated_global_output[i - 1][mask_curr],
                        global_output[mask_gt])

                if self._enable_ant_bb_subject_loss:
                    losses["anticipated_subject_boxes_loss"] += self._conf.bbox_ratio * self._bbox_loss \
                        (anticipated_subject_boxes[i - 1][mask_curr], subject_boxes_rcnn[mask_gt])

                if self._enable_ant_pred_loss:
                    losses["anticipated_spatial_relation_loss"] += self._mlm_loss \
                        (anticipated_spatial_distribution[i - 1][mask_curr], spatial_label[mask_gt])
                    losses["anticipated_contact_relation_loss"] += self._mlm_loss \
                        (anticipated_contact_distribution[i - 1][mask_curr], contact_label[mask_gt])
                    losses["anticipated_attention_relation_loss"] += self._ce_loss \
                        (anticipated_attention_distribution[i - 1][mask_curr], attention_label[mask_gt])

                # if self._enable_ant_bb_object_loss:
                #     losses["anticipated_object_boxes_loss"] += self._conf.bbox_ratio * self._bbox_loss(
                #               anticipated_object_boxes[i - 1][mask_curr],
                #               object_boxes_rcnn[mask_gt]
                #               )
        else:
            losses["spatial_relation_loss"] = self._bce_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = self._bce_loss(contact_distribution, contact_label)
            for i in range(1, self._conf.max_window + 1):
                if "mask_gt_" + str(i) not in pred:
                    print("mask_gt_" + str(i) + " not in pred")
                    continue
                mask_curr = pred["mask_curr_" + str(i)]
                mask_gt = pred["mask_gt_" + str(i)]

                if self._enable_ant_recon_loss:
                    losses["anticipated_latent_loss"] += model_ratio * self._abs_loss(
                        anticipated_global_output[i - 1][mask_curr],
                        global_output[mask_gt])

                if self._enable_ant_bb_subject_loss:
                    losses["anticipated_subject_boxes_loss"] += self._conf.bbox_ratio * self._bbox_loss \
                        (anticipated_subject_boxes[i - 1][mask_curr], subject_boxes_rcnn[mask_gt])

                if self._enable_ant_pred_loss:
                    losses["anticipated_spatial_relation_loss"] += self._bce_loss \
                        (anticipated_spatial_distribution[i - 1][mask_curr], spatial_label[mask_gt])
                    losses["anticipated_contact_relation_loss"] += self._bce_loss \
                        (anticipated_contact_distribution[i - 1][mask_curr], contact_label[mask_gt])
                    losses["anticipated_attention_relation_loss"] += self._ce_loss \
                        (anticipated_attention_distribution[i - 1][mask_curr], attention_label[mask_gt])

        return losses

    # ------------------------------------------------------------------------------------------------------
    # ----------------------------------- BASELINE LOSS FUNCTIONS ------------------------------------------
    # ------------------------------------------------------------------------------------------------------

    def compute_ff_ant_loss(self, pred, losses, attention_label, spatial_label, contact_label):
        global_output = pred["global_output"]
        ant_output = pred["output"]

        cum_ant_attention_relation_loss = 0
        cum_ant_spatial_relation_loss = 0
        cum_ant_contact_relation_loss = 0
        cum_ant_latent_loss = 0

        loss_count = 0
        count = 0

        num_cf = self._conf.baseline_context
        num_tf = len(pred["im_idx"].unique())
        while num_cf + 1 <= num_tf:
            ant_spatial_distribution = ant_output[count]["spatial_distribution"]
            ant_contact_distribution = ant_output[count]["contacting_distribution"]
            ant_attention_distribution = ant_output[count]["attention_distribution"]
            ant_global_output = ant_output[count]["global_output"]

            mask_ant = ant_output[count]["mask_ant"].cpu().numpy()
            mask_gt = ant_output[count]["mask_gt"].cpu().numpy()

            if len(mask_ant) == 0:
                assert len(mask_gt) == 0
            else:
                loss_count += 1
                ant_attention_relation_loss = self._ce_loss(
                    ant_attention_distribution[mask_ant],
                    attention_label[mask_gt]
                ).mean()
                try:
                    ant_anticipated_latent_loss = self._conf.hp_recon_loss * self._abs_loss(
                        ant_global_output[mask_ant],
                        global_output[mask_gt]
                    ).mean()
                except:
                    ant_anticipated_latent_loss = 0
                    print(ant_global_output.shape, mask_ant.shape, global_output.shape, mask_gt.shape)
                    print(mask_ant)

                if not self._conf.bce_loss:
                    ant_spatial_relation_loss = self._mlm_loss(ant_spatial_distribution[mask_ant],
                                                               spatial_label[mask_gt]).mean()
                    ant_contact_relation_loss = self._mlm_loss(ant_contact_distribution[mask_ant],
                                                               contact_label[mask_gt]).mean()
                else:
                    ant_spatial_relation_loss = self._bce_loss(ant_spatial_distribution[mask_ant],
                                                               spatial_label[mask_gt]).mean()
                    ant_contact_relation_loss = self._bce_loss(ant_contact_distribution[mask_ant],
                                                               contact_label[mask_gt]).mean()
                cum_ant_attention_relation_loss += ant_attention_relation_loss
                cum_ant_spatial_relation_loss += ant_spatial_relation_loss
                cum_ant_contact_relation_loss += ant_contact_relation_loss
                cum_ant_latent_loss += ant_anticipated_latent_loss
            num_cf += 1
            count += 1

        if loss_count > 0:
            if self._enable_ant_pred_loss:
                losses["anticipated_attention_relation_loss"] = cum_ant_spatial_relation_loss / loss_count
                losses["anticipated_spatial_relation_loss"] = cum_ant_spatial_relation_loss / loss_count
                losses["anticipated_contact_relation_loss"] = cum_ant_contact_relation_loss / loss_count

            if self._enable_ant_recon_loss:
                losses["anticipated_latent_loss"] = cum_ant_latent_loss / loss_count

        return losses

    def compute_gen_loss(self, pred, losses, attention_label, spatial_label, contact_label):
        attention_distribution = pred["gen_attention_distribution"]
        spatial_distribution = pred["gen_spatial_distribution"]
        contacting_distribution = pred["gen_contacting_distribution"]

        try:
            losses["gen_attention_relation_loss"] = self._ce_loss(attention_distribution, attention_label).mean()
        except ValueError:
            attention_label = attention_label.unsqueeze(0)
            losses["gen_attention_relation_loss"] = self._ce_loss(attention_distribution, attention_label).mean()

        if not self._conf.bce_loss:
            losses["gen_spatial_relation_loss"] = self._mlm_loss(spatial_distribution, spatial_label).mean()
            losses["gen_contact_relation_loss"] = self._mlm_loss(contacting_distribution, contact_label).mean()
        else:
            losses["gen_spatial_relation_loss"] = self._bce_loss(spatial_distribution, spatial_label).mean()
            losses["gen_contact_relation_loss"] = self._bce_loss(contacting_distribution, contact_label).mean()

        return losses

    def compute_baseline_ant_loss(self, pred):
        attention_label, spatial_label, contact_label = self.compute_gt_relationship_labels(pred)

        losses = {}
        if self._conf.mode == 'sgcls' or self._conf.mode == 'sgdet':
            if self._enable_obj_class_loss:
                losses['object_loss'] = self._ce_loss(pred['distribution'], pred['labels']).mean()

        losses = self.compute_ff_ant_loss(pred, losses, attention_label, spatial_label, contact_label)
        return losses

    def compute_baseline_gen_ant_loss(self, pred):
        attention_label, spatial_label, contact_label = self.compute_gt_relationship_labels(pred)

        losses = {}
        if self._conf.mode == 'sgcls' or self._conf.mode == 'sgdet':
            if self._enable_obj_class_loss:
                losses['object_loss'] = self._ce_loss(pred['distribution'], pred['labels']).mean()

        losses = self.compute_ff_ant_loss(pred, losses, attention_label, spatial_label, contact_label)

        if self._enable_gen_pred_class_loss:
            losses = self.compute_gen_loss(pred, losses, attention_label, spatial_label, contact_label)

        return losses

    # ------------------------ Abstract Train Methods ------------------------ #

    @abstractmethod
    def process_train_video(self, entry, gt_annotation, frame_size) -> dict:
        pass

    @abstractmethod
    def compute_loss(self, pred, gt) -> dict:
        pass

    @abstractmethod
    def init_method_loss_type_params(self):
        pass

    # ------------------------ Abstract Test Methods ------------------------ #
    @abstractmethod
    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
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

        # 3. Enable/Disable loss type parameters
        self.init_method_loss_type_params()

        # 3. Initialize and load pre-trained models
        self.init_model()
        self._load_checkpoint()
        self._init_object_detector()
        self._init_optimizer()
        self._init_scheduler()

        # 4. Initialize model training
        self._train_model()
