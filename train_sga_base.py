import copy
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from constants import Constants as const
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

        # Load checkpoint name
        self._checkpoint_name = None

    def _init_loss_functions(self):
        self._bce_loss = nn.BCELoss()
        self._ce_loss = nn.CrossEntropyLoss()
        self._mlm_loss = nn.MultiLabelMarginLoss()
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
                pred = self.process_train_video(entry, frame_size)
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
                checkpoint_save_file_path=self._conf.save_path,
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
                    pred = self.process_test_video(entry, frame_size)
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

    @abstractmethod
    def process_train_video(self, video, frame_size) -> dict:
        pass

    @abstractmethod
    def process_test_video(self, video, frame_size) -> dict:
        pass

    @abstractmethod
    def compute_loss(self, pred, gt) -> dict:
        pass

    @abstractmethod
    def process_evaluation_score(self, pred, gt):
        pass

    # TODO: Fill the following methods describing various loss functions
    def compute_scene_sayer_loss(self, pred, gt):
        pass

    def compute_baseline_ant_loss(self, pred, gt):
        pass

    def compute_baseline_gen_ant_loss(self, pred, gt):
        pass
