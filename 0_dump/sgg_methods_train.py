import pickle

import torch
import torch.nn as nn
import numpy as np
import time
import os
import warnings
import pandas as pd
import copy

from tqdm import tqdm

from dataloader.action_genome.ag_features import AGFeatures
from dataloader.action_genome.ag_features import cuda_collate_fn as ag_features_cuda_collate_fn
from dataloader.action_genome.ag_dataset import AG
from dataloader.action_genome.ag_dataset import cuda_collate_fn as ag_data_cuda_collate_fn
from constants import Constants as const
from lib.supervised.config import Config
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from logger_config import get_logger, setup_logging

np.set_printoptions(precision=3)
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

setup_logging()
logger = get_logger(__name__)


def prepare_optimizer(model):
    if conf.optimizer == const.ADAMW:
        optimizer = AdamW(model.parameters(), lr=conf.lr)
    elif conf.optimizer == const.ADAM:
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.optimizer == const.SGD:
        optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise NotImplementedError

    scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4,
                                  threshold_mode="abs", min_lr=1e-7)
    return optimizer, scheduler


def train_dsg_detr():
    from lib.supervised.dsgdetr.dsgdetr import DsgDETR
    from lib.supervised.dsgdetr.track import get_sequence_with_tracking
    from lib.supervised.dsgdetr.matcher import HungarianMatcher
    from lib.object_detector import Detector

    model = DsgDETR(mode=conf.mode,
                    attention_class_num=len(ag_train_data.attention_relationships),
                    spatial_class_num=len(ag_train_data.spatial_relationships),
                    contact_class_num=len(ag_train_data.contacting_relationships),
                    obj_classes=ag_train_data.object_classes,
                    enc_layer_num=conf.enc_layer,
                    dec_layer_num=conf.dec_layer).to(device=gpu_device)

    if conf.ckpt:
        ckpt = torch.load(conf.ckpt, map_location=gpu_device)
        model.load_state_dict(ckpt[const.STATE_DICT], strict=False)

    optimizer, scheduler = prepare_optimizer(model)

    tr = []
    matcher = HungarianMatcher(0.5, 1, 1, 0.5)
    matcher.eval()

    object_detector = Detector(
        train=True,
        object_classes=ag_train_data.object_classes,
        use_SUPPLY=True,
        mode=conf.mode
    ).to(device=gpu_device)
    object_detector.eval()
    print("Finished loading object detector", flush=True)

    if conf.optimizer == const.ADAMW:
        optimizer = AdamW(model.parameters(), lr=conf.lr)
    elif conf.optimizer == const.ADAM:
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.optimizer == const.SGD:
        optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise NotImplementedError

    scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4,
                                  threshold_mode="abs", min_lr=1e-7)

    for epoch in range(conf.nepoch):
        model.train()
        train_iter = iter(dataloader_train)
        start = time.time()
        counter = 0
        for b in range(len(dataloader_train)):
            data = next(train_iter)
            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = ag_train_data.gt_annotations[data[4]]
            frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data
            with torch.no_grad():
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            get_sequence_with_tracking(entry, gt_annotation, matcher, frame_size, conf.mode)
            pred = model(entry)

            attention_distribution = pred[const.ATTENTION_DISTRIBUTION]
            spatial_distribution = pred[const.SPATIAL_DISTRIBUTION]
            contact_distribution = pred[const.CONTACTING_DISTRIBUTION]

            attention_label = torch.tensor(pred[const.ATTENTION_GT], dtype=torch.long).to(
                device=attention_distribution.device).squeeze()
            if not conf.bce_loss:
                # multi-label margin loss or adaptive loss
                spatial_label = -torch.ones([len(pred[const.SPATIAL_GT]), 6], dtype=torch.long).to(
                    device=attention_distribution.device)
                contact_label = -torch.ones([len(pred[const.CONTACTING_GT]), 17], dtype=torch.long).to(
                    device=attention_distribution.device)
                for i in range(len(pred[const.SPATIAL_GT])):
                    spatial_label[i, : len(pred[const.SPATIAL_GT][i])] = torch.tensor(pred[const.SPATIAL_GT][i])
                    contact_label[i, : len(pred[const.CONTACTING_GT][i])] = torch.tensor(pred[const.CONTACTING_GT][i])
            else:
                # bce loss
                spatial_label = torch.zeros([len(pred[const.SPATIAL_GT]), 6], dtype=torch.float32).to(
                    device=attention_distribution.device)
                contact_label = torch.zeros([len(pred[const.CONTACTING_GT]), 17], dtype=torch.float32).to(
                    device=attention_distribution.device)
                for i in range(len(pred[const.SPATIAL_GT])):
                    spatial_label[i, pred[const.SPATIAL_GT][i]] = 1
                    contact_label[i, pred[const.CONTACTING_GT][i]] = 1

            losses = {}
            if conf.mode == const.SGCLS or conf.mode == const.SGDET:
                losses[const.OBJECT_LOSS] = ce_loss(pred[const.DISTRIBUTION], pred[const.LABELS])

            losses[const.ATTENTION_RELATION_LOSS] = ce_loss(attention_distribution, attention_label)
            if not conf.bce_loss:
                losses[const.SPATIAL_RELATION_LOSS] = mlm_loss(spatial_distribution, spatial_label)
                losses[const.CONTACTING_RELATION_LOSS] = mlm_loss(contact_distribution, contact_label)
            else:
                losses[const.SPATIAL_RELATION_LOSS] = bce_loss(spatial_distribution, spatial_label)
                losses[const.CONTACTING_RELATION_LOSS] = bce_loss(contact_distribution, contact_label)

            optimizer.zero_grad()
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()

            tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

            if counter % 1000 == 0 and counter >= 1000:
                time_per_batch = (time.time() - start) / 1000
                print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, counter, len(dataloader_train),
                                                                                    time_per_batch,
                                                                                    len(dataloader_train) * time_per_batch / 60))

                mn = pd.concat(tr[-1000:], axis=1).mean(1)
                print(mn)
                start = time.time()
            counter += 1

        torch.save({const.STATE_DICT: model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
        print("*" * 40)
        print("save the checkpoint after {} epochs".format(epoch))
        with open(evaluator.save_file, "a") as f:
            f.write("save the checkpoint after {} epochs\n".format(epoch))

        model.eval()
        with torch.no_grad():
            for test_entry in tqdm(dataloader_test):
                gt_annotation = test_entry[const.GT_ANNOTATION]
                frame_size = test_entry[const.FRAME_SIZE]

                get_sequence_with_tracking(test_entry, gt_annotation, matcher, frame_size, conf.mode)
                pred = model(test_entry)

                vid_no = gt_annotation[0][0][const.FRAME].split('.')[0]
                test_intermediate_dump_directory = os.path.join(conf.data_path, f"frames_{conf.mode}",
                                                                f"test_{str(epoch)}")
                os.makedirs(test_intermediate_dump_directory, exist_ok=True)
                test_intermediate_dump_file = os.path.join(test_intermediate_dump_directory, f"{vid_no}.pkl")
                with open(test_intermediate_dump_file, 'wb') as f:
                    pickle.dump(pred, f)

                evaluator.evaluate_scene_graph(gt_annotation, pred)
            print('-----------------------------------------------------------------------------------', flush=True)
        score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
        evaluator.print_stats()
        evaluator.reset_result()
        scheduler.step(score)


def train_sttran():
    from lib.supervised.biased.sttran.sttran import STTran

    model = STTran(mode=conf.mode,
                   attention_class_num=len(ag_train_data.attention_relationships),
                   spatial_class_num=len(ag_train_data.spatial_relationships),
                   contact_class_num=len(ag_train_data.contacting_relationships),
                   obj_classes=ag_train_data.object_classes,
                   enc_layer_num=conf.enc_layer,
                   dec_layer_num=conf.dec_layer).to(device=gpu_device)

    optimizer, scheduler = prepare_optimizer(model)
    print('-----------------------------------------------------------------------------------')
    print(f"Mode : {conf.mode}")
    print('-----------------------------------------------------------------------------------')

    tr = []
    for epoch in range(conf.nepoch):
        model.train()
        start = time.time()
        # train_iter = iter(dataloader_train)
        # test_iter = iter(dataloader_test)
        counter = 0
        for train_entry in tqdm(dataloader_train):
            pred = model(train_entry)
            attention_distribution = pred[const.ATTENTION_DISTRIBUTION]
            spatial_distribution = pred[const.SPATIAL_DISTRIBUTION]
            contact_distribution = pred[const.CONTACTING_DISTRIBUTION]

            attention_label = torch.tensor(pred[const.ATTENTION_GT], dtype=torch.long).to(
                device=attention_distribution.device).squeeze()
            if not conf.bce_loss:
                # multi-label margin loss or adaptive loss
                spatial_label = -torch.ones([len(pred[const.SPATIAL_GT]), 6], dtype=torch.long).to(
                    device=attention_distribution.device)
                contact_label = -torch.ones([len(pred[const.CONTACTING_GT]), 17], dtype=torch.long).to(
                    device=attention_distribution.device)
                for i in range(len(pred[const.SPATIAL_GT])):
                    spatial_label[i, : len(pred[const.SPATIAL_GT][i])] = torch.tensor(pred[const.SPATIAL_GT][i])
                    contact_label[i, : len(pred[const.CONTACTING_GT][i])] = torch.tensor(pred[const.CONTACTING_GT][i])
            else:
                # bce loss
                spatial_label = torch.zeros([len(pred[const.SPATIAL_GT]), 6], dtype=torch.float32).to(
                    device=attention_distribution.device)
                contact_label = torch.zeros([len(pred[const.CONTACTING_GT]), 17], dtype=torch.float32).to(
                    device=attention_distribution.device)
                for i in range(len(pred[const.SPATIAL_GT])):
                    spatial_label[i, pred[const.SPATIAL_GT][i]] = 1
                    contact_label[i, pred[const.CONTACTING_GT][i]] = 1

            losses = {}
            if conf.mode == const.SGCLS or conf.mode == const.SGDET:
                losses[const.OBJECT_LOSS] = ce_loss(pred[const.DISTRIBUTION], pred[const.LABELS])

            losses[const.ATTENTION_RELATION_LOSS] = ce_loss(attention_distribution, attention_label)
            if not conf.bce_loss:
                losses[const.SPATIAL_RELATION_LOSS] = mlm_loss(spatial_distribution, spatial_label)
                losses[const.CONTACTING_RELATION_LOSS] = mlm_loss(contact_distribution, contact_label)

            else:
                losses[const.SPATIAL_RELATION_LOSS] = bce_loss(spatial_distribution, spatial_label)
                losses[const.CONTACTING_RELATION_LOSS] = bce_loss(contact_distribution, contact_label)

            optimizer.zero_grad()
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()

            tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
            
            if counter % 1000 == 0 and counter >= 1000:
                time_per_batch = (time.time() - start) / 1000
                print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, counter, len(dataloader_train),
                                                                                    time_per_batch,
                                                                                    len(dataloader_train) * time_per_batch / 60))
                
                mn = pd.concat(tr[-1000:], axis=1).mean(1)
                print(mn)
                start = time.time()
            counter += 1

        torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
        print("*" * 40)
        print("save the checkpoint after {} epochs".format(epoch))

        model.eval()
        with torch.no_grad():
            # for b in range(len(dataloader_test)):
            for test_entry in tqdm(dataloader_test):
                gt_annotation = test_entry[const.GT_ANNOTATION]
                pred = model(test_entry)
                evaluator.evaluate_scene_graph(gt_annotation, pred)
            print('-----------', flush=True)
        score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
        evaluator.print_stats()
        evaluator.reset_result()
        scheduler.step(score)


if __name__ == '__main__':
    conf = Config()
    print('The CKPT saved here:', conf.save_path)
    if not os.path.exists(conf.save_path):
        os.mkdir(conf.save_path)
    print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
    for i in conf.args:
        print(i, ':', conf.args[i])
    
    # Set the preferred device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not conf.use_raw_data:
        ag_train_data = AGFeatures(
            mode=conf.mode,
            data_split=const.TRAIN,
            device=device,
            data_path=conf.data_path,
            is_compiled_together=False,
            filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == const.PREDCLS else True,
            features_path=conf.features_path,
            additional_data_path=conf.additional_data_path
        )

        ag_test_data = AGFeatures(
            mode=conf.mode,
            data_split=const.TEST,
            device=device,
            data_path=conf.data_path,
            is_compiled_together=False,
            filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == const.PREDCLS else True,
            features_path=conf.features_path,
            additional_data_path=conf.additional_data_path
        )

        dataloader_train = DataLoader(
            ag_train_data,
            shuffle=True,
            collate_fn=ag_features_cuda_collate_fn,
            pin_memory=False,
            num_workers=0
        )

        dataloader_test = DataLoader(
            ag_test_data,
            shuffle=False,
            collate_fn=ag_features_cuda_collate_fn,
            pin_memory=False
        )
    else:
        ag_train_data = AG(
            phase="train",
            datasize=conf.datasize,
            data_path=conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == 'predcls' else True
        )

        ag_test_data = AG(
            phase="test",
            datasize=conf.datasize,
            data_path=conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == 'predcls' else True
        )

        dataloader_train = DataLoader(
            ag_train_data,
            shuffle=True,
            collate_fn=ag_data_cuda_collate_fn,
            pin_memory=True,
            num_workers=0
        )

        dataloader_test = DataLoader(
            ag_test_data,
            shuffle=False,
            collate_fn=ag_data_cuda_collate_fn,
            pin_memory=False
        )

    gpu_device = torch.device("cuda:0")

    evaluator = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=ag_train_data.object_classes,
        AG_all_predicates=ag_train_data.relationship_classes,
        AG_attention_predicates=ag_train_data.attention_relationships,
        AG_spatial_predicates=ag_train_data.spatial_relationships,
        AG_contacting_predicates=ag_train_data.contacting_relationships,
        iou_threshold=0.5,
        save_file=os.path.join(conf.save_path, const.PROGRESS_TEXT_FILE),
        constraint='with'
    )

    # loss function, default Multi-label margin loss
    if conf.bce_loss:
        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCELoss()
    else:
        ce_loss = nn.CrossEntropyLoss()
        mlm_loss = nn.MultiLabelMarginLoss()

    train_dsg_detr()
    # train_sttran()
