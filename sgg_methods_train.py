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

from dataloader.supervised.generation.action_genome.ag_features import AGFeatures, cuda_collate_fn
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
    # optimizer
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
    from lib.supervised.biased.dsgdetr.dsgdetr import DsgDETR
    from lib.supervised.biased.dsgdetr.track import get_sequence
    from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher

    model = DsgDETR(mode=conf.mode,
                    attention_class_num=len(ag_features_train.attention_relationships),
                    spatial_class_num=len(ag_features_train.spatial_relationships),
                    contact_class_num=len(ag_features_train.contacting_relationships),
                    obj_classes=ag_features_train.object_classes,
                    enc_layer_num=conf.enc_layer,
                    dec_layer_num=conf.dec_layer).to(device=gpu_device)

    if conf.ckpt:
        ckpt = torch.load(conf.ckpt, map_location=gpu_device)
        model.load_state_dict(ckpt[const.STATE_DICT], strict=False)

    optimizer, scheduler = prepare_optimizer(model)

    tr = []
    matcher = HungarianMatcher(0.5, 1, 1, 0.5)
    matcher.eval()

    for epoch in range(conf.nepoch):
        model.train()
        start = time.time()
        train_iter = iter(dataloader_train)
        test_iter = iter(dataloader_test)
        for b in range(len(dataloader_train)):
            data = next(train_iter)

            gt_annotation = ag_features_train.gt_annotations[data[4]]

            get_sequence(entry, gt_annotation, matcher, (im_info[0][:2] / im_info[0, 2]).cpu().data, conf.mode)
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

            if b % 1000 == 0 and b >= 1000:
                time_per_batch = (time.time() - start) / 1000
                print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                    time_per_batch,
                                                                                    len(dataloader_train) * time_per_batch / 60))

                mn = pd.concat(tr[-1000:], axis=1).mean(1)
                print(mn)
                start = time.time()

        torch.save({const.STATE_DICT: model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
        print("*" * 40)
        print("save the checkpoint after {} epochs".format(epoch))
        with open(evaluator.save_file, "a") as f:
            f.write("save the checkpoint after {} epochs\n".format(epoch))

        model.eval()
        with torch.no_grad():
            for test_entry in tqdm(dataloader_test):
                gt_annotation = test_entry[const.GT_ANNOTATION]

                get_sequence(test_entry, gt_annotation, matcher, (im_info[0][:2] / im_info[0, 2]).cpu().data, conf.mode)
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
                   attention_class_num=len(ag_features_train.attention_relationships),
                   spatial_class_num=len(ag_features_train.spatial_relationships),
                   contact_class_num=len(ag_features_train.contacting_relationships),
                   obj_classes=ag_features_train.object_classes,
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
                print(
                    "\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, counter, len(dataloader_train),
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

    ag_features_train = AGFeatures(
        mode=conf.mode,
        data_split=const.TRAIN,
        data_path=conf.data_path,
        is_compiled_together=False,
        filter_nonperson_box_frame=True,
        filter_small_box=False if conf.mode == const.PREDCLS else True
    )

    dataloader_train = DataLoader(
        ag_features_train,
        shuffle=True,
        num_workers=1,
        collate_fn=cuda_collate_fn,
        pin_memory=False
    )

    ag_features_test = AGFeatures(
        mode=conf.mode,
        data_split=const.TEST,
        data_path=conf.data_path,
        is_compiled_together=False,
        filter_nonperson_box_frame=True,
        filter_small_box=False if conf.mode == const.PREDCLS else True
    )

    dataloader_test = DataLoader(
        ag_features_test,
        shuffle=False,
        num_workers=1,
        collate_fn=cuda_collate_fn,
        pin_memory=False
    )

    gpu_device = torch.device("cuda:0")

    evaluator = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=ag_features_train.object_classes,
        AG_all_predicates=ag_features_train.relationship_classes,
        AG_attention_predicates=ag_features_train.attention_relationships,
        AG_spatial_predicates=ag_features_train.spatial_relationships,
        AG_contacting_predicates=ag_features_train.contacting_relationships,
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

    # train_dsg_detr()
    train_sttran()
