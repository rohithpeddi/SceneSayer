import sys
from ODE import ODE as ODE

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence
import numpy as np
np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import copy
import json
import _pickle as pickle
import logging
#import transformers.Adafactor as adafact

from tqdm import tqdm
from constants import Constants as const
from lib.supervised.config import Config
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
from torch.utils.data import DataLoader
from logger_config import get_logger, setup_logging
from AGFeatures import AGFeatures, cuda_collate_fn

from lib.supervised.biased.dsgdetr.track import get_sequence
from lib.supervised.biased.dsgdetr.matcher import HungarianMatcher

"""------------------------------------some settings----------------------------------------"""
conf = Config()
print('The CKPT saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
for i in conf.args:
    print(i,':', conf.args[i])
"""-----------------------------------------------------------------------------------------"""

AG_dataset_train = AGFeatures(
    mode=conf.mode,
    data_split=const.TRAIN,
    data_path=conf.data_path,
    is_compiled_together=True,
    filter_nonperson_box_frame=True,
    filter_small_box=False if conf.mode == const.PREDCLS else True
)
                                            
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AGFeatures(
    mode=conf.mode,
    data_split=const.TEST,
    data_path=conf.data_path,
    is_compiled_together=True,
    filter_nonperson_box_frame=True,
    filter_small_box=False if conf.mode == const.PREDCLS else True
)

dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")
max_window = conf.max_window
ode_ratio = conf.ode_ratio
bbox_ratio = conf.bbox_ratio

ode = ODE(mode=conf.mode,
            attention_class_num=len(AG_dataset_train.attention_relationships),
            spatial_class_num=len(AG_dataset_train.spatial_relationships),
            contact_class_num=len(AG_dataset_train.contacting_relationships),
            obj_classes=AG_dataset_train.object_classes,
            enc_layer_num=conf.enc_layer,
            dec_layer_num=conf.dec_layer, 
            max_window=max_window).to(device=gpu_device)
if conf.ckpt:
    ckpt = torch.load(conf.ckpt, map_location=gpu_device)
    ode.load_state_dict(ckpt['state_dict'], strict=False)

#cttran = cttran(d_model = 1024, num_layers = 3)

#model = combined_model(sttran,cttran)

evaluator = BasicSceneGraphEvaluator(mode=conf.mode,
                                    AG_object_classes=AG_dataset_train.object_classes,
                                    AG_all_predicates=AG_dataset_train.relationship_classes,
                                    AG_attention_predicates=AG_dataset_train.attention_relationships,
                                    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
                                    iou_threshold=0.5,
                                    save_file = os.path.join(conf.save_path, "progress.txt"),
                                    constraint='with')

# loss function, default Multi-label margin loss
if conf.bce_loss:
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
else:
    ce_loss = nn.CrossEntropyLoss()
    mlm_loss = nn.MultiLabelMarginLoss()

bbox_loss = nn.SmoothL1Loss()
abs_loss = nn.L1Loss()

# optimizer
if conf.optimizer == 'adamw':
    optimizer = AdamW(ode.parameters(), lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(ode.parameters(), lr=conf.lr)
elif conf.optimizer == 'adafact':
    optimizer = adafact(ode.parameters(),lr = conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(ode.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)
# some parameters
tr = []
matcher= HungarianMatcher(0.5,1,1,0.5)
matcher.eval()
for epoch in range(10):
    ode.train()
    #cttran.train()
    start = time.time()
    for entry in tqdm(dataloader_train):
        gt_annotation = entry[const.GT_ANNOTATION]
        frame_size = entry[const.FRAME_SIZE]
        get_sequence(entry, gt_annotation, matcher, frame_size, conf.mode)
        #pred = cttran(sttran(entry))
        pred = ode(entry)
        global_output = pred["global_output"]
        attention_distribution = pred["attention_distribution"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contacting_distribution"]
        subject_boxes_gt = pred["subject_boxes_gt"]
        object_boxes_gt = pred["object_boxes_gt"]
        subject_boxes_dsg = pred["subject_boxes"]
        object_boxes_dsg = pred["object_boxes"]

        anticipated_global_output = pred["anticipated_vals"]
        anticipated_attention_distribution = pred["anticipated_attention_distribution"]
        anticipated_spatial_distribution = pred["anticipated_spatial_distribution"]
        anticipated_contact_distribution = pred["anticipated_contact_distribution"]
        anticipated_subject_boxes = pred["anticipated_subject_boxes"]
        anticipated_object_boxes = pred["anticipated_object_boxes"]

        attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=attention_distribution.device).squeeze()
        if not conf.bce_loss:
            # multi-label margin loss or adaptive loss
            spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=attention_distribution.device)
            contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])

        else:
            # bce loss
            spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=attention_distribution.device)
            contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1
                contact_label[i, pred["contacting_gt"][i]] = 1

        vid_no = gt_annotation[0][0]["frame"].split('.')[0]
        #pickle.dump(pred,open('/home/cse/msr/csy227518/Dsg_masked_output/sgdet/train'+'/'+vid_no+'.pkl','wb'))

        losses = {}
        if conf.mode == 'sgcls' or conf.mode == 'sgdet':
            losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

        losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
        losses["subject_boxes_loss"] = bbox_ratio * bbox_loss(subject_boxes_dsg, subject_boxes_gt)
        losses["object_boxes_loss"] = bbox_ratio * bbox_loss(object_boxes_dsg, object_boxes_gt)
        losses["anticipated_spatial_relation_loss"] = 0
        losses["anticipated_contact_relation_loss"] = 0
        losses["anticipated_attention_relation_loss"] = 0
        if not conf.bce_loss:
            losses["spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = mlm_loss(contact_distribution, contact_label)
            for i in range(1, max_window + 1):
                mask_curr = entry["mask_curr_" + str(i)] 
                mask_gt = entry["mask_gt_" + str(i)]   
                losses["anticipated_latent_loss"] += ode_ratio * abs_loss(anticipated_global_output[i - 1][mask_curr], global_output[mask_gt])
                losses["anticipated_spatial_relation_loss"] += mlm_loss(anticipated_spatial_distribution[i - 1][mask_curr], spatial_label[mask_gt])
                losses["anticipated_contact_relation_loss"] += mlm_loss(anticipated_contact_distribution[i - 1][mask_curr], contact_label[mask_gt])
                losses["anticipated_attention_relation_loss"] += ce_loss(anticipated_attention_distribution[i - 1][mask_curr], attention_label[mask_gt])
                losses["anticipated_subject_boxes_loss"] += bbox_ratio * bbox_loss(anticipated_subject_boxes[i - 1][mask_curr], subject_boxes_gt[mask_gt])
                losses["anticipated_object_boxes_loss"] += bbox_ratio * bbox_loss(anticipated_object_boxes[i - 1][mask_curr], object_boxes_gt[mask_gt])
        else:
            losses["spatial_relation_loss"] = bce_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = bce_loss(contact_distribution, contact_label)
            for i in range(1, max_window + 1):
                mask_curr = entry["mask_curr_" + str(i)] 
                mask_gt = entry["mask_gt_" + str(i)]   
                losses["anticipated_latent_loss"] += ode_ratio * abs_loss(anticipated_global_output[i - 1][mask_curr], global_output[mask_gt])
                losses["anticipated_spatial_relation_loss"] += bce_loss(anticipated_spatial_distribution[i - 1][mask_curr], spatial_label[mask_gt])
                losses["anticipated_contact_relation_loss"] += bce_loss(anticipated_contact_distribution[i - 1][mask_curr], contact_label[mask_gt])
                losses["anticipated_attention_relation_loss"] += ce_loss(anticipated_attention_distribution[i - 1][mask_curr], attention_label[mask_gt])
                losses["anticipated_subject_boxes_loss"] += bbox_ratio * bbox_loss(anticipated_subject_boxes[i - 1][mask_curr], subject_boxes_gt[mask_gt])
                losses["anticipated_object_boxes_loss"] += bbox_ratio * bbox_loss(anticipated_object_boxes[i - 1][mask_curr], object_boxes_gt[mask_gt])
        #optimizer.zero_grad()
        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(cttran.parameters(), max_norm=5, norm_type=2)
        torch.nn.utils.clip_grad_norm_(ode.parameters(), max_norm=5, norm_type=2)
        #optimizer.step()
        optimizer.step()
        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

        if b % 1000 == 0 and b >= 1000:
            time_per_batch = (time.time() - start) / 1000
            print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60))

            mn = pd.concat(tr[-1000:], axis=1).mean(1)
            print(mn)
            start = time.time()

    torch.save({"ode_state_dict": ode.state_dict()}, os.path.join(conf.save_path, "ode_{}.tar".format(epoch)))
    #torch.save({"cttran_state_dict": cttran.state_dict()}, os.path.join(conf.save_path, "cttran_{}.tar".format(epoch)))
    #torch.save({"model_state_dict": model.state_dict()}, os.path.join(conf.save_path, "model_{}.tar".format(epoch)))
    print("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch))
    with open(evaluator.save_file, "a") as f:
        f.write("save the checkpoint after {} epochs\n".format(epoch))
    ode.eval()
    #cttran.eval()
    object_detector.is_train = False
    """with torch.no_grad():
        for entry in tqdm(dataloader_test):
            gt_annotation = entry[const.GT_ANNOTATION]
            frame_size = entry[const.FRAME_SIZE]
            get_sequence(entry, gt_annotation, matcher, frame_size, conf.mode)
            #pred = cttran(sttran(entry))
            pred = ode(entry)
            vid_no = gt_annotation[0][0]["frame"].split('.')[0]
            #pickle.dump(pred,open('/home/cse/msr/csy227518/Dsg_masked_output/sgdet/test'+'/'+vid_no+'.pkl','wb'))
            for i in range(max_window):
                evaluator.evaluate_scene_graph(gt_annotation[], pred)
        print('-----------')
        sys.stdout.flush()
    score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    evaluator.print_stats()
    evaluator.reset_result()
    #scheduler.step(score)
    scheduler.step(score)"""


# python train_try.py -mode sgcls -ckpt /home/cse/msr/csy227518/scratch/DSG/DSG-DETR/sgcls/model_9.tar -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/




""" python train_DSG_masked.py -mode sgdet -save_path sgdet_masked/  -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """
""" python train_cttran.py -mode sgdet -save_path cttran/1_temporal/ -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ -bce_loss """
