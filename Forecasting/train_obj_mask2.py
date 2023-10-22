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
from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.evaluation_forecast import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
#from lib.object_mask import STTran
from lib.forecasting import STTran
from lib.track import get_sequence
from lib.matcher import *
import pdb


"""------------------------------------some settings----------------------------------------"""
conf = Config()
print('The CKPT saved here:', conf.save_path)
if not os.path.exists(conf.save_path):
    os.mkdir(conf.save_path)
print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
for i in conf.args:
    print(i,':', conf.args[i])
"""-----------------------------------------------------------------------------------------"""

AG_dataset_train = AG(mode="train", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                      filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True, num_workers=4,
                                               collate_fn=cuda_collate_fn, pin_memory=False)
AG_dataset_test = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                     filter_small_box=False if conf.mode == 'predcls' else True)
dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, num_workers=4,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device("cuda:0")
# freeze the detection backbone
object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()

model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset_train.attention_relationships),
               spatial_class_num=len(AG_dataset_train.spatial_relationships),
               contact_class_num=len(AG_dataset_train.contacting_relationships),
               obj_classes=AG_dataset_train.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer).to(device=gpu_device)
if conf.ckpt:
    ckpt = torch.load(conf.ckpt, map_location=gpu_device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
evaluator = BasicSceneGraphEvaluator(mode=conf.mode,
                                    AG_object_classes=AG_dataset_train.object_classes,
                                    AG_all_predicates=AG_dataset_train.relationship_classes,
                                    AG_attention_predicates=AG_dataset_train.attention_relationships,
                                    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
                                    iou_threshold=0.5,
                                    save_file = os.path.join(conf.save_path, "progress.txt"),
                                    constraint='with')


# loss function, default Multi-lpython train_DSG_masked.py -mode sgdet  -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/abel margin loss
if conf.bce_loss:
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
else:
    ce_loss = nn.CrossEntropyLoss()
    mlm_loss = nn.MultiLabelMarginLoss()

# optimizer
if conf.optimizer == 'adamw':
    optimizer = AdamW(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=conf.lr)
elif conf.optimizer == 'adafact':
    optimizer = adafact(model.parameters(),lr = conf.lr)
elif conf.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

# some parameters
tr = []
matcher= HungarianMatcher(0.5,1,1,0.5)
matcher.eval()
for epoch in range(10):
    object_detector.is_train = True
    model.train()
    object_detector.train_x = True
    start_time = time.time()
    train_iter = iter(dataloader_train)
    test_iter = iter(dataloader_test)
    counter = 0
    for b in range(len(dataloader_train)):
        data = next(train_iter)
        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset_train.gt_annotations[data[4]]
    
        # prevent gradients to FasterRCNN
        with torch.no_grad():
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
        get_sequence(entry, gt_annotation, matcher, (im_info[0][:2]/im_info[0,2]).cpu().data, conf.mode)

        pred = model(entry)
        
        start =0 
        start_index = 0
        context = 2
        future = 3
        count = 0
        losses = {}

        if conf.mode == 'sgcls' or conf.mode == 'sgdet':
                losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

        losses["attention_relation_loss"] = 0
        losses["spatial_relation_loss"] = 0
        losses["contact_relation_loss"] = 0

        while (start+context+1 <= len(entry["im_idx"].unique())):

            try:
                end = int(torch.where(entry["im_idx"]==start+context)[0][0])
            except IndexError:
                start = start+1
                end = int(torch.where(entry["im_idx"]==start+context)[0][0])

            if (start+context+future> len(entry["im_idx"].unique())):
                future = len(entry["im_idx"].unique()) - context

            if (start+context+future > len(entry["im_idx"].unique())):
                break

            if (start+context+future == len(entry["im_idx"].unique())):
                future_end = len(entry["im_idx"])
            else:
                try:
                    future_end = int(torch.where(entry["im_idx"]==start+context+future)[0][0])
                except IndexError:
                    start +=1
                    future_end = int(torch.where(entry["im_idx"]==start+context+future)[0][0])

            future_frame_idx = entry["im_idx"][end:future_end]
            future_frame_len = future_frame_idx.shape[0]
                    
            attention_distribution = pred["output"][count]["attention_distribution"]
            spatial_distribution = pred["output"][count]["spatial_distribution"]
            contact_distribution = pred["output"][count]["contacting_distribution"]

            attention_label = torch.tensor(pred["attention_gt"][end:future_end], dtype=torch.long).to(device=attention_distribution.device).squeeze()
            try:    
                losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
            except ValueError:
                attention_label = attention_label.unsqueeze(0)
                losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
            
            if not conf.bce_loss:
            # multi-label margin loss or adaptive loss
                spatial_label = -torch.ones([len(pred["spatial_gt"][end:future_end]), 6], dtype=torch.long).to(device=attention_distribution.device)
                contact_label = -torch.ones([len(pred["contacting_gt"][end:future_end]), 17], dtype=torch.long).to(device=attention_distribution.device)
                for i in range(len(pred["spatial_gt"][end:future_end])):
                    spatial_label[i, : len(pred["spatial_gt"][end:future_end][i])] = torch.tensor(pred["spatial_gt"][end:future_end][i])
                    contact_label[i, : len(pred["contacting_gt"][end:future_end][i])] = torch.tensor(pred["contacting_gt"][end:future_end][i])

            else:
            # bce loss
                spatial_label = torch.zeros([len(pred["spatial_gt"][end:future_end]), 6], dtype=torch.float32).to(device=attention_distribution.device)
                contact_label = torch.zeros([len(pred["contacting_gt"][end:future_end]), 17], dtype=torch.float32).to(device=attention_distribution.device)
                for i in range(len(pred["spatial_gt"][end:future_end])):
                    spatial_label[i, pred["spatial_gt"][end:future_end][i]] = 1
                    contact_label[i, pred["contacting_gt"][end:future_end][i]] = 1

            vid_no = gt_annotation[0][0]["frame"].split('.')[0]

            # losses = {}
            # if conf.mode == 'sgcls' or conf.mode == 'sgdet':
            #     losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])

            losses["attention_relation_loss"] += ce_loss(attention_distribution, attention_label)
            if not conf.bce_loss:
                losses["spatial_relation_loss"] += mlm_loss(spatial_distribution, spatial_label)
                losses["contact_relation_loss"] += mlm_loss(contact_distribution, contact_label)

            else:
                losses["spatial_relation_loss"] += bce_loss(spatial_distribution, spatial_label)
                losses["contact_relation_loss"] += bce_loss(contact_distribution, contact_label)

            # optimizer.zero_grad()
            # loss = sum(losses.values())
            # loss.backward(retain_graph = True)
            # if counter%1000 ==0:
            #     print(f"Loss for forecasting : {loss}")
            
            count += 1
            start += 1
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            # optimizer.step()
            
        losses["attention_relation_loss"] = losses["attention_relation_loss"]/count
        losses["spatial_relation_loss"] = losses["spatial_relation_loss"]/count
        losses["contact_relation_loss"] = losses["contact_relation_loss"]/count
        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()
        # if b%100 ==0:
        #     print("Loss for forecasting : ",loss)
        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

        if b % 1000 == 0 and b >= 1000:
            time_per_batch = (time.time() - start_time) / 1000
            print("\ne{:2d}encoder_tran  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60))

            mn = pd.concat(tr[-1000:], axis=1).mean(1)
            print(mn)
            start_time = time.time()

    torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "DSG_masked_{}.tar".format(epoch)))
    print("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch))
    with open(evaluator.save_file, "a") as f:
        f.write("save the checkpoint after {} epochs\n".format(epoch))
    model.eval()
    object_detector.is_train = False
    with torch.no_grad():
        for b in range(len(dataloader_test)):
            data = next(test_iter)
            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_test.gt_annotations[data[4]]
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            get_sequence(entry, gt_annotation, matcher, (im_info[0][:2]/im_info[0,2]).cpu().data, conf.mode)
            pred = model(entry)
            count = 0
            start = 0
            context = 2
            future = 3
            while (start+context+1 <= len(entry["im_idx"].unique())):
                try:
                    end = int(torch.where(entry["im_idx"]==start+context)[0][0])
                except IndexError:
                    start = start+1
                    end = int(torch.where(entry["im_idx"]==start+context)[0][0])
                if (start+context+future> len(entry["im_idx"].unique())):
                    future = len(entry["im_idx"].unique()) - context

                if (start+context+future > len(entry["im_idx"].unique())):
                    break

                if (start+context+future == len(entry["im_idx"].unique())):
                    future_end = len(entry["im_idx"])
                else:
                    try:
                        future_end = int(torch.where(entry["im_idx"]==start+context+future)[0][0])
                    except IndexError:
                        start +=1
                        future_end = int(torch.where(entry["im_idx"]==start+context+future)[0][0])
                future_frame_idx = entry["im_idx"][end:future_end]
                future_frame_len = future_frame_idx.shape[0]
                
                gt_future = gt_annotation[start+context:start+context+future]

                vid_no = gt_annotation[0][0]["frame"].split('.')[0]
                #pickle.dump(pred,open('/home/cse/msr/csy227518/Dsg_masked_output/sgdet/test'+'/'+vid_no+'.pkl','wb'))
                evaluator.evaluate_scene_graph(gt_future, pred,end,future_end,future_frame_idx,count)
                #evaluator.print_stats()
                count += 1
                start += 1
        print('-----------', flush=True)
    score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    evaluator.print_stats()
    evaluator.reset_result()
    scheduler.step(score)


# python train_try.py -mode sgcls -ckpt /home/cse/msr/csy227518/scratch/DSG/DSG-DETR/sgcls/model_9.tar -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/




""" python train_obj_mask2.py -mode sgdet  -save_path forecasting/sgdet/ -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """

