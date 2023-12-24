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
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW
#from lib.object_mask import STTran
from lib.object_predictor import STTran
from lib.track import get_sequence
from lib.matcher import *
import torch.nn.functional as F
import pdb

CONTEXT = 4
FUTURE = 1
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

object_class = ['__background__', 'human','bag', 'bed', 'blanket', 'book',
                 'box', 'broom', 'chair', 'closet/cabinet', 'clothes', 
                 'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway', 
                 'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 
                 'paper/notebook', 'phone/camera', 'picture', 'pillow', 'refrigerator', 
                 'sandwich', 'shelf', 'shoe', 'sofa/couch', 'table', 'television', 'towel', 
                 'vacuum', 'window']

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

# loss function, default Multi-lpython train_DSG_masked.py -mode sgdet  -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/abel margin loss


evaluator = BasicSceneGraphEvaluator(mode=conf.mode,
                                    AG_object_classes=AG_dataset_train.object_classes,
                                    AG_all_predicates=AG_dataset_train.relationship_classes,
                                    AG_attention_predicates=AG_dataset_train.attention_relationships,
                                    AG_spatial_predicates=AG_dataset_train.spatial_relationships,
                                    AG_contacting_predicates=AG_dataset_train.contacting_relationships,
                                    iou_threshold=0,
                                    save_file = os.path.join(conf.save_path, "progress.txt"),
                                    constraint='with')


l1_loss = nn.SmoothL1Loss()
mse_loss = nn.MSELoss()

if conf.bce_loss:
    ce_loss = nn.CrossEntropyLoss()
    bce_pure_loss = nn.BCELoss()
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

tr = []
matcher= HungarianMatcher(0.5,1,1,0.5)
matcher.eval()
for epoch in range(0,10):
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
        dec_out,bbox_out,pred,vr,vr_out = model(entry,CONTEXT,FUTURE,epoch)
        # dec_out,pred,vr,vr_out = model(entry,CONTEXT,FUTURE,epoch)

        start =0 
        context = CONTEXT
        future =FUTURE
        count = 0
        losses = {}

        losses["object_pred_loss"] = 0
        losses["boxes"] = 0
        

        if (start+context+1>len(entry["im_idx"].unique())):
            while(start+context+1 != len(entry["im_idx"].unique()) and context >1):
                context -= 1

        while (start+context+1 <= len(entry["im_idx"].unique())):

            future_frame_start_id = entry["im_idx"].unique()[context]


            future_frame_end_id = entry["im_idx"].unique()[context+future-1]

            context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
            context_idx = entry["im_idx"][:context_end_idx]
            context_len = context_idx.shape[0]

            future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1])+1
            future_idx = entry["im_idx"][context_end_idx:future_end_idx]
            future_len = future_idx.shape[0]

            gt_labels = list(entry["labels"][entry["pair_idx"][:,1][context_end_idx:context_end_idx+future_len]])
            gt_labels.append(entry["labels"][entry["pair_idx"][:,0][context_end_idx:context_end_idx+future_len]].unique())
            labels = torch.zeros(37,1)
            
            boxes_ind = torch.where(entry["boxes"][:,0]==context)[0]
            gt_ind = torch.where(entry["labels"][boxes_ind]!=0)[0]
            gt_ind = gt_ind+boxes_ind[0]

            boxes_gt = entry["boxes"][torch.where(entry["boxes"][gt_ind][:,0]==context)[0]][:,1:]
            out_boxes =[]
            
            for i in gt_ind:
                # out_boxes.append(bbox_out[count]["box"][entry["labels"][i]])
                out_boxes.append(dec_out[count]["box"][entry["labels"][i]])

            out_boxes = [list(x) for x in out_boxes]
            out_boxes = torch.tensor(out_boxes)
            out_boxes = out_boxes.cuda()
            boxes_gt = boxes_gt.cuda()

            for idx in gt_labels:
                labels[idx] = 1
            
            labels = labels.cuda()
            weight = torch.ones(labels.shape).cuda()
            weight[labels==1]=7.0
            vid_no = gt_annotation[0][0]["frame"].split('.')[0]
            losses["boxes"] += l1_loss(out_boxes,boxes_gt)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight = weight)
            losses["object_pred_loss"] += bce_loss(dec_out[count]["object"],labels)

            context += 1
            count += 1

            if(start+context+future > len(entry["im_idx"].unique())):
                break

        attention_distribution = pred["attention_distribution"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contacting_distribution"]

        cont = 4
        if (start+cont+1>len(entry["im_idx"].unique())):
            while(start+cont+1 != len(entry["im_idx"].unique()) and cont >1):
                cont -= 1
        ind = torch.where(pred["im_idx"]==cont)[0][0]

        attention_label = torch.tensor(pred["attention_gt"][ind:], dtype=torch.long).to(device=attention_distribution.device).squeeze()
        if not conf.bce_loss:
            # multi-label margin loss or adaptive loss
            spatial_label = -torch.ones([len(pred["spatial_gt"][ind:]), 6], dtype=torch.long).to(device=attention_distribution.device)
            contact_label = -torch.ones([len(pred["contacting_gt"][ind:]), 17], dtype=torch.long).to(device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"][ind:])):
                spatial_label[i, : len(pred["spatial_gt"][ind:][i])] = torch.tensor(pred["spatial_gt"][ind:][i])
                contact_label[i, : len(pred["contacting_gt"][ind:][i])] = torch.tensor(pred["contacting_gt"][ind:][i])

        else:
            # bce loss
            spatial_label = torch.zeros([len(pred["spatial_gt"][ind:]), 6], dtype=torch.float32).to(device=attention_distribution.device)
            contact_label = torch.zeros([len(pred["contacting_gt"][ind:]), 17], dtype=torch.float32).to(device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"][ind:])):
                spatial_label[i, pred["spatial_gt"][ind:][i]] = 1
                contact_label[i, pred["contacting_gt"][ind:][i]] = 1

        if conf.mode == 'sgcls' or conf.mode == 'sgdet':
            losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])
        try:
            losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
        except ValueError:
            attention_label = attention_label.unsqueeze(0)
            losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
        if not conf.bce_loss:
            losses["spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = mlm_loss(contact_distribution, contact_label)

        else:
            losses["spatial_relation_loss"] = bce_pure_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = bce_pure_loss(contact_distribution, contact_label)

        losses["vr_loss"] = mse_loss(vr_out,vr)
    
        optimizer.zero_grad()
        losses["boxes"] = losses["boxes"]/count
        losses["object_pred_loss"] /= count
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()

        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

        if b % 1000 == 0 and b >= 1000:
            time_per_batch = (time.time() - start_time) / 1000
            print("\ne{:2d}encoder_tran  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b, len(dataloader_train),
                                                                                time_per_batch, len(dataloader_train) * time_per_batch / 60))

            mn = pd.concat(tr[-1000:], axis=1).mean(1)
            print(mn)
            start_time = time.time()

    # torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "obj_predictor_with_bbox_new_{}.tar".format(epoch)))
    torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "obj_predictor_new_sub_obj_{}.tar".format(epoch)))

    model.eval()
    object_detector.is_train = False
    with torch.no_grad():
        accuracy = 0
        for b in range(len(dataloader_test)):
            data = next(test_iter)
            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_test.gt_annotations[data[4]]
            entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            get_sequence(entry, gt_annotation, matcher, (im_info[0][:2]/im_info[0,2]).cpu().data, conf.mode)

            start =0 
            context = CONTEXT
            future = FUTURE
            count = 0
            #entry = model_ST(entry,context,future)
            dec_out,pred = model(entry,context,future,epoch)
            #pdb.set_trace()
        #     acc = 0
            if (start+context+1>len(entry["im_idx"].unique())):
                while(start+context+1 != len(entry["im_idx"].unique()) and context >1):
                    context -= 1

            gt = gt_annotation[context:]
            evaluator.evaluate_scene_graph(gt, pred)
            #evaluator.print_stats()
        print('-----------', flush=True)
    score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    evaluator.print_stats()
    evaluator.reset_result()
    scheduler.step(score)



""" python train_obj_pred.py -mode sgdet -ckpt obj_pred_forecast/sgdet/obj_predictor_new_sub_obj_10.tar -save_path obj_pred_forecast/sgdet/ -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """

""" python train_obj_pred.py -mode sgdet -save_path obj_pred_forecast/sgdet/ -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """
