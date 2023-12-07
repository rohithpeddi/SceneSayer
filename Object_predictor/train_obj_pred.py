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
from lib.object_predictor import STTran
from lib.object_pred_test  import STTran as ST
from lib.track import get_sequence
from lib.matcher import *
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

object_class = ['__background__', 'bag', 'bed', 'blanket', 'book',
                 'box', 'broom', 'chair', 'closet/cabinet', 'clothes', 
                 'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway', 
                 'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 
                 'paper/notebook', 'phone/camera', 'picture', 'pillow', 'refrigerator', 
                 'sandwich', 'shelf', 'shoe', 'sofa/couch', 'table', 'television', 'towel', 
                 'vacuum', 'window']

model = STTran()
if conf.ckpt:
    ckpt = torch.load(conf.ckpt, map_location=gpu_device)
    model.load_state_dict(ckpt['state_dict'], strict=False)

matcher= HungarianMatcher(0.5, 1, 1, 0.5)
matcher.eval()

# loss function, default Multi-lpython train_DSG_masked.py -mode sgdet  -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/abel margin loss
model_ST = ST(mode=conf.mode,
               attention_class_num=len(AG_dataset_train.attention_relationships),
               spatial_class_num=len(AG_dataset_train.spatial_relationships),
               contact_class_num=len(AG_dataset_train.contacting_relationships),
               obj_classes=AG_dataset_train.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer).to(device=gpu_device)

model_ST.eval()

bce_loss = nn.BCEWithLogitsLoss()

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
        #get_sequence(entry, gt_annotation, matcher, (im_info[0][:2]/im_info[0,2]).cpu().data, conf.mode)

        pred = model(entry,CONTEXT,FUTURE)

        start =0 
        context = CONTEXT
        future =1
        count = 0
        losses = {}

        losses["object_pred_loss"] = 0


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

            gt_labels = entry["labels"][entry["pair_idx"][:,1][context_end_idx:context_end_idx+future_len]]

            labels = torch.zeros(36,1)

            for idx in gt_labels:
                if idx!=1:
                    labels[idx-1] = 1
            
            labels = labels.cuda()

            vid_no = gt_annotation[0][0]["frame"].split('.')[0]
            losses["object_pred_loss"] += bce_loss(pred[count],labels)

            context += 1
            count += 1

            if(start+context+future > len(entry["im_idx"].unique())):
                break

        optimizer.zero_grad()
        loss = sum(losses.values())/b
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

    torch.save({"state_dict": model.state_dict()}, os.path.join(conf.save_path, "forecast_{}.tar".format(epoch)))

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
            future = 1
            count = 0
            entry = model_ST(entry,context,future)
            pred = model(entry,context,future)
            acc = 0
            if (start+context+1>len(entry["im_idx"].unique())):
                while(start+context+1 != len(entry["im_idx"].unique()) and context >1):
                    context -= 1


            while (start+context+1 <= len(entry["im_idx"].unique())):
                correct = 0
                future_frame_start_id = entry["im_idx"].unique()[context]
                

                future_frame_end_id = entry["im_idx"].unique()[context+future-1]

                context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
                context_idx = entry["im_idx"][:context_end_idx]
                context_len = context_idx.shape[0]

                future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1])+1
                future_idx = entry["im_idx"][context_end_idx:future_end_idx]
                future_len = future_idx.shape[0]
                
                gt_labels = entry["labels"][entry["pair_idx"][:,1][context_end_idx:context_end_idx+future_len]]
                labels = torch.zeros(36,1)

                for idx in gt_labels:
                    if idx>=1:
                        labels[idx-1] = 1
                    else:
                        labels[idx] = 1                
                labels = labels.cuda() 

                out = torch.sigmoid(pred[count]) 
                for i,x in enumerate(out):
                    if x>=0.5:
                        out[i]=1
                    else:
                        out[i]=0
                for i,x in enumerate(out):  
                    if out[i]==labels[i][0]:
                        correct+=1

                correct /=36

                acc += correct
                pdb.set_trace() 
                count += 1
                context += 1

                if(start+context+future > len(entry["im_idx"].unique())):
                    break
            acc=acc/count
            accuracy += acc
            if b%200==0:
                print(f"accuracy is : {accuracy/b}")
        print('-----------', flush=True)

        print(f"Final accuracy is : {accuracy/1737}")


# python train_try.py -mode sgcls -ckpt /home/cse/msr/csy227518/scratch/DSG/DSG-DETR/sgcls/model_9.tar -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/




""" python train_obj_pred.py -mode sgcls -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """

