import os
#import sys
import pandas as pd
import pickle
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
import time
import json
import logging
from dataloader import *
from model2 import *
from loss import *
from lib.object_detector import Detector
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW

DATA_PATH = "../../all_frames_final"

#############################################################################
############################# logging #######################################

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

log_file = 'val_log.json'
file_handler = logging.FileHandler(log_file, mode='w')
formatter = logging.Formatter('%(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

###################################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataloader = AG_dataloader(data = DATA_PATH, data_split = "train")
val_dataloader = AG_dataloader(data = DATA_PATH, data_split = "val")

model = SGG(max_len = 200)
# ckpt = torch.load('/home/cse/msr/csy227518/scratch/Project/SG_sota_code/sota_eval/save_model/model_mlm_1.tar')
# model.load_state_dict(ckpt['state_dict'], strict=False)
model = model.to(device)
model.to(device)

LR = 1e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS_PCT = 0.02
DECAY_STEPS_PCT = 0.2
MAX_EPOCHS = 10
SCHEDULER_GAMMA = 0.4
NUM_SANITY_VAL_STEPS = 1

# optimizer = optim.Adam( model.parameters(),
#                         lr = LR, 
#                         weight_decay=WEIGHT_DECAY, 
#                         amsgrad=True)

optimizer = AdamW(model.parameters(), lr=LR)

#total_steps = MAX_EPOCHS * len(train_dataloader)+len(val_dataloader)
total_steps = MAX_EPOCHS * len(train_dataloader)
ff = len(train_dataloader)
def warm_and_decay_lr_scheduler(step: int):
    warmup_steps = WARMUP_STEPS_PCT * total_steps
    decay_steps = DECAY_STEPS_PCT * total_steps
    assert step < total_steps
    if step < warmup_steps:
        factor = step / warmup_steps
    else:
        factor = 1
        
    factor *= SCHEDULER_GAMMA ** (step / decay_steps)
    factor = SCHEDULER_GAMMA ** (step/ff)
    factor = 1
    # print(step, ff, factor, lr*factor)
    return factor
object_class = ['__background__', 'person', 'bag', 'bed', 'blanket', 'book',
                 'box', 'broom', 'chair', 'closet/cabinet', 'clothes', 
                 'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway', 
                 'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 
                 'paper/notebook', 'phone/camera', 'picture', 'pillow', 'refrigerator', 
                 'sandwich', 'shelf', 'shoe', 'sofa/couch', 'table', 'television', 'towel', 
                 'vacuum', 'window']

relationship_classes = ['looking_at', 'not_looking_at', 'unsure', 'above', 
                    'beneath', 'in_front_of', 'behind', 'on_the_side_of', 
                    'in', 'carrying', 'covered_by', 'drinking_from', 'eating', 
                    'have_it_on_the_back', 'holding', 'leaning_on', 'lying_on', 
                    'not_contacting', 'other_relationship', 'sitting_on', 
                    'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on']

attention_relationships = ['looking_at', 'not_looking_at', 'unsure']

spatial_relationships = ['above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in']

contacting_relationships = ['carrying', 'covered_by', 'drinking_from', 'eating', 
                            'have_it_on_the_back', 'holding', 'leaning_on', 
                            'lying_on', 'not_contacting', 'other_relationship', 
                            'sitting_on', 'standing_on', 'touching', 'twisting', 
                            'wearing', 'wiping', 'writing_on']

evaluator = BasicSceneGraphEvaluator(mode='sgdet',
                                    AG_object_classes=object_class,
                                    AG_all_predicates=relationship_classes,
                                    AG_attention_predicates=attention_relationships,
                                    AG_spatial_predicates=spatial_relationships,
                                    AG_contacting_predicates=contacting_relationships,
                                    iou_threshold=0.5,
                                    save_file = os.path.join('.', "progress.txt"),
                                    constraint='with')
                            
scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

ce_loss = nn.CrossEntropyLoss()
mlm_loss = nn.MultiLabelMarginLoss()
tr=[]
for epoch in range(MAX_EPOCHS+2):
    #training Part 
    start = time.time()
    model.train()
    train_epoch_loss = 0
    train_dataloader = tqdm(train_dataloader)
    b = 0
    for batch in train_dataloader:
        b += 1
        entry = model(batch)

        #relations ground truth
        # vid_no = entry['gt_annotation'][0][0]['frame'][0].split('.')[0]
        # gt_rel = pickle.load(open(GT_RELATION_PATH+vid_no+'.pkl','rb'))

        pred_attention = entry["attention_distribution"]
        pred_spatial = entry["spatial_distribution"]
        pred_contacting = entry["contacting_distribution"]
        pred_attention = pred_attention.squeeze(0)
        pred_spatial = pred_spatial.squeeze(0)
        pred_contacting = pred_contacting.squeeze(0)

        a_gt = torch.tensor(entry["attention_gt"], dtype=torch.long).to(device).squeeze()
        s_gt = -torch.ones([len(entry["spatial_gt"]), 6], dtype=torch.long).to(device)
        c_gt = -torch.ones([len(entry["contacting_gt"]), 17], dtype=torch.long).to(device)
        for i in range(len(entry["spatial_gt"])):
            s_gt[i, : len(entry["spatial_gt"][i])] = torch.tensor(entry["spatial_gt"][i])
            c_gt[i, : len(entry["contacting_gt"][i])] = torch.tensor(entry["contacting_gt"][i])
        # a_gt = gt_rel["attention_gt"]

        # s_gt = gt_rel["spatial_gt"]
        # c_gt = gt_rel["contacting_gt"]

        # a_gt = a_gt.to(device)
        # s_gt = s_gt.to(device)
        # c_gt = c_gt.to(device)

        # pred_attention = entry["attention_distribution"]
        # pred_spatial = entry["spatial_distribution"]
        # pred_contacting = entry["contacting_distribution"]
        # pred_attention = pred_attention.squeeze(0)
        # pred_spatial = pred_spatial.squeeze(0)
        # pred_contacting = pred_contacting.squeeze(0)
        
        
        losses = {}
        
        losses["attention_relation_loss"] = ce_loss(pred_attention, a_gt)
        losses["spatial_relation_loss"] = mlm_loss(pred_spatial, s_gt)
        losses["contacting_relation_loss"] = mlm_loss(pred_contacting, c_gt)

        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        train_epoch_loss += loss.item()
        optimizer.step()
        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
        if b%1000==0 and b>=1000:
            mn = pd.concat(tr[-1000:], axis=1).mean(1)
            print(mn)

    # scheduler.step()
    end = time.time()
    train_time  = (end-start)/60
    train_loss = train_epoch_loss/len(train_dataloader)
    #print(f" Epoch : {epoch+1} || Train Loss : {train_loss} || Time : {train_time//60} mins")
    torch.save({"state_dict": model.state_dict()}, os.path.join('save_model/', "model_mlm_{}.tar".format(epoch)))
    print("*" * 40)
    print("save the checkpoint after {} epochs".format(epoch))

    """########### Validation Part ############"""

    model.eval()
    val_epoch_loss = 0

    start = time.time()
    outputs = []
    with torch.no_grad():
        val_dataloader = tqdm(val_dataloader)
        for batch in val_dataloader:
            entry = model(batch)
            entry["attention_distribution"] = entry["attention_distribution"].squeeze(0)
            entry["spatial_distribution"] = entry["spatial_distribution"].squeeze(0)
            entry["contacting_distribution"] = entry["contacting_distribution"].squeeze(0)
            entry["im_idx"] = entry["im_idx"].squeeze(0)
            entry["pair_idx"] = entry["pair_idx"].squeeze(0)
            entry["boxes"] = entry["boxes"].squeeze(0)
            entry["labels"] = entry["labels"].squeeze(0)
            entry["scores"] = entry["scores"].squeeze(0)
            entry["distribution"] = entry["distribution"].squeeze(0)
            entry["pred_labels"] = entry["pred_labels"].squeeze(0)
            entry["features"] = entry["features"].squeeze(0)
            entry["fmaps"] = entry["fmaps"].squeeze(0)
            entry["im_info"] = entry["im_info"].squeeze(0)
            entry["pred_scores"] = entry["pred_scores"].squeeze(0)
            entry["human_idx"] = entry["human_idx"].squeeze(0)
            entry["union_feat"] = entry["union_feat"].squeeze(0)
            entry["union_box"] = entry["union_box"].squeeze(0)
            entry["spatial_masks"] = entry["spatial_masks"].squeeze(0)
            gt_annotation = entry['gt_annotation']
            #outputs.append(loss_dict)
            evaluator.evaluate_scene_graph(gt_annotation,entry)
    #score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    end = time.time()
    val_time = (end-start)/60
    score = np.mean(evaluator.result_dict["sgdet" + "_recall"][20])
    evaluator.print_stats()
    evaluator.reset_result()
    scheduler.step(score)
    print(f"Validation Time : {val_time} mins")
    #print(f" Epoch : {epoch+1} || Validation Loss : {val_loss} || Accuracy : {100*accuracy} || Time : {val_time//60} mins")

"""(['boxes', 'labels', 'scores', 'distribution', 'pred_labels', 'features', 'fmaps', 'im_info', 'indices', 'pred_scores', 'pair_idx', 'im_idx', 'human_idx', 'union_feat', 'union_box', 'spatial_masks', 'attention_distribution', 'spatial_distribution', 'contacting_distribution', 'global_output', 'gt_annotation'])"""
