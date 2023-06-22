import os
#import sys
import pickle
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F 
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
from lib.object_detector import detector
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator

DATA_PATH = "../../all_frames_full"
GT_RELATION_PATH = '/home/cse/msr/csy227518/scratch/Project/gt_relations/train/'

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

model = SGG(max_len = 101)
model.to(device)

LR = 1e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS_PCT = 0.02
DECAY_STEPS_PCT = 0.2
MAX_EPOCHS = 200
SCHEDULER_GAMMA = 0.4
NUM_SANITY_VAL_STEPS = 1

optimizer = optim.Adam( model.parameters(),
                        lr = LR, 
                        weight_decay=WEIGHT_DECAY, 
                        amsgrad=True)

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
                            
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)

ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()

for epoch in range(MAX_EPOCHS):
    #training Part 
    start = time.time()
    model.train()
    train_epoch_loss = 0
    train_dataloader = tqdm(train_dataloader)
    for batch in train_dataloader:
        entry = model(batch)

        #relations ground truth
        vid_no = entry['gt_annotation'][0][0]['frame'][0].split('.')[0]
        gt_rel = pickle.load(open(GT_RELATION_PATH+vid_no+'.pkl','rb'))
        a_gt = gt_rel["a_gt"]

        s_gt = gt_rel["s_gt"]
        c_gt = gt_rel["c_gt"]

        a_gt = a_gt.to(device)
        s_gt = s_gt.to(device)
        c_gt = c_gt.to(device)

        pred_attention = entry["attention_distribution"]
        
        print("global",entry["global_output"].shape)
        print(pred_attention.shape)
        pred_spatial = entry["spatial_distribution"]
        pred_contacting = entry["contacting_distribution"]

        pred_attention = pred_attention.squeeze(0)
        
        print(a_gt.shape)
        
        losses = {}
        losses["attention_relation_loss"] = ce_loss(pred_attention, a_gt)
        losses["spatial_relation_loss"] = bce_loss(pred_spatial, s_gt)
        losses["contacting_relation_loss"] = bce_loss(pred_contacting, c_gt)

        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        train_epoch_loss += loss.item()
        optimizer.step()

    scheduler.step()
    end = time.time()
    train_time  = (end-start)
    train_loss = train_epoch_loss/len(train_dataloader)

    # # Validation Part

    # model.eval()
    # val_epoch_loss = 0
    # val_epoch_loss1 = 0
    # val_epoch_loss2 = 0
    # start = time.time()
    # # tp = 0
    # # fp = 0
    # # fn = 0
    # # tn = 0
    # # tp10_ = 0
    # # tp20_ = 0
    # # tp50_ = 0
    # # tpfn_ = 0
    # # tp10_with = 0
    # # tp20_with = 0
    # # tp50_with = 0
    # outputs = []
    # with torch.no_grad():
    #     val_dataloader = tqdm(val_dataloader)
    #     for batch in train_dataloader:
    #         future_graphs, gt_dense_sg, future_final_graphs, gt_actual_sg ,entry= model(batch,'/home/cse/msr/csy227518/scratch/Project/gt_relations/train/')
    #         loss_dict = sg_loss(future_graphs, gt_dense_sg, future_final_graphs, gt_actual_sg)
    #         loss = loss_dict["loss"]
    #         loss1 = loss_dict["loss1"]
    #         loss2 = loss_dict["loss2"]
    #         # tp += loss_dict["tp"].item()
    #         # fp += loss_dict["fp"].item()
    #         # fn += loss_dict["fn"].item()
    #         # tn += loss_dict["tn"].item()
    #         # tp10_ += loss_dict["tp10_"].item()
    #         # tp20_ += loss_dict["tp20_"].item()
    #         # tp50_ += loss_dict["tp50_"].item()
    #         # tpfn_ += loss_dict["tpfn_"].item() 
    #         # tp10_with += loss_dict["tp10_with"].item()
    #         # tp20_with += loss_dict["tp20_with"].item()
    #         # tp50_with += loss_dict["tp50_with"].item()
    #         gt_annotation = entry['gt_annotation']
    #         outputs.append(loss_dict)
    #         val_epoch_loss += loss.item()
    #         val_epoch_loss1 += loss1.item()
    #         val_epoch_loss2 += loss2.item()
    #         evaluator.evaluate_scene_graph(gt_annotation,entry)
    # #score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
    # evaluator.print_stats()
    # evaluator.reset_result()
    print(f" Epoch : {epoch+1} || Train Loss : {train_loss} || Time : {train_time//60} mins")
    #print(f" Epoch : {epoch+1} || Validation Loss : {val_loss} || Accuracy : {100*accuracy} || Time : {val_time//60} mins")

