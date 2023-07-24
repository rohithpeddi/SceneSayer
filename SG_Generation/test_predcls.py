import os
import pandas as pd
import pandas as pd
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
import json
import logging
from dataloader import *
from model2 import *
from loss import *
from lib.object_detector import detector
from lib.config import Config
from unbiasedSGG.lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW

DATA_PATH = "../../frames_predcls"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
val_dataloader = AG_dataloader(data = DATA_PATH, annot_path = '../../predcls_gt_annotations',data_split = "val")

model = SGG(max_len = 101)
ckpt = torch.load('/home/cse/msr/csy227518/scratch/Project/SG_sota_code/sota_eval/model_predcls/model_14_lr4.tar')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.to(device)
print('*'*50)
print('CKPT {} is loaded'.format(('/home/cse/msr/csy227518/scratch/Project/SG_sota_code/sota_eval/model_predcls/model_14_lr4.tar')))

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

evaluator1 = BasicSceneGraphEvaluator(mode='predcls',
                                    AG_object_classes=object_class,
                                    AG_all_predicates=relationship_classes,
                                    AG_attention_predicates=attention_relationships,
                                    AG_spatial_predicates=spatial_relationships,
                                    AG_contacting_predicates=contacting_relationships,
                                    iou_threshold=0.5,
                                    # output_dir = os.path.join('.'),
                                    constraint='with')

evaluator2 = BasicSceneGraphEvaluator(mode='predcls',
                                    AG_object_classes=object_class,
                                    AG_all_predicates=relationship_classes,
                                    AG_attention_predicates=attention_relationships,
                                    AG_spatial_predicates=spatial_relationships,
                                    AG_contacting_predicates=contacting_relationships,
                                    iou_threshold=0.5,
                                    # output_dir = os.path.join('.'),
                                    constraint='semi',semithreshold=0.9)

evaluator3 = BasicSceneGraphEvaluator(mode='predcls',
                                    AG_object_classes=object_class,
                                    AG_all_predicates=relationship_classes,
                                    AG_attention_predicates=attention_relationships,
                                    AG_spatial_predicates=spatial_relationships,
                                    AG_contacting_predicates=contacting_relationships,
                                    iou_threshold=0.5,
                                    # output_dir = os.path.join('.'),
                                    constraint='no')                                   
                            
for epoch in range(1):
    """########### Validation Part ############"""

    model.eval()
    val_epoch_loss = 0

    start = time.time()
    outputs = []
    with torch.no_grad():
        #val_dataloader = tqdm(val_dataloader)
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
            #entry["distribution"] = entry["distribution"].squeeze(0)
            entry["pred_labels"] = entry["pred_labels"].squeeze(0)
            entry["features"] = entry["features"].squeeze(0)
            #entry["fmaps"] = entry["fmaps"].squeeze(0)
            #entry["im_info"] = entry["im_info"].squeeze(0)
            #entry["pred_scores"] = entry["pred_scores"].squeeze(0)
            entry["human_idx"] = entry["human_idx"].squeeze(0)
            entry["union_feat"] = entry["union_feat"].squeeze(0)
            entry["union_box"] = entry["union_box"].squeeze(0)
            entry["spatial_masks"] = entry["spatial_masks"].squeeze(0)
            gt_annotation = entry['gt_annotation']
            #outputs.append(loss_dict)
            evaluator1.evaluate_scene_graph(gt_annotation,entry)
            evaluator2.evaluate_scene_graph(gt_annotation,entry)
            evaluator3.evaluate_scene_graph(gt_annotation,entry)
    print('-------------------------with constraint-------------------------------')
    evaluator1.print_stats()
    print('-------------------------semi constraint-------------------------------')
    evaluator2.print_stats()
    print('-------------------------no constraint-------------------------------')
    evaluator3.print_stats()

"""(['boxes', 'labels', 'scores', 'distribution', 'pred_labels', 'features', 'fmaps', 'im_info', 'indices', 'pred_scores', 'pair_idx', 'im_idx', 'human_idx', 'union_feat', 'union_box', 'spatial_masks', 'attention_distribution', 'spatial_distribution', 'contacting_distribution', 'global_output', 'gt_annotation'])"""