import os
# import sys
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

from dataloader.anticipation.action_genome.dataloader import AG_dataloader
from methods.sttran.cttran import SGA

# from dataloader.dataloader import *
# from dataloader.dataset import *
# from model.cttran import *

# from loss import *
# from lib.object_detector import detector
# from lib.config import Config
# from lib.evaluation_recall import BasicSceneGraphEvaluator
# from lib.AdamW import AdamW

# DATA_PATH = "../../frames_predcls"
# GT_RELATION_PATH = '/home/cse/msr/csy227518/scratch/Project/gt_relations_final/train/'


CONTEXT_LEN = 3
FUTURE_LEN = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataloader = AG_dataloader()
val_dataloader = AG_dataloader(data_split="val")

model = SGA(context_len=CONTEXT_LEN,
            future_len=FUTURE_LEN,
            d_model=512
            )

# ckpt = torch.load('/home/cse/msr/csy227518/scratch/Project/SG_sota_code/sota_eval/save_model/model_9_lr4.tar')
# model.load_state_dict(ckpt['state_dict'], strict=False)
model.to(device)

LR = 1e-4
WEIGHT_DECAY = 0.0
WARMUP_STEPS_PCT = 0.02
DECAY_STEPS_PCT = 0.2
MAX_EPOCHS = 10
SCHEDULER_GAMMA = 0.4
NUM_SANITY_VAL_STEPS = 1

optimizer = optim.Adam(model.parameters(),
                       lr=LR,
                       weight_decay=WEIGHT_DECAY,
                       amsgrad=True)

# optimizer = AdamW(model.parameters(), lr=2.5*1e-5)

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

# evaluator = BasicSceneGraphEvaluator(mode='predcls',
#                                     AG_object_classes=object_class,
#                                     AG_all_predicates=relationship_classes,
#                                     AG_attention_predicates=attention_relationships,
#                                     AG_spatial_predicates=spatial_relationships,
#                                     AG_contacting_predicates=contacting_relationships,
#                                     iou_threshold=0.5,
#                                     save_file = os.path.join('.', "progress.txt"),
#                                     constraint='with')

# scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=warm_and_decay_lr_scheduler)
scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4,
                              threshold_mode="abs", min_lr=1e-7)

ce_loss = nn.CrossEntropyLoss()
bce_loss = nn.BCELoss()
mse_loss = nn.MSELoss()
tr = []
for epoch in range(MAX_EPOCHS):
	# training Part
	# start = time.time()
	model.train()
	mse = 0
	ce = 0
	bce = 0
	train_dataloader = tqdm(train_dataloader)
	b = 0
	error_count = 0
	for batch in train_dataloader:
		
		global_output = batch["output"]
		at_gt = batch["attention_gt"]
		sp_gt = batch["spatial_gt"]
		cn_gt = batch["contacting_gt"]
		mask = batch["mask"]
		
		total_len = global_output.shape[1]
		count = 0
		ce_vid = 0
		bce_sp_vid = 0
		bce_cont_vid = 0
		mse_vid = 0
		for i in range(total_len - FUTURE_LEN - CONTEXT_LEN + 1):
			b += 1
			count += 1
			x = global_output[:, i:i + CONTEXT_LEN, :, :]
			future = global_output[:, i + CONTEXT_LEN:i + CONTEXT_LEN + FUTURE_LEN, :, :]
			future_mask = mask[:, i + CONTEXT_LEN:i + CONTEXT_LEN + FUTURE_LEN, :]
			out = model(x, future, future_mask)
			pred_attention = out["attention_distribution"]
			pred_spatial = out["spatial_distribution"]
			pred_contacting = out["contacting_distribution"]
			pred_attention = pred_attention.squeeze(0)
			pred_spatial = pred_spatial.squeeze(0)
			pred_contacting = pred_contacting.squeeze(0)
			
			start = torch.where(batch["im_idx"] == i + CONTEXT_LEN)[1][0]
			if i + CONTEXT_LEN + FUTURE_LEN == total_len:
				end = batch["im_idx"].shape[1]
			else:
				end = torch.where(batch["im_idx"] == i + CONTEXT_LEN + FUTURE_LEN)[1][0]
			
			a_gt = torch.tensor(at_gt[start:end], dtype=torch.long).to(device).squeeze()
			s_gt = torch.zeros([len(sp_gt[start:end]), 6], dtype=torch.float32).to(device)
			c_gt = torch.zeros([len(cn_gt[start:end]), 17], dtype=torch.float32).to(device)
			for j in range(len(sp_gt[start:end])):
				s_gt[j, sp_gt[start:end][j]] = 1
				c_gt[j, cn_gt[start:end][j]] = 1
			
			losses = {}
			try:
				losses["attention_relation_loss"] = ce_loss(pred_attention, a_gt)
				losses["spatial_relation_loss"] = bce_loss(pred_spatial, s_gt.double())
				losses["contacting_relation_loss"] = bce_loss(pred_contacting, c_gt.double())
				losses["mse_loss"] = mse_loss(out["future_dense_pred"], out["future_dense_gt"].double())
				
				ce_vid += losses["attention_relation_loss"]
				bce_sp_vid += losses["spatial_relation_loss"]
				bce_cont_vid += losses["contacting_relation_loss"]
				mse_vid += losses["mse_loss"]
				
				optimizer.zero_grad()
				loss = sum(losses.values())
				loss.backward(retain_graph=True)
				optimizer.step()
				
				tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
				if b % 1000 == 0 and b >= 1000:
					mn = pd.concat(tr[-1000:], axis=1).mean(1)
					print(mn)
			except ValueError:
				error_count += 1
				# print("im_idx : ",batch["im_idx"])
				# print("start : ",start)
				# print("end : ",end)
				# print("pred at : ",pred_attention.shape)
				# print("a gt : ",a_gt.shape)
				# print("mask : ",future_mask)
				print("#" * 50)
				print("error number : ", error_count)
				print("#" * 50)
		
		if b % 500 == 0:
			print(f" vid ce loss :: {ce_vid / count} || vid mse loss :: {mse_vid / count}")
			print(f" vid sp bce loss :: {bce_sp_vid / count} || vid con bce loss :: {bce_cont_vid / count}")
	torch.save({"state_dict": model.state_dict()}, os.path.join('sgdet_model/', "model_{}.tar".format(epoch)))
	print("*" * 40)
	print("save the checkpoint after {} epochs".format(epoch))
	print("*" * 40)

"""(['boxes', 'labels', 'scores', 'distribution', 'pred_labels', 'features', 'fmaps', 'im_info', 'indices', 'pred_scores', 'pair_idx', 'im_idx', 'human_idx', 'union_feat', 'union_box', 'spatial_masks', 'attention_distribution', 'spatial_distribution', 'contacting_distribution', 'global_output', 'gt_annotation'])"""
