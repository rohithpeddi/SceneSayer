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
import pdb
from dataloader import *

DATA_PATH = "../all_frames"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sg_loss(future_graphs, gt_graphs, future_final_graphs, gt_actual_graphs):
    #loss1 = F.mse_loss(future_graphs,gt_graphs)
    future_graphs = future_graphs.to(device)
    gt_graphs = gt_graphs.to(device)
    future_final_graphs = future_final_graphs.to(device)
    gt_actual_graphs = gt_actual_graphs.to(device)
    loss2 = nn.BCELoss()(future_final_graphs.float().to(device),gt_actual_graphs.float().to(device))

    futures_ = (future_final_graphs >= 0.5).double()

    tp = torch.sum((gt_actual_graphs*futures_) == 1)
    fp = torch.sum((1-gt_actual_graphs)*futures_ == 1)
    fn = torch.sum(gt_actual_graphs*(1-futures_) == 1)
    tn = torch.sum((1-gt_actual_graphs)*(1-futures_) == 1)
    if tp==0 and fp==0:
        with open('futures_.pkl','wb') as f:
            pickle.dump(futures_,f)
        with open('gt_actual_graphs.pkl','wb') as f:
            pickle.dump(gt_actual_graphs,f)

    # For no constraint
    aca = torch.sort(future_final_graphs, dim=2, descending=True)[0]

    future10 = aca[:, :, 9:10]
    future20 = aca[:, :, 19:20]
    future50 = aca[:, :, 49:50]

    future10 = (future_final_graphs>=future10).double()
    future20 = (future_final_graphs>=future20).double()
    future50 = (future_final_graphs>=future50).double()
    
    tp10_ = torch.sum((gt_actual_graphs*future10) == 1)
    tp20_ = torch.sum((gt_actual_graphs*future20) == 1)
    tp50_ = torch.sum((gt_actual_graphs*future50) == 1)
    tpfn_ = torch.sum(gt_actual_graphs)
    
    # For with constraint below
    future_final_graphs = future_final_graphs.reshape(future_final_graphs.shape[0], future_final_graphs.shape[1], 35, 26)

    temp1 = (future_final_graphs[:, :, :, :3] == torch.max(future_final_graphs[:, :, :, :3], dim=3, keepdim=True)[0])
    temp2 = (future_final_graphs[:, :, :, 3:9] == torch.max(future_final_graphs[:, :, :, 3:9], dim=3, keepdim=True)[0])
    temp3 = (future_final_graphs[:, :, :, 9:] == torch.max(future_final_graphs[:, :, :, 9:], dim=3, keepdim=True)[0])
    aca1 = temp1*future_final_graphs[:, :, :, :3]
    aca2 = temp2*future_final_graphs[:, :, :, 3:9]
    aca3 = temp3*future_final_graphs[:, :, :, 9:]
    future_final_graphs = torch.flatten(future_final_graphs, start_dim=2, end_dim=3)
    temp = torch.flatten(torch.cat((aca1, aca2, aca3), 3), start_dim=2, end_dim=3)
    aca = torch.sort(temp, dim=2, descending=True)[0]
    future10 = aca[:, :, 9:10]
    future20 = aca[:, :, 19:20]
    future50 = aca[:, :, 49:50]
    future10 = (temp>=future10).double()
    future20 = (temp>=future20).double()
    future50 = (temp>=future50).double()
    tp10_with = torch.sum((gt_actual_graphs*future10) == 1)
    tp20_with = torch.sum((gt_actual_graphs*future20) == 1)
    tp50_with = torch.sum((gt_actual_graphs*future50) == 1)

    return {
    "loss": loss2,
    "loss1": loss2,
    "loss2": loss2,
    # "loss_other": loss_other,
    # "loss2_other": loss2_other,
    'tp': tp,
    'fp': fp,
    'fn': fn,
    'tn': tn,
    'tp10_': tp10_, # true positive in top 10 prediction
    'tp20_': tp20_,
    'tp50_': tp50_,
    'tpfn_': tpfn_, # true positive with false negative
    'tp10_with': tp10_with, # with : with constraints
    'tp20_with': tp20_with,
    'tp50_with': tp50_with
    }