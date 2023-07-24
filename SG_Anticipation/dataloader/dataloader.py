import os
import sys
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
from dataloader.dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def AG_dataloader(
    data:str = 'data_preparation', 
    batch_size:int = 1, 
    num_workers:int = 0, 
    data_split = "train"
    ):
    dataset = AG_Dataset(data_root = data, split = data_split)
    if  data_split=="train":
        return DataLoader(
            dataset, 
            batch_size = batch_size, 
            shuffle = True, 
            num_workers = num_workers, 
            #pin_memory = True
        )
    else: 
         return DataLoader(
            dataset, 
            batch_size = batch_size, 
            shuffle = False, 
            num_workers = num_workers, 
            #pin_memory = True
        ) 


    

