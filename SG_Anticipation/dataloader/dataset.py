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

class AG_Dataset(Dataset):
    def __init__(
        self,
        data_root:str,
        split: str = "train",
        ):
        super().__init__()
        self.data_root = data_root
        if split != "val":
            self.data_path = os.path.join(data_root,split+'5')
        else:
            self.data_path = os.path.join(data_root,'test'+'5')
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        self.files = self.get_files()

    def __getitem__(
        self,
        index: int,
        ):
        graph_path = self.files[index]
        temp = pickle.load(open(graph_path,'rb'))
        return temp

    def __len__(self):
        return len(self.files)

    def get_files(self):
        paths = []
        graphs = [graph.split('.')[0] for graph in os.listdir(self.data_path)]
        #graphs.sort()
        for graph in graphs:
            paths.append(os.path.join(self.data_path,str(graph)+'.pkl'))
        return list(filter(None, paths))

# dataset = AG_Dataset(data_root = '../data_preparation',split = 'train')
# data = dataset.__getitem__(0)
# print(data.keys())