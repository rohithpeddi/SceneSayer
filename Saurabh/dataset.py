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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AG_Dataset(Dataset):
    def __init__(
        self,
        data_root:str,
        annot_root = '../../gt_annotations',
        max_num_graphs: Optional[int] = None,
        split: str = "train",
        ):
        super().__init__()
        self.data_root = data_root
        self.annot_root = annot_root
        self.max_num_graphs = max_num_graphs
        if split != "val":
            self.data_path = os.path.join(data_root,split)
            self.annot_path = os.path.join(annot_root,split)
        else:
            self.data_path = os.path.join(data_root,'test')
            self.annot_path = os.path.join(annot_root,'test')
        self.split = split
        assert os.path.exists(self.data_root), f"Path {self.data_root} does not exist"
        assert self.split == "train" or self.split == "val" or self.split == "test"
        assert os.path.exists(self.data_path), f"Path {self.data_path} does not exist"
        self.files = self.get_files()
        self.gt_annotations = self.get_annot()

    def __getitem__(
        self,
        index: int,
        ):
        entry={}
        graph_path = self.files[index]
        print("files ", graph_path)
        annot_path = self.gt_annotations[index]
        print("annot ", annot_path)
        temp = pickle.load(open(graph_path,'rb'))
        gt = pickle.load(open(annot_path,'rb'))
        entry['global_output'] = temp['global_output']
        entry['gt_annotation'] = gt
        entry["boxes"] = temp['boxes']
        entry["labels"] = temp['labels']
        entry["scores"] = temp['scores']
        entry['frames'] = temp['frames']
        entry["pred_labels"] = temp['pred_labels']
        entry["pred_scores"] = temp['pred_scores']
        entry["pair_idx"] = temp['pair_idx']
        entry["im_idx"] = temp['im_idx']
        entry["human_idx"] = temp['human_idx']
        dense_graph = temp['dense_graph']
        pred_graph = temp["pred_graph"]
        gt_graph = temp["graph"]
        mask = np.ones(101)
        for i in range(len(dense_graph),101,1):
            mask[i] *= 0
        pad_dense= np.zeros(1)
        pad= np.zeros(1)
        

        return torch.from_numpy(dense_graph), torch.from_numpy(pred_graph),torch.from_numpy(gt_graph),entry

    def __len__(self):
        return len(self.files)

    def get_files(self):
        paths = []
        graphs = [graph.split('.')[0]for graph in os.listdir(self.data_path)]
        graphs.sort()
        for graph in graphs:
            paths.append(os.path.join(self.data_path,str(graph)+'.pkl'))
        return list(filter(None, paths))

    def get_annot(self):
        paths = []
        graphs = [graph.split('.')[0]for graph in os.listdir(self.annot_path)]
        graphs.sort()
        for graph in graphs:
            paths.append(os.path.join(self.annot_path,str(graph)+'.pkl'))
        return list(filter(None, paths))

# ag = AG_Dataset(data_root = "../../all_frames_full")


# for i in range(1):
#     dg,_,_,entry = ag.__getitem__(i)

#     print(entry['gt_annotation'][0][0]['frame'].split('.')[0])
#     print(entry.keys())
