import os
# import sys
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
from dataset import *

DATA_PATH = "../all_frames_final"


def AG_dataloader(
		data: str = DATA_PATH,
		batch_size: int = 1,
		num_workers: int = 0,
		annot_path: str = '../../gt_annotations',
		data_split="train"
):
	if data_split == "train":
		dataset = AG_Dataset(data_root=data, annot_root=annot_path, split=data_split)
		return DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=num_workers,
			# pin_memory = True
		)
	else:
		dataset = AG_Dataset(data_root=data, annot_root=annot_path, mode='testing', split=data_split)
		return DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			# pin_memory = True
		)
		
		# ag = AG_dataloader()
	# print(ag)
