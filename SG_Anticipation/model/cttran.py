import os
#import sys
import pickle
from typing import Optional
import pdb
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import math
from dataloader import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SGA(nn.Module):
    def __init__(
        self,
        context_len,
        future_len,
        d_model = 128,
        num_heads = 8,
        num_layers = 4,
        ffn_dim = 512,
        num_features = 71632
        #num_features = 1936
    ):
        super().__init__()
        self.context_len = context_len
        self.future_len = future_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.ffn_dim = ffn_dim
        self.num_features = num_features

        self.in_proj = nn.Linear(self.num_features,self.d_model).double()
        self.out_proj = nn.Linear(self.d_model,self.num_features).double()
        self.y1 = nn.Linear(26,26).double()
        self.y0 = nn.Linear(26,26).double()
        self.a_rel_compress = nn.Linear(1936,3).double()
        self.s_rel_compress = nn.Linear(1936,6).double()
        self.c_rel_compress = nn.Linear(1936,17).double()

        a_weight = torch.from_numpy(pickle.load(open('pretrained_weights/a_weight.pkl', 'rb'))).double()
        a_bias = torch.from_numpy(pickle.load(open('pretrained_weights/a_bias.pkl', 'rb'))).double()
        s_weight = torch.from_numpy(pickle.load(open('pretrained_weights/s_weight.pkl', 'rb'))).double()
        s_bias = torch.from_numpy(pickle.load(open('pretrained_weights/s_bias.pkl', 'rb'))).double()
        c_weight = torch.from_numpy(pickle.load(open('pretrained_weights/c_weight.pkl', 'rb'))).double()
        c_bias = torch.from_numpy(pickle.load(open('pretrained_weights/c_bias.pkl', 'rb'))).double()

        for name, params in self.a_rel_compress.named_parameters():
            if name == 'weight':
                params.data = a_weight
            if name == 'bias':
                params.data = a_bias
        for name, params in self.s_rel_compress.named_parameters():
            if name == 'weight':
                params.data = s_weight
            if name == 'bias':
                params.data = s_bias
        for name, params in self.c_rel_compress.named_parameters():
            if name == 'weight':
                params.data = c_weight
            if name == 'bias':
                params.data = c_bias

        enc_layer = nn.TransformerEncoderLayer(
            d_model = self.d_model,
            nhead = self.num_heads,
            dim_feedforward = self.ffn_dim,
            #norm_first = True,
            batch_first = True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = enc_layer, num_layers = self.num_layers)
        self.transformer_encoder = self.transformer_encoder.double()

    def get_positional_encoding(self,seq_len, d_model):
        pos = torch.arange(0, seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        encoding = torch.zeros(seq_len, d_model)
        encoding[:, 0::2] = torch.sin(pos * div_term)
        encoding[:, 1::2] = torch.cos(pos * div_term)
        return encoding.unsqueeze(0)

    def forward(self,x,future,future_mask):
        entry  = x
        context = torch.flatten(entry,start_dim = 2, end_dim = 3)

        """############ Frame wise temporal PE #############"""
        encoding_dim = self.d_model
        seq_len = self.context_len
        context = context.to(device)
        pos_enc = self.get_positional_encoding(seq_len,encoding_dim)
        
        # pe = torch.zeros((in_x.shape[0],in_x.shape[1],in_x.shape[2]))
        # start = 0
        # for i,id in enumerate(uniq_im):
        #     start
        #     le = im_idx[im_idx == id].shape[0]
        #     pe[:,start:start+le,:] = pos_enc[:,i,:]
        #     start = start+le
        # pe = pe.to(device)


        pred_out = []

        for i in range(self.future_len):
            in_x = self.in_proj(context.double())
            pos_enc = pos_enc.to(device)
            in_x = in_x + pos_enc
            in_x = in_x.to(device)
            out = self.transformer_encoder(in_x)
            pred = out[:,-1:]
            pred = self.out_proj(pred)
            pred_out.append(pred)
            context = torch.cat([context[:,1:], pred], dim = 1)
        future_graphs = torch.stack(pred_out, dim=1)
        future_graphs = future_graphs.reshape((
                                            context.shape[0],
                                            self.future_len,
                                            37,
                                            1936
                                            ))

        # print("fg : ",future_graphs.shape)
        # print("fm : ",future_mask.unsqueeze(-1).shape)
        
        future_graphs = future_graphs[future_mask]
        future = future[future_mask]

        a_pred = self.a_rel_compress(future_graphs)
        s_pred = self.s_rel_compress(future_graphs)
        c_pred = self.c_rel_compress(future_graphs)

        out_dict ={}

        out_dict["attention_distribution"] = a_pred
        out_dict["spatial_distribution"] = s_pred
        out_dict["contacting_distribution"] = c_pred


        out_dict["spatial_distribution"] = torch.sigmoid(out_dict["spatial_distribution"])
        out_dict["contacting_distribution"] = torch.sigmoid(out_dict["contacting_distribution"])
        out_dict["future_dense_pred"]  = future_graphs
        out_dict["future_dense_gt"]  = future


        return out_dict
