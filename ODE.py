"""
Let's get the relationships yo
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from lib.word_vectors import obj_edge_vectors
from torchvision.ops.boxes import box_area
from torchdiffeq import odeint_adjoint as odeint
from itertools import combinations
import pdb

# define the transformer backbone here
EncoderLayer = nn.TransformerEncoderLayer
Encoder = nn.TransformerEncoder

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor, indices=None) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if indices is None:
            x = x + self.pe[:,:x.size(1)]
        else:
            pos = torch.cat([self.pe[:, index] for index in indices])            
            x = x + pos
        return self.dropout(x)

class PositionalEncodingLearn(nn.Module):
    def __init__(self, d_model: int, max_len: int=5000):
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)
                
    def forward(self, x, indices = None):
        # x: b, l, d
        r = torch.arange(x.shape[1], device=x.device)
        embed = self.embed(r) # seq_len, embedding_dim
        return x + embed.repeat(x.shape[0], 1, 1)
           
    
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x    

class GetBoxes(nn.Module):
    def __init__(self, d_model):
        super(GetBoxes, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.ReLU()
        )
    
    def forward(self, so_rep):
        return self.model(so_rep)

class STTran(nn.Module):

    def __init__(self, mode='train',
                 attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None, rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(STTran, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        assert mode in ('train', 'ode_train')
        self.mode = mode
        self.num_features = 1936

        self.get_subj_boxes = GetBoxes(1936)
        self.get_obj_boxes = GetBoxes(1936)
        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.01),
        )
        self.subj_fc = nn.Linear(2376, 512)
        self.obj_fc = nn.Linear(2376, 512)
        self.vr_fc = nn.Linear(256*7*7, 512)
        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        d_model=1936
        self.positional_encoder = PositionalEncoding(d_model, max_len=400)
        # temporal encoder
        global_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.global_transformer = Encoder(global_encoder, num_layers=3)
        # spatial encoder
        local_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.local_transformer = Encoder(local_encoder, num_layers=1)  
        
        self.a_rel_compress = nn.Linear(d_model, self.attention_class_num)
        self.s_rel_compress = nn.Linear(d_model, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(d_model, self.contact_class_num)
    
        


    def forward(self, entry, testing = False):
        
        # visual part
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        entry["subj_rep_actual"] = subj_rep
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7))
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
        # semantic part
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class)
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        rel_features = torch.cat((x_visual, x_semantic), dim=1)
        
        # Spatial-Temporal Transformer
        # spatial message passing
        frames = []
        im_indices = entry["boxes"][entry["pair_idx"][:,1],0] # im_indices -> centre cordinate of all objects in a video 
        for l in im_indices.unique():
            frames.append(torch.where(im_indices==l)[0])   
        frame_features = pad_sequence([rel_features[index] for index in frames], batch_first=True)
        masks = (1-pad_sequence([torch.ones(len(index)) for index in frames], batch_first=True)).bool()
        rel_ = self.local_transformer(frame_features, src_key_padding_mask=masks.cuda())
        rel_features = torch.cat([rel_[i,:len(index)] for i, index in enumerate(frames)])
        #subj_dec = self.subject_decoder(rel_features, src_key_padding_mask=masks.cuda())
        #subj_dec = subj_compress(subj_dec)
        #entry["subj_rep_decoded"] = subj_dec
        #entry["spatial_encoder_out"] = rel_features
        # temporal message passing
        sequences = []
        for l in obj_class.unique():
            k = torch.where(obj_class.view(-1)==l)[0]
            if len(k) > 0:
                sequences.append(k)
        pos_index = []
        for index in sequences:
            im_idx, counts = torch.unique(entry["pair_idx"][index][:,0].view(-1), return_counts=True, sorted=True)
            counts = counts.tolist()
            pos = torch.cat([torch.LongTensor([im]*count) for im, count in zip(range(len(counts)), counts)])
            pos_index.append(pos)
        
        sequence_features = pad_sequence([rel_features[index] for index in sequences], batch_first=True)
        in_mask = (1-torch.tril(torch.ones(sequence_features.shape[1],sequence_features.shape[1]),diagonal = 0)).type(torch.bool)
        #in_mask = (1-torch.ones(sequence_features.shape[1],sequence_features.shape[1])).type(torch.bool)
        in_mask = in_mask.cuda()
        masks = (1-pad_sequence([torch.ones(len(index)) for index in sequences], batch_first=True)).bool()
        pos_index = pad_sequence(pos_index, batch_first=True) if self.mode == "sgdet" else None
        sequence_features = self.positional_encoder(sequence_features, pos_index)
        #out = torch.zeros(sequence_features.shape)
        seq_len = sequence_features.shape[1]
        mask_input = sequence_features
        out = self.global_transformer(mask_input,src_key_padding_mask = masks.cuda(),mask = in_mask)

        rel_ = out
        in_mask = None
        rel_ = rel_.cuda()
        rel_flat = torch.cat([rel[:len(index)] for index, rel in zip(sequences,rel_)])
        rel_ = None
        indices_flat = torch.cat(sequences).unsqueeze(1).repeat(1,rel_features.shape[1])
        assert len(indices_flat) == len(entry["pair_idx"])
        global_output = torch.zeros_like(rel_features).to(rel_features.device)
        global_output.scatter_(0, indices_flat, rel_flat)

        if self.mode == "ode_train":
            entry["attention_distribution"] = self.a_rel_compress(global_output)
            entry["spatial_distribution"] = self.s_rel_compress(global_output)
            entry["contacting_distribution"] = self.c_rel_compress(global_output)

            entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
            entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])
            entry["subject_boxes"] = self.get_subj_boxes(global_output)
            entry["object_boxes"] = self.get_obj_boxes(global_output)
        gt_annotation = entry["gt_annotation"]
        pair_idx = entry["pair_idx"]
        k = im_idx.size(0)
        subject_boxes = torch.zeros(k, 4)
        object_boxes = torch.zeros(k, 4)
        ctr = 0
        prev = 0
        for frame in gt_annotation:
            while pair[ctr, 0] < len(frame) + prev:
                subject_boxes[ctr] = frame[pair[ctr, 0] - prev]['person_bbox']
                object_boxes[ctr] = frame[pair[ctr, 1] - prev]['bbox']
                ctr += 1
            prev = ctr
        entry['global_output'] = global_output
        entry['subject_boxes_gt'] = subject_boxes
        entry['object_boxes_gt'] = object_boxes
        return entry

class get_derivatives(nn.Module):
    def __init__(self):
        super(get_derivatives, self).__init__()

        self.net = nn.Sequential(nn.Linear(1936, 2048), nn.Tanh(), nn.Dropout(0.2), 
                            nn.Linear(2048, 2048), nn.Tanh(), nn.Dropout(0.2), 
                            nn.Linear(2048, 1936))

    def forward(self, t, y):
        out = self.net(y)
        return out

class ODE(nn.Module):

    def __init__(self, mode, attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None, rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None, max_window=None):
        super(ODE, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.d_model = 1936
        self.max_window = max_window

        #self.object_classifier = ObjectClassifier(mode="sgdet", obj_classes=self.obj_classes)
        self.dsgdetr = STTran("ode_train",
               attention_class_num=attention_class_num,
               spatial_class_num=spatial_class_num,
               contact_class_num=contact_class_num,
               obj_classes=obj_classes)
        self.func = get_derivatives()

    def forward(self, entry, testing = False):
        entry = self.dsgdetr(entry)
        rem = torch.unique(entry["pair_idx"][ :, 0])
        labels = entry["labels"]
        mask = torch.ones(labels.size(0), dtype = bool)
        mask[rem] = False
        entry["obj_mask"] = mask
        labels = labels[mask]
        objs = torch.unique(labels)
        num_objs = len(objs)
        im_idx = entry["frame_idx"]
        tot = im_idx.size(0)
        t = entry["times"]
        indices = torch.reshape((im_idx[ : -1] != im_idx[1 : ]).nonzero(), (-1, )) + 1
        curr_id = 0
        t_unique = torch.unique(t)
        n = t_unique.size(0)
        t_extend = torch.Tensor([t_unique[-1] + i + 1 for i in range(self.max_window)])
        t_unique = torch.cat(t_unique, t_extend)
        anticipated_vals = torch.zeros(self.max_window, indices[-1], self.d_model)
        attention_distributions = torch.zeros(self.max_window, indices[-1], self.attention_class_num)
        spatial_distributions = torch.zeros(self.max_window, indices[-1], self.spatial_class_num)
        contact_distributions = torch.zeros(self.max_window, indices[-1], self.contact_class_num)
        subj_bounding_boxes = torch.zeros(self.max_window, indices[-1], 4)
        obj_bounding_boxes = torch.zeros(self.max_window, indices[-1], 4)
        rng = torch.cat((torch.Tensor([0]), indices, torch.Tensor([tot])))
        for i in range(1, self.max_window + 1):
            mask_curr = torch.tensor([], dtype = torch.int16)
            mask_gt = torch.tensor([], dtype = torch.int16)
            for j in range(n - i):
                inter, ind1, ind2 = np.intersect1d(np.array(labels[rng[j] : rng[j + 1]]), np.array(labels[rng[j + i] : rng[j + i + 1]]),  returns_indices = True)
                ind1 += rng[j]
                ind2 += rng[j + i]
                mask_curr = torch.cat((mask_curr, torch.tensor(ind1, dtype = torch.int16)))
                mask_gt = torch.cat((mask_gt, torch.tensor(ind2, dtype = torch.int16)))
            entry["mask_curr_%d" %i] = mask_curr
            entry["mask_gt_%d" %i] = mask_gt
        for i in range(n - 1):
            end = indices[i]
            batch_y0 = entry["global_output"][curr_id : end]
            batch_times = t_unique[i : i + self.max_window]
            ret = odeint(self.func, batch_y0, batch_times, method = 'dopri8')[1 : ]
            anticipated_vals[ :, curr_id : end, : ].data.copy_(ret)
            attention_distributions[ :, curr_id : end, : ].data.copy_(self.dsgdetr.a_rel_compress(ret))
            spatial_distributions[ :, curr_id : end, : ].data.copy_(self.dsgdetr.s_rel_compress(ret))
            contact_distributions[ :, curr_id : end, : ].data.copy_(self.dsgdetr.c_rel_compress(ret))

            subj_bounding_boxes[ :, curr_id : end, : ].data.copy_(self.dsgdetr.get_subj_boxes(ret))
            obj_bounding_boxes[ :, curr_id : end, : ].data.copy_(self.dsgdetr.get_obj_boxes(ret))

            curr_id = end
        spatial_distributions = torch.sigmoid(spatial_distributions)
        contact_distributions = torch.sigmoid(contact_distributions)
        entry["anticipated_vals"] = anticipated_vals
        entry["anticipated_attention_distribution"] = attention_distributions
        entry["anticipated_spatial_distribution"] = spatial_distributions
        entry["anticipated_contact_distribution"] = contact_distributions
        entry["anticipated_subject_boxes"] = subj_bounding_boxes
        entry["anticipated_object_boxes"] = obj_bounding_boxes
        return entry



        