"""
Let's get the relationships yo
"""

import sys

sys.path.insert(0, '/home/maths/btech/mt1200841/scratch/NeSysVideoPrediction')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from lib.word_vectors import obj_edge_vectors
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from lib.fpn.box_utils import center_size
from torchvision.ops.boxes import box_area
from torchsde import sdeint_adjoint as sdeint
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

class ObjectClassifier(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='sgdet', obj_classes=None):
        super(ObjectClassifier, self).__init__()
        self.classes = obj_classes
        self.mode = mode

        #----------add nms when sgdet
        self.nms_filter_duplicates = True
        self.max_per_img =64
        self.thresh = 0.01

        #roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0/16.0, 0)

        embed_vecs = obj_edge_vectors(obj_classes[1:], wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes)-1, 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        self.obj_dim = 2048
        d_model = self.obj_dim + 200 + 128
        encoder_layer = EncoderLayer(d_model=d_model, dim_feedforward=1024, nhead=8, batch_first=True)
        self.positional_encoder = PositionalEncoding(d_model, 0.1, 600 if mode=="sgdet" else 400)
        self.encoder_tran = Encoder(encoder_layer, num_layers=3)
        
        self.decoder_lin = nn.Sequential(nn.Linear(d_model, 1024),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, len(self.classes)))
        
    def clean_class(self, entry, b, class_idx):
        final_boxes = []
        final_dists = []
        final_feats = []
        final_labels = []
        labels = []
        for i in range(b):
            scores = entry['distribution'][entry['boxes'][:, 0] == i]
            pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i]
            feats = entry['features'][entry['boxes'][:, 0] == i]
            pred_labels = entry['pred_labels'][entry['boxes'][:, 0] == i]
            labels_ = entry['labels'][entry['boxes'][:, 0] == i]

            new_box = pred_boxes[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_feats = feats[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores = scores[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_labels_ = labels_[entry['pred_labels'][entry['boxes'][:, 0] == i] == class_idx]
            new_scores[:, class_idx-1] = 0
            if new_scores.shape[0] > 0:
                new_labels = torch.argmax(new_scores, dim=1) + 1
            else:
                new_labels = torch.tensor([], dtype=torch.long).cuda(0)

            final_dists.append(scores)
            final_dists.append(new_scores)
            final_boxes.append(pred_boxes)
            final_boxes.append(new_box)
            final_feats.append(feats)
            final_feats.append(new_feats)
            final_labels.append(pred_labels)
            final_labels.append(new_labels)
            labels.append(labels_)
            labels.append(new_labels_)
            

        entry['boxes'] = torch.cat(final_boxes, dim=0)
        entry['distribution'] = torch.cat(final_dists, dim=0)
        entry['features'] = torch.cat(final_feats, dim=0)
        entry['pred_labels'] = torch.cat(final_labels, dim=0)
        entry["labels"] = torch.cat(labels, dim=0)
        return entry
    
    @torch.no_grad()
    def get_edges(self, entry):
        edges = []
        for i in entry["boxes"][:,0].unique():
            edges.extend(list(combinations(torch.where(entry["boxes"][:,0]==i)[0].cpu().numpy(),2)))
        edges = torch.LongTensor(edges).T
        return torch.cat([edges, torch.flip(edges, [0])], 1).cuda()
             
    def forward(self, entry):

        if self.mode  == 'predcls':
            obj_embed = F.one_hot(entry['labels']-1, num_classes=len(self.classes)-1).float()@ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            indices = entry["indices"]
            entry['pred_labels'] = entry['labels']
            sequence_features = pad_sequence([obj_features[index] for index in indices], batch_first=True)
            masks = (1-pad_sequence([torch.ones(len(index)) for index in indices], batch_first=True)).bool()
            obj_mask = (1-torch.tril(torch.ones(sequence_features.shape[1],sequence_features.shape[1]),diagonal = 0)).type(torch.bool)
            obj_ = self.encoder_tran(self.positional_encoder(sequence_features),src_key_padding_mask=masks.cuda(),mask = obj_mask.cuda())
            obj_flat = torch.cat([obj[:len(index)]for index, obj in zip(indices,obj_)])
            indices_flat = torch.cat(indices).unsqueeze(1).repeat(1,obj_features.shape[1])
            final_features = torch.zeros_like(obj_features).to(obj_features.device)
            final_features.scatter_(0, indices_flat, obj_flat) 
            entry["features"] = final_features
            return entry
        elif self.mode == 'sgcls':       
            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            if self.training:
                # construct tracking sequences 
                indices = entry["indices"]
                sequence_features = pad_sequence([obj_features[index] for index in indices], batch_first=True)
                masks = (1-pad_sequence([torch.ones(len(index)) for index in indices], batch_first=True)).bool()
                # object transformer
                obj_mask = (1-torch.tril(torch.ones(sequence_features.shape[1],sequence_features.shape[1]),diagonal = 0)).type(torch.bool)
                obj_ = self.encoder_tran(self.positional_encoder(sequence_features),src_key_padding_mask=masks.cuda(),mask = obj_mask.cuda())
                obj_flat = torch.cat([obj[:len(index)]for index, obj in zip(indices,obj_)])
                indices_flat = torch.cat(indices).unsqueeze(1).repeat(1,obj_features.shape[1])
                final_features = torch.zeros_like(obj_features).to(obj_features.device)
                final_features.scatter_(0, indices_flat, obj_flat)
                entry["features"] = final_features
                # object classifier
                distribution = self.decoder_lin(final_features)  
                entry["distribution"] = distribution
                entry['pred_labels'] = torch.argmax(entry["distribution"][:,1:],1) + 1
            else:
                indices = entry["indices"]
                sequence_features = pad_sequence([obj_features[index] for index in indices], batch_first=True)
                masks = (1-pad_sequence([torch.ones(len(index)) for index in indices], batch_first=True)).bool()
                obj_mask = (1-torch.tril(torch.ones(sequence_features.shape[1],sequence_features.shape[1]),diagonal = 0)).type(torch.bool)
                obj_ = self.encoder_tran(self.positional_encoder(sequence_features),src_key_padding_mask=masks.cuda(),mask = obj_mask.cuda())
                obj_flat = torch.cat([obj[:len(index)]for index, obj in zip(indices,obj_)])
                indices_flat = torch.cat(indices).unsqueeze(1).repeat(1,obj_features.shape[1])
                final_features = torch.zeros_like(obj_features).to(obj_features.device)
                final_features.scatter_(0, indices_flat, obj_flat) 
                entry["features"] = final_features
                distribution = self.decoder_lin(final_features) 
                entry["distribution"] = distribution
                box_idx = entry['boxes'][:,0].long()
                b = int(box_idx[-1] + 1)

                entry['distribution'] = torch.softmax(entry['distribution'][:, 1:], dim=1)
                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(obj_features.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry['distribution'][box_idx == i, 0]) # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                # drop repeat overlap TODO!!!!!!!!!!!!
                for i in range(b):
                    duplicate_class = torch.mode(entry['pred_labels'][entry['boxes'][:, 0] == i])[0]
                    present = entry['boxes'][:, 0] == i
                    if torch.sum(entry['pred_labels'][entry['boxes'][:, 0] == i] ==duplicate_class) > 0:
                        duplicate_position = entry['pred_labels'][present] == duplicate_class

                        ppp = torch.argsort(entry['distribution'][present][duplicate_position][:,duplicate_class - 1])[:-1]
                        for j in ppp:

                            changed_idx = global_idx[present][duplicate_position][j]
                            entry['distribution'][changed_idx, duplicate_class-1] = 0
                            entry['pred_labels'][changed_idx] = torch.argmax(entry['distribution'][changed_idx])+1
                            entry['pred_scores'][changed_idx] = torch.max(entry['distribution'][changed_idx])


                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx==j][entry['pred_labels'][box_idx==j] != 1]: # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(obj_features.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(obj_features.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx

                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat((im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                                        torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(obj_features.device)
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                entry['spatial_masks'] = spatial_masks
            return entry
        else:
            obj_embed = entry['distribution'] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)

            if self.training:
                indices = entry["indices"]
                final_features = torch.zeros_like(obj_features).to(obj_features.device)
                # save memory by filetering out single-element sequences, indices[0]
                if len(indices)>1:
                    pos_index = []
                    for index in indices[1:]:
                        #pdb.set_trace()
                        im_idx, counts = torch.unique(entry["boxes"][index][:,0].view(-1), return_counts=True, sorted=True)
                        counts = counts.tolist()
                        pos = torch.cat([torch.LongTensor([im]*count) for im, count in zip(range(len(counts)), counts)])
                        pos_index.append(pos)
                    sequence_features = pad_sequence([obj_features[index] for index in indices[1:]], batch_first=True)
                    masks = (1-pad_sequence([torch.ones(len(index)) for index in indices[1:]], batch_first=True)).bool()
                    pos_index = pad_sequence(pos_index, batch_first=True)
                    #pdb.set_trace()
                    obj_mask = (1-torch.tril(torch.ones(sequence_features.shape[1],sequence_features.shape[1]),diagonal = 0)).type(torch.bool)
                    obj_ = self.encoder_tran(self.positional_encoder(sequence_features),src_key_padding_mask=masks.cuda(),mask = obj_mask.cuda())
                    obj_flat = torch.cat([obj[:len(index)]for index, obj in zip(indices[1:],obj_)])
                    indices_flat = torch.cat(indices[1:]).unsqueeze(1).repeat(1,obj_features.shape[1]) 
                    final_features.scatter_(0, indices_flat, obj_flat)
                if len(indices[0]) > 0:
                    non_ = self.encoder_tran(self.positional_encoder(obj_features[indices[0]].unsqueeze(1)))
                    final_features.scatter_(0, indices[0].unsqueeze(1).repeat(1,obj_features.shape[1]), non_[:,0,:])
                entry["features"] = final_features
                entry["distribution"] = self.decoder_lin(final_features)
                entry['pred_labels'] = torch.argmax(entry["distribution"][:,1:],1)+1
                entry["pred_labels"] = entry["labels"]
            else:
                indices = entry["indices"]
                final_features = torch.zeros_like(obj_features).to(obj_features.device)
                if len(indices)>1:
                    pos_index = []
                    for index in indices[1:]:
                        im_idx, counts = torch.unique(entry["boxes"][index][:,0].view(-1), return_counts=True, sorted=True)
                        counts = counts.tolist()
                        pos = torch.cat([torch.LongTensor([im]*count) for im, count in zip(range(len(counts)), counts)])
                        pos_index.append(pos)
                    sequence_features = pad_sequence([obj_features[index] for index in indices[1:]], batch_first=True)
                    masks = (1-pad_sequence([torch.ones(len(index)) for index in indices[1:]], batch_first=True)).bool()
                    pos_index = pad_sequence(pos_index, batch_first=True)
                    obj_mask = (1-torch.tril(torch.ones(sequence_features.shape[1],sequence_features.shape[1]),diagonal = 0)).type(torch.bool)
                    obj_ = self.encoder_tran(self.positional_encoder(sequence_features),src_key_padding_mask=masks.cuda(),mask = obj_mask.cuda())
                    obj_flat = torch.cat([obj[:len(index)]for index, obj in zip(indices[1:],obj_)])
                    indices_flat = torch.cat(indices[1:]).unsqueeze(1).repeat(1,obj_features.shape[1])
                    final_features.scatter_(0, indices_flat, obj_flat)
                if len(indices[0]) > 0:
                    non_ = self.encoder_tran(self.positional_encoder(obj_features[indices[0]].unsqueeze(1)))           
                    final_features.scatter_(0, indices[0].unsqueeze(1).repeat(1,obj_features.shape[1]), non_[:,0,:])
                entry["features"] = final_features
                distribution = self.decoder_lin(final_features)
                #print("Decoder Distribution : ",distribution.shape)
                entry["distribution"] = torch.softmax(distribution, dim=1)[:,1:] 
                entry['pred_labels'] = torch.argmax(entry["distribution"],dim=1) + 1

                box_idx = entry['boxes'][:, 0].long()
                b = int(box_idx[-1] + 1)
                entry = self.clean_class(entry, b, 5)
                entry = self.clean_class(entry, b, 8)
                entry = self.clean_class(entry, b, 17)
                # # NMS
                final_boxes = []
                final_dists = []
                final_feats = []
                final_labels = []
                for i in range(b):
                    # images in the batch
                    scores = entry['distribution'][entry['boxes'][:, 0] == i]
                    pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i, 1:]
                    feats = entry['features'][entry['boxes'][:, 0] == i]
                    labels = entry['labels'][entry['boxes'][:, 0] == i]
                    for j in range(len(self.classes)-1):
                        # NMS according to obj categories
                        inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
                        # if there is det
                        if inds.numel() > 0:
                            cls_dists = scores[inds]
                            cls_feats = feats[inds]
                            cls_labels = labels[inds]
                            cls_scores = cls_dists[:, j]
                            _, order = torch.sort(cls_scores, 0, True)
                            cls_boxes = pred_boxes[inds]
                            cls_dists = cls_dists[order]
                            cls_feats = cls_feats[order]
                            cls_labels = cls_labels[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], 0.6)  # hyperparameter

                            final_labels.append(cls_labels[keep.view(-1).long()])
                            final_dists.append(cls_dists[keep.view(-1).long()])
                            final_boxes.append(torch.cat((torch.tensor([[i]], dtype=torch.float).repeat(keep.shape[0],
                                                                                                        1).cuda(0),
                                                          cls_boxes[order, :][keep.view(-1).long()]), 1))
                            final_feats.append(cls_feats[keep.view(-1).long()])
                            
                entry["labels"] = torch.cat(final_labels, dim=0)        
                entry['boxes'] = torch.cat(final_boxes, dim=0)
                box_idx = entry['boxes'][:, 0].long()
                #print(" Final dist : ",final_dists.shape)
                entry['distribution'] = torch.cat(final_dists, dim=0)
                #print("sstrans entry_distri : ",entry['distribution'].shape)
                entry['features'] = torch.cat(final_feats, dim=0)

                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2
                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(box_idx.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry['distribution'][
                                                       box_idx == i, 0])  # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx == j][
                        entry['pred_labels'][box_idx == j] != 1]:  # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(box_idx.device)
               
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(box_idx.device)

                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx
                entry['human_idx'] = HUMAN_IDX
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat(
                    (im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
                     torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)
                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
                entry['union_feat'] = union_feat
                entry['union_box'] = union_boxes
                pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                entry['spatial_masks'] = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(box_idx.device)

            return entry
        
class GetBoxes(nn.Module):
    def __init__(self, d_model):
        super(GetBoxes, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4),
            nn.Softplus()
        )
    
    def forward(self, so_rep):
        return self.model(so_rep)

class STTran(nn.Module):

    def __init__(self, mode='sgdet',
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
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.num_features = 1936

        self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes)

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

    def forward(self, entry):
        
        entry = self.object_classifier(entry)
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
        entry["attention_distribution"] = self.a_rel_compress(global_output)
        entry["spatial_distribution"] = self.s_rel_compress(global_output)
        entry["contacting_distribution"] = self.c_rel_compress(global_output)

        entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])
        #detached_outputs = global_output.clone().detach()
        entry["subject_boxes_dsg"] = self.get_subj_boxes(global_output)
        #entry["object_boxes_dsg"] = self.get_obj_boxes(global_output)

        pair_idx = entry["pair_idx"]
        boxes_rcnn = entry["boxes"]
        entry["global_output"] = global_output
        #entry["detached_outputs"] = detached_outputs
        entry["subject_boxes_rcnn"] = boxes_rcnn[pair_idx[ :, 0], 1 : ].to(global_output.device)
        #entry["object_boxes_rcnn"] = boxes_rcnn[pair_idx[ :, 1], 1 : ].to(global_output.device)
        return entry

class get_derivatives(nn.Module):
    noise_type = "general"
    sde_type = "stratonovich"
    def __init__(self, brownian_size):
        super(get_derivatives, self).__init__()
        self.drift = nn.Sequential(nn.Linear(1936, 2048), nn.Tanh(),
                            nn.Linear(2048, 2048), nn.Tanh(), 
                            nn.Linear(2048, 1936))
        self.diffusion = nn.Sequential(nn.Linear(1936, 2048), nn.Tanh(),
                            nn.Linear(2048, 2048), nn.Tanh(), 
                            nn.Linear(2048, 1936 * brownian_size))
        self.brownian_size = brownian_size

    def f(self, t, y):
        out = self.drift(y)
        return out
    
    def g(self, t, y):
        out = self.diffusion(y).view(-1, 1936, self.brownian_size)
        return out

class SDE(nn.Module):

    def __init__(self, mode, attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None, rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None, max_window=None, brownian_size=None):
        super(SDE, self).__init__()
        self.mode = mode
        self.diff_func = get_derivatives(brownian_size)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.d_model = 1936
        self.max_window = max_window

        self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes)
        self.dsgdetr = STTran(self.mode,
               attention_class_num=attention_class_num,
               spatial_class_num=spatial_class_num,
               contact_class_num=contact_class_num,
               obj_classes=obj_classes)
        self.gen_num = np.zeros(1750)
        self.ant_num = np.zeros((3, 1750))
        self.ctr = 0

    def forward(self, entry, testing = False):
        entry = self.dsgdetr(entry)
        obj = entry["pair_idx"][ :, 1]
        if not testing:
            labels_obj = entry["labels"][obj]
        else:
            pred_labels_obj = entry["pred_labels"][obj]
            labels_obj = entry["labels"][obj]
        #pdb.set_trace()
        im_idx = entry["im_idx"]
        pair_idx = entry["pair_idx"]
        tot = im_idx.size(0)
        t = torch.tensor(entry["frame_idx"], dtype = torch.int32)
        indices = torch.reshape((im_idx[ : -1] != im_idx[1 : ]).nonzero(), (-1, )) + 1
        curr_id = 0
        t_unique = torch.unique(t).float()
        n = torch.unique(im_idx).size(0)
        t_extend = torch.Tensor([t_unique[-1] + i + 1 for i in range(self.max_window)])
        global_output = entry["global_output"]
        t_unique = torch.cat((t_unique, t_extend)).to(device=global_output.device)
        anticipated_vals = torch.zeros(self.max_window, 0, self.d_model, device=global_output.device)
        #obj_bounding_boxes = torch.zeros(self.max_window, indices[-1], 4, device=global_output.device)
        rng = torch.cat((torch.Tensor([0]).to(device = indices.device), indices, torch.Tensor([tot]).to(device = indices.device)))
        rng = rng.long()
        #self.gen_num[self.ctr] = tot
        for i in range(1, self.max_window + 1):
            mask_curr = torch.tensor([], dtype=torch.long, device=rng.device)
            mask_gt = torch.tensor([], dtype=torch.long, device=rng.device)
            gt = entry["gt_annotation"].copy()
            for j in range(n - i):
                if testing:
                    a, b = np.array(pred_labels_obj[rng[j] : rng[j + 1]].cpu()), np.array(labels_obj[rng[j + i] : rng[j + i + 1]].cpu())
                else:
                    a, b = np.array(labels_obj[rng[j] : rng[j + 1]].cpu()), np.array(labels_obj[rng[j + i] : rng[j + i + 1]].cpu())
                intersection = np.intersect1d(a, b,  return_indices = False)
                ind1 = np.array([])
                ind2 = np.array([])
                for element in intersection:
                    tmp1, tmp2 = np.where(a == element)[0], np.where(b == element)[0]
                    mn = min(tmp1.shape[0], tmp2.shape[0])
                    ind1 = np.concatenate((ind1, tmp1[ : mn]))
                    ind2 = np.concatenate((ind2, tmp2[ : mn]))
                L = []
                if testing:
                    ctr = 0
                    for detection in gt[i + j]:
                        if "class" not in detection.keys() or detection["class"] in intersection:
                            L.append(ctr)
                        ctr += 1
                    gt[i + j] = [gt[i + j][ind] for ind in L]
                ind1 = torch.tensor(ind1, dtype=torch.long, device=rng.device)
                ind2 = torch.tensor(ind2, dtype=torch.long, device=rng.device)
                ind1 += rng[j]
                ind2 += rng[j + i]
                mask_curr = torch.cat((mask_curr, ind1))
                mask_gt = torch.cat((mask_gt, ind2))
            entry["mask_curr_%d" %i] = mask_curr
            entry["mask_gt_%d" %i] = mask_gt
            #self.ant_num[i - 1, self.ctr] = mask_curr.size(0)
            if testing:
                pair_idx_test = pair_idx[mask_curr]
                _, inverse_indices = torch.unique(pair_idx_test, sorted=True, return_inverse=True)
                entry["im_idx_test_%d" %i] = im_idx[mask_curr]
                entry["pair_idx_test_%d" %i] = inverse_indices
                if self.mode == "predcls":
                    entry["scores_test_%d" %i] = entry["scores"][_.long()]
                    entry["labels_test_%d" %i] = entry["labels"][_.long()]
                else:
                    entry["pred_scores_test_%d" %i] = entry["pred_scores"][_.long()]
                    entry["pred_labels_test_%d" %i] = entry["pred_labels"][_.long()]
                if inverse_indices.size(0) != 0:
                    mx = torch.max(inverse_indices)
                else:
                    mx = -1
                boxes_test = torch.zeros(mx + 1, 5, device=entry["boxes"].device)
                boxes_test[torch.unique_consecutive(inverse_indices[: , 0])] = entry["boxes"][torch.unique_consecutive(pair_idx[mask_gt][: , 0])]
                boxes_test[inverse_indices[: , 1]] = entry["boxes"][pair_idx[mask_gt][: , 1]]
                entry["boxes_test_%d" %i] = boxes_test
                #entry["boxes_test_%d" %i] = entry["boxes"][_.long()]
                entry["gt_annotation_%d" %i] = gt
        #self.ctr += 1
        #anticipated_latent_loss = 0
        #targets = entry["detached_outputs"]    
        for i in range(n - 1):
            end = indices[i]
            batch_y0 = global_output[curr_id : end]
            batch_times = t_unique[i : i + self.max_window + 1]
            ret = sdeint(self.diff_func, batch_y0, batch_times, method='reversible_heun', adjoint_method='adjoint_reversible_heun', dt=1)[1 : ]
            anticipated_vals = torch.cat((anticipated_vals, ret), dim=1)
            #obj_bounding_boxes[ :, curr_id : end, : ].data.copy_(self.dsgdetr.get_obj_boxes(ret))
            curr_id = end
        #for p in self.dsgdetr.get_subj_boxes.parameters():
        #    p.requires_grad_(False)
        entry["anticipated_subject_boxes"] = self.dsgdetr.get_subj_boxes(anticipated_vals)
        #for p in self.dsgdetr.get_subj_boxes.parameters():
        #    p.requires_grad_(True)
        entry["anticipated_vals"] = anticipated_vals
        entry["anticipated_attention_distribution"] = self.dsgdetr.a_rel_compress(anticipated_vals)
        entry["anticipated_spatial_distribution"] = torch.sigmoid(self.dsgdetr.s_rel_compress(anticipated_vals))
        entry["anticipated_contacting_distribution"] = torch.sigmoid(self.dsgdetr.c_rel_compress(anticipated_vals))
        #entry["anticipated_object_boxes"] = obj_bounding_boxes
        return entry



        
