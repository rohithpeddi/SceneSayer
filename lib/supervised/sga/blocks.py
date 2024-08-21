import copy
import math
from itertools import combinations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from constants import Constants as const
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.fpn.box_utils import center_size
from lib.word_vectors import obj_edge_vectors

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
            x = x + self.pe[:, :x.size(1)]
        else:
            pos = torch.cat([self.pe[:, index] for index in indices])
            x = x + pos
        return self.dropout(x)


class PositionalEncodingLearn(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.embed = nn.Embedding(max_len, d_model)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, x, indices=None):
        # x: b, l, d
        r = torch.arange(x.shape[1], device=x.device)
        embed = self.embed(r)  # seq_len, embedding_dim
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
            nn.Linear(d_model, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4),
            nn.Softplus()
        )

    def forward(self, so_rep):
        return self.model(so_rep)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class ObjectClassifierTransformer(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='sgdet', obj_classes=None):
        super(ObjectClassifierTransformer, self).__init__()
        self.classes = obj_classes
        self.mode = mode

        # ----------add nms when sgdet
        self.nms_filter_duplicates = True
        self.max_per_img = 64
        self.thresh = 0.01

        # roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0 / 16.0, 0)

        embed_vecs = obj_edge_vectors(obj_classes[1:], wv_type='glove.6B', wv_dir='data', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes) - 1, 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        self.obj_dim = 2048
        d_model = self.obj_dim + 200 + 128
        encoder_layer = EncoderLayer(d_model=d_model, dim_feedforward=1024, nhead=8, batch_first=True)
        self.positional_encoder = PositionalEncoding(d_model, 0.1, 600 if mode == "sgdet" else 400)
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
            new_scores[:, class_idx - 1] = 0
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
        for i in entry["boxes"][:, 0].unique():
            edges.extend(list(combinations(torch.where(entry["boxes"][:, 0] == i)[0].cpu().numpy(), 2)))
        edges = torch.LongTensor(edges).T
        return torch.cat([edges, torch.flip(edges, [0])], 1).cuda()

    def forward(self, entry):

        if self.mode == 'predcls':
            obj_embed = F.one_hot(entry['labels'] - 1,
                                  num_classes=len(self.classes) - 1).float() @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))
            obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1)
            indices = entry["indices"]
            entry['pred_labels'] = entry['labels']
            sequence_features = pad_sequence([obj_features[index] for index in indices], batch_first=True)
            masks = (1 - pad_sequence([torch.ones(len(index)) for index in indices], batch_first=True)).bool()
            obj_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
                                       diagonal=0)).type(torch.bool)
            obj_ = self.encoder_tran(self.positional_encoder(sequence_features), src_key_padding_mask=masks.cuda(),
                                     mask=obj_mask.cuda())
            obj_flat = torch.cat([obj[:len(index)] for index, obj in zip(indices, obj_)])
            indices_flat = torch.cat(indices).unsqueeze(1).repeat(1, obj_features.shape[1])
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
                masks = (1 - pad_sequence([torch.ones(len(index)) for index in indices], batch_first=True)).bool()
                # object transformer
                obj_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
                                           diagonal=0)).type(torch.bool)
                obj_ = self.encoder_tran(self.positional_encoder(sequence_features), src_key_padding_mask=masks.cuda(),
                                         mask=obj_mask.cuda())
                obj_flat = torch.cat([obj[:len(index)] for index, obj in zip(indices, obj_)])
                indices_flat = torch.cat(indices).unsqueeze(1).repeat(1, obj_features.shape[1])
                final_features = torch.zeros_like(obj_features).to(obj_features.device)
                final_features.scatter_(0, indices_flat, obj_flat)
                entry["features"] = final_features
                # object classifier
                distribution = self.decoder_lin(final_features)
                entry["distribution"] = distribution
                entry['pred_labels'] = torch.argmax(entry["distribution"][:, 1:], 1) + 1
            else:
                indices = entry["indices"]
                sequence_features = pad_sequence([obj_features[index] for index in indices], batch_first=True)
                masks = (1 - pad_sequence([torch.ones(len(index)) for index in indices], batch_first=True)).bool()
                obj_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
                                           diagonal=0)).type(torch.bool)
                obj_ = self.encoder_tran(self.positional_encoder(sequence_features), src_key_padding_mask=masks.cuda(),
                                         mask=obj_mask.cuda())
                obj_flat = torch.cat([obj[:len(index)] for index, obj in zip(indices, obj_)])
                indices_flat = torch.cat(indices).unsqueeze(1).repeat(1, obj_features.shape[1])
                final_features = torch.zeros_like(obj_features).to(obj_features.device)
                final_features.scatter_(0, indices_flat, obj_flat)
                entry["features"] = final_features
                distribution = self.decoder_lin(final_features)
                entry["distribution"] = distribution
                box_idx = entry['boxes'][:, 0].long()
                b = int(box_idx[-1] + 1)

                entry['distribution'] = torch.softmax(entry['distribution'][:, 1:], dim=1)
                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2

                # use the infered object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(obj_features.device)
                global_idx = torch.arange(0, entry['boxes'].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry['distribution'][
                                                       box_idx == i, 0])  # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry['pred_labels'][HUMAN_IDX.squeeze()] = 1
                entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

                # drop repeat overlap TODO!!!!!!!!!!!!
                for i in range(b):
                    duplicate_class = torch.mode(entry['pred_labels'][entry['boxes'][:, 0] == i])[0]
                    present = entry['boxes'][:, 0] == i
                    if torch.sum(entry['pred_labels'][entry['boxes'][:, 0] == i] == duplicate_class) > 0:
                        duplicate_position = entry['pred_labels'][present] == duplicate_class

                        ppp = torch.argsort(entry['distribution'][present][duplicate_position][:, duplicate_class - 1])[
                              :-1]
                        for j in ppp:
                            changed_idx = global_idx[present][duplicate_position][j]
                            entry['distribution'][changed_idx, duplicate_class - 1] = 0
                            entry['pred_labels'][changed_idx] = torch.argmax(entry['distribution'][changed_idx]) + 1
                            entry['pred_scores'][changed_idx] = torch.max(entry['distribution'][changed_idx])

                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx == j][
                        entry['pred_labels'][box_idx == j] != 1]:  # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(obj_features.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(obj_features.device)
                entry['pair_idx'] = pair
                entry['im_idx'] = im_idx

                entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
                union_boxes = torch.cat(
                    (im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
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
                # save memory by filtering out single-element sequences, indices[0]
                if len(indices) > 1:
                    pos_index = []
                    for index in indices[1:]:
                        # pdb.set_trace()
                        im_idx, counts = torch.unique(entry["boxes"][index][:, 0].view(-1), return_counts=True,
                                                      sorted=True)
                        counts = counts.tolist()
                        pos = torch.cat(
                            [torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
                        pos_index.append(pos)
                    sequence_features = pad_sequence([obj_features[index] for index in indices[1:]], batch_first=True)
                    masks = (1 - pad_sequence([torch.ones(len(index)) for index in indices[1:]],
                                              batch_first=True)).bool()
                    pos_index = pad_sequence(pos_index, batch_first=True)
                    # pdb.set_trace()
                    obj_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
                                               diagonal=0)).type(torch.bool)
                    obj_ = self.encoder_tran(self.positional_encoder(sequence_features, pos_index),
                                             src_key_padding_mask=masks.cuda(), mask=obj_mask.cuda())
                    obj_flat = torch.cat([obj[:len(index)] for index, obj in zip(indices[1:], obj_)])
                    indices_flat = torch.cat(indices[1:]).unsqueeze(1).repeat(1, obj_features.shape[1])
                    final_features.scatter_(0, indices_flat, obj_flat)
                if len(indices[0]) > 0:
                    non_ = self.encoder_tran(self.positional_encoder(obj_features[indices[0]].unsqueeze(1)))
                    final_features.scatter_(0, indices[0].unsqueeze(1).repeat(1, obj_features.shape[1]), non_[:, 0, :])
                entry["features"] = final_features
                entry["distribution"] = self.decoder_lin(final_features)
                entry['pred_labels'] = torch.argmax(entry["distribution"][:, 1:], 1) + 1
                entry["pred_labels"] = entry["labels"]
            else:
                indices = entry["indices"]
                final_features = torch.zeros_like(obj_features).to(obj_features.device)
                if len(indices) > 1:
                    pos_index = []
                    for index in indices[1:]:
                        im_idx, counts = torch.unique(entry["boxes"][index][:, 0].view(-1), return_counts=True,
                                                      sorted=True)
                        counts = counts.tolist()
                        pos = torch.cat(
                            [torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
                        pos_index.append(pos)
                    sequence_features = pad_sequence([obj_features[index] for index in indices[1:]], batch_first=True)
                    masks = (1 - pad_sequence([torch.ones(len(index)) for index in indices[1:]],
                                              batch_first=True)).bool()
                    pos_index = pad_sequence(pos_index, batch_first=True)
                    obj_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
                                               diagonal=0)).type(torch.bool)
                    obj_ = self.encoder_tran(self.positional_encoder(sequence_features, pos_index),
                                             src_key_padding_mask=masks.cuda(), mask=obj_mask.cuda())
                    obj_flat = torch.cat([obj[:len(index)] for index, obj in zip(indices[1:], obj_)])
                    indices_flat = torch.cat(indices[1:]).unsqueeze(1).repeat(1, obj_features.shape[1])
                    final_features.scatter_(0, indices_flat, obj_flat)
                if len(indices[0]) > 0:
                    non_ = self.encoder_tran(self.positional_encoder(obj_features[indices[0]].unsqueeze(1)))
                    final_features.scatter_(0, indices[0].unsqueeze(1).repeat(1, obj_features.shape[1]), non_[:, 0, :])
                entry["features"] = final_features
                distribution = self.decoder_lin(final_features)
                # print("Decoder Distribution : ",distribution.shape)
                entry["distribution"] = torch.softmax(distribution, dim=1)[:, 1:]
                entry['pred_labels'] = torch.argmax(entry["distribution"], dim=1) + 1

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
                    for j in range(len(self.classes) - 1):
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
                # print(" Final dist : ",final_dists.shape)
                entry['distribution'] = torch.cat(final_dists, dim=0)
                # print("sstrans entry_distri : ",entry['distribution'].shape)
                entry['features'] = torch.cat(final_feats, dim=0)

                entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'][:, 1:], dim=1)
                entry['pred_labels'] = entry['pred_labels'] + 2
                # use the inferred object labels for new pair idx
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


class ObjectClassifierMLP(nn.Module):
    """Module for computing the object contexts and edge contexts"""

    def __init__(self, mode=const.SGDET, obj_classes=None):
        super(ObjectClassifierMLP, self).__init__()
        self.classes = obj_classes
        self.mode = mode

        self.nms_filter_duplicates = True
        self.max_per_img = 64
        self.thresh = 0.01

        # roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0 / 16.0, 0)

        embed_vecs = obj_edge_vectors(obj_classes[1:], wv_type=const.GLOVE_6B, wv_dir='checkpoints', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes) - 1, 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        self.obj_dim = 2048
        self.decoder_lin = nn.Sequential(nn.Linear(self.obj_dim + 200 + 128, 1024),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, len(self.classes)))

    def clean_class(self, entry, b, class_idx):
        final_boxes = []
        final_dists = []
        final_feats = []
        final_labels = []
        for i in range(b):
            scores = entry[const.DISTRIBUTION][entry[const.BOXES][:, 0] == i]
            pred_boxes = entry[const.BOXES][entry[const.BOXES][:, 0] == i]
            feats = entry[const.FEATURES][entry[const.BOXES][:, 0] == i]
            pred_labels = entry[const.PRED_LABELS][entry[const.BOXES][:, 0] == i]

            new_box = pred_boxes[entry[const.PRED_LABELS][entry[const.BOXES][:, 0] == i] == class_idx]
            new_feats = feats[entry[const.PRED_LABELS][entry[const.BOXES][:, 0] == i] == class_idx]
            new_scores = scores[entry[const.PRED_LABELS][entry[const.BOXES][:, 0] == i] == class_idx]
            new_scores[:, class_idx - 1] = 0
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
        entry[const.BOXES] = torch.cat(final_boxes, dim=0)
        entry[const.DISTRIBUTION] = torch.cat(final_dists, dim=0)
        entry[const.FEATURES] = torch.cat(final_feats, dim=0)
        entry[const.PRED_LABELS] = torch.cat(final_labels, dim=0)
        return entry

    def forward(self, entry):
        if self.mode == const.PREDCLS:
            entry[const.PRED_LABELS] = entry[const.LABELS]
            return entry
        elif self.mode == const.SGCLS:
            obj_embed = entry[const.DISTRIBUTION] @ self.obj_embed.weight
            pos_embed = self.pos_embed(center_size(entry[const.BOXES][:, 1:]))
            obj_features = torch.cat((entry[const.FEATURES], obj_embed, pos_embed), 1)
            if self.training:
                entry[const.DISTRIBUTION] = self.decoder_lin(obj_features)
                entry[const.PRED_LABELS] = entry[const.LABELS]
            else:
                entry[const.DISTRIBUTION] = self.decoder_lin(obj_features)

                box_idx = entry[const.BOXES][:, 0].long()
                b = int(box_idx[-1] + 1)

                entry[const.DISTRIBUTION] = torch.softmax(entry[const.DISTRIBUTION][:, 1:], dim=1)
                entry[const.PRED_SCORES], entry[const.PRED_LABELS] = torch.max(entry[const.DISTRIBUTION][:, 1:], dim=1)
                entry[const.PRED_LABELS] = entry[const.PRED_LABELS] + 2

                # use the inferred object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(obj_features.device)
                global_idx = torch.arange(0, entry[const.BOXES].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry[const.DISTRIBUTION][
                                                       box_idx == i, 0])  # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry[const.PRED_LABELS][HUMAN_IDX.squeeze()] = 1
                entry[const.PRED_SCORES][HUMAN_IDX.squeeze()] = entry[const.DISTRIBUTION][HUMAN_IDX.squeeze(), 0]

                # drop repeat overlap TODO!!!!!!!!!!!!
                for i in range(b):
                    duplicate_class = torch.mode(entry[const.PRED_LABELS][entry[const.BOXES][:, 0] == i])[0]
                    present = entry[const.BOXES][:, 0] == i
                    if torch.sum(entry[const.PRED_LABELS][entry[const.BOXES][:, 0] == i] == duplicate_class) > 0:
                        duplicate_position = entry[const.PRED_LABELS][present] == duplicate_class

                        ppp = torch.argsort(
                            entry[const.DISTRIBUTION][present][duplicate_position][:, duplicate_class - 1])[
                              :-1]
                        for j in ppp:
                            changed_idx = global_idx[present][duplicate_position][j]
                            entry[const.DISTRIBUTION][changed_idx, duplicate_class - 1] = 0
                            entry[const.PRED_LABELS][changed_idx] = torch.argmax(
                                entry[const.DISTRIBUTION][changed_idx]) + 1
                            entry[const.PRED_SCORES][changed_idx] = torch.max(entry[const.DISTRIBUTION][changed_idx])

                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx == j][
                        entry[const.PRED_LABELS][
                            box_idx == j] != 1]:  # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(obj_features.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(obj_features.device)
                entry[const.PAIR_IDX] = pair
                entry[const.IM_IDX] = im_idx

                entry[const.BOXES][:, 1:] = entry[const.BOXES][:, 1:] * entry[const.IM_INFO]
                union_boxes = torch.cat(
                    (im_idx[:, None],
                     torch.min(entry[const.BOXES][:, 1:3][pair[:, 0]], entry[const.BOXES][:, 1:3][pair[:, 1]]),
                     torch.max(entry[const.BOXES][:, 3:5][pair[:, 0]], entry[const.BOXES][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry[const.FMAPS], union_boxes)
                entry[const.BOXES][:, 1:] = entry[const.BOXES][:, 1:] / entry[const.IM_INFO]
                pair_rois = torch.cat((entry[const.BOXES][pair[:, 0], 1:], entry[const.BOXES][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(obj_features.device)
                entry[const.UNION_FEAT] = union_feat
                entry[const.UNION_BOX] = union_boxes
                entry[const.SPATIAL_MASKS] = spatial_masks
            return entry
        else:
            if self.training:
                obj_embed = entry[const.DISTRIBUTION] @ self.obj_embed.weight
                pos_embed = self.pos_embed(center_size(entry[const.BOXES][:, 1:]))
                obj_features = torch.cat((entry[const.FEATURES], obj_embed, pos_embed), 1)

                entry[const.DISTRIBUTION] = self.decoder_lin(obj_features)
                entry[const.PRED_LABELS] = entry[const.LABELS]
            else:
                obj_embed = entry[const.DISTRIBUTION] @ self.obj_embed.weight
                pos_embed = self.pos_embed(center_size(entry[const.BOXES][:, 1:]))
                obj_features = torch.cat((entry[const.FEATURES], obj_embed, pos_embed),
                                         1)  # use the result from FasterRCNN directly

                box_idx = entry[const.BOXES][:, 0].long()
                b = int(box_idx[-1] + 1)

                entry = self.clean_class(entry, b, 5)
                entry = self.clean_class(entry, b, 8)
                entry = self.clean_class(entry, b, 17)

                # # NMS
                final_boxes = []
                final_dists = []
                final_feats = []
                for i in range(b):
                    # images in the batch
                    scores = entry[const.DISTRIBUTION][entry[const.BOXES][:, 0] == i]
                    pred_boxes = entry[const.BOXES][entry[const.BOXES][:, 0] == i, 1:]
                    feats = entry[const.FEATURES][entry[const.BOXES][:, 0] == i]

                    for j in range(len(self.classes) - 1):
                        # NMS according to obj categories
                        inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
                        # if there is det
                        if inds.numel() > 0:
                            cls_dists = scores[inds]
                            cls_feats = feats[inds]
                            cls_scores = cls_dists[:, j]
                            _, order = torch.sort(cls_scores, 0, True)
                            cls_boxes = pred_boxes[inds]
                            cls_dists = cls_dists[order]
                            cls_feats = cls_feats[order]
                            keep = nms(cls_boxes[order, :], cls_scores[order], 0.6)  # hyperparameter

                            final_dists.append(cls_dists[keep.view(-1).long()])
                            final_boxes.append(torch.cat((torch.tensor([[i]], dtype=torch.float).repeat(keep.shape[0],
                                                                                                        1).cuda(0),
                                                          cls_boxes[order, :][keep.view(-1).long()]), 1))
                            final_feats.append(cls_feats[keep.view(-1).long()])

                entry[const.BOXES] = torch.cat(final_boxes, dim=0)
                box_idx = entry[const.BOXES][:, 0].long()
                entry[const.DISTRIBUTION] = torch.cat(final_dists, dim=0)
                entry[const.FEATURES] = torch.cat(final_feats, dim=0)

                entry[const.PRED_SCORES], entry[const.PRED_LABELS] = torch.max(entry[const.DISTRIBUTION][:, 1:], dim=1)
                entry[const.PRED_LABELS] = entry[const.PRED_LABELS] + 2

                # use the inferred object labels for new pair idx
                HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(box_idx.device)
                global_idx = torch.arange(0, entry[const.BOXES].shape[0])

                for i in range(b):
                    local_human_idx = torch.argmax(entry[const.DISTRIBUTION][
                                                       box_idx == i, 0])  # the local bbox index with highest human score in this frame
                    HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

                entry[const.PRED_LABELS][HUMAN_IDX.squeeze()] = 1
                entry[const.PRED_SCORES][HUMAN_IDX.squeeze()] = entry[const.DISTRIBUTION][HUMAN_IDX.squeeze(), 0]

                im_idx = []  # which frame are the relations belong to
                pair = []
                for j, i in enumerate(HUMAN_IDX):
                    for m in global_idx[box_idx == j][
                        entry[const.PRED_LABELS][
                            box_idx == j] != 1]:  # this long term contains the objects in the frame
                        im_idx.append(j)
                        pair.append([int(i), int(m)])

                pair = torch.tensor(pair).to(box_idx.device)
                im_idx = torch.tensor(im_idx, dtype=torch.float).to(box_idx.device)
                entry[const.PAIR_IDX] = pair
                entry[const.IM_IDX] = im_idx
                entry[const.HUMAN_IDX] = HUMAN_IDX
                entry[const.BOXES][:, 1:] = entry[const.BOXES][:, 1:] * entry[const.IM_INFO]
                union_boxes = torch.cat(
                    (im_idx[:, None],
                     torch.min(entry[const.BOXES][:, 1:3][pair[:, 0]], entry[const.BOXES][:, 1:3][pair[:, 1]]),
                     torch.max(entry[const.BOXES][:, 3:5][pair[:, 0]], entry[const.BOXES][:, 3:5][pair[:, 1]])), 1)

                union_feat = self.RCNN_roi_align(entry[const.FMAPS], union_boxes)
                entry[const.BOXES][:, 1:] = entry[const.BOXES][:, 1:] / entry[const.IM_INFO]
                entry[const.UNION_FEAT] = union_feat
                entry[const.UNION_BOX] = union_boxes
                pair_rois = torch.cat((entry[const.BOXES][pair[:, 0], 1:], entry[const.BOXES][pair[:, 1], 1:]),
                                      1).data.cpu().numpy()
                entry[const.SPATIAL_MASKS] = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(box_idx.device)
            return entry
