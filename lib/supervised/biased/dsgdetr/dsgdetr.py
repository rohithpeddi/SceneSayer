"""
Let's get the relationships yo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from lib.word_vectors import obj_edge_vectors
from lib.fpn.box_utils import center_size
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from torchvision.ops.boxes import box_area
from itertools import combinations
from constants import Constants as const

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
			:param x:
			:param indices:
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
		# seq_len, embedding_dim
		r = torch.arange(x.shape[1], device=x.device)
		embed = self.embed(r)
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
		self.pos_embed = nn.Sequential(
			nn.BatchNorm1d(4, momentum=0.01 / 10.0),
			nn.Linear(4, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(0.1)
		)
		
		self.obj_dim = 2048
		d_model = self.obj_dim + 200 + 128
		encoder_layer = EncoderLayer(d_model=d_model, dim_feedforward=1024, nhead=8, batch_first=True)
		self.positional_encoder = PositionalEncoding(d_model, 0.1, 600 if mode == const.SGDET else 400)
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
			scores = entry[const.DISTRIBUTION][entry[const.BOXES][:, 0] == i]
			pred_boxes = entry[const.BOXES][entry[const.BOXES][:, 0] == i]
			feats = entry[const.FEATURES][entry[const.BOXES][:, 0] == i]
			pred_labels = entry[const.PRED_LABELS][entry[const.BOXES][:, 0] == i]
			labels_ = entry[const.LABELS][entry[const.BOXES][:, 0] == i]
			
			new_box = pred_boxes[entry[const.PRED_LABELS][entry[const.BOXES][:, 0] == i] == class_idx]
			new_feats = feats[entry[const.PRED_LABELS][entry[const.BOXES][:, 0] == i] == class_idx]
			new_scores = scores[entry[const.PRED_LABELS][entry[const.BOXES][:, 0] == i] == class_idx]
			new_labels_ = labels_[entry[const.PRED_LABELS][entry[const.BOXES][:, 0] == i] == class_idx]
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
		
		entry[const.BOXES] = torch.cat(final_boxes, dim=0)
		entry[const.DISTRIBUTION] = torch.cat(final_dists, dim=0)
		entry[const.FEATURES] = torch.cat(final_feats, dim=0)
		entry[const.PRED_LABELS] = torch.cat(final_labels, dim=0)
		entry[const.LABELS] = torch.cat(labels, dim=0)
		return entry
	
	@torch.no_grad()
	def get_edges(self, entry):
		edges = []
		for i in entry[const.BOXES][:, 0].unique():
			edges.extend(list(combinations(torch.where(entry[const.BOXES][:, 0] == i)[0].cpu().numpy(), 2)))
		edges = torch.LongTensor(edges).T
		return torch.cat([edges, torch.flip(edges, [0])], 1).cuda()
	
	def forward(self, entry):
		if self.mode == const.PREDCLS:
			obj_embed = F.one_hot(entry[const.LABELS] - 1,
			                      num_classes=len(self.classes) - 1).float() @ self.obj_embed.weight
			pos_embed = self.pos_embed(center_size(entry[const.BOXES][:, 1:]))
			obj_features = torch.cat((entry[const.FEATURES], obj_embed, pos_embed), 1)
			indices = entry[const.INDICES]
			entry[const.PRED_LABELS] = entry[const.LABELS]
			sequence_features = pad_sequence([obj_features[index] for index in indices], batch_first=True)
			masks = (1 - pad_sequence([torch.ones(len(index)) for index in indices], batch_first=True)).bool()
			obj_ = self.encoder_tran(self.positional_encoder(sequence_features), src_key_padding_mask=masks.cuda())
			obj_flat = torch.cat([obj[:len(index)] for index, obj in zip(indices, obj_)])
			indices_flat = torch.cat(indices).unsqueeze(1).repeat(1, obj_features.shape[1])
			final_features = torch.zeros_like(obj_features).to(obj_features.device)
			final_features.scatter_(0, indices_flat, obj_flat)
			entry[const.FEATURES] = final_features
			return entry
		elif self.mode == const.SGCLS:
			obj_embed = entry[const.DISTRIBUTION] @ self.obj_embed.weight
			pos_embed = self.pos_embed(center_size(entry[const.BOXES][:, 1:]))
			obj_features = torch.cat((entry[const.FEATURES], obj_embed, pos_embed), 1)
			if self.training:
				# construct tracking sequences
				indices = entry[const.INDICES]
				sequence_features = pad_sequence([obj_features[index] for index in indices], batch_first=True)
				masks = (1 - pad_sequence([torch.ones(len(index)) for index in indices], batch_first=True)).bool()
				# object transformer
				obj_ = self.encoder_tran(self.positional_encoder(sequence_features), src_key_padding_mask=masks.cuda())
				obj_flat = torch.cat([obj[:len(index)] for index, obj in zip(indices, obj_)])
				indices_flat = torch.cat(indices).unsqueeze(1).repeat(1, obj_features.shape[1])
				final_features = torch.zeros_like(obj_features).to(obj_features.device)
				final_features.scatter_(0, indices_flat, obj_flat)
				entry[const.FEATURES] = final_features
				# object classifier
				distribution = self.decoder_lin(final_features)
				entry[const.DISTRIBUTION] = distribution
				entry[const.PRED_LABELS] = torch.argmax(entry[const.DISTRIBUTION][:, 1:], 1) + 1
			else:
				indices = entry[const.INDICES]
				sequence_features = pad_sequence([obj_features[index] for index in indices], batch_first=True)
				masks = (1 - pad_sequence([torch.ones(len(index)) for index in indices], batch_first=True)).bool()
				obj_ = self.encoder_tran(self.positional_encoder(sequence_features), src_key_padding_mask=masks.cuda())
				obj_flat = torch.cat([obj[:len(index)] for index, obj in zip(indices, obj_)])
				indices_flat = torch.cat(indices).unsqueeze(1).repeat(1, obj_features.shape[1])
				final_features = torch.zeros_like(obj_features).to(obj_features.device)
				final_features.scatter_(0, indices_flat, obj_flat)
				entry[const.FEATURES] = final_features
				distribution = self.decoder_lin(final_features)
				entry[const.DISTRIBUTION] = distribution
				box_idx = entry[const.BOXES][:, 0].long()
				b = int(box_idx[-1] + 1)
				
				entry[const.DISTRIBUTION] = torch.softmax(entry[const.DISTRIBUTION][:, 1:], dim=1)
				entry[const.PRED_SCORES], entry[const.PRED_LABELS] = torch.max(entry[const.DISTRIBUTION][:, 1:], dim=1)
				entry[const.PRED_LABELS] = entry[const.PRED_LABELS] + 2
				
				# use the infered object labels for new pair idx
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
			obj_embed = entry[const.DISTRIBUTION] @ self.obj_embed.weight
			pos_embed = self.pos_embed(center_size(entry[const.BOXES][:, 1:]))
			obj_features = torch.cat((entry[const.FEATURES], obj_embed, pos_embed), 1)
			
			if self.training:
				indices = entry[const.INDICES]
				final_features = torch.zeros_like(obj_features).to(obj_features.device)
				# save memory by filetering out single-element sequences, indices[0]
				if len(indices) > 1:
					pos_index = []
					for index in indices[1:]:
						im_idx, counts = torch.unique(entry[const.BOXES][index][:, 0].view(-1), return_counts=True,
						                              sorted=True)
						counts = counts.tolist()
						pos = torch.cat(
							[torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
						pos_index.append(pos)
					sequence_features = pad_sequence([obj_features[index] for index in indices[1:]], batch_first=True)
					masks = (1 - pad_sequence([torch.ones(len(index)) for index in indices[1:]],
					                          batch_first=True)).bool()
					pos_index = pad_sequence(pos_index, batch_first=True)
					obj_ = self.encoder_tran(self.positional_encoder(sequence_features, pos_index),
					                         src_key_padding_mask=masks.cuda())
					obj_flat = torch.cat([obj[:len(index)] for index, obj in zip(indices[1:], obj_)])
					indices_flat = torch.cat(indices[1:]).unsqueeze(1).repeat(1, obj_features.shape[1])
					final_features.scatter_(0, indices_flat, obj_flat)
				if len(indices[0]) > 0:
					non_ = self.encoder_tran(self.positional_encoder(obj_features[indices[0]].unsqueeze(1)))
					final_features.scatter_(0, indices[0].unsqueeze(1).repeat(1, obj_features.shape[1]), non_[:, 0, :])
				entry[const.FEATURES] = final_features
				entry[const.DISTRIBUTION] = self.decoder_lin(final_features)
				entry[const.PRED_LABELS] = torch.argmax(entry[const.DISTRIBUTION][:, 1:], 1) + 1
				entry[const.PRED_LABELS] = entry[const.LABELS]
			else:
				indices = entry[const.INDICES]
				final_features = torch.zeros_like(obj_features).to(obj_features.device)
				if len(indices) > 1:
					pos_index = []
					for index in indices[1:]:
						im_idx, counts = torch.unique(entry[const.BOXES][index][:, 0].view(-1), return_counts=True,
						                              sorted=True)
						counts = counts.tolist()
						pos = torch.cat(
							[torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
						pos_index.append(pos)
					sequence_features = pad_sequence([obj_features[index] for index in indices[1:]], batch_first=True)
					masks = (1 - pad_sequence([torch.ones(len(index)) for index in indices[1:]],
					                          batch_first=True)).bool()
					pos_index = pad_sequence(pos_index, batch_first=True)
					obj_ = self.encoder_tran(self.positional_encoder(sequence_features, pos_index),
					                         src_key_padding_mask=masks.cuda())
					obj_flat = torch.cat([obj[:len(index)] for index, obj in zip(indices[1:], obj_)])
					indices_flat = torch.cat(indices[1:]).unsqueeze(1).repeat(1, obj_features.shape[1])
					final_features.scatter_(0, indices_flat, obj_flat)
				if len(indices[0]) > 0:
					non_ = self.encoder_tran(self.positional_encoder(obj_features[indices[0]].unsqueeze(1)))
					final_features.scatter_(0, indices[0].unsqueeze(1).repeat(1, obj_features.shape[1]), non_[:, 0, :])
				entry[const.FEATURES] = final_features
				distribution = self.decoder_lin(final_features)
				entry[const.DISTRIBUTION] = torch.softmax(distribution, dim=1)[:, 1:]
				entry[const.PRED_LABELS] = torch.argmax(entry[const.DISTRIBUTION], dim=1) + 1
				
				box_idx = entry[const.BOXES][:, 0].long()
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
					scores = entry[const.DISTRIBUTION][entry[const.BOXES][:, 0] == i]
					pred_boxes = entry[const.BOXES][entry[const.BOXES][:, 0] == i, 1:]
					feats = entry[const.FEATURES][entry[const.BOXES][:, 0] == i]
					labels = entry[const.LABELS][entry[const.BOXES][:, 0] == i]
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
				
				entry[const.LABELS] = torch.cat(final_labels, dim=0)
				entry[const.BOXES] = torch.cat(final_boxes, dim=0)
				box_idx = entry[const.BOXES][:, 0].long()
				entry[const.DISTRIBUTION] = torch.cat(final_dists, dim=0)
				entry[const.FEATURES] = torch.cat(final_feats, dim=0)
				
				entry[const.PRED_SCORES], entry[const.PRED_LABELS] = torch.max(entry[const.DISTRIBUTION][:, 1:], dim=1)
				entry[const.PRED_LABELS] = entry[const.PRED_LABELS] + 2
				# use the infered object labels for new pair idx
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


class DsgDETR(nn.Module):
	
	def __init__(
			self,
			mode=const.SGDET,
			attention_class_num=None,
			spatial_class_num=None,
			contact_class_num=None,
			obj_classes=None,
			rel_classes=None,
			enc_layer_num=None,
			dec_layer_num=None
	):
		
		"""
		:param classes: Object classes
		:param rel_classes: Relationship classes. None if were not using rel mode
		:param mode: (sgcls, predcls, or sgdet)
		"""
		super(DsgDETR, self).__init__()
		self.obj_classes = obj_classes
		self.rel_classes = rel_classes
		self.attention_class_num = attention_class_num
		self.spatial_class_num = spatial_class_num
		self.contact_class_num = contact_class_num
		assert mode in (const.SGDET, const.SGCLS, const.PREDCLS)
		self.mode = mode
		
		self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes)
		
		###################################
		self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
		self.conv = nn.Sequential(
			nn.Conv2d(2, 256 // 2, kernel_size=7, stride=2, padding=3, bias=True),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(256 // 2, momentum=0.01),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
			nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(256, momentum=0.01),
		)
		self.subj_fc = nn.Linear(2376, 512)
		self.obj_fc = nn.Linear(2376, 512)
		self.vr_fc = nn.Linear(256 * 7 * 7, 512)
		embed_vecs = obj_edge_vectors(obj_classes, wv_type=const.GLOVE_6B, wv_dir='checkpoints', wv_dim=200)
		self.obj_embed = nn.Embedding(len(obj_classes), 200)
		self.obj_embed.weight.data = embed_vecs.clone()
		
		self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
		self.obj_embed2.weight.data = embed_vecs.clone()
		
		d_model = 1936
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
		subj_rep = entry[const.FEATURES][entry[const.PAIR_IDX][:, 0]]
		subj_rep = self.subj_fc(subj_rep)
		obj_rep = entry[const.FEATURES][entry[const.PAIR_IDX][:, 1]]
		obj_rep = self.obj_fc(obj_rep)
		vr = self.union_func1(entry[const.UNION_FEAT]) + self.conv(entry[const.SPATIAL_MASKS])
		vr = self.vr_fc(vr.view(-1, 256 * 7 * 7))
		x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
		
		# semantic part
		subj_class = entry[const.PRED_LABELS][entry[const.PAIR_IDX][:, 0]]
		obj_class = entry[const.PRED_LABELS][entry[const.PAIR_IDX][:, 1]]
		subj_emb = self.obj_embed(subj_class)
		obj_emb = self.obj_embed2(obj_class)
		x_semantic = torch.cat((subj_emb, obj_emb), 1)
		
		rel_features = torch.cat((x_visual, x_semantic), dim=1)
		
		# Spatial-Temporal Transformer
		# spatial message passing
		frames = []
		im_indices = entry[const.BOXES][entry[const.PAIR_IDX][:, 1], 0]
		for l in im_indices.unique():
			frames.append(torch.where(im_indices == l)[0])
		frame_features = pad_sequence([rel_features[index] for index in frames], batch_first=True)
		masks = (1 - pad_sequence([torch.ones(len(index)) for index in frames], batch_first=True)).bool()
		rel_ = self.local_transformer(frame_features, src_key_padding_mask=masks.cuda())
		rel_features = torch.cat([rel_[i, :len(index)] for i, index in enumerate(frames)])
		
		# temporal message passing
		sequences = []
		for l in obj_class.unique():
			k = torch.where(obj_class.view(-1) == l)[0]
			if len(k) > 0:
				sequences.append(k)
		pos_index = []
		for index in sequences:
			im_idx, counts = torch.unique(entry[const.PAIR_IDX][index][:, 0].view(-1), return_counts=True, sorted=True)
			counts = counts.tolist()
			pos = torch.cat([torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
			pos_index.append(pos)
		sequence_features = pad_sequence([rel_features[index] for index in sequences], batch_first=True)
		masks = (1 - pad_sequence([torch.ones(len(index)) for index in sequences], batch_first=True)).bool()
		pos_index = pad_sequence(pos_index, batch_first=True) if self.mode == const.SGDET else None
		rel_ = self.global_transformer(self.positional_encoder(sequence_features, pos_index),
		                               src_key_padding_mask=masks.cuda())
		rel_flat = torch.cat([rel[:len(index)] for index, rel in zip(sequences, rel_)])
		indices_flat = torch.cat(sequences).unsqueeze(1).repeat(1, rel_features.shape[1])
		assert len(indices_flat) == len(entry[const.PAIR_IDX])
		global_output = torch.zeros_like(rel_features).to(rel_features.device)
		global_output.scatter_(0, indices_flat, rel_flat)
		
		entry[const.GLOBAL_OUTPUT] = global_output
		entry[const.ATTENTION_DISTRIBUTION] = self.a_rel_compress(global_output)
		entry[const.SPATIAL_DISTRIBUTION] = self.s_rel_compress(global_output)
		entry[const.CONTACTING_DISTRIBUTION] = self.c_rel_compress(global_output)
		
		entry[const.SPATIAL_DISTRIBUTION] = torch.sigmoid(entry[const.SPATIAL_DISTRIBUTION])
		entry[const.CONTACTING_DISTRIBUTION] = torch.sigmoid(entry[const.CONTACTING_DISTRIBUTION])
		return entry
