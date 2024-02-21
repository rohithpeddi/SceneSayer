import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from lib.supervised.biased.sga.base_transformer import BaseTransformer
from lib.supervised.biased.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifierMLP
from lib.word_vectors import obj_edge_vectors


class BaselineWithAnticipation(BaseTransformer):
	
	def __init__(
			self,
			mode='sgdet',
			attention_class_num=None,
			spatial_class_num=None,
			contact_class_num=None,
			obj_classes=None,
			rel_classes=None,
			enc_layer_num=None,
			dec_layer_num=None
	):
		super(BaselineWithAnticipation, self).__init__()
		
		self.obj_classes = obj_classes
		self.rel_classes = rel_classes
		self.attention_class_num = attention_class_num
		self.spatial_class_num = spatial_class_num
		self.contact_class_num = contact_class_num
		assert mode in ('sgdet', 'sgcls', 'predcls')
		self.mode = mode
		self.d_model = 128
		self.num_features = 1936
		
		self.object_classifier = ObjectClassifierMLP(mode=self.mode, obj_classes=self.obj_classes)
		
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
		self.subj_fc = nn.Linear(2048, 512)
		self.obj_fc = nn.Linear(2048, 512)
		self.vr_fc = nn.Linear(256 * 7 * 7, 512)
		embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data', wv_dim=200)
		self.obj_embed = nn.Embedding(len(obj_classes), 200)
		self.obj_embed.weight.data = embed_vecs.clone()
		
		self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
		self.obj_embed2.weight.data = embed_vecs.clone()
		
		d_model = 1936
		self.positional_encoder = PositionalEncoding(d_model, max_len=400)
		
		# spatial encoder
		spatial_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
		self.spatial_transformer = Encoder(spatial_encoder, num_layers=1)
		
		# temporal encoder
		temporal_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
		self.anti_temporal_transformer = Encoder(temporal_encoder, num_layers=3)
		
		self.a_rel_compress = nn.Linear(d_model, self.attention_class_num)
		self.s_rel_compress = nn.Linear(d_model, self.spatial_class_num)
		self.c_rel_compress = nn.Linear(d_model, self.contact_class_num)
	
	def generate_future_frame_embeddings(self, entry, num_cf, num_ff, obj_seqs_tf, so_rels_feats_tf):
		ff_start_id = entry["im_idx"].unique()[num_cf]
		ff_end_id = entry["im_idx"].unique()[num_cf + num_ff - 1]
		
		objects_ff_start_id = int(torch.where(entry["im_idx"] == ff_start_id)[0][0])
		objects_cf = entry["im_idx"][:objects_ff_start_id]
		num_objects_cf = objects_cf.shape[0]
		
		objects_ff_end_id = int(torch.where(entry["im_idx"] == ff_end_id)[0][-1]) + 1
		objects_ff = entry["im_idx"][objects_ff_start_id:objects_ff_end_id]
		num_objects_ff = objects_ff.shape[0]
		
		obj_seqs_cf = []
		obj_seqs_ff = []
		for i, obj_seq_tf in enumerate(obj_seqs_tf):
			obj_seq_cf = obj_seq_tf[(obj_seq_tf < num_objects_cf)]
			obj_seq_ff = obj_seq_tf[
				(obj_seq_tf >= num_objects_cf) & (obj_seq_tf < (num_objects_cf + num_objects_ff))]
			if len(obj_seq_cf) != 0:
				obj_seqs_cf.append(obj_seq_cf)
				obj_seqs_ff.append(obj_seq_ff)
		
		so_seqs_feats_cf = pad_sequence([so_rels_feats_tf[obj_seq_cf] for obj_seq_cf in obj_seqs_cf], batch_first=True)
		causal_mask = (1 - torch.tril(torch.ones(so_seqs_feats_cf.shape[1], so_seqs_feats_cf.shape[1]),
		                              diagonal=0)).type(torch.bool)
		causal_mask = causal_mask.cuda()
		masks = (1 - pad_sequence([torch.ones(len(index)) for index in obj_seqs_cf], batch_first=True)).bool()
		
		positional_encoding = self.fetch_positional_encoding_for_obj_seqs(obj_seqs_cf, entry)
		so_seqs_feats_cf = self.positional_encoder(so_seqs_feats_cf, positional_encoding)
		
		so_seqs_feats_ff = []
		for i in range(num_ff):
			so_seqs_feats_cf_temp_attd = self.anti_temporal_transformer(
				so_seqs_feats_cf,
				src_key_padding_mask=masks.cuda(),
				mask=causal_mask
			)
			if i == 0:
				mask2 = (~masks).int()
				ind_col = torch.sum(mask2, dim=1) - 1
				so_feats_nf = []
				for j, ind in enumerate(ind_col):
					so_feats_nf.append(so_seqs_feats_cf_temp_attd[j, ind, :])
				so_feats_nf = torch.stack(so_feats_nf).unsqueeze(1)
				so_seqs_feats_ff.append(so_feats_nf)
				so_seqs_feats_cf = torch.cat([so_seqs_feats_cf, so_feats_nf], 1)
			else:
				# TODO: Check for compression
				so_seqs_feats_ff.append(so_seqs_feats_cf_temp_attd[:, -1, :].unsqueeze(1))
				so_feats_nf = torch.stack([so_seqs_feats_cf_temp_attd[:, -1, :]], dim=1)
				so_seqs_feats_cf = torch.cat([so_seqs_feats_cf, so_feats_nf], 1)
			
			positional_encoding = self.update_positional_encoding(positional_encoding)
			so_seqs_feats_cf = self.positional_encoder(so_seqs_feats_cf, positional_encoding)
			causal_mask = (1 - torch.tril(torch.ones(so_seqs_feats_cf.shape[1], so_seqs_feats_cf.shape[1]),
			                              diagonal=0)).type(torch.bool)
			causal_mask = causal_mask.cuda()
			masks = torch.cat([masks, torch.full((masks.shape[0], 1), False, dtype=torch.bool)], dim=1)
		
		so_rels_feats_ff = torch.cat(so_seqs_feats_ff, dim=1).cuda()
		updated_so_rels_feats_ff = []
		
		if self.training:
			for index, rel in zip(obj_seqs_ff, so_rels_feats_ff):
				if len(index) == 0:
					continue
				updated_so_rels_feats_ff.extend(rel[:len(index)])
		else:
			for index, rel in zip(obj_seqs_ff, so_rels_feats_ff):
				if len(index) == 0:
					continue
				
				ob_frame_idx = entry["im_idx"][index]
				rel_temp = torch.zeros(len(index), rel.shape[1])
				# For each frame in ob_frame_idx, if the value repeats then add the relation value of previous frame
				k = 0  # index for rel
				for i, frame in enumerate(ob_frame_idx):
					if i == 0:
						rel_temp[i] = rel[k]
						k += 1
					elif frame == ob_frame_idx[i - 1]:
						rel_temp[i] = rel_temp[i - 1]
					else:
						rel_temp[i] = rel[k]
						k += 1
				
				updated_so_rels_feats_ff.extend(rel_temp)
		
		so_rels_feats_ff_flat = torch.tensor([tensor.tolist() for tensor in updated_so_rels_feats_ff]).cuda()
		obj_seqs_ff_flat = torch.cat(obj_seqs_ff).unsqueeze(1).repeat(1, so_rels_feats_tf.shape[1])
		aligned_so_rels_feats_tf = torch.zeros_like(so_rels_feats_tf).to(so_rels_feats_tf.device)
		
		return aligned_so_rels_feats_tf, so_rels_feats_ff_flat, obj_seqs_ff_flat, num_objects_cf, num_objects_ff
	
	def forward(self, entry, num_cf, num_ff):
		"""
		# -------------------------------------------------------------------------------------------------------------
		# Anticipation Module
		# -------------------------------------------------------------------------------------------------------------
		# 1. This section maintains starts from a set context predicts the future relations corresponding to the last
		# frame in the context
		# 2. Then it moves the context by one frame and predicts the future relations corresponding to the
		# last frame in the new context
		# 3. This is repeated until the end of the video, loss is calculated for each
		# future relation prediction and the loss is back-propagated
        :param entry: Dictionary from object classifier
        :param num_cf: Number of context frames
        :param num_ff: Number of next frames to anticipate
        :return:
        """
		entry, so_rels_feats_tf, obj_seqs_tf = self.generate_spatial_predicate_embeddings(entry)
		
		count = 0
		result = {}
		num_tf = len(entry["im_idx"].unique())
		num_cf = min(num_cf, num_tf - 1)
		num_ff = min(num_ff, num_tf - num_cf)
		while num_cf + 1 <= num_tf:
			(aligned_so_rels_feats_tf, so_rels_feats_ff_flat,
			 obj_seqs_ff_flat, num_objects_cf, num_objects_ff) = self.generate_future_frame_embeddings(
				entry, num_cf, num_ff, obj_seqs_tf, so_rels_feats_tf)
			
			temp = {"scatter_flag": 0}
			try:
				aligned_so_rels_feats_tf.scatter_(0, obj_seqs_ff_flat, so_rels_feats_ff_flat)
			except RuntimeError:
				num_cf += 1
				temp["scatter_flag"] = 1
				result[count] = temp
				count += 1
				continue
			
			aligned_so_rels_feats_ff = aligned_so_rels_feats_tf[num_objects_cf:num_objects_cf + num_objects_ff]
			num_cf += 1
			temp["attention_distribution"] = self.a_rel_compress(aligned_so_rels_feats_ff)
			temp["spatial_distribution"] = torch.sigmoid(self.s_rel_compress(aligned_so_rels_feats_ff))
			temp["contacting_distribution"] = torch.sigmoid(self.c_rel_compress(aligned_so_rels_feats_ff))
			temp["global_output"] = aligned_so_rels_feats_ff
			temp["original"] = aligned_so_rels_feats_tf
			temp["spatial_latents"] = so_rels_feats_tf[num_objects_cf:num_objects_cf + num_objects_ff]
			result[count] = temp
			count += 1
		entry["output"] = result
		return entry
	
	def forward_single_entry(self, context_fraction, entry):
		"""
        Forward method for the baseline
        :param context_fraction:
        :param entry: Dictionary from object classifier
        :return:
        """
		entry, so_rels_feats_tf, obj_seqs_tf = self.generate_spatial_predicate_embeddings(entry)
		
		result = {}
		count = 0
		num_tf = len(entry["im_idx"].unique())
		num_cf = min(int(math.ceil(context_fraction * num_tf)), num_tf - 1)
		num_ff = num_tf - num_cf
		
		(aligned_so_rels_feats_tf, so_rels_feats_ff_flat,
		 obj_seqs_ff_flat, num_objects_cf, num_objects_ff) = self.generate_future_frame_embeddings(
			entry, num_cf, num_ff, obj_seqs_tf, so_rels_feats_tf)
		
		temp = {"scatter_flag": 0}
		try:
			aligned_so_rels_feats_tf.scatter_(0, obj_seqs_ff_flat, so_rels_feats_ff_flat)
		except RuntimeError:
			temp["scatter_flag"] = 1
			result[count] = temp
			entry["output"] = result
			return entry
		
		aligned_so_rels_feats_ff = aligned_so_rels_feats_tf[num_objects_cf:num_objects_cf + num_objects_ff]
		temp["attention_distribution"] = self.a_rel_compress(aligned_so_rels_feats_ff)
		temp["spatial_distribution"] = torch.sigmoid(self.s_rel_compress(aligned_so_rels_feats_ff))
		temp["contacting_distribution"] = torch.sigmoid(self.c_rel_compress(aligned_so_rels_feats_ff))
		temp["global_output"] = aligned_so_rels_feats_ff
		temp["original"] = aligned_so_rels_feats_tf
		temp["spatial_latents"] = so_rels_feats_tf[num_objects_cf:num_objects_cf + num_objects_ff]
		result[count] = temp
		entry["output"] = result
		
		return entry
