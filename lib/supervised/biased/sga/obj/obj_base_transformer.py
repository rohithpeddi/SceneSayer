import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class ObjBaseTransformer(nn.Module):
	
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
		super(ObjBaseTransformer, self).__init__()
		
		self.mode = mode
		self.attention_class_num = attention_class_num
		self.spatial_class_num = spatial_class_num
		self.contact_class_num = contact_class_num
		self.obj_classes = obj_classes
		self.rel_classes = rel_classes
		self.enc_layer_num = enc_layer_num
		self.dec_layer_num = dec_layer_num
		
		assert mode in ('sgdet', 'sgcls', 'predcls')
	
	# 1. Compute features of future anticipated objects
	# 2. Compute corresponding union futures
	# 3. Prepare corresponding im_idx, pair_idx, pred_labels, boxes, and scores
	
	def generate_spatial_predicate_embeddings(self, entry):
		"""
		Entry can correspond to output of all the frames in the video
		:param entry:
		:return:
		"""
		# Visual Part
		subj_rep = entry['features'][entry['pair_idx'][:, 0]]
		subj_rep = self.subj_fc(subj_rep)
		obj_rep = entry['features'][entry['pair_idx'][:, 1]]
		obj_rep = self.obj_fc(obj_rep)
		vr = self.union_func1(entry['union_feat']) + self.conv(entry['spatial_masks'])
		vr = self.vr_fc(vr.view(-1, 256 * 7 * 7))
		x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
		
		# semantic part
		subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
		obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
		subj_emb = self.obj_embed(subj_class)
		obj_emb = self.obj_embed2(obj_class)
		x_semantic = torch.cat((subj_emb, obj_emb), 1)
		spa_so_rels_feats = torch.cat((x_visual, x_semantic), dim=1)
		
		# Spatial-Temporal Transformer
		# Spatial message passing
		# im_indices -> centre coordinate of all objects excluding subjects in a video
		frames = []
		im_indices = entry["boxes"][entry["pair_idx"][:, 1], 0]
		for l in im_indices.unique():
			frames.append(torch.where(im_indices == l)[0])
		frame_features = pad_sequence([spa_so_rels_feats[index] for index in frames], batch_first=True)
		masks = (1 - pad_sequence([torch.ones(len(index)) for index in frames], batch_first=True)).bool()
		rel_ = self.spatial_transformer(frame_features, src_key_padding_mask=masks.cuda())
		spa_so_rels_feats = torch.cat([rel_[i, :len(index)] for i, index in enumerate(frames)])
		
		# Temporal message passing
		obj_seqs = []
		for l in obj_class.unique():
			k = torch.where(obj_class.view(-1) == l)[0]
			obj_seqs.append(k)
		
		return entry, spa_so_rels_feats, obj_seqs
	
	def generate_spatio_temporal_predicate_embeddings(self, entry, spa_so_rels_feats, obj_seqs, temporal_transformer):
		"""
			Generate spatio-temporal predicate embeddings
			Case-1: For generation temporal transformer
				entry would correspond to output of all the frames in the video
				temporal_transformer: Would be generation temporal transformer

			Case-2: For anticipation temporal transformer
				entry would correspond to output of all frames that are in the context and the anticipated future frames
				temporal_transformer: Would be anticipation temporal transformer
		"""
		# Temporal message passing
		sequence_features = pad_sequence([spa_so_rels_feats[index] for index in obj_seqs], batch_first=True)
		padding_mask, causal_mask = self.fetch_temporal_masks(obj_seqs, sequence_features)
		
		positional_encoding = self.fetch_positional_encoding_for_gen_obj_seqs(obj_seqs, entry)
		sequence_features = self.gen_positional_encoder(sequence_features, positional_encoding)
		rel_ = temporal_transformer(sequence_features, src_key_padding_mask=padding_mask, mask=causal_mask)
		
		rel_flat = torch.cat([rel[:len(index)] for index, rel in zip(obj_seqs, rel_)])
		indices_flat = torch.cat(obj_seqs).unsqueeze(1).repeat(1, spa_so_rels_feats.shape[1])
		
		assert len(indices_flat) == len(entry["pair_idx"])
		spa_temp_rels_feats = torch.zeros_like(spa_so_rels_feats).to(spa_so_rels_feats.device)
		spa_temp_rels_feats.scatter_(0, indices_flat, rel_flat)
		
		return entry, spa_so_rels_feats, obj_seqs, spa_temp_rels_feats
	
	@staticmethod
	def fetch_positional_encoding(obj_seqs, entry):
		positional_encoding = []
		for obj_seq in obj_seqs:
			im_idx, counts = torch.unique(entry["pair_idx"][obj_seq][:, 0].view(-1), return_counts=True, sorted=True)
			counts = counts.tolist()
			if im_idx.numel() == 0:
				pos = torch.tensor(
					[torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
			else:
				pos = torch.cat([torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
			positional_encoding.append(pos)
		
		positional_encoding = [torch.tensor(seq, dtype=torch.long) for seq in positional_encoding]
		return positional_encoding
	
	def fetch_positional_encoding_for_gen_obj_seqs(self, obj_seqs, entry):
		positional_encoding = self.fetch_positional_encoding(obj_seqs, entry)
		positional_encoding = pad_sequence(positional_encoding, batch_first=True) if self.mode == "sgdet" else None
		return positional_encoding
	
	@staticmethod
	def fetch_temporal_masks(obj_seqs, sequence_features):
		causal_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
		                              diagonal=0)).type(torch.bool).cuda()
		padding_mask = (
				1 - pad_sequence([torch.ones(len(index)) for index in obj_seqs], batch_first=True)).bool().cuda()
		return padding_mask, causal_mask
	
	def generate_future_ff_rels_for_context(self, entry, entry_cf_ff, num_cf, num_tf, num_ff):
		"""
		1. Pass the entry through object classification layer.
		2. Generate representations for objects in the future frames
		3. Construct complete entry for these items and add them as values to the keys in the entry
		4. Generate spatial embeddings for all frames
		5. Generate spatial embeddings for objects in future frames
		6. Augment the context spatial embeddings and future frame spatial embeddings
		7. Generate temporal embeddings for relations in future frames
		8. Pass them through linear layers to get the final output and maintaining same code for losses.
		"""
		entry_cf_ff, spa_so_rels_feats_cf_ff, obj_seqs_cf_ff = self.generate_spatial_predicate_embeddings(entry_cf_ff)
		entry_cf_ff, spa_so_rels_feats_cf_ff, obj_seqs_cf_ff, spa_temp_rels_feats_cf_ff = self.generate_spatio_temporal_predicate_embeddings(
			entry_cf_ff, spa_so_rels_feats_cf_ff, obj_seqs_cf_ff, self.anti_temporal_transformer)
		
		# 1. Truncate spa_so_rels_feats_cf_ff and obj_seqs_cf_ff for the future frames
		# ------------------- From dictionary corresponding to the anticipated future frames -------------------
		num_ff = min(num_ff, num_tf - num_cf)
		ant_ff_start_id = entry_cf_ff["im_idx"].unique()[num_cf]
		ant_obj_ff_start_id = int(torch.where(entry_cf_ff["im_idx"] == ant_ff_start_id)[0][0])
		ant_spa_temp_rels_feats_ff = spa_temp_rels_feats_cf_ff[ant_obj_ff_start_id:]
		ant_obj_cf = entry_cf_ff["im_idx"][:ant_obj_ff_start_id]
		
		ant_num_obj_cf = ant_obj_cf.shape[0]
		ant_num_obj_cf_ff = entry_cf_ff["im_idx"].shape[0]
		ant_num_obj_ff = ant_num_obj_cf_ff - ant_num_obj_cf
		
		# ------------------- From dictionary corresponding to the ground truth frames -------------------
		
		gt_ff_start_id = entry["im_idx"].unique()[num_cf]
		gt_ff_end_id = entry["im_idx"].unique()[num_cf + num_ff - 1]
		
		gt_obj_ff_start_id = int(torch.where(entry["im_idx"] == gt_ff_start_id)[0][0])
		gt_obj_ff_end_id = int(torch.where(entry["im_idx"] == gt_ff_end_id)[0][-1]) + 1
		
		# 2. Truncate pair_idx, im_dx, pred_labels, boxes, and scores for the future frames
		ant_im_idx_ff = entry_cf_ff["im_idx"][ant_obj_ff_start_id:]
		ant_pair_idx_ff = entry_cf_ff["pair_idx"][ant_obj_ff_start_id:]
		ant_obj_labels_ff_start_id = ant_pair_idx_ff.min()
		
		ant_pred_labels_ff = entry_cf_ff["pred_labels"][ant_obj_labels_ff_start_id:]
		ant_boxes_ff = entry_cf_ff["boxes"][ant_obj_labels_ff_start_id:]
		ant_scores_ff = entry_cf_ff["scores"][ant_obj_labels_ff_start_id:]
		
		assert ant_num_obj_ff == ant_im_idx_ff.shape[0]
		assert ant_num_obj_ff + num_ff == ant_pred_labels_ff.shape[0]
		assert ant_num_obj_ff + num_ff == ant_boxes_ff.shape[0]
		assert ant_num_obj_ff + num_ff == ant_scores_ff.shape[0]
		
		# ------------------- Construct mask_ant and mask_gt -------------------
		# 3. Construct mask_ant and mask_gt for the future frames
		# mask_ant: indices in only anticipated frames
		# mask_gt: indices in all frames
		# ----------------------------------------------------------------------
		if self.training:
			"""
			Fetch all labels of the predicted objects in the future frames.
			Fetch all the labels of the ground truth objects in the future frames.
			
			Look at the intersection of the object labels in the predicted future frames and the ground truth future frames.
			
			Maintain the indices of the intersection in the predicted future frames and the ground truth future frames.
			This can be used for application of loss function, where the loss is calculated only for the intersection.
			"""
			obj_idx_ff = entry_cf_ff["pair_idx"][ant_obj_ff_start_id:][:, 1]
			pred_obj_labels_ff = entry_cf_ff["pred_labels"][ant_obj_labels_ff_start_id:][obj_idx_ff]
			
			gt_obj_labels_ff = entry["pred_labels"][entry["pair_idx"][gt_obj_ff_start_id:gt_obj_ff_end_id][:, 1]]
			
			np_pred_obj_labels_ff = pred_obj_labels_ff.cpu().numpy()
			np_gt_obj_labels_ff = gt_obj_labels_ff.cpu().numpy()
			
			int_obj_labels, gt_ff_obj_labels_in_pred_ff, pred_ff_obj_labels_in_gt_ff = (
				np.intersect1d(np_pred_obj_labels_ff, np_gt_obj_labels_ff, return_indices=True))
			
			gt_ff_start_id = entry["im_idx"].unique()[num_cf]
			gt_ff_end_id = entry["im_idx"].unique()[num_cf + num_ff - 1]
			gt_ff_obj_idx = (entry["im_idx"] >= gt_ff_start_id) & (entry["im_idx"] <= gt_ff_end_id)
			gt_ff_obj_idx = gt_ff_obj_idx.nonzero(as_tuple=False).squeeze()
			
			assert gt_ff_obj_idx.shape[0] >= pred_ff_obj_labels_in_gt_ff.shape[0]
			
			mask_ant = torch.tensor(gt_ff_obj_labels_in_pred_ff).cuda()
			mask_gt = torch.tensor(gt_ff_obj_idx[pred_ff_obj_labels_in_gt_ff]).cuda()
			
			assert mask_ant.shape[0] == mask_gt.shape[0]
			assert mask_ant.shape[0] <= ant_spa_temp_rels_feats_ff.shape[0]
		
		# 4. Construct attention_distribution, spatial_distribution, and contacting_distribution for the future frames
		temp = {
			"attention_distribution": self.a_rel_compress(ant_spa_temp_rels_feats_ff),
			"spatial_distribution": torch.sigmoid(self.s_rel_compress(ant_spa_temp_rels_feats_ff)),
			"contacting_distribution": torch.sigmoid(self.c_rel_compress(ant_spa_temp_rels_feats_ff)),
			"global_output": ant_spa_temp_rels_feats_ff,
			"pair_idx": ant_pair_idx_ff,
			"im_idx": ant_im_idx_ff,
			"labels": ant_pred_labels_ff,
			"pred_labels": ant_pred_labels_ff,
			"scores": ant_scores_ff,
			"pred_scores": ant_scores_ff,
			"boxes": ant_boxes_ff
		}
		
		if self.training:
			temp["mask_ant"] = mask_ant
			temp["mask_gt"] = mask_gt
		
		return temp
