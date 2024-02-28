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
	
	def generate_future_ff_obj_for_context(self, entry, num_cf, num_tf, num_ff):
		pass
