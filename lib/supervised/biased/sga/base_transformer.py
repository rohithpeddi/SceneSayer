import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class BaseTransformer(nn.Module):
	
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
		super(BaseTransformer, self).__init__()
		
		self.mode = mode
		self.attention_class_num = attention_class_num
		self.spatial_class_num = spatial_class_num
		self.contact_class_num = contact_class_num
		self.obj_classes = obj_classes
		self.rel_classes = rel_classes
		self.enc_layer_num = enc_layer_num
		self.dec_layer_num = dec_layer_num
		
		assert mode in ('sgdet', 'sgcls', 'predcls')
	
	def generate_spatial_predicate_embeddings(self, entry):
		entry = self.object_classifier(entry)
		
		# visual part
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
		rel_features = torch.cat((x_visual, x_semantic), dim=1)
		
		# Spatial-Temporal Transformer
		# spatial message passing
		# im_indices -> centre coordinate of all objects in a video
		frames = []
		im_indices = entry["boxes"][entry["pair_idx"][:, 1], 0]
		for l in im_indices.unique():
			frames.append(torch.where(im_indices == l)[0])
		frame_features = pad_sequence([rel_features[index] for index in frames], batch_first=True)
		masks = (1 - pad_sequence([torch.ones(len(index)) for index in frames], batch_first=True)).bool()
		rel_ = self.spatial_transformer(frame_features, src_key_padding_mask=masks.cuda())
		rel_features = torch.cat([rel_[i, :len(index)] for i, index in enumerate(frames)])
		# temporal message passing
		sequences = []
		for l in obj_class.unique():
			k = torch.where(obj_class.view(-1) == l)[0]
			if len(k) > 0:
				sequences.append(k)
		
		return entry, rel_features, sequences
	
	def fetch_positional_encoding_for_obj_seqs(self, obj_seqs, entry):
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
		positional_encoding = pad_sequence(positional_encoding, batch_first=True) if self.mode == "sgdet" else None
		return positional_encoding
	
	def update_positional_encoding(self, positional_encoding):
		if positional_encoding is not None:
			max_values = torch.max(positional_encoding, dim=1)[0] + 1
			max_values = max_values.unsqueeze(1)
			positional_encoding = torch.cat((positional_encoding, max_values), dim=1)
		return positional_encoding
