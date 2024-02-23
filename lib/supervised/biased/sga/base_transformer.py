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
	
	def generate_future_ff_rels_for_context(self, entry, so_rels_feats_tf, obj_seqs_tf, num_cf, num_tf, num_ff):
		num_ff = min(num_ff, num_tf - num_cf)
		ff_start_id = entry["im_idx"].unique()[num_cf]
		ff_end_id = entry["im_idx"].unique()[num_cf + num_ff - 1]
		cf_end_id = entry["im_idx"].unique()[num_cf - 1]
		
		objects_ff_start_id = int(torch.where(entry["im_idx"] == ff_start_id)[0][0])
		objects_clf_start_id = int(torch.where(entry["im_idx"] == cf_end_id)[0][0])
		
		objects_ff_end_id = int(torch.where(entry["im_idx"] == ff_end_id)[0][-1]) + 1
		obj_labels_tf_unique = entry['pred_labels'][entry['pair_idx'][:, 1]].unique()
		objects_pcf = entry["im_idx"][:objects_clf_start_id]
		objects_cf = entry["im_idx"][:objects_ff_start_id]
		objects_ff = entry["im_idx"][objects_ff_start_id:objects_ff_end_id]
		num_objects_cf = objects_cf.shape[0]
		num_objects_pcf = objects_pcf.shape[0]
		num_objects_ff = objects_ff.shape[0]
		
		# 1. Refine object sequences to take only those objects that are present in the current frame.
		# 2. Fetch future representations for those objects.
		# 3. Construct im_idx, pair_idx, and labels for the future frames accordingly.
		# 4. Send everything along with ground truth for evaluation.

		cf_obj_seqs_in_clf = []
		obj_seqs_ff = []
		object_labels_clf = []
		for i, s in enumerate(obj_seqs_tf):
			if len(s) == 0:
				continue
			context_index = s[(s < num_objects_cf)]
			if len(context_index) > 0:
				prev_context_index = s[(s >= num_objects_pcf) & (s < num_objects_cf)]
				if len(prev_context_index) > 0:
					object_labels_clf.append(obj_labels_tf_unique[i])
					cf_obj_seqs_in_clf.append(context_index)

			future_index = s[(s >= num_objects_cf) & (s < (num_objects_cf + num_objects_ff))]
			if len(future_index) > 0:
				obj_seqs_ff.append(future_index - num_objects_cf)
		
		sequence_features = pad_sequence([so_rels_feats_tf[index] for index in cf_obj_seqs_in_clf], batch_first=True)
		in_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
		                          diagonal=0)).type(torch.bool)
		in_mask = in_mask.cuda()
		masks = (1 - pad_sequence([torch.ones(len(index)) for index in cf_obj_seqs_in_clf], batch_first=True)).bool()
		
		positional_encoding = self.fetch_positional_encoding_for_obj_seqs(cf_obj_seqs_in_clf, entry)
		sequence_features = self.positional_encoder(sequence_features, positional_encoding)
		mask_input = sequence_features
		
		if self.training:
			output = []
			for i in range(num_ff):
				out = self.anti_temporal_transformer(mask_input, src_key_padding_mask=masks.cuda(), mask=in_mask)
				if i == 0:
					mask2 = (~masks).int()
					ind_col = torch.sum(mask2, dim=1) - 1
					out2 = []
					for j, ind in enumerate(ind_col):
						out2.append(out[j, ind, :])
					out3 = torch.stack(out2)
					out3 = out3.unsqueeze(1)
					output.append(out3)
					mask_input = torch.cat([mask_input, out3], 1)
				else:
					output.append(out[:, -1, :].unsqueeze(1))
					out_last = [out[:, -1, :]]
					pred = torch.stack(out_last, dim=1)
					mask_input = torch.cat([mask_input, pred], 1)
				
				positional_encoding = self.update_positional_encoding(positional_encoding)
				mask_input = self.positional_encoder(mask_input, positional_encoding)
				in_mask = (1 - torch.tril(torch.ones(mask_input.shape[1], mask_input.shape[1]), diagonal=0)).type(
					torch.bool)
				in_mask = in_mask.cuda()
				masks = torch.cat([masks, torch.full((masks.shape[0], 1), False, dtype=torch.bool)], dim=1)
			
			output = torch.cat(output, dim=1)
			rel_ = output
			rel_ = rel_.cuda()
			rel_flat1 = []
			
			for index, rel in zip(obj_seqs_ff, rel_):
				if len(index) == 0:
					continue
				rel_flat1.extend(rel[:len(index)])
			
			rel_flat1 = [tensor.tolist() for tensor in rel_flat1]
			rel_flat = torch.tensor(rel_flat1)
			rel_flat = rel_flat.to('cuda:0')
			indices_flat = torch.cat(obj_seqs_ff).unsqueeze(1).repeat(1, so_rels_feats_tf.shape[1])
			global_output = torch.zeros_like(so_rels_feats_tf).to(so_rels_feats_tf.device)
			
			temp = {"scatter_flag": 0}
			try:
				global_output.scatter_(0, indices_flat, rel_flat)
			except RuntimeError:
				num_cf += 1
				temp["scatter_flag"] = 1
				return temp
			gb_output = global_output[num_objects_cf:num_objects_cf + num_objects_ff]
			num_cf += 1
			
			temp["attention_distribution"] = self.a_rel_compress(gb_output)
			temp["spatial_distribution"] = torch.sigmoid(self.s_rel_compress(gb_output))
			temp["contacting_distribution"] = torch.sigmoid(self.c_rel_compress(gb_output))
			temp["global_output"] = gb_output
			temp["original"] = global_output
			
			return temp
		else:
			obj, obj_ind = entry["pred_labels"][
				entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 1]].sort()
			obj = [o.item() for o in obj]
			obj_dict = {}
			for o in object_lab:
				obj_dict[o] = 1
			
			for o in obj:
				if o in obj_dict:
					obj_dict[o] += 1
			
			for o in torch.tensor(obj).unique():
				if o.item() in obj_dict:
					obj_dict[o.item()] -= 1
			
			output = []
			col_num = list(obj_dict.values())
			max_non_pad_len = max(col_num)
			latent_mask = torch.tensor(list(object_track.values()))
			for i in range(num_ff):
				out = self.anti_temporal_transformer(mask_input, src_key_padding_mask=masks.cuda(), mask=in_mask)
				if i == 0:
					mask2 = (~masks).int()
					ind_col = torch.sum(mask2, dim=1)
					out2 = []
					for j, ind in enumerate(ind_col):
						out2.append(out[j, ind - col_num[j]:ind, :])
					out2 = pad_sequence(out2, batch_first=True)
					out4 = out2[latent_mask == 1]
					output.append(out4)
					mask_input = torch.cat([mask_input, out2], 1)
				else:
					out2 = []
					len_seq = out.shape[1]
					start_seq = len_seq - max_non_pad_len
					for j in range(out.shape[0]):
						out2.append(out[j, start_seq:start_seq + col_num[j], :])
					out2 = pad_sequence(out2, batch_first=True)
					out4 = out2[latent_mask == 1]
					output.append(out4)
					mask_input = torch.cat([mask_input, out2], 1)
				
				in_mask = (1 - torch.tril(torch.ones(mask_input.shape[1], mask_input.shape[1]), diagonal=0)).type(
					torch.bool)
				in_mask = in_mask.cuda()
				src_pad = []
				for j in col_num:
					src_pad.append(torch.zeros(j))
				src_mask = pad_sequence(src_pad, batch_first=True, padding_value=1).bool()
				masks = torch.cat([masks, src_mask], dim=1)
			
			output = torch.cat(output, dim=1)
			rel_ = output
			rel_ = rel_.cuda()
			
			if self.mode == 'predcls':
				obj, obj_ind = entry["labels"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 1]].sort()
				num_obj_unique = len(obj.unique())
				obj = [o.item() for o in obj]
				num_obj = len(obj)
			else:
				obj, obj_ind = entry["pred_labels"][
					entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 1]].sort()
				num_obj_unique = len(obj.unique())
				obj = [o.item() for o in obj]
				num_obj = len(obj)
			
			gb_output = torch.zeros(len(obj) * num_ff, rel_.shape[2])
			obj_dict = {}
			for o in obj:
				if o not in obj_dict:
					obj_dict[o] = 1
				else:
					obj_dict[o] += 1
			
			col_num = list(obj_dict.values())
			col_idx = 0
			row_idx = 0
			i = 0
			while i < gb_output.shape[0]:
				for j in range(col_idx, col_idx + col_num[row_idx]):
					gb_output[i] = rel_[row_idx, j, :]
					i += 1
				row_idx += 1
				row_idx = row_idx % num_obj_unique
				if row_idx % num_obj_unique == 0:
					col_idx = col_idx + max_non_pad_len
			
			im_idx = torch.tensor(list(range(num_ff)))
			im_idx = im_idx.repeat_interleave(num_obj)
			
			pred_labels = [1]
			pred_labels.extend(obj)
			pred_labels = torch.tensor(pred_labels)
			pred_labels = pred_labels.repeat(num_ff)
			
			pair_idx = []
			for i in range(1, num_obj + 1):
				pair_idx.append([0, i])
			pair_idx = torch.tensor(pair_idx)
			p_i = pair_idx
			for i in range(num_ff - 1):
				p_i = p_i + num_obj + 1
				pair_idx = torch.cat([pair_idx, p_i])
			
			if self.mode == 'predcls':
				sc_human = entry["scores"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 0]][0]
				sc_obj = entry["scores"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 1]]
			else:
				sc_human = entry["pred_scores"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 0]][0]
				sc_obj = entry["pred_scores"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 1]]
			
			sc_obj = torch.index_select(sc_obj, 0, torch.tensor(obj_ind))
			sc_human = sc_human.unsqueeze(0)
			scores = torch.cat([sc_human, sc_obj])
			scores = scores.repeat(num_ff)
			
			box_human = entry["boxes"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 0]][0]
			box_obj = entry["boxes"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 1]]
			
			box_obj = torch.index_select(box_obj, 0, torch.tensor(obj_ind))
			box_human = box_human.unsqueeze(0)
			boxes = torch.cat([box_human, box_obj])
			boxes = boxes.repeat(num_ff, 1)
			
			gb_output = gb_output.cuda()
			temp = {
				"attention_distribution": self.a_rel_compress(gb_output),
				"spatial_distribution": torch.sigmoid(self.s_rel_compress(gb_output)),
				"contacting_distribution": torch.sigmoid(self.c_rel_compress(gb_output)),
				"global_output": gb_output,
				"pair_idx": pair_idx.cuda(),
				"im_idx": im_idx.cuda(),
				"labels": pred_labels.cuda(),
				"pred_labels": pred_labels.cuda(),
				"scores": scores.cuda(),
				"pred_scores": scores.cuda(),
				"boxes": boxes.cuda()
			}
			
			return temp
