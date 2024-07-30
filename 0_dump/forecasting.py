import pdb

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from lib.supervised.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifier
from lib.word_vectors import obj_edge_vectors


class STTran(nn.Module):
	
	def __init__(self, mode='sgdet',
	             attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None,
	             rel_classes=None,
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
		self.d_model = 128
		self.num_features = 1936
		
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
		embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='data', wv_dim=200)
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
	
	def forward(self, entry, context, future):
		
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
		frames = []
		im_indices = entry["boxes"][
			entry["pair_idx"][:, 1], 0]  # im_indices -> centre cordinate of all objects in a video
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
		
		# for index in sequences:
		#     im_idx, counts = torch.unique(entry["pair_idx"][index][:,0].view(-1), return_counts=True, sorted=True)
		#     counts = counts.tolist()
		#     pos = torch.cat([torch.LongTensor([im]*count) for im, count in zip(range(len(counts)), counts)])
		#     pos_index.append(pos)
		
		""" ################# changes regarding forecasting #################### """
		start = 0
		# context = context
		# future = future
		error_count = 0
		count = 0
		result = {}
		if start + context + 1 > len(entry["im_idx"].unique()):
			while start + context + 1 != len(entry["im_idx"].unique()) and context > 1:
				context -= 1
			future = 1
		if (start + context + future > len(entry["im_idx"].unique()) and start + context < len(
				entry["im_idx"].unique())):
			future = len(entry["im_idx"].unique()) - (start + context)
		
		while start + context + 1 <= len(entry["im_idx"].unique()):
			future_frame_start_id = entry["im_idx"].unique()[context]
			
			if (start + context + future > len(entry["im_idx"].unique()) and start + context < len(
					entry["im_idx"].unique())):
				future = len(entry["im_idx"].unique()) - (start + context)
			
			future_frame_end_id = entry["im_idx"].unique()[context + future - 1]
			
			context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
			context_idx = entry["im_idx"][:context_end_idx]
			context_len = context_idx.shape[0]
			
			future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
			future_idx = entry["im_idx"][context_end_idx:future_end_idx]
			future_len = future_idx.shape[0]
			
			seq = []
			
			for s in sequences:
				index = s[(s < (context_len))]
				seq.append(index)
			
			future_seq = []
			for s in sequences:
				index = s[(s >= context_len) & (s < (context_len + future_len))]
				future_seq.append(index)
			
			pos_index = []
			for index in seq:
				im_idx, counts = torch.unique(entry["pair_idx"][index][:, 0].view(-1), return_counts=True, sorted=True)
				counts = counts.tolist()
				if im_idx.numel() == 0:
					pos = torch.tensor(
						[torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
				else:
					pos = torch.cat([torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
				pos_index.append(pos)
			
			# pdb.set_trace()
			sequence_features = pad_sequence([rel_features[index] for index in seq], batch_first=True)
			in_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
			                          diagonal=0)).type(torch.bool)
			in_mask = in_mask.cuda()
			masks = (1 - pad_sequence([torch.ones(len(index)) for index in seq], batch_first=True)).bool()
			
			pos_index = [torch.tensor(seq, dtype=torch.long) for seq in pos_index]
			# pos_index = pos_index.clone().detach
			pos_index = pad_sequence(pos_index, batch_first=True) if self.mode == "sgdet" else None
			sequence_features = self.positional_encoder(sequence_features, pos_index)
			seq_len = sequence_features.shape[1]
			mask_input = sequence_features
			
			output = []
			
			for i in range(future):
				# out = self.global_transformer(mask_input,src_key_padding_mask = masks.cuda(),mask = in_mask)
				out = self.global_transformer(mask_input, mask=in_mask)
				output.append(out[:, -1, :].unsqueeze(1))
				out_last = [out[:, -1, :]]
				pred = torch.stack(out_last, dim=1)
				mask_input = torch.cat([mask_input, pred], 1)
				in_mask = (1 - torch.tril(torch.ones(mask_input.shape[1], mask_input.shape[1]), diagonal=0)).type(
					torch.bool)
				in_mask = in_mask.cuda()
			
			output = torch.cat(output, dim=1)
			rel_ = output
			rel_ = rel_.cuda()
			rel_flat1 = []
			
			for index, rel in zip(future_seq, rel_):
				if len(index) == 0:
					continue
				for i in range(len(index)):
					rel_flat1.extend(rel)
			
			rel_flat1 = [tensor.tolist() for tensor in rel_flat1]
			rel_flat = torch.tensor(rel_flat1)
			rel_flat = rel_flat.to('cuda:0')
			# rel_flat = torch.cat([rel[:len(index)] for index, rel in zip(future_seq,rel_)])
			indices_flat = torch.cat(future_seq).unsqueeze(1).repeat(1, rel_features.shape[1])
			# assert len(indices_flat) == len(entry["pair_idx"])
			
			global_output = torch.zeros_like(rel_features).to(rel_features.device)
			try:
				global_output.scatter_(0, indices_flat, rel_flat)
			except RuntimeError:
				error_count += 1
				print("global_scatter : ", error_count)
				pdb.set_trace()
			
			gb_output = global_output[context_len:context_len + future_len]
			context += 1
			
			temp = {}
			temp["attention_distribution"] = self.a_rel_compress(gb_output)
			spatial_distribution = self.s_rel_compress(gb_output)
			contacting_distribution = self.c_rel_compress(gb_output)
			
			temp["spatial_distribution"] = torch.sigmoid(spatial_distribution)
			temp["contacting_distribution"] = torch.sigmoid(contacting_distribution)
			temp["global_output"] = gb_output
			temp["original"] = global_output
			
			result[count] = temp
			count += 1
			if (start + context + future > len(entry["im_idx"].unique())):
				break
		
		entry["output"] = result
		
		# pdb.set_trace()
		return entry
