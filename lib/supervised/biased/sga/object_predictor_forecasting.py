"""
Let's get the relationships yo
"""

import pdb

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from lib.supervised.biased.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifier, PositionalEncodingLearn, TransformerDecoderLayer, TransformerDecoder
from lib.word_vectors import obj_edge_vectors


class obj_decoder(nn.Module):
	
	def __init__(self, mode='sgdet', obj_classes=None):
		
		"""
        :param classes: Object classes
        :param mode: (sgcls, predcls, or sgdet)
        """
		super(obj_decoder, self).__init__()
		self.d_model = 2376
		self.obj_classes = obj_classes
		self.mode = mode
		self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes)
		self.PositionalEncodingLearn = PositionalEncodingLearn(self.d_model).cuda()
		# self.enc_layer = nn.TransformerEncoderLayer(d_model=2048,nhead=8,dim_feedforward=512).cuda()
		# self.encoder = nn.TransformerEncoder(self.enc_layer,num_layers = 3).cuda()
		self.decoder_layer = TransformerDecoderLayer(d_model=self.d_model, nhead=4, dim_feedforward=512).cuda()
		self.decoder = TransformerDecoder(self.decoder_layer, num_layers=1).cuda()
		self.query_embed = nn.Embedding(37, self.d_model).cuda()
		self.linear1 = nn.Linear(self.d_model, 512).cuda()
		self.rel = nn.ReLU().cuda()
		self.linear2 = nn.Linear(512, 1).cuda()
		self.linear_box1 = nn.Linear(self.d_model, 512).cuda()
		self.linear_box2 = nn.Linear(512, 4).cuda()
	
	def forward(self, entry, context, future):
		
		entry = self.object_classifier(entry)
		""" ################# changes regarding forecasting #################### """
		start = 0
		error_count = 0
		count = 0
		result = {}
		if (start + context + 1 > len(entry["im_idx"].unique())):
			while (start + context + 1 != len(entry["im_idx"].unique()) and context > 1):
				context -= 1
		
		while (start + context + 1 <= len(entry["im_idx"].unique())):
			res = {}
			future_frame_start_id = entry["im_idx"].unique()[context]
			
			future_frame_end_id = entry["im_idx"].unique()[context + future - 1]
			
			context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
			context_idx = entry["im_idx"][:context_end_idx]
			context_len = context_idx.shape[0]
			
			future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
			future_idx = entry["im_idx"][context_end_idx:future_end_idx]
			future_len = future_idx.shape[0]
			
			# inp = entry["features"][entry["pair_idx"][:len(context_idx)][:,1]]
			ob_idx = entry["pair_idx"][:len(context_idx)][:, 1]
			sub_idx = entry["pair_idx"][:len(context_idx)][:, 0].unique()
			feat_idx, _ = torch.sort(torch.cat((ob_idx, sub_idx)))
			inp = entry["features"][feat_idx]
			inp = inp.unsqueeze(0)
			
			inp = inp.permute(1, 0, 2)
			mem = inp
			self.query = self.query_embed.weight.unsqueeze(1)
			tgt = torch.zeros_like(self.query)
			out = self.decoder(tgt, mem, query_pos=self.query)
			out = out.squeeze(0)
			out_decoder = out.squeeze(1)
			
			out = self.linear2(self.rel(self.linear1(out_decoder)))
			out_box = self.linear_box2(self.rel(self.linear_box1(out_decoder)))
			
			res["decoder"] = out_decoder
			res["object"] = out
			res["box"] = out_box
			result[count] = res
			count += 1
			context += 1
		return result


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
		
		self.object_decoder = obj_decoder(mode=self.mode, obj_classes=self.obj_classes)
		
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
		self.vr_lin1 = nn.Linear(1024, 512)
		self.vr_lin2 = nn.Linear(512, 512)
		self.relu = nn.ReLU()
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
	
	def forward(self, entry, context, future, epoch):
		
		dec_out = self.object_decoder(entry, context, future)
		
		start = 0
		if (start + context + 1 > len(entry["im_idx"].unique())):
			while (start + context + 1 != len(entry["im_idx"].unique()) and context > 1):
				context -= 1
		
		if self.training:
			features = []
			for idx in dec_out.keys():
				frame_num = idx + context
				label_ind = torch.where(entry["boxes"][:, 0] == frame_num)[0]
				for i in label_ind:
					features.append(dec_out[idx]["decoder"][entry["labels"][i]])
			
			features = torch.stack(features).cuda()
			
			pair_start = torch.where(entry["im_idx"] == context)[0][0]
			# visual part
			minus = torch.where(entry["boxes"][:, 0] == context)[0][0]
			subj_rep = features[entry['pair_idx'][pair_start:, 0] - minus]
			subj_rep = self.subj_fc(subj_rep)
			obj = entry["features"][entry['pair_idx'][pair_start:, 0] - minus]
			subj = entry["features"][entry['pair_idx'][pair_start:, 0] - minus]
			with torch.no_grad():
				subj = self.subj_fc(subj)
				obj = self.obj_fc(obj)
			
			vr_temp = torch.cat((subj, obj), dim=1)
			vr_out = self.vr_lin2(self.relu(self.vr_lin1(vr_temp)))
			
			obj_rep = features[entry['pair_idx'][pair_start:, 1] - minus]
			obj_rep = self.obj_fc(obj_rep)
			vr = self.union_func1(entry['union_feat'][pair_start:]) + self.conv(entry['spatial_masks'][pair_start:])
			vr = self.vr_fc(vr.view(-1, 256 * 7 * 7))
			x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
			# semantic part
			subj_class = entry['pred_labels'][entry['pair_idx'][pair_start:, 0]]
			obj_class = entry['pred_labels'][entry['pair_idx'][pair_start:, 1]]
			subj_emb = self.obj_embed(subj_class)
			obj_emb = self.obj_embed2(obj_class)
			x_semantic = torch.cat((subj_emb, obj_emb), 1)
			
			rel_features = torch.cat((x_visual, x_semantic), dim=1)
			
			# Spatial-Temporal Transformer
			# spatial message passing
			frames = []
			im_indices = entry["boxes"][
				entry["pair_idx"][pair_start:, 1], 0]  # im_indices -> centre cordinate of all objects in a video
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
				im_idx, counts = torch.unique(entry["pair_idx"][index][:, 0].view(-1), return_counts=True, sorted=True)
				counts = counts.tolist()
				pos = torch.cat([torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
				pos_index.append(pos)
			
			sequence_features = pad_sequence([rel_features[index] for index in sequences], batch_first=True)
			in_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
			                          diagonal=0)).type(torch.bool)
			# in_mask = (1-torch.ones(sequence_features.shape[1],sequence_features.shape[1])).type(torch.bool)
			in_mask = in_mask.cuda()
			# pdb.set_trace()
			masks = (1 - pad_sequence([torch.ones(len(index)) for index in sequences], batch_first=True)).bool()
			pos_index = pad_sequence(pos_index, batch_first=True) if self.mode == "sgdet" else None
			sequence_features = self.positional_encoder(sequence_features, pos_index)
			# out = torch.zeros(sequence_features.shape)
			seq_len = sequence_features.shape[1]
			mask_input = sequence_features
			out = self.global_transformer(mask_input, src_key_padding_mask=masks.cuda(), mask=in_mask)
			
			rel_ = out
			in_mask = None
			rel_ = rel_.cuda()
			rel_flat = torch.cat([rel[:len(index)] for index, rel in zip(sequences, rel_)])
			rel_ = None
			indices_flat = torch.cat(sequences).unsqueeze(1).repeat(1, rel_features.shape[1])
			assert len(indices_flat) == len(entry["pair_idx"][pair_start:])
			global_output = torch.zeros_like(rel_features).to(rel_features.device)
			global_output.scatter_(0, indices_flat, rel_flat)
			
			entry["attention_distribution"] = self.a_rel_compress(global_output)
			entry["spatial_distribution"] = self.s_rel_compress(global_output)
			entry["contacting_distribution"] = self.c_rel_compress(global_output)
			
			entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
			entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])
			entry['global_output'] = global_output
			
			# pdb.set_trace()
			return dec_out, entry, vr, vr_out
		else:
			count = dec_out.keys()
			length = 0
			pair_idx = []
			im_idx = []
			pred_labels = []
			pred_scores = []
			boxes = []
			out_dict = {}
			thres = 0.01
			if epoch > 2 and epoch <= 5:
				thres = 0.05
			if epoch > 5:
				thres = 0.1
			for c in count:
				obj_prob = torch.sigmoid(dec_out[c]["object"])
				human_score = obj_prob[1][0]
				obj_prob = obj_prob.squeeze(1)
				obj_prob[0] = -0.4
				obj_prob[1] = 0.9
				labels = torch.where(obj_prob >= 0.5)[0]
				pred_labels.append(labels)
				scores = obj_prob[labels]
				scores[0] = human_score
				pred_scores.append(scores)
				pair = []
				for idx, i in enumerate(labels):
					if idx == 0:
						continue
					else:
						pair_idx.append([length, idx + length])
				for i in range(len(labels) - 1):
					im_idx.append(c)
				length += len(labels)
				new_col = torch.full((37, 1), c).cuda()
				box = torch.cat((new_col, dec_out[c]["box"]), dim=1)
				boxes.append(box[labels])
			
			boxes = torch.cat(boxes, dim=0).cuda()
			pred_labels = torch.cat(pred_labels, dim=0).cuda()
			pred_scores = torch.cat(pred_scores, dim=0).cuda()
			pair_idx = torch.tensor(pair_idx).cuda()
			im_idx = torch.tensor(im_idx).cuda()
			
			out_dict["boxes"] = boxes
			out_dict["pred_labels"] = pred_labels
			out_dict["pred_scores"] = pred_scores
			out_dict["pair_idx"] = pair_idx
			out_dict["im_idx"] = im_idx
			
			features = []
			for idx in dec_out.keys():
				frame_num = idx
				label_ind = torch.where(out_dict["boxes"][:, 0] == frame_num)[0]
				for i in label_ind:
					features.append(dec_out[idx]["decoder"][out_dict["pred_labels"][i]])
			
			# features = [list(x) for x in features]
			# features = torch.tensor(features)
			# features = features.cuda()
			f = features
			features = torch.stack(features, dim=0).cuda()
			
			# visual part
			try:
				subj_rep = features[out_dict['pair_idx'][:, 0]]
			except IndexError:
				pdb.set_trace()
			subj_rep = self.subj_fc(subj_rep)
			obj_rep = features[out_dict['pair_idx'][:, 1]]
			obj_rep = self.obj_fc(obj_rep)
			
			vr_temp = torch.cat((subj_rep, obj_rep), dim=1)
			vr = self.vr_lin2(self.relu(self.vr_lin1(vr_temp)))
			
			x_visual = torch.cat((subj_rep, obj_rep, vr), 1)
			# semantic part
			subj_class = out_dict['pred_labels'][out_dict['pair_idx'][:, 0]]
			obj_class = out_dict['pred_labels'][out_dict['pair_idx'][:, 1]]
			subj_emb = self.obj_embed(subj_class)
			obj_emb = self.obj_embed2(obj_class)
			x_semantic = torch.cat((subj_emb, obj_emb), 1)
			
			rel_features = torch.cat((x_visual, x_semantic), dim=1)
			
			# Spatial-Temporal Transformer
			# spatial message passing
			frames = []
			im_indices = out_dict["boxes"][
				out_dict["pair_idx"][:, 1], 0]  # im_indices -> centre cordinate of all objects in a video
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
				im_idx, counts = torch.unique(out_dict["pair_idx"][index][:, 0].view(-1), return_counts=True,
				                              sorted=True)
				counts = counts.tolist()
				pos = torch.cat([torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
				pos_index.append(pos)
			
			sequence_features = pad_sequence([rel_features[index] for index in sequences], batch_first=True)
			in_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
			                          diagonal=0)).type(torch.bool)
			# in_mask = (1-torch.ones(sequence_features.shape[1],sequence_features.shape[1])).type(torch.bool)
			in_mask = in_mask.cuda()
			# pdb.set_trace()
			masks = (1 - pad_sequence([torch.ones(len(index)) for index in sequences], batch_first=True)).bool()
			pos_index = pad_sequence(pos_index, batch_first=True) if self.mode == "sgdet" else None
			sequence_features = self.positional_encoder(sequence_features, pos_index)
			# out = torch.zeros(sequence_features.shape)
			seq_len = sequence_features.shape[1]
			mask_input = sequence_features
			out = self.global_transformer(mask_input, src_key_padding_mask=masks.cuda(), mask=in_mask)
			
			rel_ = out
			in_mask = None
			rel_ = rel_.cuda()
			rel_flat = torch.cat([rel[:len(index)] for index, rel in zip(sequences, rel_)])
			rel_ = None
			indices_flat = torch.cat(sequences).unsqueeze(1).repeat(1, rel_features.shape[1])
			assert len(indices_flat) == len(out_dict["pair_idx"][:])
			global_output = torch.zeros_like(rel_features).to(rel_features.device)
			global_output.scatter_(0, indices_flat, rel_flat)
			
			out_dict["attention_distribution"] = self.a_rel_compress(global_output)
			out_dict["spatial_distribution"] = self.s_rel_compress(global_output)
			out_dict["contacting_distribution"] = self.c_rel_compress(global_output)
			
			out_dict["spatial_distribution"] = torch.sigmoid(out_dict["spatial_distribution"])
			out_dict["contacting_distribution"] = torch.sigmoid(out_dict["contacting_distribution"])
			out_dict['global_output'] = global_output
			
			return dec_out, out_dict
