"""
Let's get the relationships yo
"""

import sys

sys.path.insert(0, '/home/maths/btech/mt1200841/scratch/NeSysVideoPrediction')

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from lib.word_vectors import obj_edge_vectors
from torchsde import sdeint_adjoint as sdeint
from lib.supervised.biased.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifier, GetBoxes


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
		self.num_features = 1936
		
		self.object_classifier = ObjectClassifier(mode=self.mode, obj_classes=self.obj_classes)
		
		self.get_subj_boxes = GetBoxes(1936)
		self.get_obj_boxes = GetBoxes(1936)
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
	
	def forward(self, entry):
		
		entry = self.object_classifier(entry)
		# visual part
		subj_rep = entry['features'][entry['pair_idx'][:, 0]]
		subj_rep = self.subj_fc(subj_rep)
		entry["subj_rep_actual"] = subj_rep
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
		# subj_dec = self.subject_decoder(rel_features, src_key_padding_mask=masks.cuda())
		# subj_dec = subj_compress(subj_dec)
		# entry["subj_rep_decoded"] = subj_dec
		# entry["spatial_encoder_out"] = rel_features
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
		in_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]), diagonal=0)).type(
			torch.bool)
		# in_mask = (1-torch.ones(sequence_features.shape[1],sequence_features.shape[1])).type(torch.bool)
		in_mask = in_mask.cuda()
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
		assert len(indices_flat) == len(entry["pair_idx"])
		global_output = torch.zeros_like(rel_features).to(rel_features.device)
		global_output.scatter_(0, indices_flat, rel_flat)
		entry["attention_distribution"] = self.a_rel_compress(global_output)
		entry["spatial_distribution"] = self.s_rel_compress(global_output)
		entry["contacting_distribution"] = self.c_rel_compress(global_output)
		
		entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
		entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"])
		# detached_outputs = global_output.clone().detach()
		entry["subject_boxes_dsg"] = self.get_subj_boxes(global_output)
		# entry["object_boxes_dsg"] = self.get_obj_boxes(global_output)
		
		pair_idx = entry["pair_idx"]
		boxes_rcnn = entry["boxes"]
		entry["global_output"] = global_output
		# entry["detached_outputs"] = detached_outputs
		entry["subject_boxes_rcnn"] = boxes_rcnn[pair_idx[:, 0], 1:].to(global_output.device)
		# entry["object_boxes_rcnn"] = boxes_rcnn[pair_idx[ :, 1], 1 : ].to(global_output.device)
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
	
	def __init__(self, mode, attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None,
	             rel_classes=None,
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
	
	def forward(self, entry, testing=False):
		entry = self.dsgdetr(entry)
		obj = entry["pair_idx"][:, 1]
		if not testing:
			labels_obj = entry["labels"][obj]
		else:
			pred_labels_obj = entry["pred_labels"][obj]
			labels_obj = entry["labels"][obj]
		# pdb.set_trace()
		im_idx = entry["im_idx"]
		pair_idx = entry["pair_idx"]
		tot = im_idx.size(0)
		t = torch.tensor(entry["frame_idx"], dtype=torch.int32)
		indices = torch.reshape((im_idx[: -1] != im_idx[1:]).nonzero(), (-1,)) + 1
		curr_id = 0
		t_unique = torch.unique(t).float()
		n = torch.unique(im_idx).size(0)
		t_extend = torch.Tensor([t_unique[-1] + i + 1 for i in range(self.max_window)])
		global_output = entry["global_output"]
		t_unique = torch.cat((t_unique, t_extend)).to(device=global_output.device)
		anticipated_vals = torch.zeros(self.max_window, 0, self.d_model, device=global_output.device)
		# obj_bounding_boxes = torch.zeros(self.max_window, indices[-1], 4, device=global_output.device)
		rng = torch.cat(
			(torch.Tensor([0]).to(device=indices.device), indices, torch.Tensor([tot]).to(device=indices.device)))
		rng = rng.long()
		# self.gen_num[self.ctr] = tot
		for i in range(1, self.max_window + 1):
			mask_curr = torch.tensor([], dtype=torch.long, device=rng.device)
			mask_gt = torch.tensor([], dtype=torch.long, device=rng.device)
			gt = entry["gt_annotation"].copy()
			for j in range(n - i):
				if testing:
					a, b = np.array(pred_labels_obj[rng[j]: rng[j + 1]].cpu()), np.array(
						labels_obj[rng[j + i]: rng[j + i + 1]].cpu())
				else:
					a, b = np.array(labels_obj[rng[j]: rng[j + 1]].cpu()), np.array(
						labels_obj[rng[j + i]: rng[j + i + 1]].cpu())
				intersection = np.intersect1d(a, b, return_indices=False)
				ind1 = np.array([])
				ind2 = np.array([])
				for element in intersection:
					tmp1, tmp2 = np.where(a == element)[0], np.where(b == element)[0]
					mn = min(tmp1.shape[0], tmp2.shape[0])
					ind1 = np.concatenate((ind1, tmp1[: mn]))
					ind2 = np.concatenate((ind2, tmp2[: mn]))
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
			entry["mask_curr_%d" % i] = mask_curr
			entry["mask_gt_%d" % i] = mask_gt
			# self.ant_num[i - 1, self.ctr] = mask_curr.size(0)
			if testing:
				pair_idx_test = pair_idx[mask_curr]
				_, inverse_indices = torch.unique(pair_idx_test, sorted=True, return_inverse=True)
				entry["im_idx_test_%d" % i] = im_idx[mask_curr]
				entry["pair_idx_test_%d" % i] = inverse_indices
				if self.mode == "predcls":
					entry["scores_test_%d" % i] = entry["scores"][_.long()]
					entry["labels_test_%d" % i] = entry["labels"][_.long()]
				else:
					entry["pred_scores_test_%d" % i] = entry["pred_scores"][_.long()]
					entry["pred_labels_test_%d" % i] = entry["pred_labels"][_.long()]
				if inverse_indices.size(0) != 0:
					mx = torch.max(inverse_indices)
				else:
					mx = -1
				boxes_test = torch.zeros(mx + 1, 5, device=entry["boxes"].device)
				boxes_test[torch.unique_consecutive(inverse_indices[:, 0])] = entry["boxes"][
					torch.unique_consecutive(pair_idx[mask_gt][:, 0])]
				boxes_test[inverse_indices[:, 1]] = entry["boxes"][pair_idx[mask_gt][:, 1]]
				entry["boxes_test_%d" % i] = boxes_test
				# entry["boxes_test_%d" %i] = entry["boxes"][_.long()]
				entry["gt_annotation_%d" % i] = gt
		# self.ctr += 1
		# anticipated_latent_loss = 0
		# targets = entry["detached_outputs"]
		for i in range(n - 1):
			end = indices[i]
			batch_y0 = global_output[curr_id: end]
			batch_times = t_unique[i: i + self.max_window + 1]
			ret = sdeint(self.diff_func, batch_y0, batch_times, method='reversible_heun',
			             adjoint_method='adjoint_reversible_heun', dt=1)[1:]
			anticipated_vals = torch.cat((anticipated_vals, ret), dim=1)
			# obj_bounding_boxes[ :, curr_id : end, : ].data.copy_(self.dsgdetr.get_obj_boxes(ret))
			curr_id = end
		# for p in self.dsgdetr.get_subj_boxes.parameters():
		#    p.requires_grad_(False)
		entry["anticipated_subject_boxes"] = self.dsgdetr.get_subj_boxes(anticipated_vals)
		# for p in self.dsgdetr.get_subj_boxes.parameters():
		#    p.requires_grad_(True)
		entry["anticipated_vals"] = anticipated_vals
		entry["anticipated_attention_distribution"] = self.dsgdetr.a_rel_compress(anticipated_vals)
		entry["anticipated_spatial_distribution"] = torch.sigmoid(self.dsgdetr.s_rel_compress(anticipated_vals))
		entry["anticipated_contacting_distribution"] = torch.sigmoid(self.dsgdetr.c_rel_compress(anticipated_vals))
		# entry["anticipated_object_boxes"] = obj_bounding_boxes
		return entry
