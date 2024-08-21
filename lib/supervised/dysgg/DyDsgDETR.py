import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from constants import Constants as const
from lib.supervised.dysgg.blocks import ObjectClassifierTransformer, PositionalEncoding, EncoderLayer, Encoder
from lib.word_vectors import obj_edge_vectors


class DyDsgDETR(nn.Module):
	
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
		super(DyDsgDETR, self).__init__()
		self.obj_classes = obj_classes
		self.rel_classes = rel_classes
		self.attention_class_num = attention_class_num
		self.spatial_class_num = spatial_class_num
		self.contact_class_num = contact_class_num
		assert mode in (const.SGDET, const.SGCLS, const.PREDCLS)
		self.mode = mode
		
		self.object_classifier = ObjectClassifierTransformer(mode=self.mode, obj_classes=self.obj_classes)
		
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
		self.temporal_transformer = Encoder(global_encoder, num_layers=3)
		# spatial encoder
		local_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
		self.spatial_transformer = Encoder(local_encoder, num_layers=1)
		
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
		# Spatial message passing
		frames = []
		im_indices = entry[const.BOXES][entry[const.PAIR_IDX][:, 1], 0]
		for l in im_indices.unique():
			frames.append(torch.where(im_indices == l)[0])
		frame_features = pad_sequence([rel_features[index] for index in frames], batch_first=True)
		masks = (1 - pad_sequence([torch.ones(len(index)) for index in frames], batch_first=True)).bool()
		rel_ = self.spatial_transformer(frame_features, src_key_padding_mask=masks.cuda())
		rel_features = torch.cat([rel_[i, :len(index)] for i, index in enumerate(frames)])
		
		# Temporal message passing
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
		sequence_features_position_embedded = self.positional_encoder(sequence_features, pos_index)
		future_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
		                              diagonal=0)).type(torch.bool)
		future_mask = future_mask.cuda()
		rel_ = self.temporal_transformer(sequence_features_position_embedded, src_key_padding_mask=masks.cuda(),
		                                 mask=future_mask)
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
