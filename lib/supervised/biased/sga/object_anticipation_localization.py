import torch
import torch.nn as nn

from lib.supervised.biased.sga.blocks import ObjectClassifier, \
	PositionalEncodingLearn, TransformerDecoderLayer, TransformerDecoder


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
		# 1. Construct Tracklet for each context
		
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
			# 1. Iteration - Tracklet for each context
			# a. Object distribution
			# b. New Tracklet for new object - Noisy bbox - Predicted bbox + Latent + Predicted Category
			# c. Old Tracklet - Updated latent + Predicted BBOX
			# 2. For next iteration -
			# a. Object distribution
			# b. Update weighted sum for latent
			# c. Update all tracklets with new objects
			
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