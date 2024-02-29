from torch import nn

from lib.supervised.biased.sga.blocks import ObjectClassifierMLP, EncoderLayer, Encoder, PositionalEncoding, ObjectAnt
from lib.supervised.biased.sga.obj.obj_base_transformer import ObjBaseTransformer
from lib.word_vectors import obj_edge_vectors

"""
1. ObjectClassifierTransformer
2. Enabled Tracking for past sequences
3. Uses spatial transformer for generating embeddings
4. Uses temporal transformer for anticipation
"""


class ObjSTTranAnt(ObjBaseTransformer):
	
	def __init__(self,
	             mode='sgdet',
	             attention_class_num=None,
	             spatial_class_num=None,
	             contact_class_num=None,
	             obj_classes=None,
	             rel_classes=None,
	             enc_layer_num=None,
	             dec_layer_num=None
	             ):
		super(ObjSTTranAnt, self).__init__()
		self.obj_classes = obj_classes
		self.rel_classes = rel_classes
		self.attention_class_num = attention_class_num
		self.spatial_class_num = spatial_class_num
		self.contact_class_num = contact_class_num
		assert mode in ('sgdet', 'sgcls', 'predcls')
		self.mode = mode
		self.num_features = 1936
		
		self.object_classifier = ObjectClassifierMLP(mode=self.mode, obj_classes=self.obj_classes)
		
		self.obj_anti_positional_encoder = PositionalEncoding(2376, 0.1, 600 if mode == "sgdet" else 400)
		self.obj_anti_temporal_transformer = ObjectAnt(mode=self.mode, obj_classes=self.obj_classes)
		
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
		
		# spatial encoder
		spatial_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
		self.spatial_transformer = Encoder(spatial_encoder, num_layers=1)
		
		# Anticipation Positional Encoding
		self.anti_positional_encoder = PositionalEncoding(d_model, max_len=400)
		
		# temporal encoder
		temporal_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
		self.anti_temporal_transformer = Encoder(temporal_encoder, num_layers=3)
		
		self.a_rel_compress = nn.Linear(d_model, self.attention_class_num)
		self.s_rel_compress = nn.Linear(d_model, self.spatial_class_num)
		self.c_rel_compress = nn.Linear(d_model, self.contact_class_num)
	
	def forward(self, entry, num_cf, num_ff):
		"""
		1. Pass the entry through object classification layer.
		2. Generate representations for objects in the future frames
		3. Construct complete entry for these items and add them as values to the keys in the entry
		4. Generate spatial embeddings for all frames
		5. Generate spatial embeddings for objects in future frames
		6. Augment the context spatial embeddings and future frame spatial embeddings
		7. Generate temporal embeddings for relations in future frames
		8. Pass them through linear layers to get the final output and maintaining same code for losses.
		:param entry:
		:param num_cf:
		:param num_ff:
		:return:
		"""
		
		entry = self.object_classifier(entry)
		
		# -----------------------------------------------------------------------------
		# Generate representations for objects in the future frames
		
		count = 0
		result = {}
		num_tf = len(entry["im_idx"].unique())
		num_cf = min(num_cf, num_tf - 1)
		while num_cf + 1 <= num_tf:
			num_ff = min(num_ff, num_tf - num_cf)
			entry_cf_ff = self.generate_future_ff_obj_for_context(entry, num_cf, num_tf, num_ff)
			entry_cf_ff = self.generate_future_ff_rels_for_context(entry, entry_cf_ff, num_cf, num_tf, num_ff)
			result[count] = entry_cf_ff
			count += 1
			num_cf += 1
		
		entry["output"] = result
		return entry
	
	def forward_single_entry(self, context_fraction, entry):
		pass
