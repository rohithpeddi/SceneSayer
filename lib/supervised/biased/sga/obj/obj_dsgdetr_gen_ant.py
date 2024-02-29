from lib.supervised.biased.sga.obj.obj_base_transformer import ObjBaseTransformer

"""
1. ObjectClassifierTransformer
2. Enabled Tracking for past sequences
3. Uses spatial temporal transformer for generating embeddings
4. Uses temporal transformer for anticipation
5. Also use generation transformer for classification of current inputs
6. Uses object anticipation decoder for generating future embeddings
"""


class ObjDsgDetrGenAnt(ObjBaseTransformer):
	
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
		super(ObjDsgDetrGenAnt, self).__init__()
		pass
	
	def forward(self, entry, num_cf, num_ff):
		pass
	
	def forward_single_entry(self, context_fraction, entry):
		pass
