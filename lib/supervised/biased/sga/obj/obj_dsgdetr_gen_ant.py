from lib.supervised.biased.sga.obj.obj_base_transformer import ObjBaseTransformer


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
