import math

from torch import nn

from lib.supervised.biased.sga.blocks import ObjectClassifierMLP, EncoderLayer, Encoder, PositionalEncoding, \
    ObjectAnticipation
from lib.supervised.biased.sga.obj.obj_base_transformer import ObjBaseTransformer
from lib.word_vectors import obj_edge_vectors

"""
1. ObjectClassifierMLP
2. No Tracking
3. Uses spatial transformer for embeddings
4. Uses temporal transformer for anticipation
5. Uses object anticipation decoder for generating future embeddings
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
        self.obj_anti_temporal_transformer = ObjectAnticipation(mode=self.mode, obj_classes=self.obj_classes)

        self.subj_fc = nn.Linear(2048, 512)
        self.obj_fc = nn.Linear(2048, 512)

        self.vr_fc = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
        )

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
        entry = self.object_classifier(entry)
        count = 0
        result_ff = {}
        result_cf_ff = {}
        num_tf = len(entry["im_idx"].unique())
        num_cf = min(num_cf, num_tf - 1)
        while num_cf + 1 <= num_tf:
            num_ff = min(num_ff, num_tf - num_cf)
            entry_cf_ff, entry_ff = self.obj_anti_temporal_transformer(entry, num_cf, num_tf, num_ff)
            entry_cf_ff = self.generate_future_ff_rels_for_context(entry, entry_cf_ff, num_cf, num_tf, num_ff)
            result_cf_ff[count] = entry_cf_ff
            result_ff[count] = entry_ff
            count += 1
            num_cf += 1
        entry["output_cf_ff"] = result_cf_ff
        entry["output_ff"] = result_ff
        return entry

    def forward_single_entry(self, context_fraction, entry):
        entry = self.object_classifier(entry)
        result_cf_ff = {}
        result_ff = {}
        count = 0
        num_tf = len(entry["im_idx"].unique())
        num_cf = min(int(math.ceil(context_fraction * num_tf)), num_tf - 1)
        num_ff = num_tf - num_cf
        entry_cf_ff, entry_ff = self.obj_anti_temporal_transformer(entry, num_cf, num_tf, num_ff)
        entry_cf_ff = self.generate_future_ff_rels_for_context(entry, entry_cf_ff, num_cf, num_tf, num_ff)
        result_cf_ff[count] = entry_cf_ff
        result_ff[count] = entry_ff
        entry["output_cf_ff"] = result_cf_ff
        entry["output_ff"] = result_ff
        return entry
