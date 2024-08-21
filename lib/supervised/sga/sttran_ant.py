import math

import torch.nn as nn

from lib.supervised.sga.base_transformer import BaseTransformer
from lib.supervised.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifierMLP
from lib.word_vectors import obj_edge_vectors

"""
1. ObjectClassifierMLP
2. No Tracking
3. Uses spatial transformer for embeddings
4. Uses temporal transformer for anticipation
"""


class STTranAnt(BaseTransformer):

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
        super(STTranAnt, self).__init__()

        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.d_model = 128
        self.num_features = 1936

        self.object_classifier = ObjectClassifierMLP(mode=self.mode, obj_classes=self.obj_classes)

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
        # self.positional_encoder = PositionalEncoding(d_model, max_len=400)

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

    def forward_single_entry(self, context_fraction, entry):
        """
        Forward method for the baseline
        :param context_fraction:
        :param entry: Dictionary from object classifier
        :return:
        """

        entry, spa_so_rels_feats_tf, obj_seqs_tf = self.generate_spatial_predicate_embeddings(entry)

        result = {}
        count = 0
        num_tf = len(entry["im_idx"].unique())
        num_cf = min(int(math.ceil(context_fraction * num_tf)), num_tf - 1)
        num_ff = num_tf - num_cf

        result[count] = self.generate_future_ff_rels_for_context(entry, spa_so_rels_feats_tf, obj_seqs_tf, num_cf,
                                                                 num_tf, num_ff)
        entry["output"] = result
        entry["global_output"] = spa_so_rels_feats_tf
        return entry

    def forward(self, entry, num_cf, num_ff):
        """
        # -------------------------------------------------------------------------------------------------------------
        # Anticipation Module
        # -------------------------------------------------------------------------------------------------------------
        # 1. This section maintains starts from a set context predicts the future relations corresponding to the last
        # frame in the context
        # 2. Then it moves the context by one frame and predicts the future relations corresponding to the
        # last frame in the new context
        # 3. This is repeated until the end of the video, loss is calculated for each
        # future relation prediction and the loss is back-propagated
        Forward method for the baseline
        :param entry: Dictionary from object classifier
        :param num_cf: Frame idx for context
        :param num_ff: Number of next frames to anticipate
        :return:
        """

        entry, spa_so_rels_feats_tf, obj_seqs_tf = self.generate_spatial_predicate_embeddings(entry)

        count = 0
        result = {}
        num_tf = len(entry["im_idx"].unique())
        num_cf = min(num_cf, num_tf - 1)
        while num_cf + 1 <= num_tf:
            num_ff = min(num_ff, num_tf - num_cf)
            result[count] = self.generate_future_ff_rels_for_context(entry, spa_so_rels_feats_tf, obj_seqs_tf,
                                                                     num_cf, num_tf, num_ff)
            count += 1
            num_cf += 1

        entry["output"] = result
        entry["global_output"] = spa_so_rels_feats_tf
        return entry
