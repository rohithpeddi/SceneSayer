import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from lib.supervised.sga.base_transformer import BaseTransformer
from lib.supervised.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifierTransformer
from lib.word_vectors import obj_edge_vectors

"""
1. ObjectClassifierTransformer
2. Enabled Tracking for past sequences
3. Uses spatial temporal transformer for generating embeddings
4. Uses temporal transformer for anticipation
"""


class DsgDetrGenAnt(BaseTransformer):

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
        super(DsgDetrGenAnt, self).__init__()

        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.d_model = 128
        self.num_features = 1936

        self.object_classifier = ObjectClassifierTransformer(mode=self.mode, obj_classes=self.obj_classes)

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

        # Spatial encoder
        spatial_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.spatial_transformer = Encoder(spatial_encoder, num_layers=1)

        # Generation Positional Encoding
        self.gen_positional_encoder = PositionalEncoding(d_model, max_len=400)

        # Generation temporal encoder
        gen_temporal_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.gen_temporal_transformer = Encoder(gen_temporal_encoder, num_layers=3)

        # Anticipation Positional Encoding
        self.anti_positional_encoder = PositionalEncoding(d_model, max_len=400)

        # Anticipation temporal encoder
        anti_temporal_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.anti_temporal_transformer = Encoder(anti_temporal_encoder, num_layers=1)

        self.a_rel_compress = nn.Linear(d_model, self.attention_class_num)
        self.s_rel_compress = nn.Linear(d_model, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(d_model, self.contact_class_num)

        self.gen_a_rel_compress = nn.Linear(d_model, self.attention_class_num)
        self.gen_s_rel_compress = nn.Linear(d_model, self.spatial_class_num)
        self.gen_c_rel_compress = nn.Linear(d_model, self.contact_class_num)

    def generate_spatio_temporal_predicate_embeddings(self, entry):
        entry, spa_rels_feats_tf, sequences = self.generate_spatial_predicate_embeddings(entry)
        # Temporal message passing
        sequence_features = pad_sequence([spa_rels_feats_tf[index] for index in sequences], batch_first=True)
        causal_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
                                      diagonal=0)).type(torch.bool).cuda()
        padding_mask = (
                    1 - pad_sequence([torch.ones(len(index)) for index in sequences], batch_first=True)).bool().cuda()
        positional_encoding = self.fetch_positional_encoding_for_gen_obj_seqs(sequences, entry)
        rel_ = self.gen_temporal_transformer(self.gen_positional_encoder(sequence_features, positional_encoding),
                                             src_key_padding_mask=padding_mask, mask=causal_mask)

        rel_flat = torch.cat([rel[:len(index)] for index, rel in zip(sequences, rel_)])
        indices_flat = torch.cat(sequences).unsqueeze(1).repeat(1, spa_rels_feats_tf.shape[1])

        assert len(indices_flat) == len(entry["pair_idx"])
        spa_temp_rels_feats_tf = torch.zeros_like(spa_rels_feats_tf).to(spa_rels_feats_tf.device)
        spa_temp_rels_feats_tf.scatter_(0, indices_flat, rel_flat)

        return entry, spa_rels_feats_tf, sequences, spa_temp_rels_feats_tf

    def forward_single_entry(self, context_fraction, entry):
        """
        Forward method for the baseline
        :param context_fraction:
        :param entry: Dictionary from object classifier
        :return:
        """
        (entry, spa_so_rels_feats_tf,
         obj_seqs_tf, spa_temp_so_rels_feats_tf) = self.generate_spatio_temporal_predicate_embeddings(entry)

        result = {}
        count = 0
        num_tf = len(entry["im_idx"].unique())
        num_cf = min(int(math.ceil(context_fraction * num_tf)), num_tf - 1)
        num_ff = num_tf - num_cf

        result[count] = self.generate_future_ff_rels_for_context(entry, spa_temp_so_rels_feats_tf,
                                                                 obj_seqs_tf, num_cf, num_tf, num_ff)
        entry["output"] = result
        entry["global_output"] = spa_temp_so_rels_feats_tf
        entry["gen_attention_distribution"] = self.gen_a_rel_compress(spa_temp_so_rels_feats_tf)
        entry["gen_spatial_distribution"] = torch.sigmoid(self.gen_s_rel_compress(spa_temp_so_rels_feats_tf))
        entry["gen_contacting_distribution"] = torch.sigmoid(self.gen_c_rel_compress(spa_temp_so_rels_feats_tf))

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
        (entry, spa_so_rels_feats_tf, obj_seqs_tf,
         spa_temp_so_rels_feats_tf) = self.generate_spatio_temporal_predicate_embeddings(entry)

        count = 0
        result = {}
        num_tf = len(entry["im_idx"].unique())
        num_cf = min(num_cf, num_tf - 1)
        while num_cf + 1 <= num_tf:
            num_ff = min(num_ff, num_tf - num_cf)
            result[count] = self.generate_future_ff_rels_for_context(entry, spa_temp_so_rels_feats_tf, obj_seqs_tf,
                                                                     num_cf, num_tf, num_ff)
            count += 1
            num_cf += 1
        entry["gen_attention_distribution"] = self.gen_a_rel_compress(spa_temp_so_rels_feats_tf)
        entry["gen_spatial_distribution"] = torch.sigmoid(self.gen_s_rel_compress(spa_temp_so_rels_feats_tf))
        entry["gen_contacting_distribution"] = torch.sigmoid(self.gen_c_rel_compress(spa_temp_so_rels_feats_tf))
        entry["global_output"] = spa_temp_so_rels_feats_tf
        entry["output"] = result

        return entry
