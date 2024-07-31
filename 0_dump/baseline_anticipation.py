import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from lib.supervised.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifierMLP
from lib.word_vectors import obj_edge_vectors


class BaselineWithAnticipation(nn.Module):

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
        super(BaselineWithAnticipation, self).__init__()

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
        self.positional_encoder = PositionalEncoding(d_model, max_len=400)

        # spatial encoder
        spatial_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.spatial_transformer = Encoder(spatial_encoder, num_layers=1)

        # temporal encoder
        temporal_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.anticipation_temporal_transformer = Encoder(temporal_encoder, num_layers=3)

        self.a_rel_compress = nn.Linear(d_model, self.attention_class_num)
        self.s_rel_compress = nn.Linear(d_model, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(d_model, self.contact_class_num)

    def process_entry(self, entry):
        entry = self.object_classifier(entry)

        # visual part
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
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
        # im_indices -> centre coordinate of all objects in a video
        frames = []
        im_indices = entry["boxes"][entry["pair_idx"][:, 1], 0]
        for l in im_indices.unique():
            frames.append(torch.where(im_indices == l)[0])
        frame_features = pad_sequence([rel_features[index] for index in frames], batch_first=True)
        masks = (1 - pad_sequence([torch.ones(len(index)) for index in frames], batch_first=True)).bool()
        rel_ = self.spatial_transformer(frame_features, src_key_padding_mask=masks.cuda())
        rel_features = torch.cat([rel_[i, :len(index)] for i, index in enumerate(frames)])
        # temporal message passing
        sequences = []
        for l in obj_class.unique():
            k = torch.where(obj_class.view(-1) == l)[0]
            if len(k) > 0:
                sequences.append(k)

        return entry, rel_features, sequences

    def fetch_initial_positional_encoding(self, obj_seqs_cf, entry):
        positional_encoding = []
        for obj_seq_cf in obj_seqs_cf:
            im_idx, counts = torch.unique(entry["pair_idx"][obj_seq_cf][:, 0].view(-1), return_counts=True,
                                          sorted=True)
            counts = counts.tolist()
            pos = torch.cat([torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
            positional_encoding.append(pos)

        positional_encoding = [torch.tensor(seq, dtype=torch.long) for seq in positional_encoding]
        positional_encoding = pad_sequence(positional_encoding, batch_first=True) if self.mode == "sgdet" else None
        return positional_encoding

    def update_positional_encoding(self, positional_encoding):
        if positional_encoding is not None:
            max_values = torch.max(positional_encoding, dim=1)[0] + 1
            max_values = max_values.unsqueeze(1)
            positional_encoding = torch.cat((positional_encoding, max_values), dim=1)
        return positional_encoding

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
        :param entry: Dictionary from object classifier
        :param num_cf: Frame idx for context
        :param num_ff: Number of next frames to anticipate
        :return:
        """
        entry, so_rels_feats_tf, obj_seqs_tf = self.process_entry(entry)

        count = 0
        result = {}
        num_tf = len(entry["im_idx"].unique())
        num_cf = min(num_cf, num_tf - 1)

        while num_cf + 1 <= num_tf:
            num_ff = min(num_ff, num_tf - num_cf)
            ff_start_id = entry["im_idx"].unique()[num_cf]
            ff_end_id = entry["im_idx"].unique()[num_cf + num_ff - 1]

            objects_ff_start_id = int(torch.where(entry["im_idx"] == ff_start_id)[0][0])
            objects_cf = entry["im_idx"][:objects_ff_start_id]
            num_objects_cf = objects_cf.shape[0]

            objects_ff_end_id = int(torch.where(entry["im_idx"] == ff_end_id)[0][-1]) + 1
            objects_ff = entry["im_idx"][objects_ff_start_id:objects_ff_end_id]
            num_objects_ff = objects_ff.shape[0]

            obj_seqs_cf = []
            obj_seqs_ff = []
            for i, obj_seq_tf in enumerate(obj_seqs_tf):
                obj_seq_cf = obj_seq_tf[(obj_seq_tf < num_objects_cf)]
                obj_seq_ff = obj_seq_tf[
                    (obj_seq_tf >= num_objects_cf) & (obj_seq_tf < (num_objects_cf + num_objects_ff))]
                if len(obj_seq_cf) != 0:
                    obj_seqs_cf.append(obj_seq_cf)
                    obj_seqs_ff.append(obj_seq_ff)

            so_seqs_feats_cf = pad_sequence([so_rels_feats_tf[obj_seq_cf] for obj_seq_cf in obj_seqs_cf],
                                            batch_first=True)
            causal_mask = (1 - torch.tril(torch.ones(so_seqs_feats_cf.shape[1], so_seqs_feats_cf.shape[1]),
                                          diagonal=0)).type(torch.bool)
            causal_mask = causal_mask.cuda()
            masks = (1 - pad_sequence([torch.ones(len(index)) for index in obj_seqs_cf], batch_first=True)).bool()

            positional_encoding = self.fetch_initial_positional_encoding(obj_seqs_cf, entry)
            so_seqs_feats_cf = self.positional_encoder(so_seqs_feats_cf, positional_encoding)

            so_seqs_feats_ff = []
            for i in range(num_ff):
                so_seqs_feats_cf_temp_attd = self.anticipation_temporal_transformer(
                    so_seqs_feats_cf,
                    src_key_padding_mask=masks.cuda(),
                    mask=causal_mask
                )
                if i == 0:
                    mask2 = (~masks).int()
                    ind_col = torch.sum(mask2, dim=1) - 1
                    so_feats_nf = []
                    for j, ind in enumerate(ind_col):
                        so_feats_nf.append(so_seqs_feats_cf_temp_attd[j, ind, :])
                    so_feats_nf = torch.stack(so_feats_nf).unsqueeze(1)
                    so_seqs_feats_ff.append(so_feats_nf)
                    so_seqs_feats_cf = torch.cat([so_seqs_feats_cf, so_feats_nf], 1)
                else:
                    # TODO: Check for compression
                    so_seqs_feats_ff.append(so_seqs_feats_cf_temp_attd[:, -1, :].unsqueeze(1))
                    so_feats_nf = torch.stack([so_seqs_feats_cf_temp_attd[:, -1, :]], dim=1)
                    so_seqs_feats_cf = torch.cat([so_seqs_feats_cf, so_feats_nf], 1)

                positional_encoding = self.update_positional_encoding(positional_encoding)
                so_seqs_feats_cf = self.positional_encoder(so_seqs_feats_cf, positional_encoding)
                causal_mask = (1 - torch.tril(torch.ones(so_seqs_feats_cf.shape[1], so_seqs_feats_cf.shape[1]),
                                              diagonal=0)).type(torch.bool)
                causal_mask = causal_mask.cuda()
                masks = torch.cat([masks, torch.full((masks.shape[0], 1), False, dtype=torch.bool)], dim=1)

            so_rels_feats_ff = torch.cat(so_seqs_feats_ff, dim=1).cuda()
            updated_so_rels_feats_ff = []

            if self.training:
                for index, rel in zip(obj_seqs_ff, so_rels_feats_ff):
                    if len(index) == 0:
                        continue
                    updated_so_rels_feats_ff.extend(rel[:len(index)])
            else:
                for index, rel in zip(obj_seqs_ff, so_rels_feats_ff):
                    if len(index) == 0:
                        continue

                    ob_frame_idx = entry["im_idx"][index]
                    rel_temp = torch.zeros(len(index), rel.shape[1])
                    # For each frame in ob_frame_idx, if the value repeats then add the relation value of previous frame
                    k = 0  # index for rel
                    for i, frame in enumerate(ob_frame_idx):
                        if i == 0:
                            rel_temp[i] = rel[k]
                            k += 1
                        elif frame == ob_frame_idx[i - 1]:
                            rel_temp[i] = rel_temp[i - 1]
                        else:
                            rel_temp[i] = rel[k]
                            k += 1

                    updated_so_rels_feats_ff.extend(rel_temp)

            so_rels_feats_ff_flat = torch.tensor([tensor.tolist() for tensor in updated_so_rels_feats_ff]).cuda()
            obj_seqs_ff_flat = torch.cat(obj_seqs_ff).unsqueeze(1).repeat(1, so_rels_feats_tf.shape[1])
            aligned_so_rels_feats_tf = torch.zeros_like(so_rels_feats_tf).to(so_rels_feats_tf.device)

            temp = {"scatter_flag": 0}
            try:
                aligned_so_rels_feats_tf.scatter_(0, obj_seqs_ff_flat, so_rels_feats_ff_flat)
            except RuntimeError:
                num_cf += 1
                temp["scatter_flag"] = 1
                result[count] = temp
                count += 1
                continue

            aligned_so_rels_feats_ff = aligned_so_rels_feats_tf[num_objects_cf:num_objects_cf + num_objects_ff]
            num_cf += 1
            temp["attention_distribution"] = self.a_rel_compress(aligned_so_rels_feats_ff)
            temp["spatial_distribution"] = torch.sigmoid(self.s_rel_compress(aligned_so_rels_feats_ff))
            temp["contacting_distribution"] = torch.sigmoid(self.c_rel_compress(aligned_so_rels_feats_ff))
            temp["global_output"] = aligned_so_rels_feats_ff
            temp["original"] = aligned_so_rels_feats_tf
            temp["spatial_latents"] = so_rels_feats_tf[num_objects_cf:num_objects_cf + num_objects_ff]
            result[count] = temp
            count += 1
        entry["output"] = result
        return entry

    def forward_single_entry(self, context_fraction, entry):
        """
        Forward method for the baseline
        :param context_fraction:
        :param entry: Dictionary from object classifier
        :return:
        """
        entry, rel_features, sequences = self.process_entry(entry)

        count = 0
        total_frames = len(entry["im_idx"].unique())
        context = min(int(math.ceil(context_fraction * total_frames)), total_frames - 1)
        future = total_frames - context

        result = {}
        future_frame_start_id = entry["im_idx"].unique()[context]

        if context + future > total_frames > context:
            future = total_frames - context

        future_frame_end_id = entry["im_idx"].unique()[context + future - 1]

        context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
        context_idx = entry["im_idx"][:context_end_idx]
        context_len = context_idx.shape[0]

        future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
        future_idx = entry["im_idx"][context_end_idx:future_end_idx]
        future_len = future_idx.shape[0]

        context_seq = []
        future_seq = []
        new_future_seq = []
        for i, s in enumerate(sequences):
            context_index = s[(s < context_len)]
            future_index = s[(s >= context_len) & (s < (context_len + future_len))]
            future_seq.append(future_index)
            if len(context_index) != 0:
                context_seq.append(context_index)
                new_future_seq.append(future_index)

        pos_index = []
        for index in context_seq:
            im_idx, counts = torch.unique(entry["pair_idx"][index][:, 0].view(-1), return_counts=True, sorted=True)
            counts = counts.tolist()
            if im_idx.numel() == 0:
                pos = torch.tensor(
                    [torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
            else:
                pos = torch.cat([torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
            pos_index.append(pos)

        sequence_features = pad_sequence([rel_features[index] for index in context_seq], batch_first=True)
        in_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
                                  diagonal=0)).type(torch.bool)
        in_mask = in_mask.cuda()
        masks = (1 - pad_sequence([torch.ones(len(index)) for index in context_seq], batch_first=True)).bool()
        pos_index = [torch.tensor(seq, dtype=torch.long) for seq in pos_index]
        pos_index = pad_sequence(pos_index, batch_first=True) if self.mode == "sgdet" else None
        sequence_features = self.positional_encoder(sequence_features, pos_index)
        mask_input = sequence_features

        output = []
        for i in range(future):
            out = self.anticipation_temporal_transformer(mask_input, src_key_padding_mask=masks.cuda(), mask=in_mask)
            if i == 0:
                mask2 = (~masks).int()
                ind_col = torch.sum(mask2, dim=1) - 1
                out2 = []
                for j, ind in enumerate(ind_col):
                    out2.append(out[j, ind, :])
                out3 = torch.stack(out2)
                out3 = out3.unsqueeze(1)
                output.append(out3)
                mask_input = torch.cat([mask_input, out3], 1)
            else:
                output.append(out[:, -1, :].unsqueeze(1))
                out_last = [out[:, -1, :]]
                pred = torch.stack(out_last, dim=1)
                mask_input = torch.cat([mask_input, pred], 1)
            max_values = torch.max(pos_index, dim=1)[0] + 1
            max_values = max_values.unsqueeze(1)
            pos_index = torch.cat((pos_index, max_values), dim=1)
            mask_input = self.positional_encoder(mask_input, pos_index)
            in_mask = (1 - torch.tril(torch.ones(mask_input.shape[1], mask_input.shape[1]), diagonal=0)).type(
                torch.bool)
            in_mask = in_mask.cuda()
            masks = torch.cat([masks, torch.full((masks.shape[0], 1), False, dtype=torch.bool)], dim=1)

        output = torch.cat(output, dim=1)
        rel_ = output
        rel_ = rel_.cuda()
        rel_flat1 = []

        for index, rel in zip(new_future_seq, rel_):
            if len(index) == 0:
                continue
            ob_frame_idx = entry["im_idx"][index]
            rel_temp = torch.zeros(len(index), rel.shape[1])
            # For each frame in ob_frame_idx, if the value repeats then add the relation value of previous frame
            k = 0  # index for rel
            for i, frame in enumerate(ob_frame_idx):
                if i == 0:
                    rel_temp[i] = rel[k]
                    k += 1
                elif frame == ob_frame_idx[i - 1]:
                    rel_temp[i] = rel_temp[i - 1]
                else:
                    rel_temp[i] = rel[k]
                    k += 1

            rel_flat1.extend(rel_temp)

        rel_flat1 = [tensor.tolist() for tensor in rel_flat1]
        rel_flat = torch.tensor(rel_flat1)
        rel_flat = rel_flat.to('cuda:0')
        # rel_flat = torch.cat([rel[:len(index)] for index, rel in zip(future_seq,rel_)])
        indices_flat = torch.cat(new_future_seq).unsqueeze(1).repeat(1, rel_features.shape[1])
        global_output = torch.zeros_like(rel_features).to(rel_features.device)

        temp = {"scatter_flag": 0}
        try:
            global_output.scatter_(0, indices_flat, rel_flat)
        except RuntimeError:
            context += 1
            temp["scatter_flag"] = 1
            result[count] = temp
            entry["output"] = result
            return entry

        gb_output = global_output[context_len:context_len + future_len]

        temp["attention_distribution"] = self.a_rel_compress(gb_output)
        temp["spatial_distribution"] = torch.sigmoid(self.s_rel_compress(gb_output))
        temp["contacting_distribution"] = torch.sigmoid(self.c_rel_compress(gb_output))
        temp["global_output"] = gb_output
        temp["original"] = global_output
        temp["spatial_latents"] = rel_features[context_len:context_len + future_len]
        result[count] = temp
        entry["output"] = result

        return entry
