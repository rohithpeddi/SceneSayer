import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class BaseTransformer(nn.Module):

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
        super(BaseTransformer, self).__init__()

        self.mode = mode
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.enc_layer_num = enc_layer_num
        self.dec_layer_num = dec_layer_num

        assert mode in ('sgdet', 'sgcls', 'predcls')

    def generate_spatial_predicate_embeddings(self, entry):
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
            sequences.append(k)

        return entry, rel_features, sequences

    def fetch_positional_encoding_for_obj_seqs(self, obj_seqs, entry):
        positional_encoding = []
        for obj_seq in obj_seqs:
            im_idx, counts = torch.unique(entry["pair_idx"][obj_seq][:, 0].view(-1), return_counts=True, sorted=True)
            counts = counts.tolist()
            if im_idx.numel() == 0:
                pos = torch.tensor(
                    [torch.LongTensor([im] * count) for im, count in zip(range(len(counts)), counts)])
            else:
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

    def generate_future_ff_rels_for_context(self, entry, so_rels_feats_tf, obj_seqs_tf, num_cf, num_tf, num_ff):
        num_ff = min(num_ff, num_tf - num_cf)
        ff_start_id = entry["im_idx"].unique()[num_cf]
        ff_end_id = entry["im_idx"].unique()[num_cf + num_ff - 1]
        cf_end_id = entry["im_idx"].unique()[num_cf - 1]

        objects_ff_start_id = int(torch.where(entry["im_idx"] == ff_start_id)[0][0])
        objects_clf_start_id = int(torch.where(entry["im_idx"] == cf_end_id)[0][0])

        objects_ff_end_id = int(torch.where(entry["im_idx"] == ff_end_id)[0][-1]) + 1
        obj_labels_tf_unique = entry['pred_labels'][entry['pair_idx'][:, 1]].unique()
        objects_pcf = entry["im_idx"][:objects_clf_start_id]
        objects_cf = entry["im_idx"][:objects_ff_start_id]
        objects_ff = entry["im_idx"][objects_ff_start_id:objects_ff_end_id]
        num_objects_cf = objects_cf.shape[0]
        num_objects_pcf = objects_pcf.shape[0]
        num_objects_ff = objects_ff.shape[0]

        # 1. Refine object sequences to take only those objects that are present in the current frame.
        # 2. Fetch future representations for those objects.
        # 3. Construct im_idx, pair_idx, and labels for the future frames accordingly.
        # 4. Send everything along with ground truth for evaluation.

        cf_obj_seqs_in_clf = []
        obj_seqs_ff = []
        obj_labels_clf = []
        for i, s in enumerate(obj_seqs_tf):
            if len(s) == 0:
                continue
            context_index = s[(s < num_objects_cf)]
            if len(context_index) > 0:
                prev_context_index = s[(s >= num_objects_pcf) & (s < num_objects_cf)]
                if len(prev_context_index) > 0:
                    obj_labels_clf.append(obj_labels_tf_unique[i])
                    cf_obj_seqs_in_clf.append(context_index)

            future_index = s[(s >= num_objects_cf) & (s < (num_objects_cf + num_objects_ff))]
            if len(future_index) > 0:
                obj_seqs_ff.append(future_index - num_objects_cf)

        so_rels_feats_cf = pad_sequence([so_rels_feats_tf[cf_obj_seq.flip(dims=[0])]
                                         for cf_obj_seq in cf_obj_seqs_in_clf], batch_first=True).flip(dims=[1])
        causal_mask = torch.triu(torch.ones(so_rels_feats_cf.shape[1], so_rels_feats_cf.shape[1]),
                                 diagonal=1).bool().cuda()
        padding_mask = (1 - pad_sequence([torch.ones(len(cf_obj_seq)).flip(dims=[0])
                                          for cf_obj_seq in cf_obj_seqs_in_clf], batch_first=True)).flip(
            dims=[1]).bool().cuda()

        # TODO: Change Positional Encoding Scheme
        positional_encoding = self.fetch_positional_encoding_for_obj_seqs(cf_obj_seqs_in_clf, entry)
        so_rels_feats_cf = self.positional_encoder(so_rels_feats_cf, positional_encoding)

        output = []
        for i in range(num_ff):
            out = self.anti_temporal_transformer(so_rels_feats_cf, src_key_padding_mask=padding_mask.cuda(),
                                                 mask=causal_mask)
            output.append(out[:, -1, :].unsqueeze(1))
            out_last = [out[:, -1, :]]
            pred = torch.stack(out_last, dim=1)
            so_rels_feats_cf = torch.cat([so_rels_feats_cf, pred], 1)
            # TODO: Change padding schemes
            positional_encoding = self.update_positional_encoding(positional_encoding)
            so_rels_feats_cf = self.positional_encoder(so_rels_feats_cf, positional_encoding)
            causal_mask = (1 - torch.tril(torch.ones(so_rels_feats_cf.shape[1], so_rels_feats_cf.shape[1]), diagonal=0)).type(
                torch.bool)
            causal_mask = causal_mask.cuda()
            padding_mask = torch.cat(
                [padding_mask, torch.full((padding_mask.shape[0], 1), False, dtype=torch.bool)], dim=1)
        output = torch.cat(output, dim=1)

        # TODO: Change the way the output is handled
        indices_flat = torch.cat(obj_seqs_ff).unsqueeze(1).repeat(1, so_rels_feats_tf.shape[1])
        global_output = torch.zeros_like(so_rels_feats_tf).to(so_rels_feats_tf.device)

        # TODO: Correct this code
        im_idx = torch.tensor(list(range(num_ff)))
        im_idx = im_idx.repeat_interleave(num_obj)

        pred_labels = [1]
        pred_labels.extend(obj)
        pred_labels = torch.tensor(pred_labels)
        pred_labels = pred_labels.repeat(num_ff)

        pair_idx = []
        for i in range(1, num_obj + 1):
            pair_idx.append([0, i])
        pair_idx = torch.tensor(pair_idx)
        p_i = pair_idx
        for i in range(num_ff - 1):
            p_i = p_i + num_obj + 1
            pair_idx = torch.cat([pair_idx, p_i])

        if self.mode == 'predcls':
            sc_human = entry["scores"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 0]][0]
            sc_obj = entry["scores"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 1]]
        else:
            sc_human = entry["pred_scores"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 0]][0]
            sc_obj = entry["pred_scores"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 1]]

        sc_obj = torch.index_select(sc_obj, 0, torch.tensor(obj_ind))
        sc_human = sc_human.unsqueeze(0)
        scores = torch.cat([sc_human, sc_obj])
        scores = scores.repeat(num_ff)

        box_human = entry["boxes"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 0]][0]
        box_obj = entry["boxes"][entry["pair_idx"][objects_clf_start_id:objects_ff_start_id][:, 1]]

        box_obj = torch.index_select(box_obj, 0, torch.tensor(obj_ind))
        box_human = box_human.unsqueeze(0)
        boxes = torch.cat([box_human, box_obj])
        boxes = boxes.repeat(num_ff, 1)

        gb_output = gb_output.cuda()
        temp = {
            "attention_distribution": self.a_rel_compress(gb_output),
            "spatial_distribution": torch.sigmoid(self.s_rel_compress(gb_output)),
            "contacting_distribution": torch.sigmoid(self.c_rel_compress(gb_output)),
            "global_output": gb_output,
            "pair_idx": pair_idx.cuda(),
            "im_idx": im_idx.cuda(),
            "labels": pred_labels.cuda(),
            "pred_labels": pred_labels.cuda(),
            "scores": scores.cuda(),
            "pred_scores": scores.cuda(),
            "boxes": boxes.cuda()
        }

        return temp
