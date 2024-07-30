import numpy as np
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

    def fetch_positional_encoding(self, obj_seqs, entry):
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
        return positional_encoding

    def fetch_positional_encoding_for_gen_obj_seqs(self, obj_seqs, entry):
        positional_encoding = self.fetch_positional_encoding(obj_seqs, entry)
        positional_encoding = pad_sequence(positional_encoding, batch_first=True) if self.mode == "sgdet" else None
        return positional_encoding

    def fetch_positional_encoding_for_ant_obj_seqs(self, obj_seqs, entry):
        positional_encoding = self.fetch_positional_encoding(obj_seqs, entry)
        positional_encoding = pad_sequence([pos_enc.flip(dims=[0]) for pos_enc in positional_encoding],
                                           batch_first=True).flip(dims=[1])
        return positional_encoding if self.mode == "sgdet" else None

    def update_positional_encoding(self, positional_encoding):
        if positional_encoding is not None:
            max_values = torch.max(positional_encoding, dim=1)[0] + 1
            max_values = max_values.unsqueeze(1)
            positional_encoding = torch.cat((positional_encoding, max_values), dim=1)
        return positional_encoding

    @staticmethod
    def fetch_masks(obj_seqs):
        padding_mask = (1 - pad_sequence([torch.ones(len(obj_seq)).flip(dims=[0])
                                          for obj_seq in obj_seqs], batch_first=True)).flip(
            dims=[1]).bool().cuda()
        return padding_mask

    @staticmethod
    def fetch_updated_masks(padding_mask):
        additional_column = torch.full((padding_mask.shape[0], 1), False, dtype=torch.bool).cuda()
        padding_mask = torch.cat([padding_mask, additional_column], dim=1)
        return padding_mask

    def generate_future_ff_rels_for_context(self, entry, so_rels_feats_tf, obj_seqs_tf, num_cf, num_tf, num_ff):
        num_ff = min(num_ff, num_tf - num_cf)
        ff_start_id = entry["im_idx"].unique()[num_cf]
        cf_end_id = entry["im_idx"].unique()[num_cf - 1]

        objects_ff_start_id = int(torch.where(entry["im_idx"] == ff_start_id)[0][0])
        objects_clf_start_id = int(torch.where(entry["im_idx"] == cf_end_id)[0][0])

        obj_labels_tf_unique = entry['pred_labels'][entry['pair_idx'][:, 1]].unique()
        objects_pcf = entry["im_idx"][:objects_clf_start_id]
        objects_cf = entry["im_idx"][:objects_ff_start_id]
        num_objects_cf = objects_cf.shape[0]
        num_objects_pcf = objects_pcf.shape[0]

        # 1. Refine object sequences to take only those objects that are present in the current frame.
        cf_obj_seqs_in_clf = []
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

        # ------------------- Masks for anticipation transformer -------------------
        # Causal mask is not required for anticipation transformer. It is required for the generation transformer.
        # In anticipation transformer we only take the last output from the transformer, so mask is not required.

        so_rels_feats_cf = pad_sequence([so_rels_feats_tf[cf_obj_seq.flip(dims=[0])]
                                         for cf_obj_seq in cf_obj_seqs_in_clf], batch_first=True).flip(dims=[1])
        padding_mask = self.fetch_masks(cf_obj_seqs_in_clf)
        positional_encoding = self.fetch_positional_encoding_for_ant_obj_seqs(cf_obj_seqs_in_clf, entry)
        so_rels_feats_cf = self.anti_positional_encoder(so_rels_feats_cf, positional_encoding)

        # 2. Fetch future representations for those objects.
        so_rels_feats_ff = []
        for i in range(num_ff):
            out = self.anti_temporal_transformer(so_rels_feats_cf, src_key_padding_mask=padding_mask)

            so_rels_feats_ff.append(out[:, -1, :].unsqueeze(1))
            so_rels_feats_ant_one_step = torch.stack([out[:, -1, :]], dim=1)

            so_rels_feats_cf = torch.cat([so_rels_feats_cf, so_rels_feats_ant_one_step], 1)
            positional_encoding = self.update_positional_encoding(positional_encoding)
            so_rels_feats_cf = self.anti_positional_encoder(so_rels_feats_cf, positional_encoding)
            padding_mask = self.fetch_updated_masks(padding_mask)
        so_rels_feats_ff = torch.cat(so_rels_feats_ff, dim=1)

        # 3. Construct im_idx, pair_idx, and labels for the future frames accordingly.
        ff_obj_list = [1] + [obj_label.cpu().item() for obj_label in obj_labels_clf]
        pred_labels = (torch.tensor(ff_obj_list).repeat(num_ff)).cuda()
        im_idx = (torch.tensor(list(range(num_ff))).repeat_interleave(len(cf_obj_seqs_in_clf))).cuda()

        pair_idx = []
        ff_sub_idx = [i * (len(cf_obj_seqs_in_clf) + 1) for i in range(num_ff)]
        for j in ff_sub_idx:
            for i in range(len(cf_obj_seqs_in_clf)):
                pair_idx.append([j, i + 1 + j])

        pair_idx = torch.tensor(pair_idx).cuda()
        boxes = torch.tensor([[0.5] * 5 for _ in range(len(pred_labels))]).cuda()
        scores = torch.tensor([1] * len(pred_labels)).cuda()

        obj_range = torch.tensor(list(range(len(cf_obj_seqs_in_clf)))).reshape((-1, 1))
        obj_range_ff = torch.repeat_interleave(obj_range, num_ff, dim=1)
        range_list = torch.tensor(list(range(num_ff))) * len(cf_obj_seqs_in_clf)
        indices_flat = (range_list + obj_range_ff).flatten().cuda()

        so_rels_feats_ff_flat = so_rels_feats_ff.view(-1, so_rels_feats_ff.shape[-1])
        so_rels_feats_ff_flat_ord = torch.zeros_like(so_rels_feats_ff_flat).cuda()
        indices_flat = indices_flat.unsqueeze(1).repeat(1, so_rels_feats_ff_flat_ord.shape[1])
        so_rels_feats_ff_flat_ord.scatter_(0, indices_flat, so_rels_feats_ff_flat)

        # ------------------- Construct mask_ant and mask_gt -------------------
        # mask_ant: indices in only anticipated frames
        # mask_gt: indices in all frames
        # ----------------------------------------------------------------------
        if self.training:
            obj_idx_tf = entry["pair_idx"][:, 1]
            pred_obj_labels_tf = entry["pred_labels"][obj_idx_tf]
            gt_obj_labels_tf = entry["pred_labels"][obj_idx_tf]
            # obj_labels_tf = entry["labels"][obj_idx_tf] if self.training else entry["pred_labels"][obj_idx_tf]
            assert cf_end_id.cpu().numpy() in entry["im_idx"].cpu().numpy()
            clf_obj_idx = torch.where(entry["im_idx"] == cf_end_id)[0]
            clf_obj_labels = pred_obj_labels_tf[clf_obj_idx].unique(sorted=True)
            num_clf_obj = clf_obj_labels.shape[0]

            mask_ant = []
            mask_gt = []
            for ff_id in range(num_ff):
                ff_obj_idx = torch.where(entry["im_idx"] == num_cf + ff_id)[0]
                ff_obj_labels = gt_obj_labels_tf[ff_obj_idx]

                np_clf_obj_labels = clf_obj_labels.cpu().numpy()
                np_ff_obj_labels = ff_obj_labels.cpu().numpy()

                int_obj_labels, ff_idx_obj_in_clf, clf_idx_obj_in_ff = np.intersect1d(np_clf_obj_labels, np_ff_obj_labels,
                                                                                      return_indices=True)
                mask_ant.append(ff_id * num_clf_obj + ff_idx_obj_in_clf)
                mask_gt.append(ff_obj_idx[clf_idx_obj_in_ff])

            mask_ant_flat = []
            for sublist in mask_ant:
                mask_ant_flat.extend(sublist)

            mask_gt_flat = torch.tensor([], dtype=torch.int).cuda()
            for sublist in mask_gt:
                mask_gt_flat = torch.cat((mask_gt_flat, sublist.cuda()))

            mask_ant = torch.tensor(mask_ant_flat).cuda()
            mask_gt = mask_gt_flat

            assert mask_ant.shape[0] == mask_gt.shape[0]
            assert mask_ant.shape[0] <= so_rels_feats_ff_flat_ord.shape[0]

        temp = {
            "attention_distribution": self.a_rel_compress(so_rels_feats_ff_flat_ord),
            "spatial_distribution": torch.sigmoid(self.s_rel_compress(so_rels_feats_ff_flat_ord)),
            "contacting_distribution": torch.sigmoid(self.c_rel_compress(so_rels_feats_ff_flat_ord)),
            "global_output": so_rels_feats_ff_flat_ord,
            "pair_idx": pair_idx.cuda(),
            "im_idx": im_idx.cuda(),
            "labels": pred_labels.cuda(),
            "pred_labels": pred_labels.cuda(),
            "scores": scores.cuda(),
            "pred_scores": scores.cuda(),
            "boxes": boxes.cuda(),
        }

        if self.training:
            temp["mask_ant"] = mask_ant
            temp["mask_gt"] = mask_gt

        return temp
