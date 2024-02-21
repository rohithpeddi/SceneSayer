import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from lib.supervised.biased.sga.base_transformer import BaseTransformer
from lib.supervised.biased.sga.blocks import EncoderLayer, Encoder, PositionalEncoding, ObjectClassifierMLP
from lib.word_vectors import obj_edge_vectors


class BaselineWithAnticipationGenLoss(BaseTransformer):
    
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
        super(BaselineWithAnticipationGenLoss, self).__init__()
        
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
        
        # Spatial encoder
        spatial_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.spatial_transformer = Encoder(spatial_encoder, num_layers=1)
        
        # Generation temporal encoder
        gen_temporal_encoder = EncoderLayer(d_model=d_model, dim_feedforward=2048, nhead=8, batch_first=True)
        self.gen_temporal_transformer = Encoder(gen_temporal_encoder, num_layers=3)
        
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
        entry, rel_features, sequences = self.generate_spatial_predicate_embeddings(entry)
        # Temporal message passing
        # TODO: Check if object tracking and all is necessary?
        sequences = []
        obj_class = entry["pred_labels"][entry["pair_idx"][:, 1]]
        object_tracker = {}
        object_labels = []
        for i, l in enumerate(obj_class.unique()):
            k = torch.where(obj_class.view(-1) == l)[0]
            if len(k) > 0:
                object_tracker[i] = 0
                object_labels.append(l)
                sequences.append(k)
        	
        sequence_features = pad_sequence([rel_features[index] for index in sequences], batch_first=True)
        in_mask_dsg = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
                                      diagonal=0)).type(torch.bool)
        in_mask_dsg = in_mask_dsg.cuda()
        masks = (1 - pad_sequence([torch.ones(len(index)) for index in sequences], batch_first=True)).bool()
        positional_encoding = self.fetch_positional_encoding_for_obj_seqs(sequences, entry)
        rel_ = self.gen_temporal_transformer(self.positional_encoder(sequence_features, positional_encoding),
                                             src_key_padding_mask=masks.cuda(), mask=in_mask_dsg)
        
        rel_flat = torch.cat([rel[:len(index)] for index, rel in zip(sequences, rel_)])
        indices_flat = torch.cat(sequences).unsqueeze(1).repeat(1, rel_features.shape[1])
        
        assert len(indices_flat) == len(entry["pair_idx"])
        dsg_global_output = torch.zeros_like(rel_features).to(rel_features.device)
        dsg_global_output.scatter_(0, indices_flat, rel_flat)
        
        return entry, rel_features, sequences, object_labels, dsg_global_output
    
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
         object_labels, spa_temp_so_rels_feats_tf) = self.generate_spatio_temporal_predicate_embeddings(entry)
        
        count = 0
        result = {}
        num_tf = len(entry["im_idx"].unique())
        num_cf = min(num_cf, num_tf - 1)
        num_ff = min(num_ff, num_tf - num_cf)
        while num_cf + 1 <= num_tf:
            ff_start_id = entry["im_idx"].unique()[num_cf]
            ff_end_id = entry["im_idx"].unique()[num_cf + num_ff - 1]
            cf_end_id = entry["im_idx"].unique()[num_cf - 1]
            
            context_end_idx = int(torch.where(entry["im_idx"] == ff_start_id)[0][0])
            prev_context_end_idx = int(torch.where(entry["im_idx"] == cf_end_id)[0][0])
            prev_context_idx = entry["im_idx"][:prev_context_end_idx]
            context_idx = entry["im_idx"][:context_end_idx]
            context_len = context_idx.shape[0]
            prev_context_len = prev_context_idx.shape[0]
            
            future_end_idx = int(torch.where(entry["im_idx"] == ff_end_id)[0][-1]) + 1
            future_idx = entry["im_idx"][context_end_idx:future_end_idx]
            future_len = future_idx.shape[0]
            
            obj_seqs_cf = []
            obj_seqs_ff = []
            object_track = {}
            object_lab = []
            obj_ind = 0
            
            for i, s in enumerate(obj_seqs_tf):
                context_index = s[(s < context_len)]
                future_index = s[(s >= context_len) & (s < (context_len + future_len))]
                if len(context_index) != 0:
                    object_track[obj_ind] = 0
                    object_lab.append(object_labels[i].item())
                    obj_seqs_cf.append(context_index)
                    obj_seqs_ff.append(future_index)
                    obj_ind += 1
            
            obj_ind = 0
            for i, s in enumerate(obj_seqs_cf):
                prev_context_index = s[(s >= prev_context_len) & (s < context_len)]
                if len(prev_context_index) != 0:
                    object_track[i] = 1
               
            sequence_features = pad_sequence([spa_temp_so_rels_feats_tf[index] for index in obj_seqs_cf], batch_first=True)
            in_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
                                      diagonal=0)).type(torch.bool)
            in_mask = in_mask.cuda()
            masks = (1 - pad_sequence([torch.ones(len(index)) for index in obj_seqs_cf], batch_first=True)).bool()
            
            positional_encoding = self.fetch_positional_encoding_for_obj_seqs(obj_seqs_cf, entry)
            sequence_features = self.positional_encoder(sequence_features, positional_encoding)
            mask_input = sequence_features
            
            if self.training:
                output = []
                for i in range(num_ff):
                    out = self.anti_temporal_transformer(mask_input, src_key_padding_mask=masks.cuda(), mask=in_mask)
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
                    
                    positional_encoding = self.update_positional_encoding(positional_encoding)
                    mask_input = self.positional_encoder(mask_input, positional_encoding)
                    in_mask = (1 - torch.tril(torch.ones(mask_input.shape[1], mask_input.shape[1]), diagonal=0)).type(
                        torch.bool)
                    in_mask = in_mask.cuda()
                    masks = torch.cat([masks, torch.full((masks.shape[0], 1), False, dtype=torch.bool)], dim=1)
                
                output = torch.cat(output, dim=1)
                rel_ = output
                rel_ = rel_.cuda()
                rel_flat1 = []
                
                for index, rel in zip(obj_seqs_ff, rel_):
                    if len(index) == 0:
                        continue
                    rel_flat1.extend(rel[:len(index)])
                
                rel_flat1 = [tensor.tolist() for tensor in rel_flat1]
                rel_flat = torch.tensor(rel_flat1)
                rel_flat = rel_flat.to('cuda:0')
                indices_flat = torch.cat(obj_seqs_ff).unsqueeze(1).repeat(1, spa_temp_so_rels_feats_tf.shape[1])
                global_output = torch.zeros_like(spa_temp_so_rels_feats_tf).to(spa_temp_so_rels_feats_tf.device)
                
                temp = {"scatter_flag": 0}
                try:
                    global_output.scatter_(0, indices_flat, rel_flat)
                except RuntimeError:
                    num_cf += 1
                    temp["scatter_flag"] = 1
                    result[count] = temp
                    count += 1
                    continue
                gb_output = global_output[context_len:context_len + future_len]
                num_cf += 1
                
                temp["attention_distribution"] = self.a_rel_compress(gb_output)
                temp["spatial_distribution"] = torch.sigmoid(self.s_rel_compress(gb_output))
                temp["contacting_distribution"] = torch.sigmoid(self.c_rel_compress(gb_output))
                temp["global_output"] = gb_output
                temp["original"] = global_output
                
                result[count] = temp
                count += 1
            else:
                obj, obj_ind = entry["pred_labels"][
                    entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]].sort()
                obj = [o.item() for o in obj]
                obj_dict = {}
                for o in object_lab:
                    obj_dict[o] = 1
                
                for o in obj:
                    if o in obj_dict:
                        obj_dict[o] += 1
                
                for o in torch.tensor(obj).unique():
                    if o.item() in obj_dict:
                        obj_dict[o.item()] -= 1
                
                output = []
                col_num = list(obj_dict.values())
                max_non_pad_len = max(col_num)
                latent_mask = torch.tensor(list(object_track.values()))
                for i in range(num_ff):
                    out = self.anti_temporal_transformer(mask_input, src_key_padding_mask=masks.cuda(), mask=in_mask)
                    if i == 0:
                        mask2 = (~masks).int()
                        ind_col = torch.sum(mask2, dim=1)
                        out2 = []
                        for j, ind in enumerate(ind_col):
                            out2.append(out[j, ind - col_num[j]:ind, :])
                        out2 = pad_sequence(out2, batch_first=True)
                        out4 = out2[latent_mask == 1]
                        output.append(out4)
                        mask_input = torch.cat([mask_input, out2], 1)
                    else:
                        out2 = []
                        len_seq = out.shape[1]
                        start_seq = len_seq - max_non_pad_len
                        for j in range(out.shape[0]):
                            out2.append(out[j, start_seq:start_seq + col_num[j], :])
                        out2 = pad_sequence(out2, batch_first=True)
                        out4 = out2[latent_mask == 1]
                        output.append(out4)
                        mask_input = torch.cat([mask_input, out2], 1)
                    
                    in_mask = (1 - torch.tril(torch.ones(mask_input.shape[1], mask_input.shape[1]), diagonal=0)).type(
                        torch.bool)
                    in_mask = in_mask.cuda()
                    src_pad = []
                    for j in col_num:
                        src_pad.append(torch.zeros(j))
                    src_mask = pad_sequence(src_pad, batch_first=True, padding_value=1).bool()
                    masks = torch.cat([masks, src_mask], dim=1)
                
                output = torch.cat(output, dim=1)
                rel_ = output
                rel_ = rel_.cuda()
                
                if self.mode == 'predcls':
                    obj, obj_ind = entry["labels"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]].sort()
                    num_obj_unique = len(obj.unique())
                    obj = [o.item() for o in obj]
                    num_obj = len(obj)
                else:
                    obj, obj_ind = entry["pred_labels"][
                        entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]].sort()
                    num_obj_unique = len(obj.unique())
                    obj = [o.item() for o in obj]
                    num_obj = len(obj)
                
                gb_output = torch.zeros(len(obj) * num_ff, rel_.shape[2])
                obj_dict = {}
                for o in obj:
                    if o not in obj_dict:
                        obj_dict[o] = 1
                    else:
                        obj_dict[o] += 1
                
                col_num = list(obj_dict.values())
                col_idx = 0
                row_idx = 0
                i = 0
                while i < gb_output.shape[0]:
                    for j in range(col_idx, col_idx + col_num[row_idx]):
                        gb_output[i] = rel_[row_idx, j, :]
                        i += 1
                    row_idx += 1
                    row_idx = row_idx % num_obj_unique
                    if row_idx % num_obj_unique == 0:
                        col_idx = col_idx + max_non_pad_len
                
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
                    sc_human = entry["scores"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 0]][0]
                    sc_obj = entry["scores"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]]
                else:
                    sc_human = entry["pred_scores"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 0]][0]
                    sc_obj = entry["pred_scores"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]]
                
                sc_obj = torch.index_select(sc_obj, 0, torch.tensor(obj_ind))
                sc_human = sc_human.unsqueeze(0)
                scores = torch.cat([sc_human, sc_obj])
                scores = scores.repeat(num_ff)
                
                box_human = entry["boxes"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 0]][0]
                box_obj = entry["boxes"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]]
                
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
                
                result[count] = temp
                count += 1
                num_cf += 1
        
        entry["gen_attention_distribution"] = self.gen_a_rel_compress(spa_temp_so_rels_feats_tf)
        entry["gen_spatial_distribution"] = torch.sigmoid(self.gen_s_rel_compress(spa_temp_so_rels_feats_tf))
        entry["gen_contacting_distribution"] = torch.sigmoid(self.gen_c_rel_compress(spa_temp_so_rels_feats_tf))
        entry["output"] = result
        
        return entry
    
    def forward_single_entry(self, context_fraction, entry):
        """
        Forward method for the baseline
        :param context_fraction:
        :param entry: Dictionary from object classifier
        :return:
        """
        entry, rel_features, sequences, object_labels, dsg_global_output = self.generate_spatio_temporal_predicate_embeddings(
            entry)
        
        """ ################# changes regarding forecasting #################### """
        count = 0
        total_frames = len(entry["im_idx"].unique())
        context = min(int(math.ceil(context_fraction * total_frames)), total_frames - 1)
        future = total_frames - context
        
        temp = {}
        future_frame_start_id = entry["im_idx"].unique()[context]
        prev_context_start_id = entry["im_idx"].unique()[context - 1]
        
        if context + future > total_frames > context:
            future = total_frames - context
        
        future_frame_end_id = entry["im_idx"].unique()[context + future - 1]
        
        context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
        prev_context_end_idx = int(torch.where(entry["im_idx"] == prev_context_start_id)[0][0])
        prev_context_idx = entry["im_idx"][:prev_context_end_idx]
        context_idx = entry["im_idx"][:context_end_idx]
        context_len = context_idx.shape[0]
        prev_context_len = prev_context_idx.shape[0]
        
        future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
        future_idx = entry["im_idx"][context_end_idx:future_end_idx]
        future_len = future_idx.shape[0]
        
        context_seq = []
        future_seq = []
        new_future_seq = []
        object_track = {}
        object_lab = []
        obj_ind = 0
        
        for i, s in enumerate(sequences):
            context_index = s[(s < context_len)]
            future_index = s[(s >= context_len) & (s < (context_len + future_len))]
            future_seq.append(future_index)
            if len(context_index) != 0:
                object_track[obj_ind] = 0
                object_lab.append(object_labels[i].item())
                context_seq.append(context_index)
                new_future_seq.append(future_index)
                obj_ind += 1
        
        obj_ind = 0
        for i, s in enumerate(context_seq):
            prev_context_index = s[(s >= prev_context_len) & (s < context_len)]
            if len(prev_context_index) != 0:
                object_track[i] = 1
        
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
        
        sequence_features = pad_sequence([dsg_global_output[index] for index in context_seq], batch_first=True)
        in_mask = (1 - torch.tril(torch.ones(sequence_features.shape[1], sequence_features.shape[1]),
                                  diagonal=0)).type(torch.bool)
        in_mask = in_mask.cuda()
        masks = (1 - pad_sequence([torch.ones(len(index)) for index in context_seq], batch_first=True)).bool()
        pos_index = [torch.tensor(seq, dtype=torch.long) for seq in pos_index]
        pos_index = pad_sequence(pos_index, batch_first=True) if self.mode == "sgdet" else None
        sequence_features = self.positional_encoder(sequence_features, pos_index)
        mask_input = sequence_features
        
        obj, obj_ind = entry["pred_labels"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]].sort()
        obj = [o.item() for o in obj]
        obj_dict = {}
        for o in object_lab:
            obj_dict[o] = 1
        
        for o in obj:
            if o in obj_dict:
                obj_dict[o] += 1
        
        for o in torch.tensor(obj).unique():
            if o.item() in obj_dict:
                obj_dict[o.item()] -= 1
        
        output = []
        col_num = list(obj_dict.values())
        max_non_pad_len = max(col_num)
        latent_mask = torch.tensor(list(object_track.values()))
        for i in range(future):
            out = self.anti_temporal_transformer(mask_input, src_key_padding_mask=masks.cuda(), mask=in_mask)
            if i == 0:
                mask2 = (~masks).int()
                ind_col = torch.sum(mask2, dim=1)
                out2 = []
                for j, ind in enumerate(ind_col):
                    out2.append(out[j, ind - col_num[j]:ind, :])
                out2 = pad_sequence(out2, batch_first=True)
                out4 = out2[latent_mask == 1]
                output.append(out4)
                mask_input = torch.cat([mask_input, out2], 1)
            else:
                out2 = []
                len_seq = out.shape[1]
                start_seq = len_seq - max_non_pad_len
                for j in range(out.shape[0]):
                    out2.append(out[j, start_seq:start_seq + col_num[j], :])
                out2 = pad_sequence(out2, batch_first=True)
                out4 = out2[latent_mask == 1]
                output.append(out4)
                mask_input = torch.cat([mask_input, out2], 1)
            
            in_mask = (1 - torch.tril(torch.ones(mask_input.shape[1], mask_input.shape[1]), diagonal=0)).type(
                torch.bool)
            in_mask = in_mask.cuda()
            src_pad = []
            for j in col_num:
                src_pad.append(torch.zeros(j))
            src_mask = pad_sequence(src_pad, batch_first=True, padding_value=1).bool()
            masks = torch.cat([masks, src_mask], dim=1)
        
        output = torch.cat(output, dim=1)
        rel_ = output
        rel_ = rel_.cuda()
        
        if self.mode == 'predcls':
            obj, obj_ind = entry["labels"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]].sort()
            num_obj_unique = len(obj.unique())
            obj = [o.item() for o in obj]
            num_obj = len(obj)
        else:
            obj, obj_ind = entry["pred_labels"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]].sort()
            num_obj_unique = len(obj.unique())
            obj = [o.item() for o in obj]
            num_obj = len(obj)
        
        gb_output = torch.zeros(len(obj) * future, rel_.shape[2])
        obj_dict = {}
        for o in obj:
            if o not in obj_dict:
                obj_dict[o] = 1
            else:
                obj_dict[o] += 1
        
        col_num = list(obj_dict.values())
        col_idx = 0
        row_idx = 0
        i = 0
        while i < gb_output.shape[0]:
            for j in range(col_idx, col_idx + col_num[row_idx]):
                gb_output[i] = rel_[row_idx, j, :]
                i += 1
            row_idx += 1
            row_idx = row_idx % num_obj_unique
            if (row_idx % num_obj_unique == 0):
                col_idx = col_idx + max_non_pad_len
        
        im_idx = torch.tensor(list(range(future)))
        im_idx = im_idx.repeat_interleave(num_obj)
        
        pred_labels = [1]
        pred_labels.extend(obj)
        pred_labels = torch.tensor(pred_labels)
        pred_labels = pred_labels.repeat(future)
        
        pair_idx = []
        for i in range(1, num_obj + 1):
            pair_idx.append([0, i])
        pair_idx = torch.tensor(pair_idx)
        p_i = pair_idx
        for i in range(future - 1):
            p_i = p_i + num_obj + 1
            pair_idx = torch.cat([pair_idx, p_i])
        
        if self.mode == 'predcls':
            sc_human = entry["scores"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 0]][0]
            sc_obj = entry["scores"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]]
        else:
            sc_human = entry["pred_scores"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 0]][0]
            sc_obj = entry["pred_scores"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]]
        
        sc_obj = torch.index_select(sc_obj, 0, torch.tensor(obj_ind))
        sc_human = sc_human.unsqueeze(0)
        scores = torch.cat([sc_human, sc_obj])
        scores = scores.repeat(future)
        
        box_human = entry["boxes"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 0]][0]
        box_obj = entry["boxes"][entry["pair_idx"][prev_context_end_idx:context_end_idx][:, 1]]
        
        box_obj = torch.index_select(box_obj, 0, torch.tensor(obj_ind))
        box_human = box_human.unsqueeze(0)
        boxes = torch.cat([box_human, box_obj])
        boxes = boxes.repeat(future, 1)
        
        gb_output = gb_output.cuda()
        temp["attention_distribution"] = self.a_rel_compress(gb_output)
        temp["spatial_distribution"] = torch.sigmoid(self.s_rel_compress(gb_output))
        temp["contacting_distribution"] = torch.sigmoid(self.c_rel_compress(gb_output))
        temp["global_output"] = gb_output
        temp["pair_idx"] = pair_idx.cuda()
        temp["im_idx"] = im_idx.cuda()
        temp["labels"] = pred_labels.cuda()
        temp["pred_labels"] = pred_labels.cuda()
        temp["scores"] = scores.cuda()
        temp["pred_scores"] = scores.cuda()
        temp["boxes"] = boxes.cuda()
        
        return temp
