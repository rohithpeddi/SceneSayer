import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from lib.supervised.sga.base_ldpu_diffeq import BaseLDPU


class SceneSayerODEDerivatives(nn.Module):
    def __init__(self):
        super(SceneSayerODEDerivatives, self).__init__()

        self.net = nn.Sequential(nn.Linear(1936, 2048), nn.Tanh(),
                                 nn.Linear(2048, 2048), nn.Tanh(),
                                 nn.Linear(2048, 1936))

    def forward(self, t, y):
        out = self.net(y)
        return out


class SceneSayerODE(nn.Module):

    def __init__(self, mode, attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None,
                 rel_classes=None, max_window=None):
        super(SceneSayerODE, self).__init__()
        self.mode = mode
        self.diff_func = SceneSayerODEDerivatives()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        self.d_model = 1936
        self.max_window = max_window

        self.base_ldpu = BaseLDPU(self.mode,
                                  attention_class_num=attention_class_num,
                                  spatial_class_num=spatial_class_num,
                                  contact_class_num=contact_class_num,
                                  obj_classes=obj_classes)
        self.ctr = 0

    def forward(self, entry, testing=False):
        entry = self.base_ldpu(entry)
        obj = entry["pair_idx"][:, 1]
        if not testing:
            labels_obj = entry["labels"][obj]
        else:
            pred_labels_obj = entry["pred_labels"][obj]
            labels_obj = entry["labels"][obj]
        im_idx = entry["im_idx"]
        pair_idx = entry["pair_idx"]
        gt_annotation = entry["gt_annotation"]
        num_preds = im_idx.size(0)
        times = torch.tensor(entry["frame_idx"], dtype=torch.float32)
        indices = torch.reshape((im_idx[: -1] != im_idx[1:]).nonzero(), (-1,)) + 1
        curr_id = 0
        times_unique = torch.unique(times).float()
        num_frames = len(gt_annotation)
        window = self.max_window
        if self.max_window == -1:
            window = num_frames - 1
        window = min(window, num_frames - 1)
        times_extend = torch.Tensor([times_unique[-1] + i + 1 for i in range(window)])
        global_output = entry["global_output"]
        anticipated_vals = torch.zeros(window, 0, self.d_model, device=global_output.device)
        # obj_bounding_boxes = torch.zeros(self.max_window, indices[-1], 4, device=global_output.device)
        frames_ranges = torch.cat(
            (torch.tensor([0]).to(device=indices.device), indices, torch.tensor([num_preds]).to(device=indices.device)))
        frames_ranges = frames_ranges.long()
        k = frames_ranges.size(0) - 1
        for i in range(k - 1, 0, -1):
            diff = int(im_idx[frames_ranges[i]] - im_idx[frames_ranges[i - 1]])
            if diff > 1:
                frames_ranges = torch.cat((frames_ranges[: i],
                                           torch.tensor([frames_ranges[i] for j in range(diff - 1)]).to(
                                               device=im_idx.device), frames_ranges[i:]))
        if im_idx[0] > 0:
            frames_ranges = torch.cat(
                (torch.tensor([0 for j in range(im_idx[0])]).to(device=im_idx.device), frames_ranges))
        if frames_ranges.size(0) != num_frames + 1:
            frames_ranges = torch.cat((frames_ranges, torch.tensor(
                [num_preds for j in range(num_frames + 1 - frames_ranges.size(0))]).to(device=im_idx.device)))
        entry["times"] = torch.repeat_interleave(times_unique.to(device=global_output.device),
                                                 frames_ranges[1:] - frames_ranges[: -1])
        entry["rng"] = frames_ranges
        times_unique = torch.cat((times_unique, times_extend)).to(device=global_output.device)
        for i in range(1, window + 1):
            # masks for final output latents used during loss evaluation
            mask_preds = torch.tensor([], dtype=torch.long, device=frames_ranges.device)
            mask_gt = torch.tensor([], dtype=torch.long, device=frames_ranges.device)
            gt = gt_annotation.copy()
            for j in range(num_frames - i):
                if testing:
                    a, b = np.array(pred_labels_obj[frames_ranges[j]: frames_ranges[j + 1]].cpu()), np.array(
                        labels_obj[frames_ranges[j + i]: frames_ranges[j + i + 1]].cpu())
                else:
                    a, b = np.array(labels_obj[frames_ranges[j]: frames_ranges[j + 1]].cpu()), np.array(
                        labels_obj[frames_ranges[j + i]: frames_ranges[j + i + 1]].cpu())
                # persistent object labels
                intersection = np.intersect1d(a, b, return_indices=False)
                ind1 = np.array([])  # indices of object labels from last context frame in the intersection
                ind2 = np.array(
                    [])  # indices of object labels that persist in the ith frame after the last context frame
                for element in intersection:
                    tmp1, tmp2 = np.where(a == element)[0], np.where(b == element)[0]
                    mn = min(tmp1.shape[0], tmp2.shape[0])
                    ind1 = np.concatenate((ind1, tmp1[: mn]))
                    ind2 = np.concatenate((ind2, tmp2[: mn]))
                L = []
                if testing:
                    ctr = 0
                    for detection in gt[i + j]:
                        if "class" not in detection.keys() or detection["class"] in intersection:
                            L.append(ctr)
                        ctr += 1
                    # stores modified ground truth
                    gt[i + j] = [gt[i + j][ind] for ind in L]
                ind1 = torch.tensor(ind1, dtype=torch.long, device=frames_ranges.device)
                ind2 = torch.tensor(ind2, dtype=torch.long, device=frames_ranges.device)
                # offset by subject-object pair position
                ind1 += frames_ranges[j]
                ind2 += frames_ranges[j + i]
                mask_preds = torch.cat((mask_preds, ind1))
                mask_gt = torch.cat((mask_gt, ind2))
            entry["mask_curr_%d" % i] = mask_preds
            entry["mask_gt_%d" % i] = mask_gt
            if testing:
                """pair_idx_test = pair_idx[mask_preds]
                _, inverse_indices = torch.unique(pair_idx_test, sorted=True, return_inverse=True)
                entry["im_idx_test_%d" %i] = im_idx[mask_preds]
                entry["pair_idx_test_%d" %i] = inverse_indices
                if self.mode == "predcls":
                    entry["scores_test_%d" %i] = entry["scores"][_.long()]
                    entry["labels_test_%d" %i] = entry["labels"][_.long()]
                else:
                    entry["pred_scores_test_%d" %i] = entry["pred_scores"][_.long()]
                    entry["pred_labels_test_%d" %i] = entry["pred_labels"][_.long()]
                if inverse_indices.size(0) != 0:
                    mx = torch.max(inverse_indices)
                else:
                    mx = -1
                boxes_test = torch.zeros(mx + 1, 5, device=entry["boxes"].device)
                boxes_test[torch.unique_consecutive(inverse_indices[: , 0])] = entry["boxes"][torch.unique_consecutive(pair_idx[mask_gt][: , 0])]
                boxes_test[inverse_indices[: , 1]] = entry["boxes"][pair_idx[mask_gt][: , 1]]
                entry["boxes_test_%d" %i] = boxes_test"""
                # entry["boxes_test_%d" %i] = entry["boxes"][_.long()]
                entry["last_%d" % i] = frames_ranges[-(i + 1)]
                mx = torch.max(pair_idx[: frames_ranges[-(i + 1)]]) + 1
                entry["im_idx_test_%d" % i] = entry["im_idx"][: frames_ranges[-(i + 1)]]
                entry["pair_idx_test_%d" % i] = entry["pair_idx"][: frames_ranges[-(i + 1)]]
                if self.mode == "predcls":
                    entry["scores_test_%d" % i] = entry["scores"][: mx]
                    entry["labels_test_%d" % i] = entry["labels"][: mx]
                else:
                    entry["pred_scores_test_%d" % i] = entry["pred_scores"][: mx]
                    entry["pred_labels_test_%d" % i] = entry["pred_labels"][: mx]
                entry["boxes_test_%d" % i] = torch.ones(mx, 5).to(device=im_idx.device) / 2
                entry["gt_annotation_%d" % i] = gt
        # self.ctr += 1
        # anticipated_latent_loss = 0
        # targets = entry["detached_outputs"]
        for i in range(num_frames - 1):
            end = frames_ranges[i + 1]
            if curr_id == end:
                continue
            batch_y0 = global_output[curr_id: end]
            batch_times = times_unique[i: i + window + 1]
            ret = odeint(self.diff_func, batch_y0, batch_times, method='explicit_adams',
                         options=dict(max_order=4, step_size=1))[1:]
            # ret = odeint(self.diff_func, batch_y0, batch_times, method='dopri5', rtol=1e-2, atol=1e-3)[1 : ]
            anticipated_vals = torch.cat((anticipated_vals, ret), dim=1)
            # obj_bounding_boxes[ :, curr_id : end, : ].data.copy_(self.dsgdetr.get_obj_boxes(ret))
            curr_id = end
        # for p in self.dsgdetr.get_subj_boxes.parameters():
        #    p.requires_grad_(False)
        entry["anticipated_subject_boxes"] = self.base_ldpu.get_subj_boxes(anticipated_vals)
        # for p in self.dsgdetr.get_subj_boxes.parameters():
        #    p.requires_grad_(True)
        entry["anticipated_vals"] = anticipated_vals
        entry["anticipated_attention_distribution"] = self.base_ldpu.a_rel_compress(anticipated_vals)
        entry["anticipated_spatial_distribution"] = torch.sigmoid(self.base_ldpu.s_rel_compress(anticipated_vals))
        entry["anticipated_contacting_distribution"] = torch.sigmoid(self.base_ldpu.c_rel_compress(anticipated_vals))
        # entry["anticipated_object_boxes"] = obj_bounding_boxes
        return entry

    def forward_single_entry(self, context_fraction, entry):
        # [0.3, 0.5, 0.7, 0.9]
        # end = 39
        # future_end = 140
        # future_frame_idx = [40, 41, .............139]
        # Take each entry and extrapolate it to the future
        # evaluation_recall.evaluate_scene_graph_forecasting(self, gt, pred, end, future_end, future_frame_idx, count=0)
        # entry["output"][0] = {pred_scores, pred_labels, attention_distribution, spatial_distribution, contact_distribution}
        assert context_fraction > 0
        entry = self.base_ldpu(entry)
        im_idx = entry["im_idx"]
        pair_idx = entry["pair_idx"]
        gt_annotation = entry["gt_annotation"]
        times = torch.unique(torch.tensor(entry["frame_idx"], dtype=torch.float32).to(device=im_idx.device))
        num_frames = len(gt_annotation)
        num_preds = im_idx.size(0)
        indices = torch.reshape((im_idx[: -1] != im_idx[1:]).nonzero(), (-1,)) + 1
        window = self.max_window
        window = min(window, num_frames - 1)
        global_output = entry["global_output"]
        frames_ranges = torch.cat(
            (torch.tensor([0]).to(device=indices.device), indices, torch.tensor([num_preds]).to(device=indices.device)))
        frames_ranges = frames_ranges.long()
        k = frames_ranges.size(0) - 1
        for i in range(k - 1, 0, -1):
            diff = int(im_idx[frames_ranges[i]] - im_idx[frames_ranges[i - 1]])
            if diff > 1:
                frames_ranges = torch.cat((frames_ranges[: i],
                                           torch.tensor([frames_ranges[i] for j in range(diff - 1)]).to(
                                               device=im_idx.device), frames_ranges[i:]))
        if im_idx[0] > 0:
            frames_ranges = torch.cat(
                (torch.tensor([0 for j in range(im_idx[0])]).to(device=im_idx.device), frames_ranges))
        if frames_ranges.size(0) != num_frames + 1:
            frames_ranges = torch.cat((frames_ranges, torch.tensor(
                [num_preds for j in range(num_frames + 1 - frames_ranges.size(0))]).to(device=im_idx.device)))
        entry["rng"] = frames_ranges
        frames_ranges = entry["rng"]
        pred = {}
        end = int(np.ceil(num_frames * context_fraction) - 1)
        while end > 0 and frames_ranges[end] == frames_ranges[end + 1]:
            end -= 1
        if end == num_frames - 1 or frames_ranges[end] == frames_ranges[end + 1]:
            return num_frames, pred
        ret = odeint(self.diff_func, global_output[frames_ranges[end]: frames_ranges[end + 1]], times[end:],
                     method='explicit_adams', options=dict(max_order=4, step_size=1))[1:]
        pred["attention_distribution"] = torch.flatten(self.base_ldpu.a_rel_compress(ret), start_dim=0, end_dim=1)
        pred["spatial_distribution"] = torch.flatten(torch.sigmoid(self.base_ldpu.s_rel_compress(ret)), start_dim=0,
                                                     end_dim=1)
        pred["contacting_distribution"] = torch.flatten(torch.sigmoid(self.base_ldpu.c_rel_compress(ret)), start_dim=0,
                                                        end_dim=1)
        if self.mode == "predcls":
            pred["scores"] = entry["scores"][torch.min(pair_idx[frames_ranges[end]: frames_ranges[end + 1]]): torch.max(
                pair_idx[frames_ranges[end]: frames_ranges[end + 1]]) + 1].repeat(num_frames - end - 1)
            pred["labels"] = entry["labels"][torch.min(pair_idx[frames_ranges[end]: frames_ranges[end + 1]]): torch.max(
                pair_idx[frames_ranges[end]: frames_ranges[end + 1]]) + 1].repeat(num_frames - end - 1)
        else:
            pred["pred_scores"] = entry["pred_scores"][
                                  torch.min(pair_idx[frames_ranges[end]: frames_ranges[end + 1]]): torch.max(
                                      pair_idx[frames_ranges[end]: frames_ranges[end + 1]]) + 1].repeat(
                num_frames - end - 1)
            pred["pred_labels"] = entry["pred_labels"][
                                  torch.min(pair_idx[frames_ranges[end]: frames_ranges[end + 1]]): torch.max(
                                      pair_idx[frames_ranges[end]: frames_ranges[end + 1]]) + 1].repeat(
                num_frames - end - 1)
        pred["im_idx"] = torch.tensor(
            [i for i in range(num_frames - end - 1) for j in range(frames_ranges[end + 1] - frames_ranges[end])],
            dtype=torch.int32).to(device=frames_ranges.device)
        mx = torch.max(pair_idx[frames_ranges[end]: frames_ranges[end + 1]]) - torch.min(
            pair_idx[frames_ranges[end]: frames_ranges[end + 1]]) + 1
        pred["pair_idx"] = (pair_idx[frames_ranges[end]: frames_ranges[end + 1]] - torch.min(
            pair_idx[frames_ranges[end]: frames_ranges[end + 1]])).repeat(num_frames - end - 1, 1) + mx * torch.reshape(
            pred["im_idx"], (-1, 1))
        pred["boxes"] = torch.ones(mx * (num_frames - end - 1), 5).to(device=im_idx.device) / 2
        return end + 1, pred
