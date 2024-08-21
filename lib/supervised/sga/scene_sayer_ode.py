import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

from lib.supervised.sga.base_ldpu_diffeq import BaseLDPU, BaseLDPUDiffEq


class SceneSayerODEDerivatives(nn.Module):
    def __init__(self):
        super(SceneSayerODEDerivatives, self).__init__()

        self.net = nn.Sequential(nn.Linear(1936, 2048), nn.Tanh(),
                                 nn.Linear(2048, 2048), nn.Tanh(),
                                 nn.Linear(2048, 1936))

    def forward(self, t, y):
        out = self.net(y)
        return out


class SceneSayerODE(BaseLDPUDiffEq):

    def __init__(
            self,
            mode,
            attention_class_num=None,
            spatial_class_num=None,
            contact_class_num=None,
            obj_classes=None,
            rel_classes=None,
            max_window=None
    ):
        super(SceneSayerODE, self).__init__(
            mode,
            attention_class_num,
            spatial_class_num,
            contact_class_num,
            obj_classes,
            rel_classes,
            max_window
        )
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
        entry, num_frames, frames_ranges, times_unique, window, global_output, anticipated_vals = self.process_ff_rels(
            entry, testing)
        curr_id = 0
        for i in range(num_frames - 1):
            end = frames_ranges[i + 1]
            if curr_id == end:
                continue
            batch_y0 = global_output[curr_id: end]
            batch_times = times_unique[i: i + window + 1]
            ret = odeint(
                self.diff_func,
                batch_y0,
                batch_times,
                method='explicit_adams',
                options=dict(max_order=4, step_size=1))[1:]
            anticipated_vals = torch.cat((anticipated_vals, ret), dim=1)
            curr_id = end
        entry = self.process_ff_rels_output_entry(entry, anticipated_vals)
        return entry

    def forward_single_entry(self, context_fraction, entry):
        entry, end, num_frames, frames_ranges, global_output, pair_idx, im_idx, times = self.process_context_fraction_rels(
            entry, context_fraction)

        pred = {}
        if end == num_frames - 1 or frames_ranges[end] == frames_ranges[end + 1]:
            return num_frames, pred
        ret = odeint(
            self.diff_func,
            global_output[frames_ranges[end]: frames_ranges[end + 1]],
            times[end:],
            method='explicit_adams', options=dict(max_order=4, step_size=1))[1:]
        pred = self.process_context_fraction_output_pred(entry, end, num_frames, frames_ranges, pair_idx, im_idx, ret, pred)
        return end + 1, pred
