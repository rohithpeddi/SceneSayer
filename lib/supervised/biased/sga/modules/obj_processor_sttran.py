import math

from torch import nn

from lib.supervised.biased.sga.blocks import ObjectClassifierMLP, ObjectAnticipation

"""
1. ObjectClassifierMLP
2. No Tracking
3. Uses object anticipation decoder for generating future embeddings
"""


class ObjProcessorSTTran(nn.Module):

    def __init__(self, mode='sgdet', obj_classes=None):
        super(ObjProcessorSTTran, self).__init__()
        self.obj_classes = obj_classes

        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode
        self.num_features = 1936

        self.object_classifier = ObjectClassifierMLP(mode=self.mode, obj_classes=self.obj_classes)
        self.obj_anti_temporal_transformer = ObjectAnticipation(mode=self.mode, obj_classes=self.obj_classes)

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
            result_cf_ff[count] = entry_cf_ff
            result_ff[count] = entry_ff
            count += 1
            num_cf += 1
        entry["output"] = result_cf_ff
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
        result_cf_ff[count] = entry_cf_ff
        result_ff[count] = entry_ff
        entry["output_cf_ff"] = result_cf_ff
        entry["output_ff"] = result_ff
        return entry
