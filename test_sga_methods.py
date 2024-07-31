"""
1. STTran
2. DsgDetr
3. Tempura
4. Trace
"""
import torch
from lib.supervised.config import Config
from test_sga_base import TestSGABase


def get_sequence_no_tracking(entry, task="sgcls"):
    if task == "predcls":
        indices = []
        for i in entry["labels"].unique():
            indices.append(torch.where(entry["labels"] == i)[0])
        entry["indices"] = indices
        return

    if task == "sgdet" or task == "sgcls":
        # for sgdet, use the predicted object classes, as a special case of
        # the proposed method, comment this out for general coase tracking.
        indices = [[]]
        # indices[0] store single-element sequence, to save memory
        pred_labels = torch.argmax(entry["distribution"], 1)
        for i in pred_labels.unique():
            index = torch.where(pred_labels == i)[0]
            if len(index) == 1:
                indices[0].append(index)
            else:
                indices.append(index)
        if len(indices[0]) > 0:
            indices[0] = torch.cat(indices[0])
        else:
            indices[0] = torch.tensor([])
        entry["indices"] = indices
        return


class TestODE(TestSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sga.scene_sayer_ode import SceneSayerODE
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher

        self._model = SceneSayerODE(mode=self._conf.mode,
                                    attention_class_num=len(self._test_dataset.attention_relationships),
                                    spatial_class_num=len(self._test_dataset.spatial_relationships),
                                    contact_class_num=len(self._test_dataset.contacting_relationships),
                                    obj_classes=self._test_dataset.object_classes,
                                    max_window=self._conf.max_window).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)
        self._matcher.eval()

    def process_test_video_context(self, video_entry, frame_size, gt_annotation, context_fraction) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(video_entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        video_entry["gt_annotation"] = gt_annotation
        # video_entry = self._model(video_entry, True)
        ind, pred = self._model.forward_single_entry(context_fraction=context_fraction, entry=video_entry)
        skip = False
        if ind >= len(gt_annotation):
            skip = True
        return skip, ind, pred

    def process_test_video_future_frame(self, video_entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(video_entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        video_entry["gt_annotation"] = gt_annotation
        pred = self._model(video_entry, True)
        return pred


class TestSDE(TestSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sga.scene_sayer_sde import SceneSayerSDE
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher

        self._model = SceneSayerSDE(mode=self._conf.mode,
                                    attention_class_num=len(self._test_dataset.attention_relationships),
                                    spatial_class_num=len(self._test_dataset.spatial_relationships),
                                    contact_class_num=len(self._test_dataset.contacting_relationships),
                                    obj_classes=self._test_dataset.object_classes,
                                    max_window=self._conf.max_window,
                                    brownian_size=self._conf.brownian_size).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)
        self._matcher.eval()

    def process_test_video_context(self, video_entry, frame_size, gt_annotation, context_fraction) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(video_entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        video_entry["gt_annotation"] = gt_annotation
        # video_entry = self._model(video_entry, True)
        ind, pred = self._model.forward_single_entry(context_fraction=context_fraction, entry=video_entry)
        skip = False
        if ind >= len(gt_annotation):
            skip = True
        return skip, ind, pred

    def process_test_video_future_frame(self, video_entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(video_entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        video_entry["gt_annotation"] = gt_annotation
        pred = self._model(video_entry, True)
        return pred


class TestSTTranAnt(TestSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.sga.sttran_ant import STTranAnt

        self._model = STTranAnt(mode=self._conf.mode,
                                attention_class_num=len(self._test_dataset.attention_relationships),
                                spatial_class_num=len(self._test_dataset.spatial_relationships),
                                contact_class_num=len(self._test_dataset.contacting_relationships),
                                obj_classes=self._test_dataset.object_classes,
                                enc_layer_num=self._conf.enc_layer,
                                dec_layer_num=self._conf.dec_layer).to(device=self._device)

    def process_test_video_context(self, video_entry, frame_size, gt_annotation, context_fraction) -> dict:
        import math
        get_sequence_no_tracking(video_entry, self._conf)
        video_entry["gt_annotation"] = gt_annotation
        pred = self._model.forward_single_entry(context_fraction=context_fraction, entry=video_entry)
        num_tf = len(video_entry["im_idx"].unique())
        num_cf = min(int(math.ceil(context_fraction * num_tf)), num_tf - 1)
        gt_future = gt_annotation[num_cf: num_tf]
        pred_dict = pred["output"][0]
        return gt_future, pred_dict


class TestSTTranGenAnt(TestSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.sga.sttran_gen_ant import STTranGenAnt

        self._model = STTranGenAnt(mode=self._conf.mode,
                                   attention_class_num=len(self._test_dataset.attention_relationships),
                                   spatial_class_num=len(self._test_dataset.spatial_relationships),
                                   contact_class_num=len(self._test_dataset.contacting_relationships),
                                   obj_classes=self._test_dataset.object_classes,
                                   enc_layer_num=self._conf.enc_layer,
                                   dec_layer_num=self._conf.dec_layer).to(device=self._device)

    def process_test_video_context(self, video_entry, frame_size, gt_annotation, context_fraction) -> dict:
        import math
        get_sequence_no_tracking(video_entry, self._conf)
        video_entry["gt_annotation"] = gt_annotation
        pred = self._model.forward_single_entry(context_fraction=context_fraction, entry=video_entry)
        num_tf = len(video_entry["im_idx"].unique())
        num_cf = min(int(math.ceil(context_fraction * num_tf)), num_tf - 1)
        gt_future = gt_annotation[num_cf: num_tf]
        pred_dict = pred["output"][0]
        return gt_future, pred_dict


class TestDsgDetrAnt(TestSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
        from lib.supervised.sga.dsgdetr_ant import DsgDetrAnt

        self._model = DsgDetrAnt(mode=self._conf.mode,
                                 attention_class_num=len(self._test_dataset.attention_relationships),
                                 spatial_class_num=len(self._test_dataset.spatial_relationships),
                                 contact_class_num=len(self._test_dataset.contacting_relationships),
                                 obj_classes=self._test_dataset.object_classes,
                                 enc_layer_num=self._conf.enc_layer,
                                 dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)
        self._matcher.eval()

    def process_test_video_context(self, video_entry, frame_size, gt_annotation, context_fraction) -> dict:
        import math
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(video_entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        video_entry["gt_annotation"] = gt_annotation
        pred = self._model.forward_single_entry(context_fraction=context_fraction, entry=video_entry)
        num_tf = len(video_entry["im_idx"].unique())
        num_cf = min(int(math.ceil(context_fraction * num_tf)), num_tf - 1)
        gt_future = gt_annotation[num_cf: num_tf]
        pred_dict = pred["output"][0]
        return gt_future, pred_dict


class TestDsgDetrGenAnt(TestSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
        from lib.supervised.sga.dsgdetr_gen_ant import DsgDetrGenAnt

        self._model = DsgDetrGenAnt(mode=self._conf.mode,
                                    attention_class_num=len(self._test_dataset.attention_relationships),
                                    spatial_class_num=len(self._test_dataset.spatial_relationships),
                                    contact_class_num=len(self._test_dataset.contacting_relationships),
                                    obj_classes=self._test_dataset.object_classes,
                                    enc_layer_num=self._conf.enc_layer,
                                    dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)
        self._matcher.eval()

    def process_test_video_context(self, video_entry, frame_size, gt_annotation, context_fraction) -> dict:
        import math
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(video_entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        video_entry["gt_annotation"] = gt_annotation
        pred = self._model.forward_single_entry(context_fraction=context_fraction, entry=video_entry)
        num_tf = len(video_entry["im_idx"].unique())
        num_cf = min(int(math.ceil(context_fraction * num_tf)), num_tf - 1)
        gt_future = gt_annotation[num_cf: num_tf]
        pred_dict = pred["output"][0]
        return gt_future, pred_dict


def main():
    conf = Config()
    if conf.method_name == "ode":
        evaluate_class = TestODE(conf)
    elif conf.method_name == "sde":
        evaluate_class = TestSDE(conf)
    elif conf.method_name == "sttran_ant":
        evaluate_class = TestSTTranAnt(conf)
    elif conf.method_name == "sttran_gen_ant":
        evaluate_class = TestSTTranGenAnt(conf)
    elif conf.method_name == "dsgdetr_ant":
        evaluate_class = TestDsgDetrAnt(conf)
    elif conf.method_name == "dsgdetr_gen_ant":
        evaluate_class = TestDsgDetrGenAnt(conf)
    else:
        raise NotImplementedError

    evaluate_class.init_method_evaluation()


if __name__ == "__main__":
    main()
