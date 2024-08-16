from constants import Constants as const
from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
from train_sga_base import TrainSGABase


# -------------------------------------------------------------------------------------
# ------------------------------- BASELINE METHODS ---------------------------------
# -------------------------------------------------------------------------------------

class TrainSttranAnt(TrainSGABase):

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

    def process_train_video(self, entry, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.baseline_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        self.get_sequence_no_tracking(entry)
        num_ff = self._conf.baseline_future
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


class TrainSttranGenAnt(TrainSGABase):

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

    def process_train_video(self, entry, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.baseline_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        self.get_sequence_no_tracking(entry)
        num_ff = self._conf.baseline_future
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_gen_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


class TrainDsgDetrAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._init_matcher()

    def init_model(self):
        from lib.supervised.sga.dsgdetr_ant import DsgDetrAnt

        self._model = DsgDetrAnt(mode=self._conf.mode,
                                 attention_class_num=len(self._test_dataset.attention_relationships),
                                 spatial_class_num=len(self._test_dataset.spatial_relationships),
                                 contact_class_num=len(self._test_dataset.contacting_relationships),
                                 obj_classes=self._test_dataset.object_classes,
                                 enc_layer_num=self._conf.enc_layer,
                                 dec_layer_num=self._conf.dec_layer).to(device=self._device)
        self._init_matcher()

    def process_train_video(self, entry, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.baseline_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(self._conf, entry, gt_annotation, self._matcher, frame_size)
        num_ff = self._conf.baseline_future
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


class TrainDsgDetrGenAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._init_matcher()

    def init_model(self):
        from lib.supervised.sga.dsgdetr_gen_ant import DsgDetrGenAnt

        self._model = DsgDetrGenAnt(mode=self._conf.mode,
                                    attention_class_num=len(self._test_dataset.attention_relationships),
                                    spatial_class_num=len(self._test_dataset.spatial_relationships),
                                    contact_class_num=len(self._test_dataset.contacting_relationships),
                                    obj_classes=self._test_dataset.object_classes,
                                    enc_layer_num=self._conf.enc_layer,
                                    dec_layer_num=self._conf.dec_layer).to(device=self._device)
        self._init_matcher()

    def process_train_video(self, entry, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.baseline_future)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(self._conf, entry, gt_annotation, self._matcher, frame_size)
        num_ff = self._conf.baseline_future
        num_cf = self._conf.baseline_context
        pred = self._model(entry, num_cf, num_ff)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_gen_ant_loss(pred)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_baseline_evaluation_score(pred, gt)


# -------------------------------------------------------------------------------------
# ------------------------------- SCENE SAYER METHODS ---------------------------------
# -------------------------------------------------------------------------------------

class TrainODE(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._init_matcher()

    def process_train_video(self, entry, frame_size) -> dict:
        gt_annotation = entry[const.GT_ANNOTATION]
        frame_size = entry[const.FRAME_SIZE]
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(self._conf, entry, gt_annotation, self._matcher, frame_size)
        pred = self._model(entry, True)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_scene_sayer_loss(pred, self._conf.ode_ratio)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_scene_sayer_evaluation_score(pred, gt)


class TrainSDE(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)
        self._init_matcher()

    def process_train_video(self, entry, frame_size) -> dict:
        gt_annotation = entry[const.GT_ANNOTATION]
        frame_size = entry[const.FRAME_SIZE]
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, gt_annotation, frame_size) -> dict:
        get_sequence_with_tracking(self._conf, entry, gt_annotation, self._matcher, frame_size)
        pred = self._model(entry, True)
        return pred

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_scene_sayer_loss(pred, self._conf.sde_ratio)
        return losses

    def process_evaluation_score(self, pred, gt):
        self.compute_scene_sayer_evaluation_score(pred, gt)
