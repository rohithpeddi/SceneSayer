from constants import Constants as const
from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
from train_sga_base import TrainSGABase


# -------------------------------------------------------------------------------------
# ------------------------------- BASELINE METHODS ---------------------------------
# -------------------------------------------------------------------------------------

class TrainSttranAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def process_train_video(self, entry, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.baseline_future)
        return pred

    def process_test_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_ant_loss(pred, gt)
        return losses

    def process_evaluation_score(self, pred, gt):
        raise NotImplementedError


class TrainSttranGenAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def process_train_video(self, entry, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.baseline_future)
        return pred

    def process_test_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_gen_ant_loss(pred, gt)
        return losses

    def process_evaluation_score(self, pred, gt):
        raise NotImplementedError


class TrainDsgDetrAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def process_train_video(self, entry, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.baseline_future)
        return pred

    def process_test_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_ant_loss(pred, gt)
        return losses

    def process_evaluation_score(self, pred, gt):
        raise NotImplementedError


class TrainDsgDetrGenAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def process_train_video(self, entry, frame_size) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry, self._conf.baseline_context, self._conf.baseline_future)
        return pred

    def process_test_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_baseline_gen_ant_loss(pred, gt)
        return losses

    def process_evaluation_score(self, pred, gt):
        raise NotImplementedError


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

    def process_test_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_scene_sayer_loss(pred, self._conf.ode_ratio)
        return losses

    def process_evaluation_score(self, pred, gt):
        raise NotImplementedError


class TrainSDE(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def process_train_video(self, entry, frame_size) -> dict:
        gt_annotation = entry[const.GT_ANNOTATION]
        frame_size = entry[const.FRAME_SIZE]
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_scene_sayer_loss(pred, self._conf.sde_ratio)
        return losses

    def process_evaluation_score(self, pred, gt):
        raise NotImplementedError
