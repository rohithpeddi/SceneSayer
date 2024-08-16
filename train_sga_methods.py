import torch
from lib.supervised.config import Config
from train_sga_base import TrainSGABase


# -------------------------------------------------------------------------------------
# ------------------------------- BASELINE METHODS ---------------------------------
# -------------------------------------------------------------------------------------

class TrainSttranAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def process_train_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def process_test_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def compute_loss(self, pred, gt) -> dict:
        raise NotImplementedError

    def process_evaluation_score(self, pred, gt):
        raise NotImplementedError


class TrainSttranGenAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def process_train_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def process_test_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def compute_loss(self, pred, gt) -> dict:
        raise NotImplementedError

    def process_evaluation_score(self, pred, gt):
        raise NotImplementedError


class TrainDsgDetrAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def process_train_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def process_test_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def compute_loss(self, pred, gt) -> dict:
        raise NotImplementedError

    def process_evaluation_score(self, pred, gt):
        raise NotImplementedError


class TrainDsgDetrGenAnt(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def process_train_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def process_test_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def compute_loss(self, pred, gt) -> dict:
        raise NotImplementedError

    def process_evaluation_score(self, pred, gt):
        raise NotImplementedError


# -------------------------------------------------------------------------------------
# ------------------------------- SCENE SAYER METHODS ---------------------------------
# -------------------------------------------------------------------------------------

class TrainODE(TrainSGABase):

    def __init__(self, conf):
        super().__init__(conf)

    def process_train_video(self, video, frame_size) -> dict:
        raise NotImplementedError

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

    def process_train_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def process_test_video(self, video, frame_size) -> dict:
        raise NotImplementedError

    def compute_loss(self, pred, gt) -> dict:
        losses = self.compute_scene_sayer_loss(pred, self._conf.sde_ratio)
        return losses

    def process_evaluation_score(self, pred, gt):
        raise NotImplementedError
