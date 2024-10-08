import os

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from constants import Constants as const
from dataloader.action_genome.ag_features import AGFeatures
from dataloader.action_genome.ag_features import cuda_collate_fn as ag_features_cuda_collate_fn
from dataloader.action_genome import AG
from dataloader.action_genome import cuda_collate_fn as ag_data_cuda_collate_fn

from lib.AdamW import AdamW
from lib.supervised.config import Config
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator


def prepare_optimizer(conf, model):
    if conf.optimizer == const.ADAMW:
        optimizer = AdamW(model.parameters(), lr=conf.lr)
    elif conf.optimizer == const.ADAM:
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.optimizer == const.SGD:
        optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)
    else:
        raise NotImplementedError

    scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True, threshold=1e-4,
                                  threshold_mode="abs", min_lr=1e-7)
    return optimizer, scheduler


def fetch_loss_functions():
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()
    mlm_loss = nn.MultiLabelMarginLoss()
    bbox_loss = nn.SmoothL1Loss()
    abs_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    return bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss


def fetch_transformer_loss_functions():
    bce_loss = nn.BCELoss(reduction='none')
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    mlm_loss = nn.MultiLabelMarginLoss(reduction='none')
    bbox_loss = nn.SmoothL1Loss()
    abs_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    return bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss


def fetch_train_basic_config():
    conf = Config()
    print('The CKPT saved here:', conf.save_path)
    if not os.path.exists(conf.save_path):
        os.mkdir(conf.save_path)
    print('spatial encoder layer num: {} / temporal decoder layer num: {}'.format(conf.enc_layer, conf.dec_layer))
    for i in conf.args:
        print(i, ':', conf.args[i])

    # Set the preferred device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not conf.use_raw_data:
        ag_train_data = AGFeatures(
            mode=conf.mode,
            data_split=const.TRAIN,
            device=device,
            data_path=conf.data_path,
            is_compiled_together=False,
            filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == const.PREDCLS else True,
            features_path=conf.features_path,
            additional_data_path=conf.additional_data_path
        )

        ag_test_data = AGFeatures(
            mode=conf.mode,
            data_split=const.TEST,
            device=device,
            data_path=conf.data_path,
            is_compiled_together=False,
            filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == const.PREDCLS else True,
            features_path=conf.features_path,
            additional_data_path=conf.additional_data_path
        )

        dataloader_train = DataLoader(
            ag_train_data,
            shuffle=True,
            collate_fn=ag_features_cuda_collate_fn,
            pin_memory=False,
            num_workers=0
        )

        dataloader_test = DataLoader(
            ag_test_data,
            shuffle=False,
            collate_fn=ag_features_cuda_collate_fn,
            pin_memory=False
        )
    else:
        ag_train_data = AG(
            phase="train",
            datasize=conf.datasize,
            data_path=conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == 'predcls' else True
        )

        ag_test_data = AG(
            phase="test",
            datasize=conf.datasize,
            data_path=conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == 'predcls' else True
        )

        dataloader_train = DataLoader(
            ag_train_data,
            shuffle=True,
            collate_fn=ag_data_cuda_collate_fn,
            pin_memory=True,
            num_workers=0
        )

        dataloader_test = DataLoader(
            ag_test_data,
            shuffle=False,
            collate_fn=ag_data_cuda_collate_fn,
            pin_memory=False
        )

    gpu_device = torch.device("cuda:0")

    evaluator = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=ag_train_data.object_classes,
        AG_all_predicates=ag_train_data.relationship_classes,
        AG_attention_predicates=ag_train_data.attention_relationships,
        AG_spatial_predicates=ag_train_data.spatial_relationships,
        AG_contacting_predicates=ag_train_data.contacting_relationships,
        iou_threshold=0.0,
        save_file=os.path.join(conf.save_path, const.PROGRESS_TEXT_FILE),
        constraint='with'
    )

    return conf, dataloader_train, dataloader_test, gpu_device, evaluator, ag_train_data, ag_test_data


def save_model(model, epoch, checkpoint_save_file_path, checkpoint_name, model_name):
    os.makedirs(os.path.dirname(checkpoint_save_file_path), exist_ok=True)
    torch.save({f"{model_name}_state_dict": model.state_dict()},
               os.path.join(checkpoint_save_file_path, f"{checkpoint_name}_epoch_{epoch}.tar"))
    print("*" * 40)
    print("Saved the checkpoint after {} epochs".format(epoch))


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


def load_model_from_checkpoint(model, conf, gpu_device):
    if conf.ckpt is not None:
        ckpt = torch.load(conf.ckpt, map_location=gpu_device)
        model.load_state_dict(ckpt[f'{conf.method_name}_state_dict'], strict=False)
        print(f"Loaded model from checkpoint {conf.ckpt}")
    else:
        print("No checkpoint to load from...... Training from scratch.")
    return model
