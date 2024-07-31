import os
import time

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from constants import Constants as const
from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
from lib.supervised.sga import SceneSayerODE as ODE
from lib.supervised.sga import SceneSayerSDE
from train_base import fetch_train_basic_config, prepare_optimizer, fetch_loss_functions, save_model


def train_model(conf, model, matcher, optimizer, dataloader_train, model_ratio, tr, epoch):
    model.train()
    num = 0
    start = time.time()
    bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss = fetch_loss_functions()
    for entry in tqdm(dataloader_train, position=0, leave=True):
        gt_annotation = entry[const.GT_ANNOTATION]
        frame_size = entry[const.FRAME_SIZE]
        get_sequence_with_tracking(entry, gt_annotation, matcher, frame_size, conf.mode)
        pred = model(entry)
        global_output = pred["global_output"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contacting_distribution"]
        attention_distribution = pred["attention_distribution"]
        subject_boxes_rcnn = pred["subject_boxes_rcnn"]
        # object_boxes_rcnn = pred["object_boxes_rcnn"]
        subject_boxes_dsg = pred["subject_boxes_dsg"]
        # object_boxes_dsg = pred["object_boxes_dsg"]
        
        anticipated_global_output = pred["anticipated_vals"]
        anticipated_subject_boxes = pred["anticipated_subject_boxes"]
        # targets = pred["detached_outputs"]
        anticipated_spatial_distribution = pred["anticipated_spatial_distribution"]
        anticipated_contact_distribution = pred["anticipated_contacting_distribution"]
        anticipated_attention_distribution = pred["anticipated_attention_distribution"]
        # anticipated_object_boxes = pred["anticipated_object_boxes"]
        
        attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to \
            (device=attention_distribution.device).squeeze()
        if not conf.bce_loss:
            # multi-label margin loss or adaptive loss
            spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to \
                (device=attention_distribution.device)
            contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to \
                (device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
                contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])
        
        else:
            # bce loss
            spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to \
                (device=attention_distribution.device)
            contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to \
                (device=attention_distribution.device)
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1
                contact_label[i, pred["contacting_gt"][i]] = 1
        
        losses = {}
        if conf.mode == 'sgcls' or conf.mode == 'sgdet':
            losses['object_loss'] = ce_loss(pred['distribution'], pred['labels'])
        
        losses["attention_relation_loss"] = ce_loss(attention_distribution, attention_label)
        losses["subject_boxes_loss"] = conf.bbox_ratio * bbox_loss(subject_boxes_dsg, subject_boxes_rcnn)
        # losses["object_boxes_loss"] = bbox_ratio * bbox_loss(object_boxes_dsg, object_boxes_rcnn)
        losses["anticipated_latent_loss"] = 0
        losses["anticipated_subject_boxes_loss"] = 0
        losses["anticipated_spatial_relation_loss"] = 0
        losses["anticipated_contact_relation_loss"] = 0
        losses["anticipated_attention_relation_loss"] = 0
        # losses["anticipated_object_boxes_loss"] = 0
        if not conf.bce_loss:
            losses["spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = mlm_loss(contact_distribution, contact_label)
            for i in range(1, conf.max_window + 1):
                mask_curr = entry["mask_curr_" + str(i)]
                mask_gt = entry["mask_gt_" + str(i)]
                losses["anticipated_latent_loss"] += model_ratio * abs_loss(
                    anticipated_global_output[i - 1][mask_curr],
                    global_output[mask_gt])
                losses["anticipated_subject_boxes_loss"] += conf.bbox_ratio * bbox_loss \
                    (anticipated_subject_boxes[i - 1][mask_curr], subject_boxes_rcnn[mask_gt])
                losses["anticipated_spatial_relation_loss"] += mlm_loss \
                    (anticipated_spatial_distribution[i - 1][mask_curr], spatial_label[mask_gt])
                losses["anticipated_contact_relation_loss"] += mlm_loss \
                    (anticipated_contact_distribution[i - 1][mask_curr], contact_label[mask_gt])
                losses["anticipated_attention_relation_loss"] += ce_loss \
                    (anticipated_attention_distribution[i - 1][mask_curr], attention_label[mask_gt])
        # losses["anticipated_object_boxes_loss"] += bbox_ratio * bbox_loss(anticipated_object_boxes[i - 1][mask_curr], object_boxes_rcnn[mask_gt])
        else:
            losses["spatial_relation_loss"] = bce_loss(spatial_distribution, spatial_label)
            losses["contact_relation_loss"] = bce_loss(contact_distribution, contact_label)
            for i in range(1, conf.max_window + 1):
                mask_curr = entry["mask_curr_" + str(i)]
                mask_gt = entry["mask_gt_" + str(i)]
                losses["anticipated_latent_loss"] += model_ratio * abs_loss(
                    anticipated_global_output[i - 1][mask_curr],
                    global_output[mask_gt])
                losses["anticipated_subject_boxes_loss"] += conf.bbox_ratio * bbox_loss \
                    (anticipated_subject_boxes[i - 1][mask_curr], subject_boxes_rcnn[mask_gt])
                losses["anticipated_spatial_relation_loss"] += bce_loss \
                    (anticipated_spatial_distribution[i - 1][mask_curr], spatial_label[mask_gt])
                losses["anticipated_contact_relation_loss"] += bce_loss \
                    (anticipated_contact_distribution[i - 1][mask_curr], contact_label[mask_gt])
                losses["anticipated_attention_relation_loss"] += ce_loss \
                    (anticipated_attention_distribution[i - 1][mask_curr], attention_label[mask_gt])
        # losses["anticipated_object_boxes_loss"] += bbox_ratio * bbox_loss(anticipated_object_boxes[i - 1][mask_curr], object_boxes_rcnn[mask_gt])
        # optimizer_diff.zero_grad()
        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(diff_func.parameters(), max_norm=5, norm_type=2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        # optimizer_diff.step()
        optimizer.step()
        tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
        num += 1
        if num % 1000 == 0 and num >= 1000:
            time_per_batch = (time.time() - start) / 1000
            print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, num, len(dataloader_train),
                                                                                time_per_batch,
                                                                                len(dataloader_train) * time_per_batch / 60))
            
            mn = pd.concat(tr[-1000:], axis=1).mean(1)
            print(mn)
            start = time.time()


def test_model(model, dataloader_test, evaluator, conf, matcher):
    model.eval()
    with torch.no_grad():
        for entry in tqdm(dataloader_test, position=0, leave=True):
            gt_annotation = entry[const.GT_ANNOTATION]
            frame_size = entry[const.FRAME_SIZE]
            get_sequence_with_tracking(entry, gt_annotation, matcher, frame_size, conf.mode)
            w = conf.max_window
            n = len(gt_annotation)
            w = min(w, n - 1)
            pred = model(entry, True)
            for i in range(1, w + 1):
                pred_anticipated = pred.copy()
                mask_curr = pred["mask_curr_" + str(i)]
                pred_anticipated["spatial_distribution"] = pred["anticipated_spatial_distribution"][i - 1][
                    mask_curr]
                pred_anticipated["contacting_distribution"] = pred["anticipated_contacting_distribution"][i - 1][
                    mask_curr]
                pred_anticipated["attention_distribution"] = pred["anticipated_attention_distribution"][i - 1][
                    mask_curr]
                pred_anticipated["im_idx"] = pred["im_idx_test_" + str(i)]
                pred_anticipated["pair_idx"] = pred["pair_idx_test_" + str(i)]
                if conf.mode == "predcls":
                    pred_anticipated["scores"] = pred["scores_test_" + str(i)]
                    pred_anticipated["labels"] = pred["labels_test_" + str(i)]
                else:
                    pred_anticipated["pred_scores"] = pred["pred_scores_test_" + str(i)]
                    pred_anticipated["pred_labels"] = pred["pred_labels_test_" + str(i)]
                pred_anticipated["boxes"] = pred["boxes_test_" + str(i)]
                evaluator.evaluate_scene_graph(gt_annotation[i:], pred_anticipated)


def load_ode(conf, ag_train_data, gpu_device):
    max_window = conf.max_window
    
    ode = ODE(mode=conf.mode,
              attention_class_num=len(ag_train_data.attention_relationships),
              spatial_class_num=len(ag_train_data.spatial_relationships),
              contact_class_num=len(ag_train_data.contacting_relationships),
              obj_classes=ag_train_data.object_classes,
              enc_layer_num=conf.enc_layer,
              dec_layer_num=conf.dec_layer,
              max_window=max_window).to(device=gpu_device)
    
    if conf.ckpt:
        ckpt = torch.load(conf.ckpt, map_location=gpu_device)
        ode.load_state_dict(ckpt['ode_state_dict'], strict=False)
    
    optimizer, _ = prepare_optimizer(conf, ode)
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.25, verbose=True, threshold=1e-4,
                                  threshold_mode="abs", min_lr=1e-7)
    
    return ode, optimizer, scheduler


def load_sde(conf, ag_train_data, gpu_device):
    max_window = conf.max_window
    brownian_size = conf.brownian_size
    
    sde = SDE(mode=conf.mode,
              attention_class_num=len(ag_train_data.attention_relationships),
              spatial_class_num=len(ag_train_data.spatial_relationships),
              contact_class_num=len(ag_train_data.contacting_relationships),
              obj_classes=ag_train_data.object_classes,
              enc_layer_num=conf.enc_layer,
              dec_layer_num=conf.dec_layer,
              max_window=max_window,
              brownian_size=brownian_size).to(device=gpu_device)
    
    if conf.ckpt:
        ckpt = torch.load(conf.ckpt, map_location=gpu_device)
        sde.load_state_dict(ckpt['sde_state_dict'], strict=False)
    
    optimizer, _ = prepare_optimizer(conf, sde)
    scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.25, verbose=True, threshold=1e-4,
                                  threshold_mode="abs", min_lr=1e-7)
    
    return sde, optimizer, scheduler


def main():
    conf, dataloader_train, dataloader_test, gpu_device, evaluator, ag_train_data, ag_test_data = fetch_train_basic_config()
    method_name = conf.method_name
    matcher = HungarianMatcher(0.5, 1, 1, 0.5)
    matcher.eval()
    model, optimizer, scheduler = None, None, None
    model_ratio = None
    model_name = None
    if method_name == "ode":
        model_ratio = conf.ode_ratio
        model_name = "ode"
        model, optimizer, scheduler = load_ode(conf, ag_train_data, gpu_device)
    elif method_name == "sde":
        model_ratio = conf.sde_ratio
        model_name = "sde"
        model, optimizer, scheduler = load_sde(conf, ag_train_data, gpu_device)
    
    assert model is not None and optimizer is not None and scheduler is not None
    assert model_ratio is not None
    assert model_name is not None
    
    checkpoint_name = f"{model_name}_{conf.mode}_future_{conf.max_window}"
    checkpoint_save_file_path = os.path.join(conf.save_path, model_name)
    tr = []
    for epoch in range(conf.num_epochs):
        train_model(conf, model, matcher, optimizer, dataloader_train, model_ratio, tr, epoch)
        save_model(model, epoch, checkpoint_save_file_path, checkpoint_name, model_name)
        test_model(model, dataloader_test, evaluator, conf, matcher)
        score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
        evaluator.print_stats()
        evaluator.reset_result()
        scheduler.step(score)


if __name__ == '__main__':
    # Required attributes to be set in conf:
    # method_name: ["ode", "sde"]
    main()

# python train_latent_diffeq.py -mode sgdet -method_name ode -max_window 5 -features_path /data/rohith/ag/features/ -additional_data_path /data/rohith/ag/additional_data/

""" python train_DSG_masked.py -mode sgdet -save_path sgdet_masked/  -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ """
""" python train_cttran.py -mode sgdet -save_path cttran/1_temporal/ -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ -bce_loss """
