import copy
import os
import time

import numpy as np
import pandas as pd
import torch

from train_base import fetch_train_basic_config, prepare_optimizer, save_model, fetch_transformer_loss_functions
from transformer_base_scripts import load_common_config, fetch_sequences_after_tracking


def calculate_object_decoder_losses(conf, pred, losses, abs_loss, bce_loss):
    num_tf = len(pred["im_idx"].unique())
    num_cf = conf.baseline_context

    cum_obj_pred_loss = torch.tensor(0.0, device=pred["device"])
    cum_feat_recon_loss = torch.tensor(0.0, device=pred["device"])

    pred_loss_count = num_tf - num_cf

    gt_num_obj_ff = 0
    pred_num_obj_ff = 0
    obj_ant_output = pred
    for count in range(pred_loss_count):
        obj_ant_output_ff = pred["output_ff"][count]
        gt_num_obj_ff += obj_ant_output_ff["gt_num_obj_ff"]

        # Object presence prediction loss
        obj_presence_pred_loss = bce_loss(obj_ant_output_ff["pred_obj_presence"],
                                          obj_ant_output_ff["gt_obj_presence"]).mean()

        pred_feats_mask = obj_ant_output_ff["feat_mask_ant"]
        gt_feats_mask = obj_ant_output_ff["feat_mask_gt"]

        if pred_feats_mask.numel() == 0 or gt_feats_mask.numel() == 0:
            print("Empty feature mask")
        else:
            assert pred_feats_mask.shape[0] == gt_feats_mask.shape[0]

            pred_feats_ff = obj_ant_output_ff["features"]
            gt_feats_ff = obj_ant_output["features"]

            pred_common_feats_ff = pred_feats_ff[pred_feats_mask.view(-1)]
            gt_common_feats_ff = gt_feats_ff[gt_feats_mask.view(-1)]

            feat_recon_loss = conf.hp_recon_loss * abs_loss(pred_common_feats_ff, gt_common_feats_ff).mean()
            cum_feat_recon_loss += feat_recon_loss

        cum_obj_pred_loss += obj_presence_pred_loss
        pred_num_obj_ff += obj_ant_output_ff["im_idx"].shape[0]

    cum_obj_pred_loss /= max(pred_loss_count, 1)
    cum_feat_recon_loss /= max(pred_loss_count, 1)

    if torch.isnan(cum_obj_pred_loss):
        print("Object presence prediction loss is NaN")
    if torch.isnan(cum_feat_recon_loss):
        print("Feature reconstruction loss is NaN")

    print(f"--------------------------------------------------------------------")
    print(
        f"GT Num Obj FF: {gt_num_obj_ff}, Pred Num Obj FF: {pred_num_obj_ff}")

    # Update losses
    losses.update({
        "object_prediction_loss": cum_obj_pred_loss,
        "feature_recon_loss": cum_feat_recon_loss
    })

    return losses


def process_test_video(conf, entry, model, gt_annotation, evaluator, matcher, frame_size):
    fetch_sequences_after_tracking(conf, entry, gt_annotation, matcher, frame_size)
    num_ff = conf.baseline_future
    num_cf = conf.baseline_context
    pred = model(entry, num_cf, num_ff)


def process_train_video(conf, entry, optimizer, model, epoch, num_video, tr, gpu_device, dataloader_train,
                        gt_annotation, matcher, frame_size, start_time):
    bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss = fetch_transformer_loss_functions()

    num_cf = conf.baseline_context
    num_ff = conf.baseline_future
    fetch_sequences_after_tracking(conf, entry, gt_annotation, matcher, frame_size)
    pred = model(entry, num_cf, num_ff)

    losses = {}
    if conf.mode == 'sgcls' or conf.mode == 'sgdet':
        losses['object_loss'] = ce_loss(pred['distribution'], pred['labels']).mean()

    # ----------------- Loss calculation for anticipated objects -----------------
    losses = calculate_object_decoder_losses(conf, pred, losses, abs_loss, bce_loss)

    optimizer.zero_grad()
    loss = sum(losses.values())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
    optimizer.step()
    print("Cold Start Object Loss: epoch {:2d}  batch {:5d}/{:5d}  loss {:.4f}".format(
        epoch, num_video,
        len(dataloader_train),
        loss.item()))

    num_video += 1

    tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
    if num_video % 1000 == 0 and num_video >= 1000:
        time_per_batch = (time.time() - start_time) / 1000
        print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, num_video, len(dataloader_train),
                                                                            time_per_batch,
                                                                            len(dataloader_train) * time_per_batch / 60))
        mn = pd.concat(tr[-1000:], axis=1).mean(1)
        print(mn)
    return num_video


def train_obj_processor():
    (conf, dataloader_train, dataloader_test, gpu_device,
     evaluator, ag_train_data, ag_test_data) = fetch_train_basic_config()

    method_name = conf.method_name
    checkpoint_name = f"{method_name}_{conf.mode}_future_{conf.baseline_future}"
    checkpoint_save_file_path = os.path.join(conf.save_path, method_name)
    os.makedirs(checkpoint_save_file_path, exist_ok=True)

    (model, object_detector, future_frame_loss_num,
     mode, method_name, matcher) = load_common_config(conf, ag_test_data, gpu_device)
    optimizer, scheduler = prepare_optimizer(conf, model)

    tr = []
    for epoch in range(int(conf.nepoch)):
        print("Begin epoch {:d}".format(epoch))
        assert conf.use_raw_data == True
        print('Training using raw data', flush=True)
        train_iter = iter(dataloader_train)
        object_detector.is_train = True
        model.train()
        object_detector.train_x = True
        num = 0
        start_time = time.time()
        for b in range(len(dataloader_train)):
            data = next(train_iter)
            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = ag_train_data.gt_annotations[data[4]]
            frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data

            with torch.no_grad():
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                entry["device"] = conf.device
            num = process_train_video(conf, entry, optimizer, model, epoch, num, tr, gpu_device, dataloader_train,
                                      gt_annotation, matcher, frame_size, start_time)
        print(f"Finished training an epoch {epoch}")
        save_model(model, epoch, checkpoint_save_file_path, checkpoint_name, method_name)
        print(f"Saving model after epoch {epoch}")
        test_iter = iter(dataloader_test)
        model.eval()
        object_detector.is_train = False
        with torch.no_grad():
            for b in range(len(dataloader_test)):
                data = next(test_iter)
                im_data = copy.deepcopy(data[0].cuda(0))
                im_info = copy.deepcopy(data[1].cuda(0))
                gt_boxes = copy.deepcopy(data[2].cuda(0))
                num_boxes = copy.deepcopy(data[3].cuda(0))
                gt_annotation = ag_test_data.gt_annotations[data[4]]
                frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data

                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                entry["device"] = gpu_device
                process_test_video(conf, entry, model, gt_annotation, evaluator, matcher, frame_size)
                if b % 50 == 0:
                    print(f"Finished processing {b} of {len(dataloader_test)} batches")
        # score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
        # scheduler.step(score)


if __name__ == "__main__":
    train_obj_processor()
