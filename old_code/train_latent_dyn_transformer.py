import copy
import os
import time

import numpy as np
import pandas as pd
import torch

from lib.object_detector import Detector
from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
from lib.supervised.sga import DsgDetrAnt
from lib.supervised.sga import DsgDetrGenAnt
from lib.supervised.sga import STTranAnt
from lib.supervised.sga import STTranGenAnt
from train_base import fetch_train_basic_config, prepare_optimizer, save_model, fetch_transformer_loss_functions, \
    get_sequence_no_tracking, load_model_from_checkpoint


def fetch_sequences_after_tracking(conf, entry, gt_annotation, matcher, frame_size):
    if conf.method_name in ["sttran_ant", "sttran_gen_ant"]:
        get_sequence_no_tracking(entry, conf)
    elif conf.method_name in ["dsgdetr_ant", "dsgdetr_gen_ant"]:
        get_sequence_with_tracking(entry, gt_annotation, matcher, frame_size, conf.mode)
    else:
        raise ValueError(f"Method name {conf.method_name} not recognized")


def calculate_gen_losses(conf, pred, losses, ce_loss, mlm_loss, bce_loss, attention_label, spatial_label,
                         contact_label):
    attention_distribution = pred["gen_attention_distribution"]
    spatial_distribution = pred["gen_spatial_distribution"]
    contacting_distribution = pred["gen_contacting_distribution"]

    try:
        losses["gen_attention_relation_loss"] = ce_loss(attention_distribution, attention_label).mean()
    except ValueError:
        attention_label = attention_label.unsqueeze(0)
        losses["gen_attention_relation_loss"] = ce_loss(attention_distribution, attention_label).mean()

    if not conf.bce_loss:
        losses["gen_spatial_relation_loss"] = mlm_loss(spatial_distribution, spatial_label).mean()
        losses["gen_contact_relation_loss"] = mlm_loss(contacting_distribution, contact_label).mean()
    else:
        losses["gen_spatial_relation_loss"] = bce_loss(spatial_distribution, spatial_label).mean()
        losses["gen_contact_relation_loss"] = bce_loss(contacting_distribution, contact_label).mean()

    return losses


def process_train_video(conf, entry, optimizer, model, epoch, num_video, tr, gpu_device, dataloader_train,
                        gt_annotation, matcher, frame_size, start_time):
    bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss = fetch_transformer_loss_functions()

    count = 0
    num_cf = conf.baseline_context
    num_ff = conf.max_future
    fetch_sequences_after_tracking(conf, entry, gt_annotation, matcher, frame_size)
    pred = model(entry, num_cf, num_ff)

    attention_label = torch.tensor(pred["attention_gt"], dtype=torch.long).to(device=gpu_device).squeeze()
    if not conf.bce_loss:
        spatial_label = -torch.ones([len(pred["spatial_gt"]), 6], dtype=torch.long).to(device=gpu_device)
        contact_label = -torch.ones([len(pred["contacting_gt"]), 17], dtype=torch.long).to(device=gpu_device)
        for i in range(len(pred["spatial_gt"])):
            spatial_label[i, : len(pred["spatial_gt"][i])] = torch.tensor(pred["spatial_gt"][i])
            contact_label[i, : len(pred["contacting_gt"][i])] = torch.tensor(pred["contacting_gt"][i])
    else:
        spatial_label = torch.zeros([len(pred["spatial_gt"]), 6], dtype=torch.float32).to(device=gpu_device)
        contact_label = torch.zeros([len(pred["contacting_gt"]), 17], dtype=torch.float32).to(device=gpu_device)
        for i in range(len(pred["spatial_gt"])):
            spatial_label[i, pred["spatial_gt"][i]] = 1
            contact_label[i, pred["contacting_gt"][i]] = 1

    losses = {}
    has_object_loss = False
    if conf.mode == 'sgcls' or conf.mode == 'sgdet':
        has_object_loss = True
        losses['object_loss'] = ce_loss(pred['distribution'], pred['labels']).mean()

    has_gen_loss = False
    # ----------------- Loss calculation for generation outputs -----------------
    if conf.method_name in ["sttran_gen_ant", "dsgdetr_gen_ant"]:
        has_gen_loss = True
        losses = calculate_gen_losses(conf, pred, losses, ce_loss, mlm_loss, bce_loss,
                                      attention_label, spatial_label, contact_label)

    # ----------------- Loss calculation for anticipated outputs -----------------
    global_output = pred["global_output"]
    ant_output = pred["output"]

    cum_ant_attention_relation_loss = 0
    cum_ant_spatial_relation_loss = 0
    cum_ant_contact_relation_loss = 0
    cum_ant_latent_loss = 0

    num_cf = conf.baseline_context
    num_tf = len(entry["im_idx"].unique())
    loss_count = 0
    while num_cf + 1 <= num_tf:
        ant_spatial_distribution = ant_output[count]["spatial_distribution"]
        ant_contact_distribution = ant_output[count]["contacting_distribution"]
        ant_attention_distribution = ant_output[count]["attention_distribution"]
        ant_global_output = ant_output[count]["global_output"]

        mask_ant = ant_output[count]["mask_ant"].cpu().numpy()
        mask_gt = ant_output[count]["mask_gt"].cpu().numpy()

        if len(mask_ant) == 0:
            assert len(mask_gt) == 0
        else:
            loss_count += 1
            ant_attention_relation_loss = ce_loss(ant_attention_distribution[mask_ant], attention_label[mask_gt]).mean()
            try:
                ant_anticipated_latent_loss = conf.hp_recon_loss * abs_loss(ant_global_output[mask_ant],
                                                                            global_output[mask_gt]).mean()
            except:
                ant_anticipated_latent_loss = 0
                print(ant_global_output.shape, mask_ant.shape, global_output.shape, mask_gt.shape)
                print(mask_ant)

            if not conf.bce_loss:
                ant_spatial_relation_loss = mlm_loss(ant_spatial_distribution[mask_ant], spatial_label[mask_gt]).mean()
                ant_contact_relation_loss = mlm_loss(ant_contact_distribution[mask_ant], contact_label[mask_gt]).mean()
            else:
                ant_spatial_relation_loss = bce_loss(ant_spatial_distribution[mask_ant], spatial_label[mask_gt]).mean()
                ant_contact_relation_loss = bce_loss(ant_contact_distribution[mask_ant], contact_label[mask_gt]).mean()
            cum_ant_attention_relation_loss += ant_attention_relation_loss
            cum_ant_spatial_relation_loss += ant_spatial_relation_loss
            cum_ant_contact_relation_loss += ant_contact_relation_loss
            cum_ant_latent_loss += ant_anticipated_latent_loss
        num_cf += 1
        count += 1

    if loss_count > 0:
        losses["anticipated_attention_relation_loss"] = cum_ant_spatial_relation_loss / loss_count
        losses["anticipated_spatial_relation_loss"] = cum_ant_spatial_relation_loss / loss_count
        losses["anticipated_contact_relation_loss"] = cum_ant_contact_relation_loss / loss_count
        losses["anticipated_latent_loss"] = cum_ant_latent_loss / loss_count

    if loss_count > 0 or has_gen_loss or has_object_loss:
        optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
        optimizer.step()
        print(
            "Gen Loss: {}, Ant Loss: {}, Object Loss:{}, epoch {:2d}  batch {:5d}/{:5d}  loss {:.4f}".format(has_gen_loss,
                                                                                                     loss_count, has_object_loss,
                                                                                                     epoch, num_video,
                                                                                                     len(dataloader_train),
                                                                                                     loss.item()))
    else:
        print(f"No loss to back-propagate for video: {num_video}")
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


def process_test_video(conf, entry, model, gt_annotation, evaluator, matcher, frame_size):
    # ----------------- Make predictions -----------------
    fetch_sequences_after_tracking(conf, entry, gt_annotation, matcher, frame_size)
    num_ff = conf.max_future
    num_cf = conf.baseline_context
    pred = model(entry, num_cf, num_ff)

    # ----------------- Evaluate the anticipated scene graphs -----------------
    count = 0
    num_ff = conf.max_future
    num_cf = conf.baseline_context
    num_tf = len(entry["im_idx"].unique())
    num_cf = min(num_cf, num_tf - 1)
    while num_cf + 1 <= num_tf:
        num_ff = min(num_ff, num_tf - num_cf)
        gt_future = gt_annotation[num_cf: num_cf + num_ff]
        pred_dict = pred["output"][count]
        evaluator.evaluate_scene_graph(gt_future, pred_dict)
        count += 1
        num_cf += 1


def load_sttran_ant(conf, dataset, gpu_device):
    model = STTranAnt(mode=conf.mode,
                      attention_class_num=len(dataset.attention_relationships),
                      spatial_class_num=len(dataset.spatial_relationships),
                      contact_class_num=len(dataset.contacting_relationships),
                      obj_classes=dataset.object_classes,
                      enc_layer_num=conf.enc_layer,
                      dec_layer_num=conf.dec_layer).to(device=gpu_device)
    model = load_model_from_checkpoint(model, conf, gpu_device)
    return model


def load_sttran_gen_ant(conf, dataset, gpu_device):
    model = STTranGenAnt(mode=conf.mode,
                         attention_class_num=len(dataset.attention_relationships),
                         spatial_class_num=len(dataset.spatial_relationships),
                         contact_class_num=len(dataset.contacting_relationships),
                         obj_classes=dataset.object_classes,
                         enc_layer_num=conf.enc_layer,
                         dec_layer_num=conf.dec_layer).to(device=gpu_device)
    model = load_model_from_checkpoint(model, conf, gpu_device)
    return model


def load_dsgdetr_ant(conf, dataset, gpu_device):
    model = DsgDetrAnt(mode=conf.mode,
                       attention_class_num=len(dataset.attention_relationships),
                       spatial_class_num=len(dataset.spatial_relationships),
                       contact_class_num=len(dataset.contacting_relationships),
                       obj_classes=dataset.object_classes,
                       enc_layer_num=conf.enc_layer,
                       dec_layer_num=conf.dec_layer).to(device=gpu_device)
    model = load_model_from_checkpoint(model, conf, gpu_device)
    return model


def load_dsgdetr_gen_ant(conf, dataset, gpu_device):
    model = DsgDetrGenAnt(mode=conf.mode,
                          attention_class_num=len(dataset.attention_relationships),
                          spatial_class_num=len(dataset.spatial_relationships),
                          contact_class_num=len(dataset.contacting_relationships),
                          obj_classes=dataset.object_classes,
                          enc_layer_num=conf.enc_layer,
                          dec_layer_num=conf.dec_layer).to(device=gpu_device)
    model = load_model_from_checkpoint(model, conf, gpu_device)
    return model


def load_common_config(conf, ag_train_data, gpu_device):
    method_name = conf.method_name

    print("----------------------------------------------------------")
    print(f"Method name: {method_name}")

    if conf.ckpt is not None:
        checkpoint_name = os.path.basename(conf.ckpt).split('.')[0]
        future_frame_loss_num = checkpoint_name.split('_')[-3]
        mode = checkpoint_name.split('_')[-5]
        print(f"Checkpoint name: {checkpoint_name}")
    else:
        future_frame_loss_num = conf.max_future
        mode = conf.mode
    print(f"Future frame loss num: {future_frame_loss_num}")
    print(f"Mode: {mode}")
    print("----------------------------------------------------------")

    if conf.method_name == "sttran_ant":
        model = load_sttran_ant(conf, ag_train_data, gpu_device)
    elif conf.method_name == "sttran_gen_ant":
        model = load_sttran_gen_ant(conf, ag_train_data, gpu_device)
    elif conf.method_name == "dsgdetr_ant":
        model = load_dsgdetr_ant(conf, ag_train_data, gpu_device)
    elif conf.method_name == "dsgdetr_gen_ant":
        model = load_dsgdetr_gen_ant(conf, ag_train_data, gpu_device)
    else:
        raise ValueError(f"Method name {conf.method_name} not recognized")

    model.eval()

    matcher = HungarianMatcher(0.5, 1, 1, 0.5)
    matcher.eval()

    object_detector = Detector(
        train=False,
        object_classes=ag_train_data.object_classes,
        use_SUPPLY=True,
        mode=conf.mode
    ).to(device=gpu_device)
    object_detector.eval()
    object_detector.is_train = False

    return model, object_detector, future_frame_loss_num, mode, method_name, matcher


def train_model():
    (conf, dataloader_train, dataloader_test, gpu_device,
     evaluator, ag_train_data, ag_test_data) = fetch_train_basic_config()

    method_name = conf.method_name
    checkpoint_name = f"{method_name}_{conf.mode}_future_{conf.max_future}"
    checkpoint_save_file_path = os.path.join(conf.save_path, method_name)
    os.makedirs(checkpoint_save_file_path, exist_ok=True)
    evaluator_save_file_path = os.path.join(os.path.abspath('..'), conf.results_path, method_name,
                                            f"train_{method_name}_{conf.mode}_{conf.max_future}.txt")
    os.makedirs(os.path.dirname(evaluator_save_file_path), exist_ok=True)
    evaluator.save_file = evaluator_save_file_path

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
                process_test_video(conf, entry, model, gt_annotation, evaluator, matcher, frame_size)
                print(f"Finished processing {b} of {len(dataloader_test)} videos")
        score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
        evaluator.print_stats()
        evaluator.reset_result()
        scheduler.step(score)


if __name__ == "__main__":
    train_model()
