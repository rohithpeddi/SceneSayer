import copy
import os
import time

import numpy as np
import pandas as pd
import torch
from lib.object_detector import Detector
from lib.supervised.sga.baseline_anticipation import BaselineWithAnticipation
from train_base import fetch_train_basic_config, prepare_optimizer, save_model, get_sequence_no_tracking, \
    fetch_transformer_loss_functions


def process_train_video(conf, entry, optimizer, model, epoch, num, tr, gpu_device, dataloader_train):
    bce_loss, ce_loss, mlm_loss, bbox_loss, abs_loss, mse_loss = fetch_transformer_loss_functions()
    start_time = time.time()
    start = 0
    context = conf.baseline_context
    future = conf.baseline_future
    count = 0
    total_frames = len(entry["im_idx"].unique())

    get_sequence_no_tracking(entry, conf.mode)
    pred = model(entry, conf.baseline_context, conf.baseline_future)
    losses={}
    if conf.mode == 'sgcls' or conf.mode == 'sgdet':
        losses['object_loss'] = ce_loss(pred['distribution'], pred['labels']).mean()

    losses["attention_relation_loss"] = 0
    losses["spatial_relation_loss"] = 0
    losses["contact_relation_loss"] = 0
    losses["anticipated_latent_loss"] = 0

    context = min(context, total_frames - 1)
    future = min(future, total_frames - context)
    
    while start + context + 1 <= total_frames:
        future_frame_start_id = entry["im_idx"].unique()[context]
        
        if start + context + future > total_frames > start + context:
            future = total_frames - (start + context)
        
        future_frame_end_id = entry["im_idx"].unique()[context + future - 1]
        
        context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
        future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
        if pred["output"][count]["scatter_flag"] == 0:
            attention_distribution = pred["output"][count]["attention_distribution"]
            spatial_distribution = pred["output"][count]["spatial_distribution"]
            contact_distribution = pred["output"][count]["contacting_distribution"]
            
            attention_label = torch.tensor(pred["attention_gt"][context_end_idx:future_end_idx], dtype=torch.long).to(
                device=attention_distribution.device).squeeze()
            
            if not conf.bce_loss:
                spatial_label = -torch.ones([len(pred["spatial_gt"][context_end_idx:future_end_idx]), 6],
                                            dtype=torch.long).to(device=attention_distribution.device)
                contact_label = -torch.ones([len(pred["contacting_gt"][context_end_idx:future_end_idx]), 17],
                                            dtype=torch.long).to(device=attention_distribution.device)
                for i in range(len(pred["spatial_gt"][context_end_idx:future_end_idx])):
                    spatial_label[i, : len(pred["spatial_gt"][context_end_idx:future_end_idx][i])] = torch.tensor(
                        pred["spatial_gt"][context_end_idx:future_end_idx][i])
                    contact_label[i, : len(pred["contacting_gt"][context_end_idx:future_end_idx][i])] = torch.tensor(
                        pred["contacting_gt"][context_end_idx:future_end_idx][i])
            else:
                spatial_label = torch.zeros([len(pred["spatial_gt"][context_end_idx:future_end_idx]), 6],
                                            dtype=torch.float32).to(device=attention_distribution.device)
                contact_label = torch.zeros([len(pred["contacting_gt"][context_end_idx:future_end_idx]), 17],
                                            dtype=torch.float32).to(device=attention_distribution.device)
                for i in range(len(pred["spatial_gt"][context_end_idx:future_end_idx])):
                    spatial_label[i, pred["spatial_gt"][context_end_idx:future_end_idx][i]] = 1
                    contact_label[i, pred["contacting_gt"][context_end_idx:future_end_idx][i]] = 1
            
            context_boxes_idx = torch.where(entry["boxes"][:, 0] == context)[0][0]
            context_excluding_last_frame_boxes_idx = torch.where(entry["boxes"][:, 0] == context - 1)[0][0]
            future_boxes_idx = torch.where(entry["boxes"][:, 0] == context + future - 1)[0][-1]
            
            if conf.mode == 'predcls':
                context_last_frame_labels = set(
                    pred["labels"][context_excluding_last_frame_boxes_idx:context_boxes_idx].tolist())
                future_labels = set(pred["labels"][context_boxes_idx:future_boxes_idx + 1].tolist())
                context_labels = set(pred["labels"][:context_boxes_idx].tolist())
            else:
                context_last_frame_labels = set(
                    pred["pred_labels"][context_excluding_last_frame_boxes_idx:context_boxes_idx].tolist())
                future_labels = set(pred["pred_labels"][context_boxes_idx:future_boxes_idx + 1].tolist())
                context_labels = set(pred["pred_labels"][:context_boxes_idx].tolist())
            
            appearing_object_labels = future_labels - context_last_frame_labels
            disappearing_object_labels = context_labels - context_last_frame_labels
            ignored_object_labels = appearing_object_labels.union(disappearing_object_labels)
            ignored_object_labels = list(ignored_object_labels)
            
            # Weighting loss based on appearance or disappearance of objects
            # We only consider loss on objects that are present in the last frame of the context
            weight = torch.ones(pred["output"][count]["global_output"].shape[0]).cuda()
            for object_label in ignored_object_labels:
                for idx, pair in enumerate(pred["pair_idx"][context_end_idx:future_end_idx]):
                    if conf.mode == 'predcls':
                        if pred["labels"][pair[1]] == object_label:
                            weight[idx] = 0
                    else:
                        if pred["pred_labels"][pair[1]] == object_label:
                            weight[idx] = 0
            try:
                at_loss = ce_loss(attention_distribution, attention_label)
                losses["attention_relation_loss"] += (at_loss * weight).mean()
            except ValueError:
                # If there is only one object in the last frame of the context, we need to unsqueeze the label
                attention_label = attention_label.unsqueeze(0)
                at_loss = ce_loss(attention_distribution, attention_label)
                losses["attention_relation_loss"] += (at_loss * weight).mean()
            
            if not conf.bce_loss:
                sp_loss = mlm_loss(spatial_distribution, spatial_label)
                losses["spatial_relation_loss"] += (sp_loss * weight).mean()
                con_loss = mlm_loss(contact_distribution, contact_label)
                losses["contact_relation_loss"] += (con_loss * weight).mean()
            else:
                sp_loss = bce_loss(spatial_distribution, spatial_label)
                losses["spatial_relation_loss"] += (sp_loss * weight).mean()
                con_loss = bce_loss(contact_distribution, contact_label)
                losses["contact_relation_loss"] += (con_loss * weight).mean()
            
            latent_loss = abs_loss(pred["output"][count]["global_output"],pred["output"][count]["spatial_latents"])
            losses["anticipated_latent_loss"] += (latent_loss * weight).mean() 
            
            context += 1
            count += 1
        else:
            context += 1
            count += 1
            
    losses["attention_relation_loss"] = losses["attention_relation_loss"] / count
    losses["spatial_relation_loss"] = losses["spatial_relation_loss"] / count
    losses["contact_relation_loss"] = losses["contact_relation_loss"] / count
    losses["anticipated_latent_loss"] = losses["anticipated_latent_loss"]/count
    optimizer.zero_grad()
    loss = sum(losses.values())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
    optimizer.step()
    num += 1
    
    if num % 50 == 0:
        print("epoch {:2d}  batch {:5d}/{:5d}  loss {:.4f}".format(epoch, num, len(dataloader_train), loss.item()))

    tr.append(pd.Series({x: y.item() if isinstance(y,torch.Tensor) else y for x, y in losses.items()}))
    if num % 1000 == 0 and num >= 1000:
        time_per_batch = (time.time() - start_time) / 1000
        print("\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, num, len(dataloader_train),
                                                                            time_per_batch,
                                                                            len(dataloader_train) * time_per_batch / 60))
        
        mn = pd.concat(tr[-1000:], axis=1).mean(1)
        print(mn)
    
    return num


def process_test_video(conf, entry, model, gt_annotation, evaluator):
    get_sequence_no_tracking(entry, conf.mode)
    pred = model(entry, conf.baseline_context, conf.baseline_future)
    start = 0
    count = 0
    context = conf.baseline_context
    future = conf.baseline_future
    total_frames = len(entry["im_idx"].unique())
    
    context = min(context, total_frames - 1)
    future = min(future, total_frames - context)
    
    if (start + context + 1 > total_frames):
        while (start + context + 1 != total_frames and context > 1):
            context -= 1
        future = 1
    
    if (start + context + future > total_frames > start + context):
        future = total_frames - (start + context)
    while (start + context + 1 <= total_frames):
        
        future_frame_start_id = entry["im_idx"].unique()[context]
        prev_con = entry["im_idx"].unique()[context - 1]
        
        if (start + context + future > total_frames > start + context):
            future = total_frames - (start + context)
        
        future_frame_end_id = entry["im_idx"].unique()[context + future - 1]
        
        context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
        context_idx = entry["im_idx"][:context_end_idx]
        context_len = context_idx.shape[0]
        
        future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1]) + 1
        future_idx = entry["im_idx"][context_end_idx:future_end_idx]
        future_len = future_idx.shape[0]
        
        gt_future = gt_annotation[start + context:start + context + future]
        
        vid_no = gt_annotation[0][0]["frame"].split('.')[0]
        # print(vid_no)
        ind = torch.where(entry["boxes"][:, 0] == future_frame_start_id)[0][0]
        prev_ind = torch.where(entry["boxes"][:, 0] == prev_con)[0][0]
        f_ind = torch.where(entry["boxes"][:, 0] == future_frame_end_id)[0][-1]
        if conf.mode == 'predcls':
            con = set(pred["labels"][prev_ind:ind].tolist())
            fut = set(pred["labels"][ind:f_ind + 1].tolist())
            all_con = set(pred["labels"][:ind].tolist())
        else:
            con = set(pred["pred_labels"][prev_ind:ind].tolist())
            fut = set(pred["pred_labels"][ind:f_ind + 1].tolist())
            all_con = set(pred["pred_labels"][prev_ind:ind].tolist())
        
        box_mask = torch.ones(pred["boxes"][ind:f_ind + 1].shape[0])
        frame_mask = torch.ones(future_idx.shape[0])
        
        im_idx = pred["im_idx"][context_end_idx:future_end_idx]
        im_idx = im_idx - im_idx.min()
        
        pair_idx = pred["pair_idx"][context_end_idx:future_end_idx]
        reshape_pair = pair_idx.view(-1, 2)
        min_value = reshape_pair.min()
        new_pair = reshape_pair - min_value
        pair_idx = new_pair.view(pair_idx.size())
        
        boxes = pred["boxes"][ind:f_ind + 1]
        labels = pred["labels"][ind:f_ind + 1]
        pred_labels = pred["pred_labels"][ind:f_ind + 1]
        scores = pred["scores"][ind:f_ind + 1]
        if conf.mode != 'predcls':
            pred_scores = pred["pred_scores"][ind:f_ind + 1]
        
        ob1 = fut - con
        ob2 = all_con - con
        objects = ob1.union(ob2)
        objects = list(objects)
        for obj in objects:
            for idx, pair in enumerate(pred["pair_idx"][context_end_idx:future_end_idx]):
                if conf.mode == 'predcls':
                    if pred["labels"][pair[1]] == obj:
                        frame_mask[idx] = 0
                        box_mask[pair_idx[idx, 1]] = 0
                else:
                    if pred["pred_labels"][pair[1]] == obj:
                        frame_mask[idx] = 0
                        box_mask[pair_idx[idx, 1]] = 0
        
        im_idx = im_idx[frame_mask == 1]
        removed = pair_idx[frame_mask == 0]
        mask_pair_idx = pair_idx[frame_mask == 1]
        
        flat_pair = mask_pair_idx.view(-1)
        flat_pair_copy = mask_pair_idx.view(-1).detach().clone()
        var = 1
        for pair in removed:
            idx = pair[1]
            for i, p in enumerate(flat_pair_copy):
                if p > idx:
                    flat_pair[i] -= 1
        new_pair_idx = flat_pair.view(mask_pair_idx.size())
        
        if conf.mode == 'predcls':
            scores = scores[box_mask == 1]
            labels = labels[box_mask == 1]
        pred_labels = pred_labels[box_mask == 1]
        boxes = boxes[box_mask == 1]
        if conf.mode != 'predcls':
            pred_scores = pred_scores[box_mask == 1]
        
        atten = pred["output"][count]['attention_distribution'][frame_mask == 1]
        spatial = pred["output"][count]['spatial_distribution'][frame_mask == 1]
        contact = pred["output"][count]['contacting_distribution'][frame_mask == 1]
        
        if conf.mode == 'predcls':
            pred_dict = {'attention_distribution': atten,
                         'spatial_distribution': spatial,
                         'contacting_distribution': contact,
                         'boxes': boxes,
                         'pair_idx': new_pair_idx,
                         'im_idx': im_idx,
                         'labels': labels,
                         'pred_labels': pred_labels,
                         'scores': scores
                         }
        else:
            pred_dict = {'attention_distribution': atten,
                         'spatial_distribution': spatial,
                         'contacting_distribution': contact,
                         'boxes': boxes,
                         'pair_idx': new_pair_idx,
                         'im_idx': im_idx,
                         # 'labels':labels,
                         'pred_labels': pred_labels,
                         # 'scores':scores,
                         'pred_scores': pred_scores
                         }
        evaluator.evaluate_scene_graph(gt_future, pred_dict)
        # evaluator.print_stats()
        count += 1
        context += 1


def load_object_detector(conf, gpu_device, ag_train_data):
    object_detector = Detector(
        train=True,
        object_classes=ag_train_data.object_classes,
        use_SUPPLY=True,
        mode=conf.mode
    ).to(device=gpu_device)
    object_detector.eval()
    print("Finished loading object detector", flush=True)
    return object_detector


def main():
    conf, dataloader_train, dataloader_test, gpu_device, evaluator, ag_train_data, ag_test_data = fetch_train_basic_config()
    method_name = conf.method_name
    checkpoint_name = f"{method_name}_{conf.mode}_future_{conf.baseline_future}"
    checkpoint_save_file_path = os.path.join(conf.save_path, method_name)
    os.makedirs(checkpoint_save_file_path, exist_ok=True)
    evaluator_save_file_path = os.path.join(os.path.abspath('..'), conf.results_path, method_name,
                                            f"train_{method_name}_{conf.mode}_{conf.baseline_future}.txt")
    os.makedirs(os.path.dirname(evaluator_save_file_path), exist_ok=True)
    evaluator.save_file = evaluator_save_file_path

    model = BaselineWithAnticipation(
        mode=conf.mode,
        attention_class_num=len(ag_train_data.attention_relationships),
        spatial_class_num=len(ag_train_data.spatial_relationships),
        contact_class_num=len(ag_train_data.contacting_relationships),
        obj_classes=ag_train_data.object_classes,
        enc_layer_num=conf.enc_layer,
        dec_layer_num=conf.dec_layer
    ).to(device=gpu_device)

    if conf.ckpt:
        ckpt = torch.load(conf.ckpt, map_location=gpu_device)
        model.load_state_dict(ckpt[f'{method_name}_state_dict'], strict=False)
        print(f"Loaded model from checkpoint {ckpt}")

    object_detector = load_object_detector(conf, gpu_device, ag_train_data)
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
        for b in range(len(dataloader_train)):
            data = next(train_iter)
            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = ag_train_data.gt_annotations[data[4]]
            with torch.no_grad():
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
            num = process_train_video(conf, entry, optimizer, model, epoch, num, tr, gpu_device, dataloader_train)
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
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                process_test_video(conf, entry, model, gt_annotation, evaluator)
                if b % 50 == 0:
                    print(f"Finished processing {b} of {len(dataloader_test)} batches")
        score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
        evaluator.print_stats()
        evaluator.reset_result()
        scheduler.step(score)


if __name__ == '__main__':
    main()

# python train_baseline.py -use_raw_data -mode sgdet -baseline_future 3  -save_path /home/cse/msr/csy227518/scratch/SGG/SGG_task/baseline/sgdet_f3/ -nepoch 5 -method_name baseline_so -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/
