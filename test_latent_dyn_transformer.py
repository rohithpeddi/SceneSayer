import copy
import math

import torch

from test_base import (fetch_transformer_test_basic_config, prepare_prediction_graph,
                       send_future_evaluators_stats_to_firebase, write_future_evaluators_stats,
                       write_percentage_evaluators_stats, send_percentage_evaluators_stats_to_firebase)
from transformer_base_scripts import load_common_config, fetch_sequences_after_tracking


def evaluate_model_context_fraction(model, matcher, entry, gt_annotation, frame_size,
                                    conf, context_fraction, percentage_evaluators):
    gt_future, pred_dict = fetch_model_context_pred_dict(model, matcher, entry, gt_annotation,
                                                         frame_size, conf, context_fraction)
    evaluators = percentage_evaluators[context_fraction]
    evaluators[0].evaluate_scene_graph(gt_future, pred_dict)
    evaluators[1].evaluate_scene_graph(gt_future, pred_dict)
    evaluators[2].evaluate_scene_graph(gt_future, pred_dict)


def evaluate_model_future_frames(model, matcher, entry, gt_annotation, frame_size, conf, num_ff, future_evaluators):
    # ----------------- Future frames scene graph prediction -----------------
    num_cf = conf.baseline_context
    fetch_sequences_after_tracking(conf, entry, gt_annotation, matcher, frame_size)
    pred = model(entry, num_cf, num_ff)
    # ----------------- Evaluate future scene graphs -----------------
    count = 0
    num_cf = conf.baseline_context
    num_tf = len(entry["im_idx"].unique())
    num_cf = min(num_cf, num_tf - 1)
    while num_cf + 1 <= num_tf:
        num_ff = min(num_ff, num_tf - num_cf)
        gt_future = gt_annotation[num_cf: num_cf + num_ff]
        pred_dict = pred["output"][count]
        
        evaluators = future_evaluators[num_ff]
        evaluators[0].evaluate_scene_graph(gt_future, pred_dict)
        evaluators[1].evaluate_scene_graph(gt_future, pred_dict)
        evaluators[2].evaluate_scene_graph(gt_future, pred_dict)
        count += 1
        num_cf += 1


def fetch_model_context_pred_dict(model, matcher, entry, gt_annotation, frame_size, conf, context_fraction):
    fetch_sequences_after_tracking(conf, entry, gt_annotation, matcher, frame_size)
    pred = model.forward_single_entry(context_fraction=context_fraction, entry=entry)
    num_tf = len(entry["im_idx"].unique())
    num_cf = min(int(math.ceil(context_fraction * num_tf)), num_tf - 1)
    gt_future = gt_annotation[num_cf: num_tf]
    pred_dict = pred["output"][0]
    return gt_future, pred_dict


def generate_context_qualitative_results(model, matcher, entry, gt_annotation, frame_size, conf, context_fraction,
                                         percentage_evaluators, video_id, ag_test_data):
    gt_future, pred_dict = fetch_model_context_pred_dict(model, matcher, entry, gt_annotation, frame_size, conf,
                                                         context_fraction)
    
    evaluators = percentage_evaluators[context_fraction]
    with_constraint_predictions_map = evaluators[0].fetch_pred_tuples(gt_future, pred_dict)
    no_constraint_prediction_map = evaluators[1].fetch_pred_tuples(gt_future, pred_dict)
    
    prepare_prediction_graph(
        with_constraint_predictions_map,
        ag_test_data, video_id, conf.method_name,
        "with_constraints", conf.mode, context_fraction
    )
    
    prepare_prediction_graph(
        no_constraint_prediction_map,
        ag_test_data, video_id, conf.method_name,
        "no_constraints", conf.mode, context_fraction
    )


def test_model():
    (ag_test_data, dataloader_test, gen_evaluators, future_evaluators,
     future_evaluators_modified_gt, percentage_evaluators,
     percentage_evaluators_modified_gt, gpu_device, conf) = fetch_transformer_test_basic_config()
    
    (model, object_detector, future_frame_loss_num,
     mode, method_name, matcher) = load_common_config(conf, ag_test_data, gpu_device)
    
    test_iter = iter(dataloader_test)
    model.eval()
    future_frames_list = [5]
    context_fractions = [0.3, 0.5, 0.7, 0.9]
    with torch.no_grad():
        for b in range(len(dataloader_test)):
            data = next(test_iter)
            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = ag_test_data.gt_annotations[data[4]]
            frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data
            
            for num_future_frames in future_frames_list:
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                evaluate_model_future_frames(model, matcher, entry, gt_annotation, frame_size,
                                             conf, num_future_frames, future_evaluators)
            
            for context_fraction in context_fractions:
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                evaluate_model_context_fraction(model, matcher, entry, gt_annotation, frame_size,
                                                conf, context_fraction, percentage_evaluators)
            
            if b % 50 == 0:
                print(f"Finished processing {b} of {len(dataloader_test)} batches")
        
        # Write future and gen evaluators stats
        write_future_evaluators_stats(conf.mode, future_frame_loss_num, method_name=method_name,
                                      future_evaluators=future_evaluators)
        
        # Send future evaluation and generation evaluation stats to firebase
        send_future_evaluators_stats_to_firebase(future_evaluators, conf.mode, method_name=method_name,
                                                 future_frame_loss_num=future_frame_loss_num)
        
        # Write percentage evaluation stats and send to firebase
        for context_fraction in context_fractions:
            write_percentage_evaluators_stats(
                conf.mode,
                future_frame_loss_num,
                method_name,
                percentage_evaluators,
                context_fraction
            )
            send_percentage_evaluators_stats_to_firebase(
                percentage_evaluators,
                mode,
                method_name,
                future_frame_loss_num,
                context_fraction
            )


def generate_qualitative_results():
    (ag_test_data, dataloader_test, gen_evaluators, future_evaluators,
     future_evaluators_modified_gt, percentage_evaluators,
     percentage_evaluators_modified_gt, gpu_device, conf) = fetch_transformer_test_basic_config()
    
    video_id_index_map = {}
    for index, video_gt_annotation in enumerate(ag_test_data.gt_annotations):
        video_id = video_gt_annotation[0][0]['frame'].split(".")[0]
        video_id_index_map[video_id] = index
    
    (model, object_detector,
     future_frame_loss_num, mode, method_name, matcher) = load_common_config(conf, ag_test_data, gpu_device)
    
    model.eval()
    context_fractions = [0.3, 0.5, 0.7, 0.9]
    video_id_list = ["21F9H", "X95D0", "M18XP", "0A8CF", "LUQWY", "QE4YE", "ENOLD"]
    with torch.no_grad():
        for video_id in video_id_list:
            d_im_data, d_im_info, d_gt_boxes, d_num_boxes, d_index = ag_test_data.fetch_video_data(
                video_id_index_map[video_id])
            im_data = copy.deepcopy(d_im_data.cuda(0))
            im_info = copy.deepcopy(d_im_info.cuda(0))
            gt_boxes = copy.deepcopy(d_gt_boxes.cuda(0))
            num_boxes = copy.deepcopy(d_num_boxes.cuda(0))
            gt_annotation = ag_test_data.gt_annotations[video_id_index_map[video_id]]
            frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data
            for context_fraction in context_fractions:
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
                generate_context_qualitative_results(model, matcher, entry, gt_annotation, frame_size, conf,
                                                     context_fraction, percentage_evaluators, video_id, ag_test_data)


if __name__ == '__main__':
    test_model()
