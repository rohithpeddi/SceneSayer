import os

import torch
from lib.supervised.sga import ODE as ODE

from constants import Constants as const
from tqdm import tqdm
from lib.supervised.dsgdetr.track import get_sequence_with_tracking
from lib.supervised.dsgdetr.matcher import HungarianMatcher
from lib.supervised.sga import SDE
from test_base import (fetch_diffeq_test_basic_config, write_percentage_evaluators_stats, \
                       prepare_prediction_graph,
                       modify_pred_dict_disappearance_diffeq)


def process_data(matcher, model, max_window):
    all_time = []
    # with torch.no_grad():
    #     for entry in tqdm(dataloader_test, position=0, leave=True):
    #         try:
    #             start = time()
    #             gt_annotation = entry[const.GT_ANNOTATION]
    #             frame_size = entry[const.FRAME_SIZE]
    #             get_sequence_with_tracking(entry, gt_annotation, matcher, frame_size, conf.mode)
    #             pred = model(entry, True)
    #             global_output = pred["global_output"]
    #             times = pred["times"]
    #             global_output_mod = global_output.clone().to(global_output.device)
    #             denominator = torch.zeros(global_output.size(0)).to(global_output.device) + 1.0
    #             all_time.append(time() - start)
    #             w = max_window
    #             n = len(gt_annotation)
    #             if max_window == -1:
    #                 w = n - 1
    #             w = min(w, n - 1)
    #             for i in range(1, w + 1):
    #                 pred_anticipated = pred.copy()
    #                 mask_curr = pred["mask_curr_" + str(i)]
    #                 mask_gt = pred["mask_gt_" + str(i)]
    #                 last = pred["last_" + str(i)]
    #                 pred_anticipated["spatial_distribution"] = pred["anticipated_spatial_distribution"][i - 1, : last]
    #                 pred_anticipated["contacting_distribution"] = pred["anticipated_contacting_distribution"][i - 1,
    #                                                               : last]
    #                 pred_anticipated["attention_distribution"] = pred["anticipated_attention_distribution"][i - 1,
    #                                                              : last]
    #                 pred_anticipated["im_idx"] = pred["im_idx_test_" + str(i)]
    #                 pred_anticipated["pair_idx"] = pred["pair_idx_test_" + str(i)]
    #                 if conf.mode == "predcls":
    #                     pred_anticipated["scores"] = pred["scores_test_" + str(i)]
    #                     pred_anticipated["labels"] = pred["labels_test_" + str(i)]
    #                 else:
    #                     pred_anticipated["pred_scores"] = pred["pred_scores_test_" + str(i)]
    #                     pred_anticipated["pred_labels"] = pred["pred_labels_test_" + str(i)]
    #                 pred_anticipated["boxes"] = pred["boxes_test_" + str(i)]
    #                 # evaluate_anticipated_future_frame_scene_graph(
    #                 # 	entry["gt_annotation_" + str(i)][i:],
    #                 # 	pred_anticipated,
    #                 # 	future_frame_count=i,
    #                 # 	is_modified_gt=True,
    #                 # 	future_evaluators=future_evaluators,
    #                 # 	future_evaluators_modified_gt=future_evaluators_modified_gt
    #                 # )
    #                 evaluate_anticipated_future_frame_scene_graph(
    #                     entry["gt_annotation"][i:],
    #                     pred_anticipated,
    #                     future_frame_count=i,
    #                     is_modified_gt=False,
    #                     future_evaluators=future_evaluators,
    #                     future_evaluators_modified_gt=future_evaluators_modified_gt
    #                 )
    #                 global_output_mod[mask_gt] += pred["anticipated_vals"][i - 1][mask_curr] / torch.reshape(
    #                     (times[mask_gt] - times[mask_curr] + 1), (-1, 1))
    #                 denominator[mask_gt] += 1 / (times[mask_gt] - times[mask_curr] + 1)
    #         except Exception as e:
    #             print(e)
    #             continue
    #         global_output_mod = global_output_mod / torch.reshape(denominator, (-1, 1))
    #         pred["global_output"] = global_output_mod
    #         pred["attention_distribution"] = model.dsgdetr.a_rel_compress(global_output)
    #         pred["spatial_distribution"] = model.dsgdetr.s_rel_compress(global_output)
    #         pred["contacting_distribution"] = model.dsgdetr.c_rel_compress(global_output)
    #         pred["spatial_distribution"] = torch.sigmoid(pred["spatial_distribution"])
    #         pred["contacting_distribution"] = torch.sigmoid(pred["contacting_distribution"])
    #         gen_evaluators[0].evaluate_scene_graph(gt_annotation, pred)
    #         gen_evaluators[1].evaluate_scene_graph(gt_annotation, pred)
    #         gen_evaluators[2].evaluate_scene_graph(gt_annotation, pred)
    # print('Average inference time', np.mean(all_time))
    # # Write future and gen evaluators stats
    # write_future_evaluators_stats(mode, future_frame_loss_num, method_name, future_evaluators)
    # write_gen_evaluators_stats(mode, future_frame_loss_num, method_name, gen_evaluators)
    # # Send future evaluation and generation evaluation stats to firebase
    # send_future_evaluators_stats_to_firebase(future_evaluators, mode, method_name, future_frame_loss_num)
    # send_gen_evaluators_stats_to_firebase(gen_evaluators, mode, method_name, future_frame_loss_num)

    print('*' * 50)
    print('Begin Percentage Evaluation')
    # Context Fraction Evaluation
    with torch.no_grad():
        # for context_fraction in [0.3, 0.5, 0.7, 0.9]:
        for context_fraction in [0.3, 0.9]:
            for entry in tqdm(dataloader_test, position=0, leave=True):
                try:
                    gt_annotation = entry[const.GT_ANNOTATION]
                    frame_size = entry[const.FRAME_SIZE]
                    get_sequence_with_tracking(entry, gt_annotation, matcher, frame_size, conf.mode)
                    entry = model(entry, True)
                    ind, pred = model.forward_single_entry(context_fraction=context_fraction, entry=entry)
                    if ind >= len(gt_annotation):
                        continue

                    gt_ff_annotation = gt_annotation[ind:]
                    if conf.oracle_disappearance:
                        pred = modify_pred_dict_disappearance_diffeq(pred, gt_ff_annotation)

                    percentage_evaluators[context_fraction][0].evaluate_scene_graph(gt_ff_annotation, pred)
                    percentage_evaluators[context_fraction][1].evaluate_scene_graph(gt_ff_annotation, pred)
                    percentage_evaluators[context_fraction][2].evaluate_scene_graph(gt_ff_annotation, pred)
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue
            # Write percentage evaluation stats
            write_percentage_evaluators_stats(mode, future_frame_loss_num, method_name, percentage_evaluators,
                                              context_fraction)
            # # Send percentage evaluation stats to firebase
            # send_percentage_evaluators_stats_to_firebase(percentage_evaluators, mode, method_name,
            #                                              future_frame_loss_num, context_fraction)


def load_ode(max_window):
    ode = ODE(mode=conf.mode,
              attention_class_num=len(ag_test_data.attention_relationships),
              spatial_class_num=len(ag_test_data.spatial_relationships),
              contact_class_num=len(ag_test_data.contacting_relationships),
              obj_classes=ag_test_data.object_classes,
              enc_layer_num=conf.enc_layer,
              dec_layer_num=conf.dec_layer,
              max_window=max_window).to(device=gpu_device)

    ode.eval()

    ckpt = torch.load(conf.model_path, map_location=gpu_device)
    ode.load_state_dict(ckpt['ode_state_dict'], strict=False)

    print('*' * 50)
    print('CKPT {} is loaded'.format(conf.model_path))

    return ode


def load_sde(max_window):
    brownian_size = conf.brownian_size
    sde = SDE(mode=conf.mode,
              attention_class_num=len(ag_test_data.attention_relationships),
              spatial_class_num=len(ag_test_data.spatial_relationships),
              contact_class_num=len(ag_test_data.contacting_relationships),
              obj_classes=ag_test_data.object_classes,
              enc_layer_num=conf.enc_layer,
              dec_layer_num=conf.dec_layer,
              max_window=max_window,
              brownian_size=brownian_size).to(device=gpu_device)

    sde.eval()

    ckpt = torch.load(conf.model_path, map_location=gpu_device)
    sde.load_state_dict(ckpt['sde_state_dict'], strict=False)

    print('*' * 50)
    print('CKPT {} is loaded'.format(conf.model_path))

    return sde


def main():
    max_window = conf.max_window

    matcher = HungarianMatcher(0.5, 1, 1, 0.5)
    matcher.eval()

    model = None
    if method_name == "NeuralODE":
        assert train_method == "ode"
        model = load_ode(max_window)
    elif method_name in ["NeuralSDE", "sde_wo_bb", "sde_wo_recon", "sde_wo_gen"]:
        assert train_method == "sde"
        model = load_sde(max_window)
        print(f"Model loaded: for method: {method_name}")

    assert model is not None
    process_data(matcher, model, max_window)


def generate_qualitative_results():
    model = None
    if method_name == "ode":
        assert train_method == "ode"
        model = load_ode(conf.max_window)
    elif method_name == "sde":
        assert train_method == "sde"
        model = load_sde(conf.max_window)

    matcher = HungarianMatcher(0.5, 1, 1, 0.5)
    matcher.eval()

    video_id_list = ["0A8CF", "21F9H", "X95D0", "M18XP", "LUQWY", "QE4YE", "ENOLD"]
    context_fraction_list = [0.3, 0.5, 0.7, 0.9]
    with torch.no_grad():
        for context_fraction in context_fraction_list:
            for video_id in video_id_list:
                video_name = video_id + ".mp4"
                entry = ag_test_data.fetch_video_data(video_name)
                gt_annotation = entry[const.GT_ANNOTATION]
                frame_size = entry[const.FRAME_SIZE]
                get_sequence_with_tracking(entry, gt_annotation, matcher, frame_size, conf.mode)
                entry = model(entry, True)
                ind, pred = model.forward_single_entry(context_fraction=context_fraction, entry=entry)
                if ind >= len(gt_annotation):
                    continue
                with_constraint_predictions_map = percentage_evaluators[context_fraction][0].fetch_pred_tuples(
                    gt_annotation[ind:], pred)
                no_constraint_prediction_map = percentage_evaluators[context_fraction][1].fetch_pred_tuples(
                    gt_annotation[ind:], pred)
                prepare_prediction_graph(
                    with_constraint_predictions_map,
                    ag_test_data, video_id, method_name, "with_constraints", conf.mode, context_fraction
                )

                prepare_prediction_graph(
                    no_constraint_prediction_map,
                    ag_test_data, video_id, method_name, "no_constraints", conf.mode, context_fraction
                )


if __name__ == '__main__':
    ag_test_data, dataloader_test, gen_evaluators, future_evaluators, future_evaluators_modified_gt, percentage_evaluators, percentage_evaluators_modified_gt, gpu_device, conf = fetch_diffeq_test_basic_config()
    model_name = os.path.basename(conf.model_path).split('.')[0]
    future_frame_loss_num = model_name.split('_')[-3]
    train_method = model_name.split('_')[0]
    mode = model_name.split('_')[-5]
    method_name = conf.method_name
    # generate_qualitative_results()
    main()
# python test_latent_diffeq.py -mode sgcls -max_window 5 -method_name NeuralODE -model_path /data/rohith/ag/checkpoints/ode/ode_sgcls_future_3_epoch_9.tar
