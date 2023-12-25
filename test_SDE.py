import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
from time import time
from SDE import SDE as SDE
import sys
import pdb

from dataloader.supervised.generation.action_genome.ag_features import AGFeatures, cuda_collate_fn
from constants import Constants as const
from tqdm import tqdm
from lib.supervised.config import Config
from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator
from lib.supervised.biased.dsgdetr.track import get_sequence
from lib.supervised.biased.dsgdetr.matcher import *

conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])

AG_dataset = AGFeatures(
    mode=conf.mode,
    data_split=const.TEST,
    data_path=conf.data_path,
    is_compiled_together=True,
    filter_nonperson_box_frame=True,
    filter_small_box=False if conf.mode == const.PREDCLS else True
)

dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False,
                                              collate_fn=cuda_collate_fn, pin_memory=False)

gpu_device = torch.device('cuda:0')
max_window = conf.max_window
brownian_size = conf.brownian_size

sde = SDE(mode=conf.mode,
               attention_class_num=len(AG_dataset.attention_relationships),
               spatial_class_num=len(AG_dataset.spatial_relationships),
               contact_class_num=len(AG_dataset.contacting_relationships),
               obj_classes=AG_dataset.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer,
               max_window=max_window, 
               brownian_size=brownian_size).to(device=gpu_device)

sde.eval()
#cttran.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
sde.load_state_dict(ckpt['sde_state_dict'], strict=False)

#ckpt = torch.load(conf.model_cttran_path, map_location=gpu_device)
#cttran.load_state_dict(ckpt['cttran_state_dict'], strict=False)

print('*'*50)
print('CKPT {} is loaded'.format(conf.model_path))
#
evaluator1 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.0,
    constraint='with')

evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.0,
    constraint='no')

matcher= HungarianMatcher(0.5, 1, 1, 0.5)
matcher.eval()
all_time = []

with torch.no_grad():
    for entry in tqdm(dataloader, position=0, leave=True):
        start = time()
        gt_annotation = entry[const.GT_ANNOTATION]
        frame_size = entry[const.FRAME_SIZE]
        get_sequence(entry, gt_annotation, matcher, frame_size, conf.mode)
        pred = sde(entry, True)
        vid_no = gt_annotation[0][0]["frame"].split('.')[0]
        all_time.append(time()-start)
        for i in range(1, max_window + 1):
            pred_anticipated = pred.copy()
            mask_curr = pred["mask_curr_" + str(i)]
            pred_anticipated["spatial_distribution"] = pred["anticipated_spatial_distribution"][i - 1][mask_curr]
            pred_anticipated["contacting_distribution"] = pred["anticipated_contacting_distribution"][i - 1][mask_curr]
            pred_anticipated["attention_distribution"] = pred["anticipated_attention_distribution"][i - 1][mask_curr]
            pred_anticipated["im_idx"] = pred["im_idx_test_" + str(i)]
            pred_anticipated["pair_idx"] = pred["pair_idx_test_" + str(i)]
            if conf.mode == "predcls":
                pred_anticipated["scores"] = pred["scores_test_" + str(i)]
                pred_anticipated["labels"] = pred["labels_test_" + str(i)]
            else:
                pred_anticipated["pred_scores"] = pred["pred_scores_test_" + str(i)]
                pred_anticipated["pred_labels"] = pred["pred_labels_test_" + str(i)]
            pred_anticipated["boxes"] = pred["boxes_test_" + str(i)]
            if conf.modified_gt:
                evaluator1.evaluate_scene_graph(entry["gt_annotation_" + str(i)][i : ], pred_anticipated)
                evaluator2.evaluate_scene_graph(entry["gt_annotation_" + str(i)][i : ], pred_anticipated)
            else:
                evaluator1.evaluate_scene_graph(entry["gt_annotation"][i : ], pred_anticipated)
                evaluator2.evaluate_scene_graph(entry["gt_annotation"][i : ], pred_anticipated)

print('Average inference time', np.mean(all_time))
        
print('-------------------------with constraint-------------------------------')
evaluator1.print_stats()
print('-------------------------no constraint-------------------------------')
evaluator2.print_stats()

#  python test_cttran.py -mode sgdet -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ -model_sttran_path cttran/no_temporal/sttran_9.tar -model_cttran_path cttran/no_temporal/cttran_9.tar

