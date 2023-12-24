import numpy as np
np.set_printoptions(precision=4)
import copy
import torch
from time import time
from dataloader.action_genome import AG, cuda_collate_fn

from lib.config import Config
from lib.evaluation_forecast import BasicSceneGraphEvaluator
from lib.object_detector import detector
from lib.forecasting import STTran
from lib.track import get_sequence
from lib.matcher import *
import pdb 

conf = Config()
for i in conf.args:
    print(i,':', conf.args[i])

AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
                filter_small_box=False if conf.mode == 'predcls' else True)
dataloader = torch.utils.data.DataLoader(AG_dataset, shuffle=False, num_workers=0, collate_fn=cuda_collate_fn)

gpu_device = torch.device('cuda:0')
object_detector = detector(train=False, object_classes=AG_dataset.object_classes, use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
object_detector.eval()

object_class = ['__background__', 'person', 'bag', 'bed', 'blanket', 'book',
                 'box', 'broom', 'chair', 'closet/cabinet', 'clothes', 
                 'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway', 
                 'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 
                 'paper/notebook', 'phone/camera', 'picture', 'pillow', 'refrigerator', 
                 'sandwich', 'shelf', 'shoe', 'sofa/couch', 'table', 'television', 'towel', 
                 'vacuum', 'window']

relationship_classes = ['looking_at', 'not_looking_at', 'unsure', 'above', 
                    'beneath', 'in_front_of', 'behind', 'on_the_side_of', 
                    'in', 'carrying', 'covered_by', 'drinking_from', 'eating', 
                    'have_it_on_the_back', 'holding', 'leaning_on', 'lying_on', 
                    'not_contacting', 'other_relationship', 'sitting_on', 
                    'standing_on', 'touching', 'twisting', 'wearing', 'wiping', 'writing_on']

attention_relationships = ['looking_at', 'not_looking_at', 'unsure']

spatial_relationships = ['above', 'beneath', 'in_front_of', 'behind', 'on_the_side_of', 'in']

contacting_relationships = ['carrying', 'covered_by', 'drinking_from', 'eating', 
                            'have_it_on_the_back', 'holding', 'leaning_on', 
                            'lying_on', 'not_contacting', 'other_relationship', 
                            'sitting_on', 'standing_on', 'touching', 'twisting', 
                            'wearing', 'wiping', 'writing_on']


model = STTran(mode=conf.mode,
               attention_class_num=len(AG_dataset.attention_relationships),
               spatial_class_num=len(AG_dataset.spatial_relationships),
               contact_class_num=len(AG_dataset.contacting_relationships),
               obj_classes=AG_dataset.object_classes,
               enc_layer_num=conf.enc_layer,
               dec_layer_num=conf.dec_layer).to(device=gpu_device)

model.eval()

ckpt = torch.load(conf.model_path, map_location=gpu_device)
model.load_state_dict(ckpt['state_dict'], strict=False)
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
    iou_threshold=0,
    constraint='with')

evaluator2 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0.5,
    constraint='semi', semithreshold=0.9)

evaluator3 = BasicSceneGraphEvaluator(
    mode=conf.mode,
    AG_object_classes=AG_dataset.object_classes,
    AG_all_predicates=AG_dataset.relationship_classes,
    AG_attention_predicates=AG_dataset.attention_relationships,
    AG_spatial_predicates=AG_dataset.spatial_relationships,
    AG_contacting_predicates=AG_dataset.contacting_relationships,
    iou_threshold=0,
    constraint='no')
matcher= HungarianMatcher(0.5, 1, 1, 0.5)
matcher.eval()
all_time = []
c=0
with torch.no_grad():
    for b,data in enumerate(dataloader):
        start_time = time()
        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = AG_dataset.gt_annotations[data[4]]
        vid_no = gt_annotation[0][0]["frame"].split('.')[0]


        entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
        get_sequence(entry, gt_annotation, matcher, (im_info[0][:2]/im_info[0,2]).cpu().data, conf.mode)

        count = 0
        start = 0
        future = 3
        context = 4

        pred = model(entry,context,future)
        all_time.append(time()-start_time)

        if (start+context+1>len(entry["im_idx"].unique())):
            while(start+context+1 != len(entry["im_idx"].unique()) and context >1):
                context -= 1
            future = 1

        if (start+context+future > len(entry["im_idx"].unique()) and start+context < len(entry["im_idx"].unique())):
            future = len(entry["im_idx"].unique()) - (start+context)

        while (start+context+1 <= len(entry["im_idx"].unique())):
            future_frame_start_id = entry["im_idx"].unique()[context]
            if (start+context+future > len(entry["im_idx"].unique()) and start+context < len(entry["im_idx"].unique())):
                future = len(entry["im_idx"].unique()) - (start+context)

            future_frame_end_id = entry["im_idx"].unique()[context+future-1]

            context_end_idx = int(torch.where(entry["im_idx"] == future_frame_start_id)[0][0])
            context_idx = entry["im_idx"][:context_end_idx]
            context_len = context_idx.shape[0]

            future_end_idx = int(torch.where(entry["im_idx"] == future_frame_end_id)[0][-1])+1
            future_idx = entry["im_idx"][context_end_idx:future_end_idx]
            future_len = future_idx.shape[0]

            gt_future = gt_annotation[start+context:start+context+future]
            vid_no = gt_annotation[0][0]["frame"].split('.')[0]
            #pickle.dump(pred,open('/home/cse/msr/csy227518/Dsg_masked_output/sgdet/test'+'/'+vid_no+'.pkl','wb'))
            evaluator1.evaluate_scene_graph(gt_future, pred,context_end_idx,future_end_idx,future_idx,count)
            evaluator2.evaluate_scene_graph(gt_future, pred,context_end_idx,future_end_idx,future_idx,count)
            evaluator3.evaluate_scene_graph(gt_future, pred,context_end_idx,future_end_idx,future_idx,count)

            #evaluator.print_stats()
            count += 1
            context +=1
            
            if(start+context+future > len(entry["im_idx"].unique())):
                break
                
print('Averge inference time', np.mean(all_time))
print(f'------------------------- for future = {future}--------------------------')       
print('-------------------------with constraint-------------------------------')
evaluator1.print_stats()
print('-------------------------semi constraint-------------------------------')
evaluator2.print_stats()
print('-------------------------no constraint-------------------------------')
evaluator3.print_stats()

""" python test_forecasting.py -mode sgdet -datasize large -data_path /home/cse/msr/csy227518/scratch/Datasets/action_genome/ -model_path forecasting/sgdet_full_context_f3/DSG_masked_9.tar """
