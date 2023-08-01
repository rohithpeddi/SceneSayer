import numpy as np
import pandas as pd
import os
import sys
import torch
import pickle
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--mode",type=str)
args = parser.parse_args()
folder = args.mode

DATA = "../../Project/all_frames_final/"
TARGET = args.mode
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_printoptions(edgeitems=np.inf)

if __name__=='__main__':
    if folder=='train':
        path = os.path.join(DATA,folder)
        count_frames = 0
        error = 0
        for vid_name in os.listdir(path):
            out = {}
            vid = pickle.load(open(path+'/'+vid_name,'rb'))
            num_frames = int(torch.max(vid["im_idx"]))+1
            if num_frames>4:
                output = torch.zeros((num_frames,37,1936)).to(device)
                mask = torch.full((num_frames,37),False,dtype=torch.bool).to(device)
                pred_lab = torch.zeros_like(vid["im_idx"])
                start = 0
                c =0
                se = []
                for idx in range(num_frames):
                    #pdb.set_trace()
                    x=[]
                    end = len(vid["im_idx"][vid["im_idx"]==idx])
                    for i in range(start,start+end):
                        output[idx,vid["pred_labels"][vid["pair_idx"][i,1]],:] = vid["global_output"][i]
                        mask[idx,vid["pred_labels"][vid["pair_idx"][i,1]]] = True
                        #x.append(int(vid["pred_labels"][vid["pair_idx"][i,1]]))

                    start = start+end
                mask = (mask==1)
                vid["output"] = output
                vid["mask"] = mask
                pickle.dump(vid,open(TARGET+vid_name,'wb'))
                print(f" DONE : {vid_name} || Error {error}")
            else: 
                count_frames += 1
                print(f"Less than 5 : {vid_name} count : {count_frames}")
        print("Total less than 5 frames : ",count_frames)

    else:
        path = os.path.join(DATA,folder)
        count_frames = 0
        for vid_name in os.listdir(path):
            out = {}
            vid = pickle.load(open(path+'/'+vid_name,'rb'))
            num_frames = int(torch.max(vid["im_idx"]))+1
            if num_frames>4:
                output = torch.zeros((num_frames,37,1936)).to(device)
                mask = torch.zeros((num_frames,37)).to(device)
                pred_lab = torch.zeros_like(vid["im_idx"])
                start = 0
                for idx in range(num_frames):
                    end = len(vid["im_idx"][vid["im_idx"]==idx])
                    for i in range(start,start+end):
                        output[idx,vid["pred_labels"][vid["pair_idx"][i,1]],:] = vid["global_output"][i]
                        mask[idx,vid["pred_labels"][vid["pair_idx"][i,1]]] = 1
                    start = start+end
                mask = (mask==1)
                vid["output"] = output
                vid["mask"] = mask
                pickle.dump(vid,open(TARGET+vid_name,'wb'))
                print(f" DONE : {vid_name}")
            else: 
                count_frames += 1
                print(f"Less than 5 : {vid_name} count : {count_frames}")
        print("Total less than 5 frames : ",count_frames)