import os
import pickle
import tqdm


def main():
    # Load a pkl file from path and check its content
    # path = '/data/rohith/ag/features/supervised/test/TLDYG.mp4_sgcls.pkl'
    # additional_path = '/data/rohith/ag/features/supervised/additional/test/TLDYG.mp4_frame_idx.pkl'

    features_directory = '/data/rohith/ag/features/supervised/test/'
    video_list = [video_name for video_name in os.listdir(features_directory) if "sgdet" in video_name]
    video_gt_num_frame_dict = {}
    for video in tqdm.tqdm(video_list):
        with open(os.path.join(features_directory, video), 'rb') as f:
            video_features = pickle.load(f)
            video_gt_num_frame_dict[video.split('_')[0]] = len(video_features["im_idx"].unique())

    frame_directory = '/data/rohith/ag/features/supervised/additional/test/'
    video_list = os.listdir(frame_directory)
    video_num_frame_dict = {}
    for video in tqdm.tqdm(video_list):
        with open(os.path.join(frame_directory, video), 'rb') as f:
            video_features = pickle.load(f)
            video_num_frame_dict[video.split('_')[0]] = len(video_features["frame_idx"])

    # Use tqdm.tqdm to track progress
    for video in tqdm.tqdm(video_gt_num_frame_dict.keys()):
        if video_gt_num_frame_dict[video] != video_num_frame_dict[video]:
            print(video, video_gt_num_frame_dict[video], video_num_frame_dict[video])


if __name__ == '__main__':
    main()
