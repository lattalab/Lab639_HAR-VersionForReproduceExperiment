import os
import pandas as pd
import random
import h5py
import numpy as np
import torch
from torchvision.transforms import (
    CenterCrop,
    Compose,
    RandomCrop,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    Resize,
    Normalize,
    RandomRotation,
    ToTensor,
)
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed

class Lab639DataLoader(Dataset):
    def __init__(self, config, split, seed):
        # set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False

        self.config = config
        self.split = split

        if split == 'train':
            csv_name = self.config.train_csv + self.config.csv_offset + "_train.csv"
            self.anno = os.path.join(self.config.csv_path, csv_name)
            # self.transform = Compose(
            #     [
            #         Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            #         # RandomShortSideScale(
            #         #     min_size=224,
            #         #     max_size=256,
            #         # ),
            #         RandomCrop(224),
            #         RandomHorizontalFlip(p=0.5),
            #         RandomRotation(15),
            #     ]
            # )
            self.transform = Compose([
                Resize((224, 224), antialias=True),
                # CenterCrop(224),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                RandomRotation(15),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif split == 'val':
            csv_name = self.config.test_csv + self.config.csv_offset + "_test.csv"
            self.anno = os.path.join(self.config.csv_path, csv_name)
            # self.transform = Compose(
            #     [
            #         Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            #         # ShortSideScale(
            #         #     size=256
            #         # ),
            #         CenterCrop(224)
            #     ]
            # )
            self.transform = Compose([
                Resize((224, 224), antialias=True),
                # CenterCrop(224),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif split == 'test':
            csv_name = self.config.test_csv + self.config.csv_offset + "_test.csv"
            self.anno = os.path.join(self.config.csv_path, csv_name)
            # self.transform = Compose(
            #     [
            #         Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
            #         # ShortSideScale(
            #         #     size=256
            #         # ),
            #         CenterCrop(224)
            #     ]
            # )
            self.transform = Compose([
                Resize((224, 224), antialias=True),
                # CenterCrop(224),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        # read csv
        self.anno_df = pd.read_csv(self.anno)

        # read data
        self.video_list = []
        self.video_data_dict = {}
        self.actions = []
        self.views = []
        self.video_view_map = {}

        hdf5_list = os.listdir(self.config.data_path)

        for idx, row in enumerate(open(self.anno, 'r').readlines()[1:]):
            video_id, subject, action, camera, repetition, setup = row.split(',')
            dict_action = video_id[-3:]

            if f"{video_id}.hdf5" not in hdf5_list:
                print(f"{video_id}.hdf5 not in {self.config.data_path}")
                continue
            if [subject, action, repetition, setup, dict_action] not in self.video_list:
                self.video_list.append([subject, action, repetition, setup, dict_action])
            if action not in self.actions:
                self.actions.append(action)

            # [MODIFY LOGIC HERE]
            if self.config.baseline:
                # useing --baseline to extract camera-ID
                import re
                match = re.search(r'C(\d{3})', video_id)
                if match:
                    camera = match.group(1) 
                    
            # If --baseline if False, then maintain original logic of `camera-to-region Lable`
            # ==========================================

            if camera not in self.views:
                self.views.append(camera)


            if camera not in self.views:
                self.views.append(camera)
            if f"{subject}_{action}_{repetition}_{setup}_{dict_action}" not in self.video_data_dict:
                self.video_data_dict[f"{subject}_{action}_{repetition}_{setup}_{dict_action}"] = []
            self.video_data_dict[f"{subject}_{action}_{repetition}_{setup}_{dict_action}"].append(video_id)
            self.video_view_map[video_id] = camera

        for video_id in self.video_data_dict:
            if len(self.video_data_dict[video_id]) != self.config.num_views:
                print(self.video_data_dict[video_id])
                raise ValueError(f"Number of views in {self.anno} is {len(self.video_data_dict[video_id])}, but config num_views is {self.config.num_views}")

        # check class num
        if len(self.actions) != self.config.num_classes:
            raise ValueError(f"Number of classes in {self.anno} is {len(self.actions)}, but config num_classes is {self.config.num_classes}")
        
        # add augmented training data
        if split == 'train':
            self.video_list = self.video_list * 2

        if split == 'train':
            random.shuffle(self.video_list)
        self.actions = sorted(self.actions)
        self.views = sorted(self.views)
        self.num_frames = self.config.num_frames

        if len(self.views) != self.config.num_views:
            raise ValueError(f"Number of views in {self.anno} is {len(self.views)}, but config num_views is {self.config.num_views}")
        if len(self.actions) != self.config.num_classes:
            raise ValueError(f"Number of actions in {self.anno} is {len(self.actions)}, but config num_classes is {self.config.num_classes}")

    def __len__(self):
        return len(self.video_list)

    def get_data(self, video_path):
        list16 = []
        frames = h5py.File(video_path, 'r')
        frames = frames['default'][:]
        frames = torch.from_numpy(frames).float()
        frames_len = frames.shape[0]

        for idx, frame in enumerate(frames):
            list16.append(frame)
        frames = torch.stack([frame for frame in list16])

        for i, frame in enumerate(frames):
            frames[i] = frames[i] / 255.0

        frames = self.transform(frames)

        return frames, frames_len

    def gen_combined_frames(self, video_data_dict):
        combined_frames = []
        frames_len = 0
        view_labels = []

        # check if video_data_dict's size is 4 (4 views)
        if len(video_data_dict) != 4:
            raise ValueError(f"video_data_dict should have {self.config.num_views} views, but got {video_data_dict}")

        for video_id in video_data_dict:
            view_id = self.video_view_map[video_id]
            view_label = self.views.index(view_id)
            view_labels.append(view_label)
            video_path = os.path.join(self.config.data_path, f"{video_id}.hdf5")
            frames, frames_len = self.get_data(video_path)

            combined_frames.append(frames)

        # sampling
        if self.split == 'train':
            sample_idx = random_sample(frames_len, self.num_frames)
        elif self.split == 'val' or self.split == 'test':
            # uniform sampling
            sample_idx = np.linspace(0, frames_len - 1, self.num_frames).astype(int)
        combined_frames = [view_frames[sample_idx] for view_frames in combined_frames]

        combined_frames = torch.stack(combined_frames)
        view_labels = torch.tensor(view_labels, dtype=torch.long)

        return combined_frames, view_labels

    def __getitem__(self, index):
        subject, action, repetition, setup, dict_action = self.video_list[index]
        action_label = self.actions.index(action)

        # target data
        # shuffle video_data_dict for view prediction
        random.shuffle(self.video_data_dict[f"{subject}_{action}_{repetition}_{setup}_{dict_action}"])
        target_video_dict = self.video_data_dict[f"{subject}_{action}_{repetition}_{setup}_{dict_action}"]
        # remove '\n' from setup
        setup = setup.strip()
        key = f"S{setup}P{subject}R{repetition}A{action}"

        combined_frames, view_labels = self.gen_combined_frames(target_video_dict)

        if self.split == 'val' or self.split == 'test':
            return combined_frames, action_label, view_labels, key

        # different action
        diff_action_candidate_video_dict = [candidate for candidate in self.video_data_dict if candidate.split('_')[1] != action]
        diff_action_video_dict = self.video_data_dict[random.choice(diff_action_candidate_video_dict)]

        diff_action_combined_frames, _ = self.gen_combined_frames(diff_action_video_dict)

        # same action
        same_action_candidate_video_dict = [candidate for candidate in self.video_data_dict if candidate.split('_')[1] == action and \
                                                                                               (candidate not in target_video_dict)]
        same_action_video_dict = self.video_data_dict[random.choice(same_action_candidate_video_dict)]

        same_action_combined_frames, _ = self.gen_combined_frames(same_action_video_dict)

        return combined_frames, action_label, view_labels, diff_action_combined_frames, same_action_combined_frames

def random_sample(frames_len, num_samples):
    if frames_len < num_samples:
        raise ValueError("Length of the sequence is less than the number of samples to be taken.")
    
    if frames_len == num_samples:
        return np.arange(num_samples).astype(int)
    else:
        sample_idx = np.arange(num_samples) * frames_len / num_samples
        for i in range(num_samples):
            if i < num_samples - 1:
                sample_idx[i] = np.random.choice(range(int(sample_idx[i]), int(sample_idx[i + 1])), 1)
            else:
                sample_idx[i] = np.random.choice(range(int(sample_idx[i]), frames_len), 1)

        return sample_idx.astype(int)