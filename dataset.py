import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob


class QvalueDataset(Dataset):
    def __init__(self, root_dir, done_sample_rate=1.0, dataset_len=None):
        self.done_sample_rate = done_sample_rate

        self.done_image_path_s = glob(os.path.join(root_dir, "done", "*.png"))
        self.not_done_image_path_s = glob(os.path.join(root_dir, "not_done", "*.png"))
        self.dataset_len = dataset_len or len(self.done_image_path_s) + len(self.not_done_image_path_s)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if np.random.rand() < self.done_sample_rate:
            return self.read_done_data(idx)
        else:
            return self.read_not_done_data(idx)

    def decode_name(self, image_name):
        stamp, action, reward = image_name[:-4].split("_")
        action = int(action)
        reward = float(reward)
        action = torch.eye(2)[action].bool()
        return stamp, action, reward

    def read_done_data(self, idx):
        idx = idx % len(self.done_image_path_s)
        image_path = self.done_image_path_s[idx]
        image = cv2.imread(image_path)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()

        stamp, action, reward = self.decode_name(os.path.basename(image_path))
        return image, torch.zeros_like(image), action, reward, 1

    def read_not_done_data(self, idx):
        idx = idx % len(self.not_done_image_path_s)
        image_path = self.not_done_image_path_s[idx]
        image = cv2.imread(image_path)
        image_h = image.shape[0]
        image_1 = image[:image_h // 2]
        image_2 = image[image_h // 2:]
        image_1 = image_1.transpose((2, 0, 1))
        image_2 = image_2.transpose((2, 0, 1))
        image_1 = torch.from_numpy(image_1).float()
        image_2 = torch.from_numpy(image_2).float()

        stamp, action, reward = self.decode_name(os.path.basename(image_path))
        return image_1, image_2, action, reward, 0

# class NotDoneDataset(Dataset):
#     def __init__(self, root_dir, dataset_len=None):
#         self.image_path_s = glob(os.path.join(root_dir, "not_done", "*.png"))
#         self.dataset_len = dataset_len or len(self.image_path_s)
#
#     def __len__(self):
#         return self.dataset_len
#
#     def __getitem__(self, idx):
#         idx = idx % self.dataset_len
#         image_path = self.image_path_s[idx]
#         image = cv2.imread(image_path)
#         image_h = image.shape[0]
#         image_1 = image[:image_h // 2]
#         image_2 = image[image_h // 2:]
#
#         image_1 = image_1.transpose((2, 0, 1))
#         image_1 = torch.from_numpy(image_1).float()
#         image_2 = image_2.transpose((2, 0, 1))
#         image_2 = torch.from_numpy(image_2).float()
#
#         image_name = os.path.basename(image_path)
#         stamp, action, reward = image_name.split("_")
#         action = int(action)
#         reward = float(reward)
#
#         return image_1, action, reward, image_2
