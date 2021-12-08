import os

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import yaml
import cv2


# pytorch custom dataset
class CASIA_SURF(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file), header=None, sep=" ")
        self.transform = transform

    def __getitem__(self, index):
        rgb_name, depth_name, label = self.data.iloc[index, 0], self.data.iloc[index, 1], self.data.iloc[index, 3]
        rgb_img = cv2.imread(os.path.join(self.root_dir, rgb_name), cv2.IMREAD_COLOR)
        depth_img = cv2.imread(os.path.join(self.root_dir, depth_name))
        # gbr => rgb
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
        # (h,w,c) => (c,h,w)
        if self.transform:
            rgb_img = self.transform(rgb_img)
            depth_img = self.transform(depth_img)
        return rgb_img, depth_img, label
    
    def __len__(self):
        return len(self.data)


def read_cfg(cfg_file):
    """
    Read configurations from yaml file
    Args:
        cfg_file (.yaml): path to cfg yaml
    Returns:
        (dict): configuration in dict
    """
    with open(cfg_file, 'r') as rf:
        cfg = yaml.safe_load(rf)
        return cfg