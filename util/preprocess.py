import os

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import yaml
import cv2


# pytorch custom dataset
class CASIA_SURF(Dataset):
    """
    CASIA SURF dataset
    Args:
        root_dir: root directory of train set 
        csv_file: file with label
    """
    def __init__(self, root_dir, csv_file, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join(self.root_dir, csv_file), header=None, sep=" ")
        self.transform = transform

    def __getitem__(self, index):
        """
        Returns:
            rgb_map: tensor[32, 3, 255, 255]
            depth_map: tensor[32, 1, 255, 255]
            label: 0 or 1
        """
        rgb_path, depth_path, label = self.data.iloc[index, 0], self.data.iloc[index, 1], self.data.iloc[index, 3]
        rgb_img = cv2.imread(os.path.join(self.root_dir, rgb_path), cv2.IMREAD_COLOR)
        depth_img = cv2.imread(os.path.join(self.root_dir, depth_path))
        # gbr => rgb
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)
        # (h,w,c) => (c,h,w)
        rgb_map = self.transform(rgb_img)
        depth_map = self.transform(depth_img)
        return rgb_map, depth_map, label
    
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