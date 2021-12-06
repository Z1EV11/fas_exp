import torch
from torch.utils.data import Dataset
import numpy
import pandas
import yaml


# pytorch custom dataset
class CASIA_SURF(Dataset):
    def __init__(self, root_dir, csv_file, txt_file):
        super().__init__()
        self.root_dir = root_dir

    def __getitem__(self, index):
        return super().__getitem__(index)
    
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

def read_data_list(f_path):
    pass