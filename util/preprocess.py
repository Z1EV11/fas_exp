import torch
from torch.utils.data import Dataset
import numpy


# pytorch custom dataset
class CASIA_SURF(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir

    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return len(self.data)


def get_data():
    return [1], 1

def augument_data():
    pass

def super_resolution():
    pass