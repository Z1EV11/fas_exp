import torch
import torch.nn as nn
from util.performance import get_acer

# config
mode = "test"
save_path = './model/save/{}.ckpt'.format('model')

if __name__ == '__main__':
    # load model
    rgbd_model = nn.load(save_path)
    data. label = get_data()
    # testing
    pred = rgbd_model(data)
    # visualization
    acer = get_acer()