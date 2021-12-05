import torch
import torch.nn as nn

from model.rgbd_model import rgbd_model
from util.preprocess import get_data
from util.metric import get_acer
from util.visualization import plot_res


"""
    Config
"""
mode = "test"
save_path = './model/save/{}.ckpt'.format('')


if __name__ == '__main__':
    # load model
    rgbd_model = nn.load(save_path)
    data, label = get_data()
    # testing
    pred = rgbd_model(data, mode)
    acer = get_acer(pred, label)
    # visualization
    plot_res(acer)