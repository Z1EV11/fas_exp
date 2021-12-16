from math import nan
import os
from tarfile import NUL

import torch
import torch.nn as nn
from torch.nn.modules import module
from torch.utils.data import DataLoader
from torchvision import transforms

from model.rgbd_model import RGBD_model
from util.preprocessor import CASIA_SURF, read_cfg
from util.loss import Total_loss
from util.metric import Metric


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = read_cfg(cfg_file="./model/config.yml")
data_cfg = cfg['dataset']
test_cfg = cfg['test']
root_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(root_dir, 'model', 'save', test_cfg['model'])


def calc_score():
    pass


if __name__ == '__main__':\
    # preprocessing
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(test_cfg['rgb_size']),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg['mean'], data_cfg['std']),
    ])
    test_set = CASIA_SURF(
        root_dir=os.path.join(root_dir, 'dataset', data_cfg['name'], 'val'),
        csv_file=data_cfg['val_csv'],
        transform=train_transform,
        # smoothing=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=test_cfg['batch_size'],
    )
    # testing
    model = torch.load(save_path).to(device)
    metric = Metric()
    with torch.no_grad():
        for i, (rgb_map, depth_map, label) in enumerate(test_loader):
            if i>5: break
            rgb_map, depth_map = rgb_map.to(device), depth_map.to(device) # [B,3,H,W]
            label = label.float().reshape(len(label),1).to(device) # [B,1]
            _,r,p,q = model(rgb_map, depth_map) # [B,1]
            prob = (0.5*(p+q)+0.5*r)/3
            pred = torch.where(prob>0.5, 1., 0.)
            print("--------------------------------------------------------------------------------------")
            print('r:\t',r.squeeze(1))
            print('prob:\t', prob.squeeze(1))
            print('pred:\t',pred.squeeze(1))
            print('label:\t',label.squeeze(1))
            acc = metric.calc_acc(pred, label)
            print ('i: {}\t ACC: {:.4f}\t ACER: {:.4f}'.format(i, acc, nan))
    # visualization