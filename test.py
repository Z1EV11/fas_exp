from math import nan
import os
from tarfile import NUL
from numpy import NaN

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from util.preprocessor import CASIA_SURF, read_cfg
from util.loss import Total_loss
from util.metric import Metric, calc_score


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = read_cfg(cfg_file="./config.yml")
data_cfg = cfg['dataset']
test_cfg = cfg['test']
root_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(root_dir, 'model', 'save', test_cfg['model'])


def calc_score(r, p, q):
    score = (0.5*(p+q)+0.5*r)/3
    return score


if __name__ == '__main__':\
    # preprocessing
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(test_cfg['rgb_size'][0]),
        transforms.Resize(test_cfg['rgb_size']),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg['mean'], data_cfg['std']),
    ])
    test_set = CASIA_SURF(
        root_dir=os.path.join(root_dir, 'dataset', data_cfg['name'], 'val'),
        csv_file=data_cfg['val_csv'],
        transform=[train_transform, train_transform],
        # smoothing=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=test_cfg['batch_size'],
        num_workers=2
    )
    # testing
    model = torch.load(save_path).to(device)
    metric = Metric()
    for i, (rgb_map, depth_map, label) in enumerate(test_loader):
        rgb_map, depth_map = rgb_map.to(device), depth_map.to(device) # [B,3,H,W]
        label = label.float().reshape(len(label),1).to(device) # [B,1]
        output = model(rgb_map, depth_map) # (gap, r, p, q)
        score = output[1]
        pred = torch.where(score>0.5, 1., 0.)
        print("--------------------------------------------------------------------------------------")
        print('r:\t',output[1].squeeze())
        print('score:\t',score.squeeze())
        print('pred:\t',pred.squeeze())
        print('label:\t',label.squeeze())
        metric.update(pred, label)
        print ('Batch: {}\t ACC: {:.4f}\t ACER: {:.4f}'.format(i, metric.calc_acc(pred,label), NaN))
    # visualization