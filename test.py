import math
import os

import torch
import torch.nn as nn
from torch.nn.functional import leaky_relu
from torch.utils.data import DataLoader
from torchvision import transforms
# from torch.utils.tensorboard import SummaryWriter

from util.preprocessor import CASIA_SURF, read_cfg, CASIA_CEFA
from util.metric import FASMetric


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = read_cfg(cfg_file="./config.yml")
data_cfg = cfg['dataset']
test_cfg = cfg['test']
root_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(root_dir, 'exp', 'save', test_cfg['model'])


def calc_acc(pred, label):
    """
    Args:
        perd: tensor.
        label: tensor.
    """
    err = torch.mean(torch.sum(torch.abs(label-pred)) / len(label))
    acc = 1 - err
    return acc


if __name__ == '__main__':\
    # data
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(test_cfg['rgb_size'][0]),
        transforms.Resize(test_cfg['rgb_size']),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg['mean'], data_cfg['std']),
    ])
    test_set = CASIA_SURF(
        root_dir=os.path.join(root_dir, 'dataset', data_cfg['name'], 'test'),
        csv_file=data_cfg['test_csv'],
        transform=[train_transform, train_transform]
    )
    # test_set = CASIA_CEFA(
    #     root_dir=os.path.join(root_dir, 'dataset', 'CASIA-CEFA', 'train'),
    #     csv_file='4@3_train.txt',
    #     transform=[train_transform, train_transform],
    #     # smoothing=True
    # )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=test_cfg['batch_size'],
        shuffle=True,
        num_workers=4
    )
    # testing
    model = torch.load(save_path).to(device)
    metric = FASMetric()
    # writer = SummaryWriter(cfg['log_dir'])
    for i, (rgb_map, depth_map, label) in enumerate(test_loader):
        rgb_map, depth_map = rgb_map.to(device), depth_map.to(device) # [B,3,H,W]
        label = label.float().reshape(len(label),1).to(device) # [B,1]
        output = model(rgb_map, depth_map) # (gap, r, p, q)
        pred = torch.where(output[1]>0.5, 1., 0.)
        print('r:\t',output[1].squeeze())
        print('pred:\t',pred.squeeze())
        print('label:\t',label.squeeze())
        metric.update(pred, label)
        local_acc = calc_acc(pred, label)
        print ('Batch: {}\t ACC: {:.4f}\t'.format(i, local_acc))
        print("--------------------------------------------------------------------------------------")
    hter, far, frr = metric.calc_HTER()
    acc = metric.calc_ACC()
    print('Model: {}\n ACC: {:.4f}\t EER: {:.4f}\t HTER: {:.4f}\t ACER: {:.4f}'.format(test_cfg['model'], acc, 0, hter, 0))
    # writer.close()