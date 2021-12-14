from math import nan
import os
from tarfile import NUL

import torch
import torch.nn as nn
from torch.nn.modules import module
from torch.utils.data import DataLoader
from torchvision import transforms

from model.rgbd_model import RGBD_model, RGBDMH
from util.preprocessor import CASIA_SURF, read_cfg
from util.loss import Total_loss
from util.metric import get_acc, get_acer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = read_cfg(cfg_file="./model/config.yml")
data_cfg = cfg['dataset']
test_cfg = cfg['test']
root_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(root_dir, 'model', 'save', test_cfg['model'])


def calc_metric(pred, label):
    acc = get_acc(pred, label)
    acer = NUL
    return acer, acc


if __name__ == '__main__':\
    # preprocessing
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(test_cfg['rgb_size']),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg['mean'], data_cfg['std']),
    ])
    test_set = CASIA_SURF(
        root_dir=os.path.join(root_dir, 'dataset', data_cfg['name'], 'test'),
        csv_file=data_cfg['test_csv'],
        transform=train_transform,
        # smoothing=True
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=test_cfg['batch_size'],
    )
    # testing
    model = RGBDMH().to(device)
    model.load_state_dict(torch.load(save_path))
    criterion = Total_loss().to(device)
    with torch.no_grad():
        for i, (rgb_map, depth_map, label) in enumerate(test_loader):
            rgb_map, depth_map = rgb_map.to(device), depth_map.to(device) # [32,3,h,w]
            label = label.float().reshape(len(label),1).to(device) # [b,1]
            _,r,p,q = model(rgb_map, depth_map) # [b,1]
            loss = criterion(p, q, r, label)
            prob = (0.5*(p+q)+0.5*r)/3
            pred = torch.where(prob>0.5, 1., 0.)
            print(p.squeeze(1))
            print(q.squeeze(1))
            print(r.squeeze(1))
            print(pred.squeeze(1))
            print(label.squeeze(1))
            acer, acc = calc_metric(pred, label)
            print ('Error: {:.4f}   ACC: {:.4f}     ACER: {:.4f}'.format(loss.item(), acc, nan))
            if i>3: break
    # visualization