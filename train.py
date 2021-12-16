import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from model.rgbd_model import RGBD_model
from util.preprocessor import CASIA_SURF, read_cfg
from util.loss import Total_loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = read_cfg(cfg_file="./model/config.yml")
data_cfg = cfg['dataset']
train_cfg = cfg['train']
root_dir = os.path.dirname(os.path.abspath(__file__))


def create_model(cfg):
    model = None
    if cfg['train']['from'] == 'pretrain':
        model_name = ''
        save_path = os.path.join(root_dir, 'model', 'save', model_name)
        model = nn.load(save_path)
        return model
    elif cfg['train']['from'] == 'scratch':
        model = RGBD_model(device).to(device)
        return model
    else:
        print("Missing Training's Type!!!")
        raise NotImplementedError


if __name__ == "__main__":
    # preprocessing
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(train_cfg['rgb_size']),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg['mean'], data_cfg['std']),
    ])
    train_set = CASIA_SURF(
        root_dir=os.path.join(root_dir, 'dataset', data_cfg['name'], 'train'),
        csv_file=data_cfg['train_csv'],
        transform=train_transform,
        # smoothing=True
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_cfg['batch_size'],
    )
    # training
    print('Using {} device for training.'.format(device))
    model = create_model(cfg)
    loss = Total_loss(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg['lr']), weight_decay=float(train_cfg['w_decay']), eps=float(train_cfg['eps']))
    for epoch in range(train_cfg['num_epochs']):
        for i, (rgb_map, depth_map, label) in enumerate(train_loader):
            # if i>1: break
            rgb_map, depth_map = rgb_map.to(device), depth_map.to(device) # [B,3,224,224]
            label = label.unsqueeze(1).float().to(device) # [B]
            gap_map, r, p, q = model(rgb_map, depth_map)
            # break
            error = loss(p, q, r, label)
            optimizer.zero_grad()
            error.backward()
            optimizer.step() # gradient descent
        print ('Epoch [{}/{}], Error: {:.4f}'.format(epoch+1, train_cfg['num_epochs'], error.item()))
        # break
    # save model
    save_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime()) 
    save_path = os.path.join(root_dir, 'model', 'save', '{}-{}.pth'.format(save_time, train_cfg['net']))
    torch.save(model, save_path) # torch.save(model.state_dict(), save_path)
    print('Saved model: {}'.format(save_path))