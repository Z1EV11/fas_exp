import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from model.rgbd_model import RGBD_model, RGB_net, Depth_net
from util.preprocess import CASIA_SURF, read_cfg
from util.loss import Total_loss
from model.res_net import resnet18


cfg = read_cfg(cfg_file="./model/config.yml")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_cfg = cfg['dataset']
train_cfg = cfg['train']


def create_model(cfg):
    model = None
    if cfg['train']['from'] == 'pretrain':
        print('Start fine tuning.')
        model = nn.load(save_path)
        return model
    elif cfg['train']['from'] == 'scratch':
        model = RGBD_model(resnet18, resnet18)
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
        root_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset', data_cfg['name'], 'train'),
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
    criterion = Total_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['lr'])
    for epoch in range(train_cfg['num_epochs']):
        for i, (rgb_map, depth_map, label) in enumerate(train_loader):
            """
                rgb_map: tensor[32, 3, 255, 255]
                depth_map: tensor[32, 1, 255, 255]
                label: tensor[32, 1]
            """
            label = label.float()
            label = label.reshape(32,1)
            # forward
            p, q, r = model(rgb_map, depth_map)
            loss = criterion(p, q, r, label)
            # backward & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # gradient descent
            # update metrics
        if (epoch+1) % 5 == 0:
            print ('Epoch [{}/{}], Error: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    # save model
    save_path = os.path.join(os.path.abspath(__file__), 'model', 'save', '{}-model.ckpt'.format(os.time))
    torch.save(model.sate_dict(), save_path)
    print('Saved model: {}'.format(save_path))