import os
import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

from model.rgbd_model import RGBD_model
from util.preprocessor import CASIA_SURF, read_cfg
from util.loss import Total_loss
from util.metric import Metric, calc_score


warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = read_cfg(cfg_file="./config.yml")
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
    train_transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(train_cfg['rgb_size'][0]),
        transforms.RandomRotation(data_cfg['augmentation']['rotation_range']),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(train_cfg['rgb_size']),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg['mean'], data_cfg['std']),
    ])
    train_transform_d = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(train_cfg['rgb_size'][0]),
        transforms.RandomRotation(data_cfg['augmentation']['rotation_range']),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(train_cfg['rgb_size']),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg['mean'], data_cfg['std']),
    ])
    train_set = CASIA_SURF(
        root_dir=os.path.join(root_dir, 'dataset', data_cfg['name'], 'train'),
        csv_file=data_cfg['train_csv'],
        transform=[train_transform_rgb, train_transform_d],
        # smoothing=True
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_cfg['batch_size'],
        shuffle=True
    )
    # training
    print('Using {} device for training.'.format(device))
    model = create_model(cfg)
    for name,param in  model.named_parameters():
        param.requires_grad = True
    loss = Total_loss(device, lamb=train_cfg['cmfl_lamb'], alpha=train_cfg['cmfl_alpha'], gamma=train_cfg['cmfl_gamma']).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=train_cfg['lr'], weight_decay=train_cfg['w_decay'])
    metric = Metric()
    for epoch in range(train_cfg['num_epochs']):
        # if epoch>=1: break
        # print("--------------------------------------------------------------------------------------")
        for i, (rgb_map, depth_map, label) in enumerate(train_loader):
            # if i>=5: break
            # print("--------------------------------------------------------------------------------------")
            rgb_map, depth_map = rgb_map.to(device), depth_map.to(device) # [B,3,224,224]
            output = model(rgb_map, depth_map) # (gap, r, p, q)
            label = label.float().unsqueeze(1).to(device)	# [B] -> [B,1]
            # break
            error = loss(output[2], output[3], output[1], label)
            optimizer.zero_grad()
            error.backward()
            optimizer.step() # gradient descent
            # score = calc_score(output)
            # pred = torch.where(output[1]>0.5, 1., 0.)
            # print('p:\t',output[2].squeeze(1))
            # print('q:\t',output[3].squeeze(1))
            # print('r:\t',output[1].squeeze(1))
            # print('pred:\t',pred.squeeze(1))
            # print('label:\t',label.squeeze(1))
            # metric.update(pred, label)
            # print ('Batch [{}], Error: {:.4f}, ACC: {:.4f}, Score: {:.4f}'.format(i+1, error.item(), metric.calc_acc(pred,label), score))
        print('Epoch [{}/{}], Error: {:.7f}'.format(epoch+1, train_cfg['num_epochs'], error.item()))
        # break
    # save model
    save_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime()) 
    save_path = os.path.join(root_dir, 'model', 'save', '{}-{}.pth'.format(save_time, train_cfg['net']))
    torch.save(model, save_path) # torch.save(model.state_dict(), save_path)
    print('Saved model: {}'.format(save_path))