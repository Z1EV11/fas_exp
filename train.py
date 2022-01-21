import os
import time
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
import numpy as np

from model.rgbd_model import RGBD_model
from util.preprocessor import CASIA_SURF, read_cfg
from util.loss import Total_loss
from util.metric import FASMetric
# from util.trainer import Trainer


# config
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cfg = read_cfg(cfg_file="./config.yml")
data_cfg = cfg['dataset']
train_cfg = cfg['train']
val_cfg = cfg['val']
optim_cfg = train_cfg['optim']
loss_cfg = train_cfg['cmfl']
root_dir = os.path.dirname(os.path.abspath(__file__))


# common
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

def validate(val_loader, metric):
    metric.reset()
    with torch.no_grad:
        for i, (rgb_map, depth_map, label) in enumerate(val_loader): 
            rgb_map, depth_map = rgb_map.to(device), depth_map.to(device) # [B,3,224,224]
            output = model(rgb_map, depth_map) # (gap, r, p, q)
            pred = torch.where(output[1]>0.5, 1., 0.)
            metric.update(pred, label)
    hter, far, frr = metric.calc_HTER()
    acc = metric.calc_ACC()
    print('\tACC: {:.4f}\t EER: {:.4f}\t HTER: {:.4f}\t ACER: {:.4f}'.format(acc, 0, hter, 0))


if __name__ == "__main__":
    # data
    train_transform_rgb = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(train_cfg['rgb_size'][0]),
        transforms.RandomRotation(data_cfg['augmentation']['rotation_range']),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(train_cfg['rgb_size']),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg['mean'], data_cfg['std']),
    ])
    # train_transform_d = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomResizedCrop(train_cfg['rgb_size'][0]),
    #     transforms.RandomRotation(data_cfg['augmentation']['rotation_range']),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Resize(train_cfg['rgb_size']),
    #     transforms.ToTensor(),
    #     transforms.Normalize(data_cfg['mean'], data_cfg['std']),
    # ])
    train_set = CASIA_SURF(
        root_dir=os.path.join(root_dir, 'dataset', data_cfg['name'], 'train'),
        csv_file=data_cfg['train_csv'],
        transform=[train_transform_rgb, train_transform_rgb],
        # smoothing=True
    )
    val_set = CASIA_SURF(
        root_dir=os.path.join(root_dir, 'dataset', data_cfg['name'], 'val'),
        csv_file=data_cfg['val_csv'],
        transform=[train_transform_rgb, train_transform_rgb],
        # smoothing=True
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=val_cfg['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    # model
    model = create_model(cfg)
    print('Using {} device for training.\nModel:\n{}'.format(device, list(model.children())))
    # for name,param in  model.named_parameters():
    #     param.requires_grad = True
    # optimizer = torch.optim.NAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=optim_cfg['lr'], weight_decay=optim_cfg['wd'])
    loss = Total_loss(device, lamb=loss_cfg['lamb'], alpha=loss_cfg['alpha'], gamma=loss_cfg['gamma']).to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=optim_cfg['lr'], weight_decay=optim_cfg['wd'])
    scheduler = MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=0.2)
    metric = FASMetric()
    # training
    for epoch in range(train_cfg['num_epochs']):
        # if epoch>=1: break
        print("--------------------------------------------------------------------------------------")
        for i, (rgb_map, depth_map, label) in enumerate(train_loader): 
            # if i>=1: break
            # print("--------------------------------------------------------------------------------------")
            rgb_map, depth_map = rgb_map.to(device), depth_map.to(device) # [B,3,224,224]
            output = model(rgb_map, depth_map) # (gap, r, p, q)
            label = label.float().unsqueeze(1).to(device)	# [B] -> [B,1]
            # break
            error = loss(output[2], output[3], output[1], label)
            optimizer.zero_grad()
            error.backward()
            optimizer.step() # gradient descent
        validate(val_loader, metric)
        scheduler.step() # change lr
        print('Epoch [{}/{}],\tError: {:.7f},\tlr: {:.9f}'.format(epoch+1, train_cfg['num_epochs'], error.item(), optimizer.param_groups[0]['lr']))
        # break
    # save model
    save_time = time.strftime("%Y-%m-%d %H-%M", time.localtime())
    save_path = os.path.join(root_dir, 'exp', 'save', '{}_{}.pth'.format(save_time, train_cfg['net']))
    torch.save(model, save_path) # torch.save(model.state_dict(), save_path)
    print('Saved model: {}'.format(save_path))
