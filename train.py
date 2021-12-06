import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.rgbd_model import rgbd_model, rgb_model, d_model, CMF_Loss
from util.preprocess import CASIA_SURF, read_cfg


# config
cfg = read_cfg(cfg_file="./model/config.yml")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
data_cfg = cfg['dataset']
train_cfg = cfg['train']


if __name__ == "__main__":
    # preprocessing
    train_set = CASIA_SURF(
        root_dir='./data/{}'.format(cfg['dataset']['name']),
        csv_file=cfg['dataset']['train_set'],
        # transform='',
        # smoothing=True
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=2
    )
    # training
    print('Using {} device for training.'.format(device))
    if cfg['train']['from'] == 'pretrain':
        print('Start fine tuning.')
        rgbd_model = nn.load(save_path)
    elif cfg['train']['from'] == 'scratch':
        rgbd_model = rgbd_model(rgb_model, d_model, mode='train')
    else:
        print("Missing Training's Type!!!")
        exit()
    criterion = CMF_Loss()
    optimizer = torch.optim.Adam(rgbd_model.parameters(), lr=train_cfg['lr'])
    for epoch in range(train_cfg['num_epochs']):
        for i, (img, depth_map, label) in enumerate(train_loader):
            rgb_map, d_map, label = img.to(device), depth_map.to(device), label.to(device)
            # forward
            p, q, r = rgbd_model(rgb_map, d_map)
            loss = criterion(p, q, r, label)
            # backward & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # gradient descent
            # update metrics
        if (epoch+1) % 5 == 0:
            print ('Epoch [{}/{}], Error: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    # save model
    save_path = './model/save/' + os.time + '-model.ckpt'
    torch.save(rgbd_model.sate_dict(), save_path)
    print('Saved model: {}'.format(save_path))