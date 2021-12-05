import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.rgbd_model import rgbd_model, rgb_model, d_model, CMF_Loss
from util.preprocess import CASIA_SURF


"""
    Config
"""
mode = "train"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
is_pretrain = False
learning_rate = 0.0001
step_size = 500
num_epochs = 100
batch_size = 64


if __name__ == "__main__":
    # preprocessing
    train_set = CASIA_SURF(
        root_dir='./data/CASIA-SURF',
        csv_file='',
        transform='',
        smoothing=''
    )
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )
    # training
    print('Using {} device for training.'.format(device))
    if is_pretrain:
        print('Start fine tuning.')
    rgbd_model = rgbd_model(rgb_model, d_model, mode)
    loss = CMF_Loss()
    optimizer = torch.optim.Adam(rgbd_model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (img, depth_map, label) in enumerate(train_loader):
            rgb_img, d_map, label = img.to('0'), depth_map.to('0'), label.to('0')
            # forward
            p, q, r = rgbd_model(rgb_img, d_map)
            error = loss(p, q, r, label)
            # backward & optimize
            optimizer.zero_grad()
            error.backward()
            optimizer.step() # gradient descent
        if (epoch+1) % 5 == 0:
            print ('Epoch [{}/{}], Error: {:.4f}'.format(epoch+1, num_epochs, error.item()))
    # save model
    save_path = './model/save/' + os.time + '-model.ckpt'
    torch.save(rgbd_model.sate_dict(), save_path)
    print('Saved model: {}'.format(save_path))