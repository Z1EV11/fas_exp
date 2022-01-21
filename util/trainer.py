import time
import os

import torch
from torch.optim.lr_scheduler import MultiStepLR

from util.metric import FASMetric


class Trainer():
    def __init__(self, train_loader, val_loader, train_cfg, model, loss, optimizer, device, root_dir):
        self.train_cfg = train_cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.device = device
        self.root_dir = root_dir
        # strategy
        self.scheduler = MultiStepLR(optimizer, milestones=[10,20,30,40], gamma=0.2)
        self.metric = FASMetric()

    def train(self):
        print("--------------------------------------------------------------------------------------")
        for epoch in range(self.train_cfg['num_epochs']):
            self.train_1_epoch(epoch)
            self.validate()
            print("--------------------------------------------------------------------------------------")
        self.save()
            
    def train_1_epoch(self, epoch):
        for i, (rgb_map, depth_map, label) in enumerate(self.train_loader):
            rgb_map, depth_map = rgb_map.to(self.device), depth_map.to(self.device) # [B,3,224,224]
            output = self.model(rgb_map, depth_map) # (gap, r, p, q)
            label = label.float().unsqueeze(1).to(self.device)	# [B] -> [B,1]
            error = self.loss(output[2], output[3], output[1], label)
            self.optimizer.zero_grad()
            error.backward()
            self.optimizer.step() # gradient descent
            # pred = torch.where(output[1]>0.5, 1., 0.)
        self.scheduler.step() # change lr
        print('Epoch [{}/{}],\tError: {:.7f},\tlr: {:.9f}'.format(epoch+1, self.train_cfg['num_epochs'], error.item(), self.optimizer.param_groups[0]['lr']))

    def validate(self):
        for i, (rgb_map, depth_map, label) in enumerate(self.val_loader):
            rgb_map, depth_map = rgb_map.to(self.device), depth_map.to(self.device) # [B,3,224,224]
            output = self.model(rgb_map, depth_map) # (gap, r, p, q)
            label = label.float().unsqueeze(1).to(self.device)	# [B] -> [B,1]
            self.optimizer.step() # gradient descent
            pred = torch.where(output[1]>0.5, 1., 0.)
            self.metric.update(pred, label)
        hter, far, frr = self.metric.calc_HTER()
        # err = if far==frr 1 else 0
        acc = self.metric.calc_ACC()
        print('Epoch [{}/{}],\t,\tACC: {:.9f},\tEER: {:.9f},\tHTER: {:.9f},\tACER: {:.9f}'.format(1+1, self.train_cfg['num_epochs'], acc, 0, hter, 0))

    def save(self):
        save_time = time.strftime("%Y-%m-%d %H-%M", time.localtime())
        save_path = os.path.join(self.root_dir, 'exp', 'save', '{}_{}.pth'.format(save_time, self.train_cfg['net']))
        torch.save(self.model, save_path) # torch.save(model.state_dict(), save_path)
        print('Saved model: {}'.format(save_path))