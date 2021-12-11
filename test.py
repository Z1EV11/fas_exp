import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model.rgbd_model import rgbd_model
from util.preprocess import CASIA_SURF, read_cfg
from util.preprocess import get_data
from util.metric import get_acer
from util.visualization import plot_res


device = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = read_cfg(cfg_file="./model/config.yml")
data_cfg = cfg['dataset']
test_cfg = cfg['test']
root_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(root_dir, 'model', 'save', test_cfg['model'])


if __name__ == '__main__':\
    # preprocessing
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(test_cfg['rgb_size']),
        transforms.ToTensor(),
        transforms.Normalize(data_cfg['mean'], data_cfg['std']),
    ])
    test_set = CASIA_SURF(
        root_dir=os.path.join(root_dir, 'dataset', data_cfg['name'], 'train'),
        csv_file=data_cfg['test_csv'],
        transform=train_transform,
        # smoothing=True
    )
    train_loader = DataLoader(
        dataset=test_set,
        batch_size=test_cfg['batch_size'],
    )
    # testing
    rgbd_model = nn.load(save_path)
    for i, (rgb_map, depth_map, label) in enumerate(train_loader):
        p,q,r = rgbd_model(rgb_map, depth_map)
        p = (p+q+r)/3
        pred = 1 if p>0.5 else 0
        acer = get_acer(pred, label)
    # visualization
    plot_res(acer)