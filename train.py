import torch
import torch.nn as nn
import os

from model.rgbd_model import rgbd_model, rgb_model, d_model
from util.preprocess import get_data
from util.loss import cmf_loss


# config
mode = "train"
learning_rate = 0.0001
step_size = 500
epochs = 100
batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'
finetune = False
save_path = './model/save/{}-model.ckpt'.format('')

def finetune(rgbd_model, data):
    return rgbd_model

def train(rgbd_model, data, label):
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(rgbd_model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        # forward
        pred = rgbd_model(data)
        error = loss(pred, label)
        # backward & optimize
        optimizer.zero_grad()
        error.backward()
        optimizer.step() # gradient descent
        if (epoch+1) % 5 == 0:
            print ('Epoch [{}/{}], Error: {:.4f}'.format(epoch+1, num_epochs, error.item()))
    return rgbd_model


if __name__ == "__main__":
    # preprocessing
    data, label = get_data()
    # training
    print('Using {} device for training.'.format(device))
    if finetune:
        print('Using model[{}] for fine tunring.'.format(device))
        model = nn.load(save_path)
        rgbd_model = finetune(model, data)
    else:
        print('Using a new model for training.')
        rgbd_model = rgbd_model(rgb_model, d_model, mode)
<<<<<<< HEAD
        rgbd_model = train(rgbd_model, data, label)
=======
        rgbd_model = train(rgbd_model, data)
>>>>>>> 2eac43400ccff2b47f938d06efaff3b2feb29589
    # save model
    save_path = './model/save/' + os.time + '-model.ckpt'
    torch.save(rgbd_model.sate_dict(), save_path)
    print('Saved model: {}'.format(save_path))