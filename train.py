import torch
import torch.nn as nn

from model.rgb_model import rgb_model 
from model.d_model import d_model 

class rgbd_model(nn.model):
    def __init__(self, rgb_model, d_model):
        self.rgb_model = rgbd_model
        self.d_model = d_model
        pass

def train():
    pas

if __name__ == "__main__":
    # preprocessing
    x, labels = get_date()
    # rgbd model
    learning_rate = 0.01
    num_epochs = 60
    rgbd_model = rgbd_model(rbd_model, d_model)
    criterion = nn.MSEloss()
    optimizer = torch.optim.SGD(rgbd_model.parameters(), lr=learning_rate)
    # training
    for epoch in range(num_epochs):
        # forward pass
        y = rgbd_model(x)
        loss = criterion(y, labels)
        # backward prop & optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    # save model
    torch.save(rgbd_model, 'model.ckpt')
    pass