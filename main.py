import numpy as np
import os, glob, datetime, time
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR


DATA_PATH = 'data/train'



class TensorDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data_tensor = torch.tensor(self.data[index])
        label_tensor = torch.tensor(self.label[index])
        return data_tensor, label_tensor

    def __len__(self):
        return self.data.size(0)

class DnCNN(torch.nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers += [nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Dropout()]
        layers += [nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=padding,
                             bias=False),
                   nn.BatchNorm2d(32),
                   nn.ReLU(inplace=True)]  # 卷积层后接由BatchNorm或者InstanceNorm层时，bias最好设为False
        for i in range(depth-2):
            layers += [nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size,padding=padding, bias=False),
                       nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95),
                       nn.ReLU(inplace=True)]#卷积层后接由BatchNorm或者InstanceNorm层时，bias最好设为False
        layers += [nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False)]
        self.network = nn.Sequential(*layers)
        self._initialize_weights()
    def forward(self, x):
        y = x
        out = self.network(x)
        return y - out
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)



def datagenerator(data_path):
    file_list = glob.glob(data_path + '/*.npz')
    dataX0 = np.load(file_list[0])['X']
    dataY0 = np.load(file_list[0])['y']
    for i in range(len(file_list)-1):
        tempX = np.load(file_list[i+1])['X']
        tempY = np.load(file_list[i+1])['y']
        dataX0 = np.vstack((dataX0,tempX))
        dataY0 = np.hstack((dataY0,tempY))
    dataX = dataX0.reshape(-1,30,25)
    dataY = np.zeros(len(dataX))
    for i in range(len(dataY)):
        dataY[i] = dataY0[i//len(dataX0[0])]
    data = TensorDataset(dataX,dataY)


if __name__ == '__main__':
    datagenerator(DATA_PATH)