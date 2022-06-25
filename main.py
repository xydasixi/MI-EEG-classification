import numpy as np
import os, glob, datetime, time
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

LR = 0.001
EPOCH = 20
BATCH_SIZE = 64
DATA_PATH = 'data/train'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TensorDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data_tensor = torch.tensor(self.data[index])
        label_tensor = torch.tensor(self.label[index])
        return data_tensor, label_tensor

    def __len__(self):
        return len(self.data)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        layer1 = [nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Dropout()]
        self.layer1 = nn.Sequential(*layer1)
        layer2 = [nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,
                             bias=False),
                   nn.BatchNorm2d(num_features=32),
                   nn.ReLU(inplace=True)]  # 卷积层后接由BatchNorm或者InstanceNorm层时，bias最好设为False
        self.layer2 = nn.Sequential(*layer2)
        layer3 = [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Dropout(),
                   nn.MaxPool2d(kernel_size=2)]
        self.layer3 = nn.Sequential(*layer3)
        layer4 = [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,
                            bias=False),
                  nn.BatchNorm2d(num_features=64),
                  nn.ReLU(inplace=True),
                  nn.Dropout()]  # 卷积层后接由BatchNorm或者InstanceNorm层时，bias最好设为False
        self.layer4 = nn.Sequential(*layer4)
        layer5 = [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,
                            bias=False),
                  nn.BatchNorm2d(num_features=64),
                  nn.ReLU(inplace=True)]  # 卷积层后接由BatchNorm或者InstanceNorm层时，bias最好设为False
        self.layer5 = nn.Sequential(*layer5)
        layer6 = [nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                  nn.ReLU(inplace=True),
                  nn.Dropout(),
                  nn.MaxPool2d(kernel_size=2)]
        self.layer6 = nn.Sequential(*layer6)
        layer7 = [nn.Flatten(),
                  nn.Linear(in_features=5376,out_features=512),
                  nn.BatchNorm1d(num_features=512),
                  nn.ReLU(inplace=True),
                  nn.Dropout()]
        self.layer7 = nn.Sequential(*layer7)
        layer8 = nn.Linear(in_features=512,out_features=2)
        self.layer8 = layer8
        self._initialize_weights()
    def forward(self, x):
        y0 = self.layer1(x)
        y1 = self.layer2(y0)
        y2 = self.layer3(torch.cat([y0,y1],1))
        y3 = self.layer4(y2)
        y4 = self.layer5(y3)
        y5 = self.layer6(torch.cat([y3, y4], 1))
        y6 = self.layer7(y5)
        out = self.layer8(y6)
        return out
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
    dataX = dataX0.reshape(-1,30,25).astype('float32')
    dataX = np.expand_dims(dataX,axis=1)
    dataY = np.zeros(len(dataX))
    for i in range(len(dataY)):
        dataY[i] = dataY0[i//len(dataX0[0])]
    dataY = dataY.astype('int64')
    data = TensorDataset(dataX,dataY)
    return data


if __name__ == '__main__':
    model = Network()
    model.train()
    criterion = nn.CrossEntropyLoss()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    data = datagenerator(DATA_PATH)
    DLoader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCH):
        epoch_loss = 0
        for n_count, (x, y) in enumerate(DLoader):
            batch_x, batch_y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            l = model(batch_x)
            loss = criterion(model(batch_x), batch_y)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            if n_count % 10 == 0:
                print('%4d %4d / %4d loss = %2.4f' % (
                epoch + 1, n_count, len(data) // BATCH_SIZE, loss.item() / BATCH_SIZE))