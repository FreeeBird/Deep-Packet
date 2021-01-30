import torch
from torch import nn as nn
from torch.nn import functional as F
from petastorm import make_reader
from petastorm.pytorch import DataLoader
from pathlib import Path
from pyarrow import filesystem

# base on paper in 2020
class CNN(nn.Module):
    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=200, kernel_size=4, stride=3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=200, out_channels=100, kernel_size=5, stride=1),
            nn.ReLU()
        )
        self.max_pooling = nn.MaxPool1d(kernel_size=2,stride=1)
        # flatten, calculate the output size of max pool
        # use a dummy input to calculate
        # dummy_x = torch.rand(1, 1, 1500, requires_grad=False)
        # dummy_x = self.conv1(dummy_x)
        # dummy_x = self.conv2(dummy_x)
        # dummy_x = self.max_pooling(dummy_x)
        # print("dx:", dummy_x.shape)
        # max_pool_out = dummy_x.view(1, -1).shape[1]
        # max_pool_out = dummy_x.shape[2]
        # print('mo:', max_pool_out)
        self.fc0 = nn.Sequential(
            # nn.Linear(49400, 600),
            nn.Linear(49400, 600),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(600, 500),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(500, 400),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(400, 300),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.fc4 = nn.Sequential(
            nn.Linear(300, 200),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.fc5 = nn.Sequential(
            nn.Linear(200, 100),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.fc6 = nn.Sequential(
            nn.Linear(100, 50),
            nn.Dropout(0.05),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(50, 17),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        # the input is in [batch_size, channel, signal_length]
        # where channel is 1
        # signal_length is 1500 by default
        # batch_size = x.shape[0]
        # input: torch.Size([256, 1, 1500])
        # c1: torch.Size([256, 200, 499])
        # cs: torch.Size([256, 100, 495])
        # pool: torch.Size([256, 100, 247])
        # view: torch.Size([256, 24700])
        # fc0: torch.Size([256, 600])
        # fc1: torch.Size([256, 500])
        # fc2: torch.Size([256, 400])
        # out: torch.Size([256, 17])
        # conv1D 1
        # print("input:", x.shape)
        x = self.conv1(x)
        # print("c1:", x.shape)
        # conv1D 2
        x = self.conv2(x)
        # print("cs:",x.shape)
        # Max Pooling
        x = self.max_pooling(x)

        # print("pool:",x.shape)
        # Flatten
        # x = torch.squeeze(x)
        # x = x.reshape(self.batch_size, -1)
        x = x.view(x.size(0), -1)
        # print("view:", x.shape)
        x = self.fc0(x)
        # print("fc0:", x.shape)
        # 7 FC Layer
        x = self.fc1(x)
        # print("fc1:", x.shape)
        x = self.fc2(x)
        # print("fc2:", x.shape)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        # print(x.shape)
        # softmax layer
        x = self.out(x)
        # print("out:",x.shape)
        return x


# model = CNN().float()
# learning_rate = 0.05
# epoch = 200
# loss_func = F.cross_entropy()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# for i in epoch:
#     y_pred = model(x)
#     loss = loss_func(y_pred, y)
#     if i % 100 == 99:
#         print(i, loss.item())
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
