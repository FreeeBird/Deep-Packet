import torch
from torch import nn as nn
from torch.nn import functional as F
from petastorm import make_reader
from petastorm.pytorch import DataLoader
from pathlib import Path
from pyarrow import filesystem

# base on paper in 2020
class CNN(nn.Module):
    def __init__(self, batch_size=256):
        super().__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=200,  # Filters
            kernel_size=5,
            # stride=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=200,
            out_channels=100,
            kernel_size=4,
            # stride=1
        )
        self.max_pooling = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(600, 500)
        self.fc2 = nn.Linear(500, 400)
        self.fc3 = nn.Linear(400, 300)
        self.fc4 = nn.Linear(300, 200)
        self.fc5 = nn.Linear(200, 100)
        self.fc6 = nn.Linear(100, 50)
        self.out = nn.Linear(50, 17)

    def forward(self, x):
        # conv1D 1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.05)
        # conv1D 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.05)
        # Max Pooling
        x = self.max_pooling(x)
        # Flatten
        x = x.reshape(self.batch_size, -1)
        # 7 FC Layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        x = F.relu(x)
        x = self.fc6(x)
        x = F.relu(x)
        # softmax layer
        x = self.out(x)
        x = F.softmax(x)
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
