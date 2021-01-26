import torch
from torch import nn as nn
from torch.nn import functional as F
from petastorm import make_reader
from petastorm.pytorch import DataLoader
from pathlib import Path
from pyarrow import filesystem


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=200,
            kernel_size=4,
            stride=3
        )
        self.conv2 = nn.Conv1d(
            in_channels=200,
            out_channels=200,
            kernel_size=5,
            stride=1
        )

        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(500, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 200)
        self.fc4 = nn.Linear(200, 100)
        self.fc5 = nn.Linear(100, 50)
        self.out = nn.Linear(50, 17)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.max_pool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)

print(torch.cuda.is_available())
# DATA_PATH = 'D/train_set/application_classification'
# dataloader = DataLoader(
#     make_reader(Path(DATA_PATH).as_posix(),
#                 reader_pool_type='process',
#                 workers_count=10,
#                 shuffle_row_groups=True,
#                 shuffle_row_drop_partitions=2,
#                 num_epochs=1,
#                 ),
#     batch_size=16, shuffling_queue_capacity=4096
# )
# print(dataloader.reader)
model = CNN().float()
learning_rate = 0.05
epoch = 200
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
