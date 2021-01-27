import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import CNN
from torch.utils.data.dataset import random_split

BATCH_SIZE = 256




def generate_batch(batch):
    label = torch.Tensor([entry[0] for entry in batch])
    feature = [entry[1] for entry in batch]


def trainer(X_train, y_train, X_val, y_val, X_test, y_test):
    model = CNN(batch_size=BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr_)
    loss_func = torch.nn.functional.cross_entropy
    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()
    print(model)
    model.train()
    tr_loss = 0
    x_train,y_train = '',''
    x_val,y_val = '',''


