import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
import torch.nn.functional as F
from ml.utils import train_application_classification_cnn_model
from model import CNN
from utils import *
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import multiprocessing as mp
import time

DATA_PATH = ''
lock = mp.Lock()
counter = mp.Value('i', 0)

MODE = 'train'
DEBUG = True
TASK_TYPE = 'app'
DATA_PATH = 'data'
MODEL_PATH = 'model/app.cnn.model'
GPU = torch.cuda.is_available()
EPOCH = 300

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    X_train, X_val, X_test = np.array(X_train) / 255, np.array(X_val) / 255, np.array(X_test) / 255
    y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)
    print('X_train size:', len(X_train))
    print('X_val size:', len(X_val))
    print('X_test size:', len(X_test))
    max_x = 0
    for x in X_train:
        if max_x < len(x):
            max_x = len(x)
    print('max length:', max_x)
    print('===== train =====')
    X_train = np.expand_dims(X_train, 0)
    X_val = np.expand_dims(X_val, 0)
    X_test = np.expand_dims(X_test, 0)

    train(X_train, y_train, X_val, y_val, X_test, y_test)


# =================================================
#  Train
# =================================================
def train(X_train, y_train, X_val, y_val, X_test, y_test):
    model = CNN()
    if GPU:
        model = model.cuda()
    loss_fun = F.cross_entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    testset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_loader = DataLoader(trainset,batch_size=256,num_workers=8,pin_memory=True,shuffle=True)
    test_loader = DataLoader(testset,batch_size=256,num_workers=8,pin_memory=True)
    for epoch in range(EPOCH):
        running_loss = 0.0
        for i,data in enumerate(train_loader,0):
            x, y = data
            x = x.float()
            y = y.long()
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fun(y_hat, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished Training")

    print("Beginning Testing")
    correct = 0
    total = 0
    for data in test_loader:
        x, y = data
        y_hat = model(x.float())
        _, pred = torch.max(y_hat.data, 1)
        total += y.size(0)
        correct += (pred == y).sum()
    print('Accuracy of model on test set:%d %%' % (100 * correct / total))
    # X_train, X_val = np.expand_dims(X_train, 2), np.expand_dims(X_val, 2)

def check(filename):
    return not '_class' in filename


def load_data():
    ast = time.time()
    todo_list = gen_todo_list(DATA_PATH, check=check)
    train_rate = 0.64
    val_rate = 0.16
    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []

    for counter, filename in enumerate(todo_list):
        st = time.time()
        (tmpX, tmpy) = load(filename)
        if TASK_TYPE == 'class':
            tmpy = load('.'.join(filename.split('.')[:-1]) + '_class.pickle')
        # tmpX, tmpy = tmpX[:max_data_nb], tmpy[:max_data_nb]
        assert (len(tmpX) == len(tmpy))
        tmpX = processX(tmpX)
        train_num = int(len(tmpX) * train_rate)
        val_num = int(len(tmpX) * val_rate)
        x_train.extend(tmpX[:train_num])
        y_train.extend(tmpy[:train_num])
        x_val.extend(tmpX[train_num: train_num + val_num])
        y_val.extend(tmpy[train_num: train_num + val_num])
        x_test.extend(tmpX[train_num + val_num:])
        y_test.extend(tmpy[train_num + val_num:])
        print('\rLoading... {}/{}'.format(counter + 1, len(todo_list)), end='')
        et = time.time()
        print("time:", (et - st) / 60, 'min')
    print('\r{} Data loaded.               '.format(len(todo_list)))
    aet = time.time()
    print("time:", (aet - ast) / 60, 'min')
    return x_train, y_train, x_val, y_val, x_test, y_test


def processX(X):
    X = np.array(X)
    lens = [len(x) for x in X]
    maxlen = 1500
    tmpX = np.zeros((len(X), maxlen))
    mask = np.arange(maxlen) < np.array(lens)[:, None]
    tmpX[mask] = np.concatenate(X)
    return tmpX


if __name__ == '__main__':
    main()
    # (trainer, model), data, label_encoder = main()
    # (X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot) = data
