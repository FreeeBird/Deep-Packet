import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.dataset import TensorDataset
import torch.nn.functional as F

from DataLoaderX import DataLoaderX
from model import CNN
from utils import *
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import multiprocessing as mp
import time
from tqdm import tqdm
# mp.set_start_method('spawn')
lock = mp.Lock()
counter = mp.Value('i', 0)

# config
BATCH_SIZE = 32
MODE = 'train'
DEBUG = False
TASK_TYPE = 'app'
DATA_PATH = 'data'
# DATA_PATH = '/media/rootcs412/ca23967d-1da3-4d21-a1cc-71b566c0cd38/data'
MODEL_PATH = 'model/app.cnn.model'
GPU = torch.cuda.is_available()
EPOCH = 7
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    X_train, X_val, X_test = np.array(X_train) / 255, np.array(X_val) / 255, np.array(X_test) / 255
    y_train, y_val, y_test = np.array(y_train,dtype='int64'), np.array(y_val,dtype='int64'), np.array(y_test,dtype='int64')
    print('X_train size:', len(X_train), 'y_train size:', len(y_train))
    print('X_val size:', len(X_val), 'y_val size:', len(y_val))
    print('X_test size:', len(X_test), 'y_test size:', len(y_test))
    max_x = 0
    for x in X_train:
        if max_x < len(x):
            max_x = len(x)
    print('max length:', max_x)
    print('===== train =====')
    X_train = np.expand_dims(X_train, 1)
    # X_val = np.expand_dims(X_val, 0)
    X_test = np.expand_dims(X_test, 1)
    label_encoder = LabelBinarizer()
    # y_train = np.expand_dims(y_train,1)
    # y_test = np.expand_dims(y_test,1)
    # print('yts', y_train.shape)
    # print('yts', y_test.shape)
    # y_train = label_encoder.fit_transform(y_train[1])
    # y_test = label_encoder.fit_transform(y_test[1])
    # print(y_test)
    train(X_train, y_train, X_val, y_val, X_test, y_test)

# =================================================
#  Train
# =================================================
def train(X_train, y_train, X_val, y_val, X_test, y_test):
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).long()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).long()
    # torch.distributed.init_process_group(backend="nccl")
    model = CNN(batch_size=BATCH_SIZE)
    # loss_func = F.cross_entropy
    # loss_func = loss_func.
    loss_func = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    if GPU:
        # model = DistributedDataParallel(model)
        # model = nn.DataParallel(model)
        model = model.cuda()
    print('model:', model)
    print(X_train.shape)
    print(y_train.shape)
    trainset = TensorDataset(X_train, y_train)
    # valset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    testset = TensorDataset(X_test, y_test)
    test_loader = DataLoaderX(testset,batch_size=BATCH_SIZE,num_workers=32,pin_memory=True)
    # train_size = len(X_train) / 16
    train_loader = DataLoaderX(trainset,batch_size=BATCH_SIZE,num_workers=8,pin_memory=True,shuffle=True,drop_last=True)
    for epoch in tqdm(range(EPOCH)):
        print('start epoch', epoch, ' / ', EPOCH)
        running_loss = 0.0
        e_st = time.time()
        for i, (x, y) in enumerate(tqdm(train_loader)):

            # if i >= train_size - 1 :
            #     continue
            x = x.cuda()
            y = y.cuda()
            y_hat = model(x)
            # print('y',y)
            # print('y_hat', y)
            loss = loss_func(y_hat, y).cuda()
            # print('loss:',loss)
            optimizer.zero_grad()
            # print("y", y.shape)
            # print("y_h", y_hat.shape)
            # y_pre = torch.argmax(y_hat, dim=1).cuda()
            # y_pre = torch.tensor(y_pre, dtype=torch.float)
            # print("ypre:", y_pre.type())
            # print("yhat:", y_hat.type())
            # print("y:", y.type())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        e_et = time.time()
        print('\nepoch:', epoch, ' use ', show_time(e_et-e_st))
    print("Finished Training")

    print("Beginning Testing")
    correct = 0
    total = 0
    for data in test_loader:
        x, y = data
        x = x.cuda()
        y = y.cuda()
        print('y', y)
        y_hat = model(x.float())
        print('y_hat',y_hat)
        _, pred = torch.max(y_hat.data, 1)
        print('pred:', pred)
        total += y.size(0)
        correct += (pred == y).sum()
    print('Accuracy of model on test set:%d %%' % (100 * correct // total))
    # X_train, X_val = np.expand_dims(X_train, 2), np.expand_dims(X_val, 2)

def check(filename):
    return not '_class' in filename


def load_data():
    if DEBUG:
        max_data_nb = 100
    else:
        max_data_nb = 10000
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
        tmpX, tmpy = tmpX[:max_data_nb], tmpy[:max_data_nb]
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
        print("time:", show_time(et-st))
    print('\r{} Data loaded.               '.format(len(todo_list)))
    aet = time.time()
    print("time:", show_time(aet-ast))
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
    start_time = time.time()
    main()
    end_time = time.time()
    t = (end_time-start_time)/60
    print('done. used time: %.3f ' % t)
    # (trainer, model), data, label_encoder = main()
    # (X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot) = data
