import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from ml.utils import train_application_classification_cnn_model
from utils import *
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import multiprocessing as mp
import time

lock = mp.Lock()
counter = mp.Value('i', 0)

MODE = 'train'
DEBUG = True
TASK_TYPE = 'app'
DATA_PATH='data'
MODEL_PATH='model/app.cnn.model'
GPU = True

def main():
    # load 參數
    # load 資料
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    # normalize X
    X_train, X_val, X_test = np.array(X_train) / 255, np.array(X_val) / 255, np.array(X_test) / 255
    # 把 y 的 string 做成 one hot encoding 形式

    # label_encoder = LabelBinarizer()
    # y_train_onehot = np.array(label_encoder.fit_transform(y_train))
    # y_val_onehot = label_encoder.transform(y_val)
    # y_test_onehot = label_encoder.transform(y_test)
    # 印一些有的沒的
    print('X_train size:', len(X_train))
    max_x = 0
    for x in X_train:
        if max_x < len(x):
            max_x = len(x)
    print('max length:', max_x)
    print('===== train =====')
    X_train = np.expand_dims(X_train, 1)
    X_val = np.expand_dims(X_val, 1)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    trainset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_application_classification_cnn_model(DATA_PATH, MODEL_PATH, GPU, trainset, valset)
    # return train(X_train, y_train_onehot, X_val, y_val_onehot), (X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot), label_encoder


# =================================================
#  Train
# =================================================
def train(config, X_train, y_train, X_val, y_val):
    cnn = CNN()
    X_train, X_val = np.expand_dims(X_train, 2), np.expand_dims(X_val, 2)
    print('Prepare Trainer...')
    trainer = Trainer(config, cnn, X_train, y_train, X_val, y_val,
                      loss_fn='categorical_crossentropy', metrics='f1_score')
    print('Trainer prepared.')
    trainer.train(7)
    # trainer.train(500)
    return trainer, cnn




def check(filename):
    return not '_class' in filename


def load_data():
    if DEBUG:
        max_data_nb = 10000
    else:
        max_data_nb = 10000
    directory = 'data'
    todo_list = gen_todo_list(directory, check=check)
    ### ver 1 ###
    train_rate = 0.64
    val_rate = 0.16
    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []

    for counter, filename in enumerate(todo_list):
        (tmpX, tmpy) = load(filename)
        if TASK_TYPE == 'class':
            tmpy = load('.'.join(filename.split('.')[:-1]) + '_class.pickle')
        tmpX, tmpy = tmpX[:max_data_nb], tmpy[:max_data_nb]
        assert (len(tmpX) == len(tmpy))
        tmpX = processX(tmpX)
        train_num = int(len(tmpX) * train_rate)
        val_num = int(len(tmpX) * val_rate)
        X_train.extend(tmpX[:train_num])
        y_train.extend(tmpy[:train_num])
        X_val.extend(tmpX[train_num: train_num + val_num])
        y_val.extend(tmpy[train_num: train_num + val_num])
        X_test.extend(tmpX[train_num + val_num:])
        y_test.extend(tmpy[train_num + val_num:])
        print('\rLoading... {}/{}'.format(counter + 1, len(todo_list)), end='')
    print('\r{} Data loaded.               '.format(len(todo_list)))
    return X_train, y_train, X_val, y_val, X_test, y_test


def processX(X):
    if True:
        X = np.array(X)
        lens = [len(x) for x in X]
        maxlen = 1500
        tmpX = np.zeros((len(X), maxlen))
        mask = np.arange(maxlen) < np.array(lens)[:, None]
        tmpX[mask] = np.concatenate(X)
        return tmpX
    else:
        for i, x in enumerate(X):
            tmp_x = np.zeros((1500,))
            tmp_x[:len(x)] = x
            X[i] = tmp_x
        return X


if __name__ == '__main__':
    main()
    # (trainer, model), data, label_encoder = main()
    # (X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot) = data