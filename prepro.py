import os
import time
from scapy.all import *
import numpy as np
import multiprocessing as mp
from scapy.layers.inet import UDP, TCP

from utils import *

lock = mp.Lock()
counter = mp.Value('i', 0)
# cpus = mp.cpu_count()//2
cpus = mp.cpu_count()
data_path = '/media/rootcs412/ca23967d-1da3-4d21-a1cc-71b566c0cd38/data/'
# data_path = '/home/rootcs412/PycharmProjects/Deep-Packet/data/'

with open('objs/fileName2Application.pickle', 'rb') as f:
    dict_name2label = pk.load(f)

with open('objs/fileName2Characterization.pickle', 'rb') as f:
    dict_name2class = pk.load(f)


def pkts2X(pkts):
    X = []
    # lens = []
    for p in pkts:
        r = raw(p)[14:]
        r = np.frombuffer(r, dtype=np.uint8)
        if TCP in p or UDP in p:
            if len(r) > 1500:
                pass
            else:
                X.append(r)
                # lens.append(len(r))
        else:
            pass
    return X  # , lens


def get_data_by_file(filename):
    pkts = rdpcap(filename)
    X = pkts2X(pkts)
    # save X to npy and delete the original pcap (it's too large).
    return X


def task(filename):
    global dict_name2label
    global counter
    head, tail = os.path.split(filename)
    pre = tail.split('.')[0].lower()

    cond1 = os.path.isfile(data_path + tail + '.pickle')
    cond2 = os.path.isfile(data_path + tail + '_class.pickle')
    if cond1 and cond2:
        with lock:
            counter.value += 1
        print('[{}] {} already done'.format(counter, filename))
        return '#ALREADY#'
    X = get_data_by_file(filename)
    if not cond1:
        y = [PREFIX_TO_APP_ID.get(pre)] * len(X)
        # print(y)
        with open(data_path + tail + '.pickle', 'wb') as f:
            pk.dump((X, y), f)
    if not cond2:
        y2 = [PREFIX_TO_TRAFFIC_ID.get(pre)] * len(X)
        with open(data_path + tail + '_class.pickle', 'wb') as f:
            pk.dump(y2, f)
    with lock:
        counter.value += 1
    print('[{}] {} done'.format(counter, filename))
    return 'Done'


# =========================================
# mp init
# =========================================
# pcap_path='/home/rootcs412/pcap/pcaps'
pcap_path='/media/rootcs412/ca23967d-1da3-4d21-a1cc-71b566c0cd38/pcap/CompletePCAPs'
# pcap_path='/media/rootcs412/ca23967d-1da3-4d21-a1cc-71b566c0cd38/pcap'

if __name__ == '__main__':
    start = time.time()
    print("CPU:", cpus)

    pool = mp.Pool(processes=cpus)

    todo_list = gen_todo_list(pcap_path)
    # todo_list = gen_todo_list('D:/pcaps')
    # todo_list = todo_list[:3]
    total_number = len(todo_list)  # 150
    done_list = []
    # key_point
    res = pool.map(task, todo_list)
    print(len(res))
    end = time.time()
    print('Time Used:', int((end - start) / 60), ' min')
