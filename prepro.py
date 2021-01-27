import os
import time
from scapy.all import *
import numpy as np
import multiprocessing as mp
import pickle as pk

from scapy.layers.dns import DNS
from scapy.layers.inet import UDP, TCP

from utils import *

lock = mp.Lock()
counter = mp.Value('i', 0)
cpus = mp.cpu_count()
# cpus = mp.cpu_count()

with open('objs/fileName2Application.pickle', 'rb') as f:
    dict_name2label = pk.load(f)

with open('objs/fileName2Characterization.pickle', 'rb') as f:
    dict_name2class = pk.load(f)


def pkts2X(pkts):
    X = []
    # lens = []
    for p in pkts:
        # r = transform_packet(p)
        # X.append(r)
        # ===================================
        # step 1 : remove Ether Header
        # ===================================
        # if TCP in p and (p.flags & 0x13):
        #     # not payload or contains only padding
        #     layers = p[TCP].payload.layers()
        #     if not layers or (Padding in layers and len(layers) == 1):
        #         pass
        #
        #     # DNS segment
        # if DNS in p:
        #     pass
        # if UDP in p:
        #     layer_after = p[UDP].payload.copy()
        #     # build a padding layer
        #     pad = Padding()
        #     pad.load = '\x00' * 12
        #     layer_before = p.copy()
        #     layer_before[UDP].remove_payload()
        #     p = layer_before / pad / layer_after
        r = raw(p)[14:]
        r = np.frombuffer(r, dtype=np.uint8)
        # p.show()
        # ===================================
        # step 2 : pad 0 to UDP Header
        # it seems that we need to do nothing this step
        # I found some length of raw data is larger than 1500
        # remove them.
        # ===================================

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
    cond1 = os.path.isfile(os.path.join('data', tail + '.pickle'))
    cond2 = os.path.isfile(os.path.join('data', tail + '_class.pickle'))
    if cond1 and cond2:
        with lock:
            counter.value += 1
        print('[{}] {}'.format(counter, filename))
        return '#ALREADY#'
    X = get_data_by_file(filename)
    if not cond1:
        y = [PREFIX_TO_APP_ID.get(pre)] * len(X)
        # print(y)
        with open(os.path.join('data', tail + '.pickle'), 'wb') as f:
            pk.dump((X, y), f)
    if not cond2:
        y2 = [PREFIX_TO_TRAFFIC_ID.get(pre)] * len(X)
        with open(os.path.join('data', tail + '_class.pickle'), 'wb') as f:
            pk.dump(y2, f)
    with lock:
        counter.value += 1
    print('[{}] {}'.format(counter, filename))
    return 'Done'


# =========================================
# mp init
# =========================================


if __name__ == '__main__':
    start = time.time()
    print("CPU:", cpus)

    pool = mp.Pool(processes=cpus)

    todo_list = gen_todo_list('/home/rootcs412/pcaps')
    # todo_list = gen_todo_list('D:/pcaps')
    # todo_list = todo_list[:3]
    total_number = len(todo_list)  # 150
    done_list = []
    # key_point
    res = pool.map(task, todo_list)
    print(len(res))
    end = time.time()
    print('Time Used:', int((end - start) / 60), 'mins')
