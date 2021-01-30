import os
import pickle as pk

from utils import PREFIX_TO_APP_ID

with open('../objs/fileName2Application.pickle', 'rb') as f:
    dict_name2label = pk.load(f)
    print(dict_name2label)

with open('../objs/fileName2Characterization.pickle', 'rb') as f:
    dict_name2class = pk.load(f)
    print(dict_name2class)

pcap_path='/home/rootcs412/pcap/pcaps'
# pcap_path='/media/rootcs412/ca23967d-1da3-4d21-a1cc-71b566c0cd38/pcap/CompletePCAPs'
# pcap_path='/media/rootcs412/ca23967d-1da3-4d21-a1cc-71b566c0cd38/pcap'

def check(filename):
    return not '_class' in filename

if __name__ == '__main__':
    # global dict_name2label
    check = None
    files = os.listdir(pcap_path)
    todo_list = []
    for f in files:
        fullpath = os.path.join(pcap_path, f)
        if os.path.isfile(fullpath):
            if check is not None:
                if check(f):
                    todo_list.append(fullpath)
            else:
                todo_list.append(fullpath)

    num = 0
    print(len(todo_list))
    for filename in todo_list:

        head, tail = os.path.split(filename)
        pre = tail.split('.')[0].lower()
        y = PREFIX_TO_APP_ID.get(pre)
        if y== None:
            print('[{}] -  {} '.format(pre, y))
        else: num += 1
        print(num)