import torch
from torch import nn as nn
from torch.nn import functional as F
from petastorm import make_reader
# from petastorm.pytorch import DataLoader
from pathlib import Path

print(torch.cuda.is_available())
print(torch.cuda.current_device())

print(torch.cuda.device_count())
print(torch.cuda.max_memory_allocated(0))
print(torch.cuda.memory_cached(0))
print(torch.cuda.max_memory_cached(0))


# DATA_PATH = '/media/rootcs412/ca23967d-1da3-4d21-a1cc-71b566c0cd38/dataset/temp_pro/application_classification/train.parquet'
# dataloader = DataLoader(
#     make_reader(Path(DATA_PATH).absolute().as_uri(),
#                 workers_count=10,
#                 reader_pool_type='process',
#                 shuffle_row_groups=True,
#                 shuffle_row_drop_partitions=2,
#                 num_epochs=1,
#                 ),
#     batch_size=256, shuffling_queue_capacity=4096
# )
# print(dataloader.reader)
# print(dataloader.batch_size)
# for batch, data in enumerate(dataloader):
#     print(data)
#     print(data['feature'].size())
