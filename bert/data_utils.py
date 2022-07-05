from pathlib import Path
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, Sampler

PROCESSED_DATA_DIR = "/mnt/disks/persist/bert_pretraining_shards"
PROCESSED_PATH = Path(PROCESSED_DATA_DIR)
PROCESSED_FILES = [str(i) for i in PROCESSED_PATH.glob("*.parquet")]

def train_valid_split(valid_ratio = 0.05, rng=None):  # need JAX RNG?
    train_files = []
    valid_files = []
    for f in PROCESSED_FILES:
        if np.random.random() < valid_ratio:
            valid_files.append(f)
        else:
            train_files.append(f)
    if not valid_files:
        valid_files.append(train_files.pop())

    return train_files, valid_files



class BertDataset(Dataset):
    def __init__(self, files=None, shard_size=100):

        self.files = files
        self.shard_size = shard_size
        self.shard_num = len(self.files)
        self.index2file = {idx: fname for idx, fname in enumerate(self.files)}

    def __getitem__(self, index):
        file_idx = index // self.shard_size
        file_name = self.index2file[file_index]
        return dict(pd.read_parquet(file_name).iloc[index - file_idx * self.shard_size])  # dict of np.array

    def __len__(self):
        return self.shard_size * len(self.files)



class ShardSampler(Sampler):
    """Samples elements randomly, used for sharding datasets

    Args:
        data_source (Dataset): dataset to sample from
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source, seed=None):
        self.data_source = data_source
        self.num_samples = len(self.data_source)
        if seed is None:
            seed = np.random.randint(np.iinfo(np.int32).max)
        np.random.seed(seed)

    def __iter__(self):
        n = self.data_source.shard_num
        shard_indexs = list(range(n))
        np.random.shuffle(shard_indexs)
#         print(shard_indexs)
        for shard_index in shard_indexs:
            for idx in range(shard_index * self.data_source.shard_size, (shard_index+1) * self.data_source.shard_size):
                yield idx

    def __len__(self):
        return self.num_samples


files = ['/mnt/disks/persist/bert_pretraining_webdatset/part-2021011897-000000.parquet',
        '/mnt/disks/persist/bert_pretraining_webdatset/part-2021022340-000013.parquet',
        '/mnt/disks/persist/bert_pretraining_webdatset/part-2021022340-000012.parquet']

