from pathlib import Path
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader

PROCESSED_DATA_DIR = "/mnt/disks/persist/bert_pretraining_data"
PROCESSED_PATH = Path(PROCESSED_DATA_DIR)
PROCESSED_FILES = [str(i) for i in PROCESSED_PATH.glob("*.parquet")][:5]

class BertDataset(Dataset):
    def __init__(self):
        self.files = PROCESSED_FILES
        individual_lengths = [len(pd.read_parquet(i)) for i in self.files]
        self.lengths = [sum(individual_lengths[:idx]) + item for idx, item in enumerate(individual_lengths)]
    
    def _binary_search(self, item, start=0, end=0):
        mid = (start + end) // 2
        if item >= self.lengths[mid] and item < self.lengths[mid+1]:
            return self.

        if item > self.lengths[start]

    def __getitem__(self, index):

        pass
    
    def __len__(self):
        return self.lengths[-1]
    