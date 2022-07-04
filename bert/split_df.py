import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import os

PROCESSED_DATA_DIR = "/mnt/disks/persist/bert_pretraining_data"
PROCESSED_PATH = Path(PROCESSED_DATA_DIR)
PROCESSED_FILES = [str(i) for i in PROCESSED_PATH.glob("*.parquet")]
# /mnt/disks/persist/bert_pretraining_webdatset

def split_func(file_name):
    # print(file_name)
    df = pd.read_parquet(file_name, engine="fastparquet")
    df_length = df.shape[0]

    df_list = np.array_split(df, df_length // 10000)
    base_name = file_name.split("/")[-1][:-8]
    for idx, df_shard in enumerate(df_list):
        df_shard.to_parquet("/mnt/disks/persist/bert_pretraining_webdatset/" + base_name + "-" + "{:06}".format(idx) + ".parquet", index=False)
    # os.remove(file_name)
    
Parallel(3)(delayed(split_func)(x) for x in tqdm(PROCESSED_FILES[:160]))
# Custom BatchSampler with Shard
