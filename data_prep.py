import json
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import math as m
import random as rd
import torch


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class TabularSequentialDataset(Dataset):
    def __init__(self, df: pd.DataFrame, schema: dict, max_seq_len: int = 30,
                 batch_size: int = 16, device: str = 'cpu'):
        self.schema = schema
        self.dataset = self.seq_pad(df, max_seq_len)
        self.indices = list(self.dataset.index)
        self.batch_size = batch_size
        self.num_batches = int(m.ceil(len(self.dataset) / batch_size))
        self.max_seq_len = max_seq_len
        self.device = device

    def seq_pad(self, df: pd.DataFrame, max_seq_len: int):
        num_cols = len(df.columns)
        for rid, rdata in tqdm(df.iterrows(), total=len(df)):
            rdata = np.stack(rdata.values, axis=-1)
            rdata_padded = np.zeroes((max_seq_len, num_cols))
            rdata_padded[-len(rdata):, :] = rdata
            df.loc[rid] = rdata_padded.T.tolist()
        return df

    def __len__(self):
        return self.num_batches

    def shuffle(self):
        rd.shuffle(self.indices)

    def __getitem__(self, batch_id: int):
        indices = self.indices[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
        data = self.dataset.loc[indices]
        tensors = dict()
        for col in data.columns:
            array = np.stack(data[col].values, axis=0)
            tensors[col] = torch.tensor(array,
                                        dtype=torch.int if self.shecma[col]['type'] == 'categorical' else torch.half,
                                        device=self.device)
        return tensors

