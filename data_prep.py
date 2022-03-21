import json
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import math as m
import random as rd
import torch
import torch.nn as nn
from typing import Dict, Union, Optional, Any, Iterable


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
                 batch_size: int = 16, device=torch.device('cuda')):
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
            rdata_padded = np.zeros((max_seq_len, num_cols))
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
                                        dtype=torch.int if self.schema[col]['type'] == 'categorical' else torch.half,
                                        device=self.device)
        return tensors


class FeaturePreprocessing(nn.Module):
    def __init__(self, schema: Dict[str, str], hidden_dim: int=64, training: bool=True):
        super(FeaturePreprocessing, self).__init__()
        self.training = training
        self.embedding = dict()
        self.hidden_dim = hidden_dim
        self.features_dim = 0
        self.features_order = list()
        for feat, stats in schema.items():
            if stats['type'] == 'categorical':
                self.embedding[feat] == nn.Embedding(num_embeddings=stats['max_val']+1,
                                                     embedding_dim=stats['embedding_dim'])
                self.features_dim += stats['embedding_dim']
            else:
                self.features_dim += 1
            self.features_order.append(feat)
        self.normalize = nn.BatchNorm1d(num_features=self.features_dim)
        self.regularize = nn.Dropout(p=0.1369)
        self.full_connect = nn.Linear(in_features=self.features_dim,
                                      out_features=self.hidden_dim, bias=True)
        self.activation = nn.Mish()

    def forward(self, tensor: Dict[str, torch.Tensor]):
        features = []
        for feat in self.features_order:
            if feat in self.embedding.keys():
                feat_tensor = self.embedding[feat](tensor[feat])
                feat_tensor = torch.swapaxes(feat_tensor, axis0=1, axis1=2)
            else:
                feat_tensor = torch.unsqueeze(tensor[feat], dim=1)
            features.append(feat_tensor)
        features = torch.cat(features, dim=1)
        features = self.normalize(features)
        features = torch.swapaxes(features, axis0=1, axis1=2)
        features = self.regularize(features)
        features = self.full_connect(features)
        features = self.activation(features)
        return features
