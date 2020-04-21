import pandas as pd
import numpy as np

from torch.utils.data import Dataset


class M5Dataset(Dataset):
    def __init__(self, data_path, seq_length):
        train_data = np.load(data_path)
        self.seq_length = seq_length

        self.seq = train_data[:, :-1]
        self.label = train_data[:, 1:, 0]

    def __getitem__(self, idx):
        return self.seq[idx][-self.seq_length * 2:], self.label[idx][-self.seq_length * 2:]

    def __len__(self):
        return self.seq.shape[0]

