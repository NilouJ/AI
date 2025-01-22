import torch
# print(torch.backends.mps.is_available())
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math


class MetabolomicsDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=None, label_column='Disease'):
        self.data = pd.read_csv(path)
        self.features = self.data.drop(columns=[label_column]).values
        self.labels = self.data[label_column].values
        self.transform = transform

        if self.transform:
            self.transform = self.transform.fit_transform(self.features)

    # a replication of dataset subclass maps keys to the data sample
    # overriding __getitem__ fetches a sample data for a given key
    # __len__ returns size of the dataset
    # may also include getitems for batch
    def __getitem__(self, index):
        sample_data = self.data.iloc[index]
        return sample_data

    def __len__(self):
        return len(self.data)
