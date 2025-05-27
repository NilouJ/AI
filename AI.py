import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


class MetabolomicsDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=True, label_column='Disease',
                 categorical_columns=['Gender']):
        self.data = pd.read_csv(path).drop(columns=['Samples'])
        self.categorical_columns = categorical_columns

        if categorical_columns:
            self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)

        self.label_column = label_column
        self.labels = pd.Categorical(self.data[self.label_column]).codes

        self.features = self.data.drop(columns=[label_column]).select_dtypes(include=[np.number]).values


        self.transform = transform
        if self.transform:
            self.features[self.features < 0] = 1e-6
            self.features = np.nan_to_num(self.features, nan=1e-6)


            self.features = np.log1p(self.features)

            non_constant_indices = self.features.std(axis=0) > 1e-6
            self.features = self.features[:, non_constant_indices]

            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)

        if np.isnan(self.features).any():
            raise ValueError("NaN values detected in features after transformation.")

    def __getitem__(self, index):
        sample_features = torch.tensor(self.features[index], dtype=torch.float32)
        sample_label = torch.tensor(self.labels[index], dtype=torch.long)
        return sample_features, sample_label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    import torch.multiprocessing as mp
    mp.set_start_method('spawn')
    # Loading dataset
    dataset = MetabolomicsDataset(
        path="/Users/ninoufjj/Library/CloudStorage/OneDrive-MacquarieUniversity"
             "/Nilou/AI/ST000450.csv",
        transform=True,
        label_column='Disease',
        categorical_columns=['Gender']
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4, 
        shuffle=True, 
        num_workers=0  
    )

    for epoch in range(3):  
        for batch_idx, (features, labels) in enumerate(dataloader):
           
            print(f"Epoch {epoch}, Batch {batch_idx}")
            print(features.shape, labels.shape)
            print("Labels:", labels)
