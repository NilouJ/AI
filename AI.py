import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


class MetabolomicsDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform=True, label_column='Disease',
                 categorical_columns=['Gender']):
        # Load data and drop the 'Samples' column
        self.data = pd.read_csv(path).drop(columns=['Samples'])
        self.categorical_columns = categorical_columns

        # One-hot encode categorical columns
        if categorical_columns:
            self.data = pd.get_dummies(self.data, columns=categorical_columns, drop_first=True)

        # Extract label column
        self.label_column = label_column
        self.labels = pd.Categorical(self.data[self.label_column]).codes

        # Extract numeric features (excluding the label column)
        self.features = self.data.drop(columns=[label_column]).select_dtypes(include=[np.number]).values

        # Handle transformations
        self.transform = transform
        if self.transform:
            # Replace negative values and NaNs
            self.features[self.features < 0] = 1e-6
            self.features = np.nan_to_num(self.features, nan=1e-6)

            # Log-normalize numeric features
            self.features = np.log1p(self.features)

            # Remove constant columns
            non_constant_indices = self.features.std(axis=0) > 1e-6
            self.features = self.features[:, non_constant_indices]

            # Standardize numeric features
            self.scaler = StandardScaler()
            self.features = self.scaler.fit_transform(self.features)

        # Debugging: Check for NaNs
        if np.isnan(self.features).any():
            raise ValueError("NaN values detected in features after transformation.")

    def __getitem__(self, index):
        # Convert features and labels to tensors
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
        batch_size=4,  # Number of samples per batch
        shuffle=True,  # Shuffle data at every epoch
        num_workers=0  # Number of parallel data loading workers
    )

    for epoch in range(3):  # Number of epochs
        for batch_idx, (features, labels) in enumerate(dataloader):
            # Features and labels are tensors of size [batch_size, num_features] and [batch_size]
            print(f"Epoch {epoch}, Batch {batch_idx}")
            print(features.shape, labels.shape)
            # print("Features:", features)
            print("Labels:", labels)
