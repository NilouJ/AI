# import torch
# from torch.utils.data import DataLoader
# from model import CNN1DModel  # Import the model definition
# from AI import MetabolomicsDataset  # Import your dataset class
#
# # 1. Define Paths and Hyperparameters
# dataset_path = "ST000450.csv"
# batch_size = 5
# sequence_length = 226  # Number of features in your dataset
# num_classes = 2  # Assuming binary classification
#
# # 2. Load the Dataset
# dataset = MetabolomicsDataset(
#     path=dataset_path,
#     transform=True,
#     label_column='Disease',  # Make sure this matches the label column name in your dataset
#     categorical_columns=['Gender']  # Include any categorical columns here
# )
#
# # 3. Create DataLoader
# dataloader = DataLoader(
#     dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=0  # Use 0 workers on Mac to avoid multiprocessing issues
# )
#
# # 4. Initialize the Model
# model = CNN1DModel(input_length=sequence_length, num_classes=num_classes)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
#
# # 5. Test the Model with DataLoader
# for batch_idx, (features, labels) in enumerate(dataloader):
#     # features = features.unsqueeze(1).to(device)
#     features = features.to(device)# Add channel dimension for 1D CNN
#     labels = labels.to(device)
#
#     # Forward pass
#     outputs = model(features)
#
#     print(f"Batch {batch_idx + 1}")
#     print(f"Features Shape: {features.shape}")  # Should be [batch_size, 1, sequence_length]
#     print(f"Labels Shape: {labels.shape}")  # Should be [batch_size]
#     print(f"Model Outputs Shape: {outputs.shape}")  # Should be [batch_size, num_classes]
#
#     # Break after the first batch for testing purposes
#     break


import torch
from torch.utils.tensorboard import SummaryWriter

# # Dummy data
# random_profiles = torch.randn(3, 226)  # 3 samples, 226 features
# random_labels = ["Class 1", "Class 2", "Class 3"]  # Metadata for each sample
#
# # Log embeddings
# writer = SummaryWriter("runs/test_embeddings")
# writer.add_embedding(
#     mat=random_profiles,
#     metadata=random_labels,
#     tag="Test Embeddings",
#     global_step=1
# )
# writer.close()

from torch.utils.tensorboard import SummaryWriter
from model import CNN1DModel
from AI import MetabolomicsDataset
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# # 1. Load datasets and split test train
# dataset_path = "ST000450.csv"
# dataset = MetabolomicsDataset(dataset_path)
# total_samples = len(dataset)
# test_size = int(0.2 * total_samples)
# train_size = total_samples - test_size
# [test_dataset, train_dataset] = torch.utils.data.random_split(dataset, [test_size, train_size])
# # test_labels = [label for _, label in test_dataset]
#
# # 2: Data loaders
# train_loader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)
#
#
# def train_one_epoch(epoch_index, training_loader):
#     running_loss = 0.
#
#     for i, data in enumerate(training_loader):
#         inputs, labels = data
#         loss = i
#         # Gather data and report
#         running_loss += loss
#
#     avg_epoch_loss = running_loss / len(training_loader)
#     print(f'running loss: {avg_epoch_loss}, i: {i}')
#
#     return running_loss / len(training_loader)
#
#
# for epoch_index in range(0, 3):
#     train_one_epoch(epoch_index, training_loader=train_loader)

# import requests
# from bs4 import BeautifulSoup
# import pandas as pd
#
# url = ("https://docs.google.com/document/u/0/d/e/2PACX-1vRMx5YQlZNa3ra8dYYxmv-QIQ3YJe8tbI3kqcuC7lQiZm"
#        "-CSEznKfN_HYNSpoXcZIV3Y_O3YoUB1ecq/pub?pli=1")
# response = requests.get(url)
# if response.status_code == 200:
#     soup = BeautifulSoup(response.text, "html.parser")
#     tables = soup.find_all("table")
#     if tables:
#         data_frames = []
#         for index, table in enumerate(tables):
#             rows = table.find_all("tr")
#             table_data = []


import requests
from bs4 import BeautifulSoup


def decode_secret_message(url):
    # Fetch HTML content from the Google Docs URL
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to retrieve document.")
        return

    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the table
    table = soup.find("table")
    if not table:
        print("No table found in the document.")
        return

    # Extract rows from the table
    rows = table.find_all("tr")
    data = []

    for row in rows[1:]:  # Skip the header row
        cols = row.find_all("td")
        if len(cols) == 3:
            x = int(cols[0].text.strip())
            char = cols[1].text.strip()
            y = int(cols[2].text.strip())
            data.append((x, y, char))

    # Determine grid size
    max_x = max(x for x, y, _ in data)
    max_y = max(y for x, y, _ in data)

    # Create empty grid filled with spaces
    grid = [[" " for _ in range(max_x + 1)] for _ in range(max_y + 1)]

    # Populate the grid with characters
    for x, y, char in data:
        grid[y][x] = char  # Assign character at the correct position

    # Print the decoded message
    for row in grid:
        print("".join(row))


# Example usage (Replace with actual URL)
url = ("https://docs.google.com/document/u/0/d/e/2PACX-1vRMx5YQlZNa3ra8dYYxmv-QIQ3YJe8tbI3kqcuC7lQiZm"
       "-CSEznKfN_HYNSpoXcZIV3Y_O3YoUB1ecq/pub?pli=1")
decode_secret_message(url)
