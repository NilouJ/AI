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
