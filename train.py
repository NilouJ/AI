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
from torch.cuda.amp import GradScaler, autocast
import torch.nn.utils.prune as prune
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler

# 1. Load datasets and split test train
dataset_path = "ST000450.csv"
dataset = MetabolomicsDataset(dataset_path)
total_samples = len(dataset)
test_size = int(0.2 * total_samples)
train_size = total_samples - test_size
[test_dataset, train_dataset] = torch.utils.data.random_split(dataset, [test_size, train_size])
# test_labels = [label for _, label in test_dataset]

# 2: Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=True)

# 3: setting up tensorboard
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f"metabolomics_CNN1D_{timestamp}"
writer = SummaryWriter(f"runs/{run_name}")

# 4: instantiate and initialize model
model = CNN1DModel().to('mps')
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
        prune.l1_unstructured(module, name='weight', amount=0.3)

# 5: define loss and set optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 6: Log heatmap of metabolic profiles on tensorboard
def log_tb_heatmap(writer, train_loader):
    dataiter = iter(train_loader)
    metabolic_profiles, labels = next(dataiter)

    plt.figure(figsize=(12, 8))
    profiles_array = np.array([profile.cpu().numpy() for profile in metabolic_profiles])
    labels_list = [f"Label {label.item()}" for label in labels]
    sns.heatmap(
        profiles_array,
        cmap="viridis",
        annot=False,
        # xticklabels=[f"Feature {i}" for i in range(profiles_array.shape[1])],
        yticklabels=labels_list
    )
    # Add labels and title
    plt.title("Heatmap of Metabolic Profiles")
    plt.xlabel("Features")
    plt.ylabel("Samples (Labels)")

    writer.add_figure("Heatmap of Sample Data", plt.gcf())
    writer.flush()


log_tb_heatmap(writer, train_loader)


# data_sample = pd.DataFrame({
#     "Label": labels.numpy(),
#     "Metabolic Profile": [profile.numpy().tolist() for profile in metabolic_profiles]
# })

# # log df to tensorboard
# writer.add_text("Sample Metabolic Profiles", data_sample.to_markdown(index=False))
# writer.flush()


# 7: add model graph to tensorboard
def log_tb_computation_graph(writer, train_loader):
    dataiter = iter(train_loader)
    metabolic_profiles, labels = next(dataiter)
    metabolic_profiles = metabolic_profiles.to('mps')
    writer.add_graph(model, metabolic_profiles)
    writer.flush()


log_tb_computation_graph(writer, train_loader)


#
# # 8: Adding projector to view high dimensional data in lower dimension
# def select_n_random_samples(metabolic_profiles, labels, n):
#     assert len(metabolic_profiles) == len(labels)
#     perm = torch.randperm(len(labels))
#     return metabolic_profiles[perm][:n], labels[perm][:n]
#
#
# random_profiles, random_labels = select_n_random_samples(metabolic_profiles, labels, n=3)
# print(random_profiles.shape)  # Should print: torch.Size([N, D])
# print(len(random_labels))  # Should match N
# writer.add_embedding(mat=random_profiles,
#                      # metadata=random_labels.tolist(),
#                      metadata=[f"Label {label.item()}" for label in random_labels],
#                      tag="Metabolomics reduction",
#                      )
# writer.close()


# Train tracking helper functions getting results
def profiles_to_probe(model, metabolic_profiles, labels):
    outputs = model(metabolic_profiles)
    _, pred_tensor = torch.max(outputs, 1)
    preds = np.squeeze(pred_tensor.detach().cpu().numpy())
    labels = np.squeeze(labels.detach().cpu().numpy())
    return preds, labels


# Training loop PER EPOCH
def train_one_epoch(epoch_index, tb_writer, model, train_loader, optimizer, criterion):
    # model.train()
    running_loss = 0
    for batch_idx, (profiles, labels) in enumerate(train_loader):
        profiles = profiles.to('mps')
        labels = labels.to('mps')
        optimizer.zero_grad()
        output = model(profiles)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_epoch_loss = running_loss / len(train_loader)
    tb_writer.add_scalar('Avg Epoch Loss', avg_epoch_loss, epoch_index)

    return avg_epoch_loss

# Importance analysis feature importance: Gradient calculation
def compute_gradients(model, input_data, target_label):
    model.eval()
    input_data = input_data.to('mps').requires_grad_()  # Enable gradients for inputs

    # Forward pass
    output = model(input_data)
    loss = output[0, target_label]  # Select the loss for the target class

    # Backward pass to compute gradients for the selected loss
    loss.backward()

    # Extract gradients
    gradients = input_data.grad.detach().cpu().numpy()  # Shape: [batch_size, input_features]
    return gradients

## Train loop + Eval/Inference loop (Intra epoch)
# timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
# writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

EPOCHS = 5
best_vloss = float('inf')
for epoch_index in range(EPOCHS):
    print('Epoch {}: '.format(epoch_index + 1))

    # Training mode
    model.train(True)
    avg_train_loss = train_one_epoch(epoch_index, writer, model, train_loader, optimizer, criterion)

    # Inference mode evaluate model
    model.eval()
    running_tloss = 0
    with torch.no_grad():
        for tbatch_idx, (tprofiles, tlabels) in enumerate(test_loader):
            tprofiles, tlabels = tprofiles.to('mps'), tlabels.to('mps')
            toutput = model(tprofiles)
            tloss = criterion(toutput, tlabels)
            running_tloss += tloss.item()
        avg_test_loss = running_tloss / (tbatch_idx + 1)
        print(f"Test loss: {avg_test_loss: } , Test Accuracy: {(100 * avg_test_loss): }%")
        #Log losses to tensorboard
        writer.add_scalar('Loss/Train', avg_train_loss, epoch_index + 1)
        writer.add_scalar('Loss/Test', avg_test_loss, epoch_index + 1)
        writer.flush()

        #Track the best performance, and save model's state
        if avg_test_loss < best_vloss:
            best_vloss = avg_test_loss
            best_model_path = 'model_{}_{}'.format(timestamp, epoch_index)
            torch.save(model.state_dict(), best_model_path)

        # Flush after each epoch
        writer.flush()
writer.close()

# Remove pruning masks after training (optional)
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
        prune.remove(module, 'weight')




# # Quantize the model
# model.eval()  # Ensure evaluation mode
# model.cpu()
# model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
# torch.quantization.prepare(model, inplace=True)
#
# # Calibrate the model
# for profiles, labels in train_loader:
#     profiles = profiles.to('cpu')
#     model(profiles)
#
# # Convert to quantized version
# torch.quantization.convert(model, inplace=True)
# print("Quantization complete!")
#
# # Save the quantized model
# torch.save(model.state_dict(), f'model_quantized_{timestamp}.pth')


# Save the trained model
torch.save(model.state_dict(), f'model_{timestamp}.pth')
print(f"Model saved as 'model_{timestamp}.pth'")