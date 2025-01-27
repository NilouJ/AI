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

# 1. Load datasets and split test train
dataset_path = "ST000450.csv"
dataset = MetabolomicsDataset(dataset_path)
dataset.data = StandardScaler().fit_transform(dataset.data)
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 4: instantiate and initialize model
model = CNN1DModel().to(device)

# 5: define loss and set optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 6: Write to tensorboard convert the tensors to list and numpy to add to tensorboard
dataiter = iter(train_loader)
metabolic_profiles, labels = next(dataiter)

plt.figure(figsize=(12, 8))
profiles_array = np.array([profile.numpy() for profile in metabolic_profiles])
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
# Log the heatmap as an image to TensorBoard
writer.add_figure("Heatmap of Sample Data", plt.gcf())
writer.flush()

# data_sample = pd.DataFrame({
#     "Label": labels.numpy(),
#     "Metabolic Profile": [profile.numpy().tolist() for profile in metabolic_profiles]
# })

# # log df to tensorboard
# writer.add_text("Sample Metabolic Profiles", data_sample.to_markdown(index=False))
# writer.flush()


# 7: using tensorboard to inspect the model
writer.add_graph(model, metabolic_profiles)
writer.close()


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
    batch_loss = 0
    for batch_idx, (profiles, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(profiles)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 5 == 0:
            batch_loss = running_loss / 5
            print('  batch {} loss: {}'.format(batch_idx + 1, batch_loss))
            tb_x = epoch_index * len(train_loader) + batch_idx
            tb_writer.add_scalar('Loss/train', batch_loss, tb_x)
            running_loss = 0

    return batch_loss


# Train loop + Eval/Inference loop (Intra epoch)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

EPOCHS = 5
best_vloss = 1_000_000
for epoch_index in range(EPOCHS):
    print('Epoch {}: '.format(epoch_index + 1))

    # Training mode
    model.train(True)
    avg_train_loss = train_one_epoch(epoch_index, writer, model, train_loader, optimizer, criterion)

    # Inference mode
    running_tloss = 0
    model.eval()
    with torch.no_grad():
        for tbatch_idx, (tprofiles, tlabels) in enumerate(test_loader):
            toutput = model(tprofiles)
            tloss = criterion(toutput, tlabels)
            running_tloss += tloss.item()
        avg_test_loss = running_tloss / (tbatch_idx+1)
        print('Train LOSS: {}  Test LOSS: {}'.format(avg_train_loss, avg_test_loss))
        writer.add_scalar('Train vs Test Loss',
                          {'Training': avg_train_loss, 'Test Loss': avg_test_loss},
                          epoch_index+1)
        writer.flush()

        #Track the best performance, and save model's state
        if avg_test_loss < best_vloss:
            best_vloss = avg_test_loss
            best_model_path = 'model_{}_{}'.format(timestamp, epoch_index)
            torch.save(model.state_dict(), best_model_path)






