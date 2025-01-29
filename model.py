import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DModel(nn.Module):
    def __init__(self, input_length=226, num_classes=2, filters=[32, 64], kernel_sizes=[3, 3],
                 strides=[1, 1], dropout_rate=0, fc_hidden_units=[128]):
        super(CNN1DModel, self).__init__()

        # First Convolutional Layer
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters[0], kernel_size=kernel_sizes[0],
                               stride=strides[0], padding=1)  # Padding to keep dimensions
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # Halves the sequence length

        # Second Convolutional Layer
        self.conv2 = nn.Conv1d(in_channels=filters[0], out_channels=filters[1], kernel_size=kernel_sizes[1],
                               stride=strides[1], padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Compute the output size after convolution and pooling
        self.conv_output_length = self.compute_conv_output(input_length, kernel_sizes, strides)

        # Fully Connected Layers
        self.fc1 = nn.Linear(filters[1] * self.conv_output_length, fc_hidden_units[0])
        self.fc2 = nn.Linear(fc_hidden_units[0], num_classes)

        # Dropout for regularization FOR HIDDEN LAYER between fc1 and fc2
        self.dropout = nn.Dropout(dropout_rate)

    def compute_conv_output(self, input_length, kernel_sizes, strides):
        # First Conv + Pool
        length = (input_length + 2 * 1 - kernel_sizes[0]) // strides[0] + 1  # Conv1 output length
        length = length // 2  # Pool1 output length

        # Second Conv + Pool
        length = (length + 2 * 1 - kernel_sizes[1]) // strides[1] + 1  # Conv2 output length
        length = length // 2  # Pool2 output length

        return length

    def forward(self, x):
        # Input shape: [batch_size, sequence_length]
        x = x.unsqueeze(1)  # Add channel dimension (batch_size, 1, sequence_length)

        # Convolutional layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Final layer (logits)

        return x


# Example usage
if __name__ == "__main__":
    batch_size = 4
    sequence_length = 226
    num_classes = 2

    model = CNN1DModel(input_length=sequence_length, num_classes=num_classes)
    print(model)

    # Simulated batch of data
    sample_input = torch.randn(batch_size, sequence_length)
    output = model(sample_input)
    print("Output shape:", output.shape)  # Should be [batch_size, num_classes]
