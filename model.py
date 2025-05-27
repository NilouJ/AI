import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DModel(nn.Module):
    def __init__(self, input_length=226, num_classes=2, filters=[32, 64], kernel_sizes=[3, 3],
                 strides=[1, 1], dropout_rate=0, fc_hidden_units=[128]):
        super(CNN1DModel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=filters[0], kernel_size=kernel_sizes[0],
                               stride=strides[0], padding=1) 
        self.pool1 = nn.MaxPool1d(kernel_size=2) 

     
        self.conv2 = nn.Conv1d(in_channels=filters[0], out_channels=filters[1], kernel_size=kernel_sizes[1],
                               stride=strides[1], padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.conv_output_length = self.compute_conv_output(input_length, kernel_sizes, strides)

        self.fc1 = nn.Linear(filters[1] * self.conv_output_length, fc_hidden_units[0])
        self.fc2 = nn.Linear(fc_hidden_units[0], num_classes)

        self.dropout = nn.Dropout(dropout_rate)

    def compute_conv_output(self, input_length, kernel_sizes, strides):
        length = (input_length + 2 * 1 - kernel_sizes[0]) // strides[0] + 1  
        length = length // 2  

        # Second Conv + Pool
        length = (length + 2 * 1 - kernel_sizes[1]) // strides[1] + 1 
        length = length // 2 

        return length

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)  

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  

        return x


if __name__ == "__main__":
    batch_size = 4
    sequence_length = 226
    num_classes = 2

    model = CNN1DModel(input_length=sequence_length, num_classes=num_classes)
    print(model)

    # USE IROA DATA
    sample_input = torch.randn(batch_size, sequence_length)
    output = model(sample_input)
    print("Output shape:", output.shape)  
