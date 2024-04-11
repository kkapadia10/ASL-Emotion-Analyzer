"""
This class is responsible for creating a machine learning model that trains
upon our specified dataset. Converts data into tensors
from which the machine learning model learns over 50
epochs. Accuracy of the training is tested and the learned model is saved.
"""

import pickle
import numpy as np
import torch
import torch.nn as nn  # This imports the neural network module
import torch.optim as optim  # This imports the optimizer module

# sklearn used for encoding labels and splitting data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Torch is mainly the library that actually handles the dataset
from torch.utils.data import DataLoader, TensorDataset

# Loads data from serialized pickle object
data_dict = pickle.load(open('data.pickle', 'rb'))

# Creates arrays of data and labels that contains training data
# dtype=object for variable-length sequences
data = np.array(data_dict['data'], dtype=object)
labels = np.array(data_dict['labels'])

# Convert string labels into integers for easier processing
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Ensuring all data points have the same dimensionality
# Find the max length of any data point
max_len = max(len(item) for item in data)

# Pad sequences so that all are the same length
data_padded = np.zeros((len(data), max_len))

for i, row in enumerate(data):
    data_padded[i, :len(row)] = row

# Converting from NumPy arrays to PyTorch tensors
data_tensor = torch.tensor(data_padded, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(data_tensor, labels_tensor,
                                                    test_size=0.2, shuffle=True, stratify=labels_tensor)

# Create datasets and dataloaders
train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define a simple neural network for classification
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()

        # creates neural network based on number of classes/inputs
        self.fc1 = nn.Linear(input_size, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    # responsible for forward pass method of neural network.
    # performs a transformation, passes onto next tensor layer,
    # which does another set of linear transformations. finally
    # the output is returned.
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Creating machine learning model based on neural network defined above
# After padding, use data_padded.shape[1] for the input size
model = SimpleNN(input_size=data_padded.shape[1], num_classes=18)

# Creates essential components of ml model: loss function & optimizer.
# Loss function measures difference between actual and predicted output
criterion = nn.CrossEntropyLoss()

# optimizer adjusts model's parameters to minimize loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Runs 50 loops through the training data to compute loss and
# optimize parameters
for epoch in range(50):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Tests the model to ensure 100% interpretation accuracy
# during training.
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    # computes number of correct samples from the total
    # the data provides a batch of input data and their
    # related target labels
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

accuracy = 100 * correct / total
print(f'{accuracy}% of samples were classified correctly!')

# Saves the model
torch.save(model.state_dict(), 'model.pth')
