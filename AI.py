# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:48:56 2024

@author: ruira
"""

import os
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt


# Install torchvision if not installed
# !pip install torchvision

# Define the Dataset class
class CustomDataset(Dataset):
    def __init__(self, transform=None, train=True):
        directory = "./"
        positive = "Positive_tensors"
        negative = 'Negative_tensors'

        positive_file_path = os.path.join(directory, positive)
        negative_file_path = os.path.join(directory, negative)
        positive_files = [os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path) if file.endswith(".pt")]
        negative_files = [os.path.join(negative_file_path, file) for file in os.listdir(negative_file_path) if file.endswith(".pt")]
        number_of_samples = len(positive_files) + len(negative_files)
        self.all_files = [None] * number_of_samples
        self.all_files[::2] = positive_files
        self.all_files[1::2] = negative_files 
        self.transform = transform
        self.Y = torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2] = 1
        self.Y[1::2] = 0

        if train:
            self.all_files = self.all_files[0:30000]
            self.Y = self.Y[0:30000]
            self.len = len(self.all_files)
        else:
            self.all_files = self.all_files[30000:]
            self.Y = self.Y[30000:]
            self.len = len(self.all_files)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = torch.load(self.all_files[idx])
        y = self.Y[idx]

        if self.transform:
            image = self.transform(image)

        return image, y

# Load pre-trained ResNet18 model
model = models.resnet18(pretrained=True)

# Modify the fully connected layer (fc) to match the number of classes
model.fc = nn.Linear(512, 2)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create DataLoaders for training and validation datasets
train_dataset = CustomDataset(transform=transforms.ToTensor(), train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

validation_dataset = CustomDataset(transform=transforms.ToTensor(), train=False)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=100, shuffle=False)

# Training loop
n_epochs = 1
loss_list = []
start_time = time.time()

for epoch in range(n_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if batch_idx % 100 == 0:
            print('Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Validation loop
model.eval()
correct = 0
total = 0
incorrect_samples = []
with torch.no_grad():
    for data, target in validation_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        incorrect_preds_mask = predicted != target
        if torch.sum(incorrect_preds_mask) > 0:
            incorrect_samples.append({
                'predicted': predicted[incorrect_preds_mask].tolist(),
                'actual': target[incorrect_preds_mask].tolist()
            })

accuracy = 100. * correct / total
print('Accuracy on validation set: {:.2f}%'.format(accuracy))

# Plot the training loss
plt.figure(figsize=(10, 6))
plt.plot(loss_list)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.show()

# Print incorrect predictions
for i, incorrect in enumerate(incorrect_samples):
    print(f'Sample {i} predicted: {incorrect["predicted"]}, actual: {incorrect["actual"]}')
