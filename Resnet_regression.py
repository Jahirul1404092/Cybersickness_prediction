# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 06:09:06 2024

@author: Jahirul
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import Utils as utils
from torchsummary import summary
# Residual Block for ResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# ResNet-inspired architecture for time-series regression
class ResNet1D(nn.Module):
    def __init__(self, block, layers, output_dim=1):
        super(ResNet1D, self).__init__()
        self.in_channels = 64  # Adjusted to match the input feature dimension
        self.conv1 = nn.Conv1d(14, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 14 input channels
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, output_dim)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load regression data
X_train_eye, Y_train_eye, X_val_eye, Y_val_eye, X_test_eye, Y_test_eye, x_scale, y_scale = utils.getregdata()

X_train_eye = torch.tensor(X_train_eye, dtype=torch.float32)
Y_train_eye = torch.tensor(Y_train_eye, dtype=torch.float32)  # Change to float for regression
X_val_eye = torch.tensor(X_val_eye, dtype=torch.float32)
Y_val_eye = torch.tensor(Y_val_eye, dtype=torch.float32)  # Change to float for regression
X_test_eye = torch.tensor(X_test_eye, dtype=torch.float32)
Y_test_eye = torch.tensor(Y_test_eye, dtype=torch.float32)  # Change to float for regression

X_test_hp = utils.get_hp_data(x_scale)
X_test_hp = torch.tensor(X_test_hp[:X_test_eye.shape[0]], dtype=torch.float32)

train_dataset = TensorDataset(X_train_eye, Y_train_eye)
val_dataset = TensorDataset(X_val_eye, Y_val_eye)
test_dataset = TensorDataset(X_test_eye, Y_test_eye)

test_dataset_hp = TensorDataset(X_test_hp, Y_test_eye)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_loader_hp = DataLoader(test_dataset_hp, batch_size=32, shuffle=False)

# Instantiate the new ResNet model
model = ResNet1D(ResidualBlock, [2, 2, 2, 2], output_dim=1)  # Adjusted output_dim to 1 for regression

# Define loss function and optimizer with weight decay
criterion = nn.MSELoss()  # Change to MSELoss for regression
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # Adjusted learning rate and added L2 regularization

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# Implement early stopping
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

early_stopping = EarlyStopping(patience=10, delta=0.0001)

# Train the model for more epochs with early stopping and gradient clipping
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), Y_batch)  # Squeeze outputs to match Y_batch shape
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        running_loss += loss.item()
    
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), Y_batch)  # Squeeze outputs to match Y_batch shape
            val_loss += loss.item()
    
    scheduler.step(val_loss / len(val_loader))
    early_stopping(val_loss / len(val_loader))

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss/len(val_loader):.4f}')
    
    if early_stopping.early_stop:
        print("Early stopping")
        break

def prediction(test_loader):
    # Evaluate on test data
    model.eval()
    test_loss = 0.0
    predictions = []
    
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), Y_batch)  # Squeeze outputs to match Y_batch shape
            test_loss += loss.item()
            
            outputs = outputs.squeeze().cpu().numpy()
            if outputs.ndim == 0:
                predictions.append(outputs)
            else:
                predictions.extend(outputs)
    
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
    return predictions

# Convert predictions to numpy array for further analysis if needed
predictions = np.array(prediction(test_loader))

results = utils.Error_metrics(Y_test_eye, predictions)

print(results)

# Plot predictions vs ground truth
Y_test_eye_np = Y_test_eye.numpy()
plt.figure(figsize=(10, 5))
plt.plot(Y_test_eye_np, label='True Values')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
