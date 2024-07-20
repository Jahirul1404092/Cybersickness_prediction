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
from torchsummary import summary
# Define a new model architecture with convolutional layers and batch normalization
class NHITS_CNN_BN(nn.Module):
    def __init__(self):
        super(NHITS_CNN_BN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=14, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * 7, 128)  
        self.dropout = nn.Dropout(0.6)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


import Utils as utils
# X_train_eye,Y_train_eye,X_val_eye,Y_val_eye,X_test_eye,Y_test_eye,x_scale, y_scale=utils.getregdata()
X_train_eye,Y_train_eye,X_val_eye,Y_val_eye,X_test_eye,Y_test_eye,x_scale, le=utils.getclassificationdata()
X_train_eye = torch.tensor(X_train_eye, dtype=torch.float32)
Y_train_eye = torch.tensor(Y_train_eye, dtype=torch.float32)
X_val_eye = torch.tensor(X_val_eye, dtype=torch.float32)
Y_val_eye = torch.tensor(Y_val_eye, dtype=torch.float32)
X_test_eye = torch.tensor(X_test_eye, dtype=torch.float32)
Y_test_eye = torch.tensor(Y_test_eye, dtype=torch.float32)

X_test_hp= utils.get_hp_data(x_scale)
X_test_hp= torch.tensor(X_test_hp[:65], dtype=torch.float32)

train_dataset = TensorDataset(X_train_eye, Y_train_eye)
val_dataset = TensorDataset(X_val_eye, Y_val_eye)
test_dataset = TensorDataset(X_test_eye, Y_test_eye)

test_dataset_hp = TensorDataset(X_test_hp, Y_test_eye)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

test_loader_hp = DataLoader(test_dataset_hp, batch_size=16, shuffle=False)

# Instantiate the new model
model = NHITS_CNN_BN()
summary(model, input_size=(28, 14)) 
# Define loss function and optimizer with weight decay
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

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
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
###############model architecture
summary(model, input_size=(28,14))

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for X_batch, Y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), Y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        running_loss += loss.item()
    
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for X_batch, Y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), Y_batch)
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
            loss = criterion(outputs.squeeze(), Y_batch)
            test_loss += loss.item()
            
            # Ensure outputs are handled correctly for both batch and single predictions
            outputs = outputs.squeeze().cpu().numpy()
            if outputs.ndim == 0:
                predictions.append(outputs)
            else:
                predictions.extend(outputs)
    
    print(f'Test Loss: {test_loss/len(test_loader):.4f}')
    return predictions

# Convert predictions to numpy array for further analysis if needed
predictions = np.array(prediction(test_loader))

# a,p,r,f=utils.performance_metrics(Y_test_eye, predictions)
results=utils.Error_metrics(Y_test_eye, predictions)
# Analyze results

Y_test_hp=utils.get_hp_data(x_scale)
predictions_hp= np.array(prediction(test_loader_hp))
results=utils.Error_metrics(Y_test_eye, predictions_hp)

Y_test_eye_np = Y_test_eye.numpy()

# Plot predictions vs ground truth
plt.figure(figsize=(10, 5))
plt.plot(Y_test_eye_np, label='True Values')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
