import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Load the data
with h5py.File("qam_data.h5", "r") as f:
    qam_signal = np.array(f["qam_signal"])

# Convert data to (Real, Imag) format
X = np.column_stack((qam_signal.real, qam_signal.imag))
y = np.array([i % 16 for i in range(len(X))])  # Labels (0-15)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Split the data
dataset = TensorDataset(X_tensor, y_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define the improved CNN model
class QAM_CNN(nn.Module):
    def __init__(self):
        super(QAM_CNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 16)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # No ReLU here since we'll use softmax later
        return x

# Define Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

# Training the model
model = QAM_CNN()
criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Training loop
epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "qam_cnn_model.pth")
print("✅ The trained model has been saved in qam_cnn_model.pth")