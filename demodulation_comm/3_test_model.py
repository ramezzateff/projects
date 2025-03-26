#!/usr/bin/python3
import torch
import torch.nn as nn
import numpy as np
import h5py
from gnuradio import digital, blocks, gr, analog
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Define the same model used in training.
class QAM_CNN(nn.Module):
    def __init__(self):
        super(QAM_CNN, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 16)  # 16-QAM output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Load the form
model = QAM_CNN()
model.load_state_dict(torch.load("qam_cnn_model.pth", weights_only=True))
model.eval()

# Load test data
with h5py.File("qam_data.h5", "r") as f:
    qam_signal = np.array(f["qam_signal"])

X_test = np.column_stack((qam_signal.real, qam_signal.imag))
X_tensor = torch.tensor(X_test, dtype=torch.float32)

# Classification using the model
with torch.no_grad():
    predictions = model(X_tensor)
    predicted_labels = torch.argmax(predictions, dim=1).numpy()

print("✅ Demodulation complete, here are some of the predicted values:", predicted_labels[:20])
