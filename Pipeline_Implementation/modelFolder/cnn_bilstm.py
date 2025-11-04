"""
CNN-BiLSTM Model (Andersen et al. 2019)
Real-time AFib detection with ~159K parameters
CORRECTED forward pass shape handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBiLSTM(nn.Module):
    """
    Efficient CNN-BiLSTM for real-time AFib detection
    Reference: Andersen et al. 2019
    """

    def __init__(self, config):
        super(CNNBiLSTM, self).__init__()

        # CNN blocks
        # Input shape: (Batch, 1, 30) for RR intervals
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=60, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(num_features=60)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=60, out_channels=80, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_features=80)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Calculate LSTM input size dynamically based on pooling
        # Input length = 30
        # After pool1 (kernel=2): 30 / 2 = 15
        # After pool2 (kernel=2): 15 / 2 = 7 (integer division)
        # So, the sequence length after CNN is 7
        # The feature size after conv2 is 80
        lstm_input_size = 80 # Features from CNN

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=100,
            num_layers=1,
            batch_first=True, # Input shape: (Batch, SeqLen, Features)
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.2)
        # Output size from BiLSTM = hidden_size * 2 (bidirectional)
        self.fc = nn.Linear(in_features=100 * 2, out_features=2) # 2 output classes (Normal, AFib)

    def forward(self, x):
        # --- THIS LINE IS NEEDED ---
        # Input x from DataLoader is (Batch, Length=30).
        # Conv1d expects (Batch, Channels, Length). Add the channel dimension.
        x = x.unsqueeze(1) # (Batch, 30) -> (Batch, 1, 30)
        # --- END NEEDED LINE ---

        # CNN
        x = F.relu(self.bn1(self.conv1(x))) # Output: (Batch, 60, 30)
        x = self.pool1(x)                   # Output: (Batch, 60, 15)
        x = F.relu(self.bn2(self.conv2(x))) # Output: (Batch, 80, 15)
        x = self.pool2(x)                   # Output: (Batch, 80, 7)

        # BiLSTM expects (Batch, SeqLen, Features)
        # Current shape: (Batch, Features=80, SeqLen=7)
        x = x.transpose(1, 2)               # Output: (Batch, 7, 80)
        lstm_out, _ = self.lstm(x)          # Output: (Batch, 7, 200)

        # Use the output of the last time step or average pooling
        # Andersen et al. likely used average pooling or concatenation, let's average
        lstm_out = self.dropout(lstm_out)
        pooled = torch.mean(lstm_out, dim=1) # Output: (Batch, 200)

        # Output Layer
        logits = self.fc(pooled)            # Output: (Batch, 2)
        return logits


# This is what the universal trainer looks for:
MODEL_CLASS = CNNBiLSTM

MODEL_CONFIG = {
    'name': 'CNN-BiLSTM',
    'description': 'Efficient real-time model for AFib detection',
    'reference': 'Andersen et al. 2019, Expert Systems with Applications',
    'performance': 'Se=98.98%, Sp=96.95%',
    'num_classes': 2,
    'optimizer': {
        'type': 'Adam', # Usually Adam works better than SGD for these models
        # 'momentum': 0.99, # Momentum is for SGD
        # 'weight_decay': 0.000017 # Weight decay can be added to Adam too
    }
}