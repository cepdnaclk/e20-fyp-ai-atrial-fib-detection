"""

# ==============================================================================
# FILE 1: models/cnn_bilstm.py
# ==============================================================================

CNN-BiLSTM Model (Andersen et al. 2019)
Real-time AFib detection with ~159K parameters
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
        self.conv1 = nn.Conv1d(1, 60, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(60)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(60, 80, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(80)
        self.pool2 = nn.MaxPool1d(2)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=80,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(200, 2)
    
    def forward(self, x):
        # CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # BiLSTM
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        pooled = torch.mean(lstm_out, dim=1)
        
        # Output
        logits = self.fc(pooled)
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
        'type': 'SGD',
        'momentum': 0.99,
        'weight_decay': 0.000017
    }
}