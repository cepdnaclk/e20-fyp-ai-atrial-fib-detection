"""
CNN-LSTM with Focal Loss (Petmezas et al. 2021)
Best specificity: 99.29%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMFocal(nn.Module):
    """Petmezas et al. 2021 - Highest specificity model"""
    
    def __init__(self, config):
        super(CNNLSTMFocal, self).__init__()
        
        # CNN triplets
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(2)
        
        # LSTM
        self.lstm = nn.LSTM(64, 64, 1, batch_first=True)
        
        # Classifier
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        if len(x.shape) ==2:# If shape is (batch, 187)
            x = x.unsqueeze(1) # Make it (batch, 1, 187)
        # CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # LSTM
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # Output
        x = self.dropout(last_output)
        logits = self.fc(x)
        return logits


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced data"""
    def __init__(self, alpha=0.65, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets_one_hot[:, 1] * self.alpha + targets_one_hot[:, 0] * (1 - self.alpha)
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        return (alpha_weight * focal_weight * ce_loss).mean()


MODEL_CLASS = CNNLSTMFocal

MODEL_CONFIG = {
    'name': 'CNN-LSTM-Focal',
    'description': 'Highest specificity model with focal loss',
    'reference': 'Petmezas et al. 2021, Biomedical Signal Processing',
    'performance': 'Se=97.87%, Sp=99.29%',
    'num_classes': 2,
    'optimizer': {
        'type': 'Adam',
        'weight_decay': 0.001
    }
}

LOSS_FUNCTION = FocalLoss
