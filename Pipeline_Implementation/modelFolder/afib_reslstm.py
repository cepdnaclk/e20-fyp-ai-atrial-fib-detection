"""
AFib-ResLSTM - Novel Architecture
Multi-scale Conv + ResNet-34 + BiLSTM + Self-Attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Basic Residual Block for ResNet"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add shortcut
        out += self.shortcut(identity)
        out = self.relu(out)
        
        return out

class MultiScaleConv1D(nn.Module):
    """Multi-scale parallel convolutions"""
    def __init__(self):
        super(MultiScaleConv1D, self).__init__()
        
        # Three parallel branches
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 32, 15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv1d(96, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = torch.cat([b1, b2, b3], dim=1)
        return self.fusion(concat)


class AFibResLSTM(nn.Module):
    """
    Novel AFib-ResLSTM Architecture
    Your contributions: Multi-scale + Self-Attention + Hybrid Fusion
    """
    
    def __init__(self, config):
        super(AFibResLSTM, self).__init__()
        
        # Multi-scale feature extraction
        self.multiscale = MultiScaleConv1D()
        
        # ResNet-34 backbone (simplified)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Bi-LSTM
        self.lstm1 = nn.LSTM(512, 128, 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, 1, batch_first=True, bidirectional=True)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        
        self.dropout = nn.Dropout(0.2)
        
        # Hybrid fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(640, 256),  # 512 (ResNet) + 128 (LSTM-Attention)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: ECG input, shape (batch, 7500) or (batch, 1, 7500)
        
        Returns:
            logits: (batch, 2)
            attention_weights: Attention weights for interpretability
        """
        #  Ensure input has channel dimension for Conv1d
        if len(x.shape) == 2:  # If shape is (batch, length)
            x = x.unsqueeze(1)  # Make it (batch, 1, length)
        
        # Multi-scale features
        x = self.multiscale(x)
        
        # ResNet backbone
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        resnet_pooled = self.avgpool(x).squeeze(-1)
        
        # BiLSTM
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout(lstm_out)
        lstm_out, _ = self.lstm2(lstm_out)
        
        # Self-attention
        attended, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_features = torch.mean(attended, dim=1)
        
        # Hybrid fusion
        fused = torch.cat([resnet_pooled, lstm_features], dim=1)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, attention_weights


MODEL_CLASS = AFibResLSTM

MODEL_CONFIG = {
    'name': 'AFib-ResLSTM (Novel)',
    'description': 'Multi-scale + ResNet-34 + BiLSTM + Self-Attention + Hybrid Fusion',
    'reference': 'Your Research 2025',
    'performance': 'Target: AUROC > 0.96, Se > 95%, Sp > 95%',
    'num_classes': 2,
    'optimizer': {
        'type': 'Adam',
        'weight_decay': 0.00001
    },
    'innovations': [
        'Multi-scale parallel convolutions (3, 7, 15 kernels)',
        'Self-attention on BiLSTM output',
        'Hybrid feature fusion (ResNet + LSTM-Attention)'
    ]
}

# Use focal loss for this model
class FocalLoss(nn.Module):   #alpha=0.25, gamma=1.0
    def __init__(self, alpha=0.25, gamma=1.0):
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

LOSS_FUNCTION = FocalLoss