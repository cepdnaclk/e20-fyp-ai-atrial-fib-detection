"""
ResNet-BiLSTM with Attention (Jia et al. 2020)
Good balance of performance and interpretability
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
class ResNetBiLSTMAttention(nn.Module):
    """Jia et al. 2020 - ResNet-20 with BiLSTM and Attention"""
    
    def __init__(self, config):
        super(ResNetBiLSTMAttention, self).__init__()
        
        # Initial conv
        self.conv1 = nn.Conv1d(1, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(3, 2, 1)
        
        # ResNet layers (simplified)
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 3, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        
        # BiLSTM
        self.bilstm = nn.LSTM(256, 32, 1, batch_first=True, bidirectional=True)
        
        # Attention
        self.attention = nn.Linear(64, 1)
        
        # Classifier
        self.fc = nn.Linear(64, 2)
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_ch, out_ch, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if len(x.shape) ==2:# If shape is (batch, 187)
            x = x.unsqueeze(1) # Make it (batch, 1, 187)
        # ResNet
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # BiLSTM
        x = x.transpose(1, 2)
        lstm_out, _ = self.bilstm(x)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output
        logits = self.fc(attended)
        return logits, attention_weights


MODEL_CLASS = ResNetBiLSTMAttention

MODEL_CONFIG = {
    'name': 'ResNet-BiLSTM-Attention',
    'description': 'Good balance with attention for interpretability',
    'reference': 'Jia et al. 2020, End-to-end Deep Learning Scheme',
    'performance': 'F1=95.5% (duration-based)',
    'num_classes': 2,
    'optimizer': {
        'type': 'Adam',
        'weight_decay': 0.0001
    }
}