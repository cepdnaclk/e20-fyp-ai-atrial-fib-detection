# ============================================================================
# models/model_architecture.py
# AFib-ResLSTM Model Architecture
# Extracted from 04_model_architecture.ipynb
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Configuration
# ============================================================================

class ModelConfig:
    """Configuration for AFib-ResLSTM model"""
    
    # Input parameters
    INPUT_LENGTH = 2500
    INPUT_CHANNELS = 1
    NUM_CLASSES = 2
    
    # Multi-scale convolution parameters
    MULTISCALE_FILTERS = [32, 32, 32]
    MULTISCALE_KERNELS = [3, 7, 15]
    FUSION_FILTERS = 64
    
    # ResNet backbone parameters
    RESNET_INITIAL_FILTERS = 64
    RESNET_LAYERS = [3, 4, 6, 3]  # ResNet-34
    RESNET_FILTERS = [64, 128, 256, 512]
    
    # Bi-LSTM parameters
    LSTM_HIDDEN_1 = 128
    LSTM_HIDDEN_2 = 64
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    
    # Self-Attention parameters
    ATTENTION_HEADS = 4
    ATTENTION_DIM = 128
    
    # Classification head parameters
    FC_HIDDEN = 256
    DROPOUT = 0.5
    
    # Focal Loss parameters
    FOCAL_ALPHA = 0.65
    FOCAL_GAMMA = 2.0


# ============================================================================
# Multi-Scale Convolution Block
# ============================================================================

class MultiScaleConv1D(nn.Module):
    """Multi-scale parallel convolutions"""
    
    def __init__(self, in_channels, kernels=[3, 7, 15], filters=[32, 32, 32], fusion_filters=64):
        super(MultiScaleConv1D, self).__init__()
        
        self.branches = nn.ModuleList()
        
        for kernel, num_filters in zip(kernels, filters):
            branch = nn.Sequential(
                nn.Conv1d(in_channels, num_filters, kernel_size=kernel, padding=kernel//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        total_filters = sum(filters)
        self.fusion = nn.Sequential(
            nn.Conv1d(total_filters, fusion_filters, kernel_size=1),
            nn.BatchNorm1d(fusion_filters),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        concatenated = torch.cat(branch_outputs, dim=1)
        fused = self.fusion(concatenated)
        return fused


# ============================================================================
# ResNet Building Blocks
# ============================================================================

class ResidualBlock1D(nn.Module):
    """Basic Residual Block for 1D signals"""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNetBackbone1D(nn.Module):
    """1D ResNet-34 backbone"""
    
    def __init__(self, in_channels=64, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512]):
        super(ResNetBackbone1D, self).__init__()
        
        self.in_channels = in_channels
        
        self.layer1 = self._make_layer(filters[0], layers[0], stride=1)
        self.layer2 = self._make_layer(filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(filters[3], layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        pooled = self.avgpool(x).squeeze(-1)
        
        return x, pooled


# ============================================================================
# Self-Attention Mechanism
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention"""
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        attended = torch.matmul(attention_weights, V)
        
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        output = self.out_proj(attended)
        
        return output, attention_weights


# ============================================================================
# Bi-LSTM with Attention
# ============================================================================

class BiLSTMWithAttention(nn.Module):
    """Bidirectional LSTM with self-attention"""
    
    def __init__(self, input_size, hidden_1=128, hidden_2=64, num_heads=4, dropout=0.3):
        super(BiLSTMWithAttention, self).__init__()
        
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_1 * 2,
            hidden_size=hidden_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        attention_dim = hidden_2 * 2
        self.attention = MultiHeadSelfAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.output_dim = hidden_2 * 2
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout(lstm2_out)
        
        attended, attention_weights = self.attention(lstm2_out)
        
        pooled = torch.mean(attended, dim=1)
        
        return pooled, attention_weights


# ============================================================================
# Complete AFib-ResLSTM Model
# ============================================================================

class AFibResLSTM(nn.Module):
    """AFib-ResLSTM: Multi-Scale ResNet + Bi-LSTM + Self-Attention"""
    
    def __init__(self, config):
        super(AFibResLSTM, self).__init__()
        
        self.config = config
        
        # Block 1: Multi-Scale Feature Extraction
        self.multiscale = MultiScaleConv1D(
            in_channels=config.INPUT_CHANNELS,
            kernels=config.MULTISCALE_KERNELS,
            filters=config.MULTISCALE_FILTERS,
            fusion_filters=config.FUSION_FILTERS
        )
        
        # Block 2: ResNet-34 Backbone
        self.resnet = ResNetBackbone1D(
            in_channels=config.FUSION_FILTERS,
            layers=config.RESNET_LAYERS,
            filters=config.RESNET_FILTERS
        )
        
        # Block 3: Bi-LSTM with Self-Attention
        self.bilstm_attention = BiLSTMWithAttention(
            input_size=config.RESNET_FILTERS[-1],
            hidden_1=config.LSTM_HIDDEN_1,
            hidden_2=config.LSTM_HIDDEN_2,
            num_heads=config.ATTENTION_HEADS,
            dropout=config.LSTM_DROPOUT
        )
        
        # Block 4: Classification Head
        fusion_dim = config.RESNET_FILTERS[-1] + (config.LSTM_HIDDEN_2 * 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.FC_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.FC_HIDDEN, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT),
            nn.Linear(128, config.NUM_CLASSES)
        )
        
    def forward(self, x):
        # Multi-scale feature extraction
        multiscale_features = self.multiscale(x)
        
        # ResNet backbone
        resnet_seq, resnet_pooled = self.resnet(multiscale_features)
        
        # Bi-LSTM with attention
        lstm_features, attention_weights = self.bilstm_attention(resnet_seq)
        
        # Hybrid feature fusion
        fused_features = torch.cat([resnet_pooled, lstm_features], dim=1)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, attention_weights
    
    def get_attention_maps(self, x):
        """Extract attention weights for visualization"""
        with torch.no_grad():
            _, attention_weights = self.forward(x)
        return attention_weights


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.65, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)
        
        focal_weight = (1 - pt) ** self.gamma
        
        alpha_weight = targets_one_hot[:, 1] * self.alpha + targets_one_hot[:, 0] * (1 - self.alpha)
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss