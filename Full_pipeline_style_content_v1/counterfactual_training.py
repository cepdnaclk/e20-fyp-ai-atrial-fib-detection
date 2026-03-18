# ============================================================================
# Cell 1: Imports and Setup
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import AutoModel
# import wandb
import numpy as np
import os
import wandb

# PASTE YOUR KEY INSIDE THE QUOTES BELOW
os.environ["WANDB_API_KEY"] = "c9ad7c37426c5b72128b923af50ee87a44014fd1"
wandb.login()

import matplotlib.pyplot as plt
from pathlib import Path

# Check PyTorch and CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("Imports complete and seeds set")

# ============================================================================
# Cell 2: Model Configuration
# ============================================================================


class ModelConfig:
    """Configuration for AFib-ResLSTM model"""
    
    # Input parameters
    INPUT_LENGTH = 2500  # 10 seconds @ 250 Hz
    INPUT_CHANNELS = 1   # Single-lead ECG
    NUM_CLASSES = 2      # Binary: Normal vs AFib
    
    # Multi-scale convolution parameters
    MULTISCALE_FILTERS = [32, 32, 32]  # Filters per branch
    MULTISCALE_KERNELS = [3, 7, 15]    # Different scales
    FUSION_FILTERS = 64
    
    # ResNet backbone parameters
    RESNET_INITIAL_FILTERS = 64
    RESNET_LAYERS = [3, 4, 6, 3]  # ResNet-34 configuration
    RESNET_FILTERS = [64, 128, 256, 512]
    
    # Bi-LSTM parameters
    LSTM_HIDDEN_1 = 128  # First Bi-LSTM layer
    LSTM_HIDDEN_2 = 64   # Second Bi-LSTM layer
    LSTM_LAYERS = 2
    LSTM_DROPOUT = 0.3
    
    # Self-Attention parameters
    ATTENTION_HEADS = 4
    ATTENTION_DIM = 128  # Same as Bi-LSTM output
    
    # Classification head parameters
    FC_HIDDEN = 256
    DROPOUT = 0.5
    
    # Focal Loss parameters (from Petmezas 2021)
    FOCAL_ALPHA = 0.65  # Weight for minority class (AFib)
    FOCAL_GAMMA = 2.0   # Focusing parameter

# ============================================================================
# Cell 3: Multi-Scale Feature Extraction Block
# ============================================================================

class MultiScaleConv1D(nn.Module):
    """
    Multi-scale parallel convolutions to capture features at different scales.
    Innovation #1: Captures P-wave, QRS, and RR intervals simultaneously.
    """
    def __init__(self, in_channels, kernels=[3, 7, 15], filters=[32, 32, 32], fusion_filters=64):
        super(MultiScaleConv1D, self).__init__()
        
        self.branches = nn.ModuleList()
        
        # Create parallel convolution branches
        for kernel, num_filters in zip(kernels, filters):
            branch = nn.Sequential(
                nn.Conv1d(in_channels, num_filters, kernel_size=kernel, padding=kernel//2),
                nn.BatchNorm1d(num_filters),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)
        
        # Fusion layer (1x1 convolution)
        total_filters = sum(filters)
        self.fusion = nn.Sequential(
            nn.Conv1d(total_filters, fusion_filters, kernel_size=1),
            nn.BatchNorm1d(fusion_filters),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Apply all branches in parallel
        branch_outputs = [branch(x) for branch in self.branches]
        
        # Concatenate along channel dimension
        concatenated = torch.cat(branch_outputs, dim=1)
        
        # Fuse features
        fused = self.fusion(concatenated)
        
        return fused


# ============================================================================
# Cell 4: 1D ResNet Building Blocks
# ============================================================================

class ResidualBlock1D(nn.Module):
    """
    Basic Residual Block for 1D signals (ECG)
    """
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
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Downsample identity if needed (for skip connection)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        
        return out
class ResNetBackbone1D(nn.Module):
    """
    1D ResNet-34 backbone for ECG feature extraction
    Based on: Jia et al. 2020, Ben-Moshe et al. 2023
    """
    def __init__(self, in_channels=64, layers=[3, 4, 6, 3], filters=[64, 128, 256, 512]):
        super(ResNetBackbone1D, self).__init__()
        
        self.in_channels = in_channels
        
        # Initial convolution (already done by multi-scale, so this is identity)
        # We'll start from the multi-scale output (64 channels)
        
        # ResNet stages
        self.layer1 = self._make_layer(filters[0], layers[0], stride=1)
        self.layer2 = self._make_layer(filters[1], layers[1], stride=2)
        self.layer3 = self._make_layer(filters[2], layers[2], stride=2)
        self.layer4 = self._make_layer(filters[3], layers[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        
        # If dimensions change, create downsample path
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        layers = []
        # First block (may downsample)
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Pass through ResNet stages
        x = self.layer1(x)  # Output: (batch, 64, length)
        x = self.layer2(x)  # Output: (batch, 128, length/2)
        x = self.layer3(x)  # Output: (batch, 256, length/4)
        x = self.layer4(x)  # Output: (batch, 512, length/8)
        
        # Global pooling
        pooled = self.avgpool(x).squeeze(-1)  # (batch, 512)
        
        return x, pooled  # Return both sequence and pooled features

# ============================================================================
# Cell 5: Multi-Head Self-Attention Mechanism
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for temporal feature weighting.
    Innovation #2: Identifies critical time segments + provides interpretability.
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, embed_dim)
        Returns:
            attended: (batch, seq_len, embed_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear projections and reshape for multi-head
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch, heads, seq_len, head_dim) x (batch, heads, head_dim, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape back
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output projection
        output = self.out_proj(attended)
        
        return output, attention_weights

 
# ============================================================================
# Cell 6: Bidirectional LSTM with Self-Attention
# ============================================================================

class BiLSTMWithAttention(nn.Module):
    """
    Bidirectional LSTM with multi-head self-attention for temporal modeling.
    Combines ideas from Andersen 2019, Jia 2020, with attention innovation.
    """
    def __init__(self, input_size, hidden_1=128, hidden_2=64, num_heads=4, dropout=0.3):
        super(BiLSTMWithAttention, self).__init__()
        
        # First Bi-LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # No dropout for single layer
        )
        
        # Second Bi-LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_1 * 2,  # *2 for bidirectional
            hidden_size=hidden_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Self-attention (applied to LSTM output)
        attention_dim = hidden_2 * 2  # *2 for bidirectional
        self.attention = MultiHeadSelfAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Final dimension
        self.output_dim = hidden_2 * 2
        
    def forward(self, x):
        """
        Args:
            x: (batch, channels, length) - from ResNet
        Returns:
            attended_output: (batch, hidden_2 * 2)
            attention_weights: for visualization
        """
        # Reshape for LSTM: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, length, channels)
        
        # First Bi-LSTM
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)
        
        # Second Bi-LSTM
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout(lstm2_out)
        
        # Apply self-attention
        attended, attention_weights = self.attention(lstm2_out)
        
        # Global average pooling over time
        pooled = torch.mean(attended, dim=1)  # (batch, hidden_2 * 2)
        
        return pooled, attention_weights


 
# ============================================================================
# Cell 7: Complete AFib-ResLSTM Architecture
# ============================================================================

class AFibResLSTM(nn.Module):
    """
    AFib-ResLSTM: Multi-Scale ResNet + Bi-LSTM + Self-Attention
    
    Innovations:
    1. Multi-scale parallel convolutions (captures P-wave, QRS, RR intervals)
    2. Self-attention on Bi-LSTM output (interpretable + performance boost)
    3. Hybrid feature fusion (ResNet morphology + LSTM-Attention temporal)
    
    Architecture inspired by:
    - Jia et al. 2020 (ResNet-LSTM hybrid)
    - Petmezas et al. 2021 (focal loss for imbalance)
    - Ben-Moshe et al. 2023 (deep residual + BiGRU)
    + Your novel contributions
    """
    def __init__(self, config):
        super(AFibResLSTM, self).__init__()
        
        self.config = config
        
        # BLOCK 1: Multi-Scale Feature Extraction
        self.multiscale = MultiScaleConv1D(
            in_channels=config.INPUT_CHANNELS,
            kernels=config.MULTISCALE_KERNELS,
            filters=config.MULTISCALE_FILTERS,
            fusion_filters=config.FUSION_FILTERS
        )
        
        # BLOCK 2: ResNet-34 Backbone
        self.resnet = ResNetBackbone1D(
            in_channels=config.FUSION_FILTERS,
            layers=config.RESNET_LAYERS,
            filters=config.RESNET_FILTERS
        )
        
        # BLOCK 3: Bi-LSTM with Self-Attention
        self.bilstm_attention = BiLSTMWithAttention(
            input_size=config.RESNET_FILTERS[-1],
            hidden_1=config.LSTM_HIDDEN_1,
            hidden_2=config.LSTM_HIDDEN_2,
            num_heads=config.ATTENTION_HEADS,
            dropout=config.LSTM_DROPOUT
        )
        
        # BLOCK 4: Adaptive Decision Head (Hybrid Feature Fusion)
        # Concatenate ResNet pooled features + LSTM-Attention features
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
        """
        Args:
            x: (batch, 1, 2500) - Raw ECG signal
        Returns:
            logits: (batch, 2) - Class logits
            attention_weights: (batch, heads, seq_len, seq_len) - For visualization
        """
        # Multi-scale feature extraction
        multiscale_features = self.multiscale(x)  # (batch, 64, 2500)
        
        # ResNet backbone
        resnet_seq, resnet_pooled = self.resnet(multiscale_features)
        # resnet_seq: (batch, 512, ~312)
        # resnet_pooled: (batch, 512)
        
        # Bi-LSTM with attention
        lstm_features, attention_weights = self.bilstm_attention(resnet_seq)
        # lstm_features: (batch, 128)
        
        # Hybrid feature fusion
        fused_features = torch.cat([resnet_pooled, lstm_features], dim=1)
        # fused_features: (batch, 512 + 128 = 640)
        
        # Classification
        logits = self.classifier(fused_features)  # (batch, 2)
        
        return logits, attention_weights
    
    def get_attention_maps(self, x):
        """
        Extract attention weights for visualization (XAI Phase 2)
        """
        with torch.no_grad():
            _, attention_weights = self.forward(x)
        return attention_weights


 
# ============================================================================
# Cell 8: Focal Loss for Class Imbalance (Petmezas et al. 2021)
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in AFib detection.
    
    Reference: Petmezas et al. 2021 - achieved Sp=99.29% with focal loss
    
    Formula: FL(pt) = -α(1-pt)^γ * log(pt)
    
    Args:
        alpha: Weight for positive class (AFib). Set to ~0.65 for your 35.6% AFib data
        gamma: Focusing parameter. Higher gamma = more focus on hard examples
    """
    def __init__(self, alpha=0.65, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch, num_classes) - Raw model outputs
            targets: (batch,) - Class indices (0 or 1)
        """
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get probability of true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        pt = (probs * targets_one_hot).sum(dim=1)  # Probability of true class
        
        # Focal loss formula
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_weight = targets_one_hot[:, 1] * self.alpha + targets_one_hot[:, 0] * (1 - self.alpha)
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Final focal loss
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# STYLE ENCODER
class StyleEncoderWrapper(nn.Module):
    """
    Properly extracts 128D style features from trained AFibResLSTM
    """
    def __init__(self, weights_path, device='cuda'):
        super().__init__()
        
        # Load your trained classifier
        config = ModelConfig()
        self.model = AFibResLSTM(config)
        
        # Load weights on CPU first (safer)
        checkpoint = torch.load(weights_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        # Set to eval mode BEFORE moving to device
        self.model.eval()
        
        # Now move to device
        self.model = self.model.to(device)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False #Frozen weights
    
    def forward(self, x):
        """
        Extract 128D style features from LSTM output
        
        Args:
            x: [batch, 1, 2500] - ECG signal
        Returns:
            features: [batch, 128] - Style features
        """
        # Ensure eval mode (sometimes switches)
        self.model.eval()
        
        with torch.no_grad():
            # Extract intermediate features step by step
            multiscale_features = self.model.multiscale(x)
            resnet_seq, resnet_pooled = self.model.resnet(multiscale_features)
            lstm_features, _ = self.model.bilstm_attention(resnet_seq)
            
            # Return ONLY the LSTM features (128D)
            # This captures the temporal/rhythm style
            return lstm_features

# CONTENT ENCODER
class PretrainedContentEncoder(nn.Module):
    def __init__(self, model_name="Edoardo-BS/hubert-ecg-base", out_dim=512):
        super().__init__()
        print(f"Loading Pretrained ECG Content Encoder: {model_name}")
        self.net = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        # Freeze weights so we don't break the medical knowledge
        for param in self.net.parameters():
            param.requires_grad = False 

        # Projector: The model outputs 768 features, we need 512 for U-Net
        self.proj = nn.Linear(768, out_dim)

    def forward(self, x):
        # x comes in as [Batch, 1, 2500] or [Batch, 1, 1, 2500]
        
        # 1. FLATTEN to [Batch, 2500]
        # This removes ALL channel dimensions (1s) and leaves just the raw signal.
        # It's like unboxing the data so the model can read it directly.
        x = x.reshape(x.shape[0], -1) 
        
        # 2. Pass directly to model
        # The model expects [Batch, Sequence_Length], which is exactly [16, 2500]
        outputs = self.net(x)

        # 3. Extract features
        features = outputs.last_hidden_state
        return self.proj(features)

 
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaGN(nn.Module):
    """
    Adaptive Group Normalization: The "Smart" conditioning.
    Instead of just adding the condition, we use it to scale and shift the normalized features.
    """
    def __init__(self, num_channels, cond_dim, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False) # affine=False because we learn it manually
        self.proj = nn.Linear(cond_dim, num_channels * 2) # Predicts Scale and Shift
        self.act = nn.SiLU()

    def forward(self, x, cond):
        # Predict scale and shift from conditioning vector
        # cond shape: (Batch, cond_dim)
        params = self.proj(cond)
        scale, shift = params.chunk(2, dim=1)
        
        # Reshape for 1D convolution: (Batch, Channels, 1)
        scale = scale.unsqueeze(-1)
        shift = shift.unsqueeze(-1)
        
        # Apply Norm -> Scale -> Shift
        h = self.norm(x)
        h = h * (1 + scale) + shift
        return h

class UNet1DConditional(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, 
                 time_embed_dim=128, cond_embed_dim=512):
        super().__init__()
        
        # 1. Time & Condition Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.cond_proj = nn.Linear(cond_embed_dim, time_embed_dim)
        
        # 2. Encoder
        self.enc1 = self._make_block(in_channels, 64, time_embed_dim)
        self.enc2 = self._make_block(64, 128, time_embed_dim)
        self.enc3 = self._make_block(128, 256, time_embed_dim)
        self.enc4 = self._make_block(256, 512, time_embed_dim)
        
        # 3. Bottleneck
        self.bottleneck = self._make_block(512, 512, time_embed_dim)
        
        # 4. Decoder
        self.dec4 = self._make_block(512 + 512, 256, time_embed_dim)
        self.dec3 = self._make_block(256 + 256, 128, time_embed_dim)
        self.dec2 = self._make_block(128 + 128, 64, time_embed_dim)
        self.dec1 = self._make_block(64 + 64, 64, time_embed_dim)
        
        self.out = nn.Conv1d(64, out_channels, kernel_size=1)
        self.pool = nn.MaxPool1d(2)

    def _make_block(self, in_ch, out_ch, cond_dim):
        """Creates a block with AdaGN"""
        return nn.ModuleDict({
            'conv1': nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            'norm1': AdaGN(out_ch, cond_dim), # <--- Using AdaGN
            'conv2': nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            'norm2': AdaGN(out_ch, cond_dim), # <--- Using AdaGN
            'act': nn.SiLU(),
            # Residual connection if channels match
            'skip': nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1)
        })

    def _apply_block(self, x, block, cond):
        """Helper to run a block"""
        h = block['conv1'](x)
        h = block['norm1'](h, cond) # Pass condition to Norm
        h = block['act'](h)
        
        h = block['conv2'](h)
        h = block['norm2'](h, cond) # Pass condition to Norm
        h = block['act'](h)
        
        return h + block['skip'](x) # Residual connection

    def _upsample_and_concat(self, x, skip):
        x_up = F.interpolate(x, scale_factor=2, mode='linear', align_corners=True)
        if x_up.shape[2] != skip.shape[2]:
            x_up = F.interpolate(x_up, size=skip.shape[2], mode='linear', align_corners=True)
        return torch.cat([x_up, skip], dim=1)

    def forward(self, x, timestep, conditioning):
        # Embed Time
        t = timestep.float().unsqueeze(-1) / 1000.0
        time_emb = self.time_mlp(t)
        
        # Embed Condition
        cond_emb = conditioning.mean(dim=1) 
        cond_emb = self.cond_proj(cond_emb)
        
        # Combine (AdaGN will use this combined vector)
        emb = time_emb + cond_emb 
        
        # Pass 'emb' to every block
        e1 = self._apply_block(x, self.enc1, emb)
        e2 = self._apply_block(self.pool(e1), self.enc2, emb)
        e3 = self._apply_block(self.pool(e2), self.enc3, emb)
        e4 = self._apply_block(self.pool(e3), self.enc4, emb)
        
        b = self._apply_block(self.pool(e4), self.bottleneck, emb)
        
        d4 = self._apply_block(self._upsample_and_concat(b, e4), self.dec4, emb)
        d3 = self._apply_block(self._upsample_and_concat(d4, e3), self.dec3, emb)
        d2 = self._apply_block(self._upsample_and_concat(d3, e2), self.dec2, emb)
        d1 = self._apply_block(self._upsample_and_concat(d2, e1), self.dec1, emb)
        
        return self.out(d1)
    
class EMA:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

##### Adding frequency loss and smoothness loss in training #####
def compute_frequency_loss(pred_noise, true_noise):
    """
    Forces the model to learn the correct frequency components (heartbeat structure)
    instead of just random static.
    """
    # Convert to Float32 for FFT stability
    pred_noise_fp32 = pred_noise.float()
    true_noise_fp32 = true_noise.float()
    
    # FFT (Fast Fourier Transform)
    pred_fft = torch.fft.rfft(pred_noise_fp32, dim=-1)
    true_fft = torch.fft.rfft(true_noise_fp32, dim=-1)
    
    # Compare Magnitude spectra
    pred_mag = torch.abs(pred_fft)
    true_mag = torch.abs(true_fft)
    
    # L1 loss in frequency domain
    return F.l1_loss(pred_mag, true_mag)

def compute_temporal_smoothness_loss(pred_x0):
    """
    Penalizes jagged/spiky lines to ensure smooth ECG signals.
    """
    # 1. Calculate first derivative (velocity)
    diff1 = pred_x0[:, :, 1:] - pred_x0[:, :, :-1]
    
    # 2. Calculate second derivative (acceleration)
    diff2 = diff1[:, :, 1:] - diff1[:, :, :-1]
    
    # 3. Penalize high acceleration (smoothness)
    smoothness_loss = torch.mean(diff2 ** 2)
    
    # Clamp to prevent explosion (Friend's specific trick)
    return torch.clamp(smoothness_loss, max=10.0)


# ============================================================================
# COUNTERFACTUAL-FRIENDLY ECG DIFFUSION TRAINING
# ============================================================================
"""
Key Design Decisions for Counterfactual Generation:

✅ KEEP:
   - Frequency loss (preserves ECG frequency content)
   - Early stopping (prevents overfitting)
   - Diagnostic tools (PSD, NeuroKit2, correlation)

❌ REMOVE:
   - Phase shift loss (breaks counterfactual flexibility)
   - Smoothness loss (may harm sharp ECG features)
   - Hard temporal alignment

🎯 GOAL: Model learns ECG STRUCTURE (morphology, rhythm patterns)
         NOT exact timing/phase of the input
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import DDIMScheduler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy import signal

# ============================================================================
# EARLY STOPPING (KEEP THIS)
# ============================================================================

class EarlyStopping:
    """Early stopping based on validation metric"""
    def __init__(self, patience=7, min_delta=0.005, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠️  Early stopping! No improvement for {self.patience} epochs.")
                print(f"   Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
        
        return self.early_stop

# ============================================================================
# DIAGNOSTIC FUNCTIONS (KEEP - BUT RELAXED INTERPRETATION)
# ============================================================================

def diagnose_sample(original, generated, epoch, save_dir, prefix=""):
    """
    Diagnostic analysis - BUT interpreted for counterfactual context
    
    Key difference: Low raw Pearson is OK (we want different rhythms!)
                   We only care about ECG-likeness, not exact match
    """
    print(f"\n{'='*70}")
    print(f"{prefix}DIAGNOSTIC ANALYSIS - EPOCH {epoch}")
    print(f"{'='*70}")
    
    # Basic statistics
    print(f"\n📊 Signal Statistics:")
    print(f"  Original  → Mean: {original.mean():7.2f}, Std: {original.std():7.2f}, Range: [{original.min():7.2f}, {original.max():7.2f}]")
    print(f"  Generated → Mean: {generated.mean():7.2f}, Std: {generated.std():7.2f}, Range: [{generated.min():7.2f}, {generated.max():7.2f}]")
    
    fs = 250
    
    # ============================================
    # CHECK 1: PSD Analysis (CRITICAL)
    # ============================================
    f_orig, psd_orig = signal.welch(original, fs=fs, nperseg=512)
    f_gen, psd_gen = signal.welch(generated, fs=fs, nperseg=512)
    
    idx_50hz = np.argmin(np.abs(f_orig - 50))
    power_above_50 = np.sum(psd_gen[idx_50hz:])
    total_power = np.sum(psd_gen)
    noise_ratio = power_above_50 / (total_power + 1e-10)
    
    print(f"\n🔍 CHECK 1: Power Spectral Density")
    print(f"  Noise ratio (>50Hz): {noise_ratio*100:5.2f}%  ", end="")
    if noise_ratio < 0.15:
        print("✅ ECG-like!")
    elif noise_ratio < 0.30:
        print("⚠️  Moderate noise")
    else:
        print("❌ High noise")
    
    # ============================================
    # CHECK 2: NeuroKit2 (CRITICAL)
    # ============================================
    try:
        import neurokit2 as nk
        
        print(f"\n🔍 CHECK 2: NeuroKit2 R-peak Detection")
        
        try:
            _, info_orig = nk.ecg_process(original, sampling_rate=fs)
            r_peaks_orig = info_orig['ECG_R_Peaks']
            hr_orig = len(r_peaks_orig) / 10 * 60
            print(f"  Original  → {len(r_peaks_orig):2d} peaks ({hr_orig:5.1f} BPM)")
        except:
            print(f"  Original  → Failed to detect")
            r_peaks_orig = []
        
        try:
            _, info_gen = nk.ecg_process(generated, sampling_rate=fs)
            r_peaks_gen = info_gen['ECG_R_Peaks']
            hr_gen = len(r_peaks_gen) / 10 * 60
            print(f"  Generated → {len(r_peaks_gen):2d} peaks ({hr_gen:5.1f} BPM)  ✅ Structure detected!")
        except Exception as e:
            print(f"  Generated → ❌ Failed: {str(e)[:40]}")
    
    except ImportError:
        print(f"\n⚠️  NeuroKit2 not installed")
    
    # ============================================
    # CHECK 3: Correlation (INFO ONLY - NOT A TARGET)
    # ============================================
    print(f"\n🔍 CHECK 3: Correlation Analysis (For Info Only)")
    
    pearson_raw = np.corrcoef(original, generated)[0, 1]
    print(f"  Raw Pearson: {pearson_raw:6.4f}")
    print(f"  ⚠️  Note: Low correlation is OK for counterfactuals!")
    print(f"     We want DIFFERENT rhythms, not exact copies.")
    
    # ============================================
    # VISUALIZATION
    # ============================================
    create_diagnostic_plot_counterfactual(
        original, generated, epoch, save_dir,
        psd_orig=(f_orig, psd_orig),
        psd_gen=(f_gen, psd_gen),
        noise_ratio=noise_ratio,
        pearson_raw=pearson_raw
    )
    
    print(f"  📊 Plot saved: {save_dir}/plots/epoch_{epoch:03d}_diagnostics.png")
    print(f"{'='*70}\n")
    
    return {
        'noise_ratio': noise_ratio,
        'pearson_raw': pearson_raw,
        'ecg_quality_score': 1.0 - noise_ratio  # Higher is better
    }

def create_diagnostic_plot_counterfactual(original, generated, epoch, save_dir,
                                         psd_orig, psd_gen, noise_ratio, pearson_raw):
    """Visualization focused on ECG quality, not exact matching"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # Plot 1: Full signal
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(original, label='Original', color='black', alpha=0.7, linewidth=1)
    ax1.plot(generated, label='Generated', color='#ff7f0e', alpha=0.8, linewidth=1)
    ax1.set_title(f'Epoch {epoch} - Full Signal (Pearson: {pearson_raw:.4f} - Low OK for counterfactuals)', 
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Amplitude')
    
    # Plot 2: Zoom original
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(original[:500], color='black', alpha=0.7, linewidth=1.5)
    ax2.set_title('Original - First 2s', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel('Amplitude')
    
    # Plot 3: Zoom generated
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(generated[:500], color='#ff7f0e', alpha=0.8, linewidth=1.5)
    ax3.set_title('Generated - First 2s (Check for ECG structure)', fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel('Amplitude')
    
    # Plot 4: PSD (MOST IMPORTANT)
    ax4 = fig.add_subplot(gs[2, :])
    f_orig, psd_o = psd_orig
    f_gen, psd_g = psd_gen
    ax4.semilogy(f_orig, psd_o, label='Original PSD', color='black', alpha=0.7, linewidth=2)
    ax4.semilogy(f_gen, psd_g, label='Generated PSD', color='#ff7f0e', alpha=0.8, linewidth=2)
    ax4.axvline(50, color='red', linestyle='--', linewidth=2, label='50 Hz cutoff')
    ax4.set_xlim([0, 125])
    ax4.set_title(f'PSD - Noise ratio: {noise_ratio*100:.1f}% (MOST IMPORTANT METRIC)', 
                 fontsize=11, fontweight='bold')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Histogram
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.hist(original, bins=50, alpha=0.5, label='Original', color='black', density=True)
    ax5.hist(generated, bins=50, alpha=0.5, label='Generated', color='orange', density=True)
    ax5.set_title('Amplitude Distribution', fontsize=11)
    ax5.set_xlabel('Amplitude')
    ax5.set_ylabel('Density')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary
    ax6 = fig.add_subplot(gs[3, 1])
    ax6.axis('off')
    
    summary_text = f"""
    COUNTERFACTUAL GENERATION METRICS
    ═══════════════════════════════════
    
    PRIMARY GOALS:
    
    1. Noise Ratio:      {noise_ratio*100:5.2f}%
       {'  ✅ ECG-like' if noise_ratio < 0.2 else '  ❌ High noise'}
    
    2. ECG Quality:      {(1-noise_ratio)*100:5.2f}%
       Target: > 80%
    
    SECONDARY (Info Only):
    
    3. Raw Pearson:      {pearson_raw:6.4f}
       (Low is OK - we want different rhythms!)
    
    STATUS:
    {'  ✅ Model generating ECG' if noise_ratio < 0.2 else '  ❌ Still learning'}
    """
    
    ax6.text(0.05, 0.5, summary_text, fontsize=9, family='monospace',
            verticalalignment='center', 
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.savefig(f"{save_dir}/plots/epoch_{epoch:03d}_diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()

# ============================================================================
# TRAINING FUNCTION (SIMPLIFIED - NO SMOOTHNESS, NO PHASE LOSS)
# ============================================================================

def train_counterfactual_model(
    dataset_path="./ecg_afib_data/X_combined.npy",
    weights_path="./best_model/best_model.pth",
    save_dir="./counterfactual_training",
    epochs=50,
    batch_size=16,
    val_split=0.2,
    use_ema=True,
    use_cfg=True,
    guidance_scale=1.5,
    learning_rate=5e-5,
    early_stopping_patience=10,
    freq_loss_weight=0.05  # Keep frequency loss - it helps quality
):
    """
    Training for counterfactual generation
    
    Loss: MSE + Frequency (NO smoothness, NO phase alignment)
    Metric: ECG quality (PSD-based), NOT correlation
    """
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║     COUNTERFACTUAL ECG DIFFUSION TRAINING                       ║
╚══════════════════════════════════════════════════════════════════╝

🎯 GOAL: Generate different rhythms (AFib ↔ Normal)
         NOT exact reconstruction!

Configuration:
  Device:          {DEVICE}
  Epochs:          {epochs}
  Batch Size:      {batch_size}
  Learning Rate:   {learning_rate}
  
  EMA:             {'✅' if use_ema else '❌'}
  CFG:             {'✅' if use_cfg else '❌'}
  Guidance:        {guidance_scale}
  
  Loss Components:
    ✅ MSE (noise prediction)
    ✅ Frequency ({freq_loss_weight}x)
    ❌ Smoothness (removed - preserves sharp features)
    ❌ Phase loss (removed - allows rhythm changes)
  
  Early Stopping:  ✅ Patience={early_stopping_patience}
  
""")
    
    # ========================================
    # DATA LOADING
    # ========================================
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    signals_raw = np.load(dataset_path)
    signals_normalized = (signals_raw - signals_raw.mean(1, keepdims=True)) / \
                        (signals_raw.std(1, keepdims=True) + 1e-6)
    
    print(f"Dataset size: {len(signals_normalized):,}")
    
    # Save normalization params
    norm_params = {
        'means': signals_raw.mean(1),
        'stds': signals_raw.std(1)
    }
    np.save(f"{save_dir}/norm_params.npy", norm_params)
    
    # Train/Val split
    n_samples = len(signals_normalized)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    
    train_data = signals_normalized[train_indices]
    val_data = signals_normalized[val_indices]
    
    class ECGDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
    
    train_dataset = ECGDataset(train_data)
    val_dataset = ECGDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_data):,}, Val: {len(val_data):,}")
    
    # ========================================
    # MODELS
    # ========================================
    print("\n" + "="*70)
    print("INITIALIZING MODELS")
    print("="*70)
    
    unet = UNet1DConditional().to(DEVICE)
    style_net = StyleEncoderWrapper(weights_path, DEVICE)
    content_net = PretrainedContentEncoder().to(DEVICE)
    content_net.eval()
    
    style_proj = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512)
    ).to(DEVICE)
    
    for layer in style_proj:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
    
    if use_ema:
        model_container = nn.ModuleList([unet, style_proj])
        ema = EMA(model_container, decay=0.9999)
        ema.register()
    else:
        ema = None
    
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(style_proj.parameters()),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    diffusion_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=False
    )
    
    # Validation sample
    val_sample_idx = val_indices[0]
    val_sample_raw = signals_raw[val_sample_idx]
    val_sample_norm = signals_normalized[val_sample_idx]
    val_sample_tensor = torch.tensor(val_sample_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    
    # Early stopping based on ECG quality (NOT correlation!)
    early_stopper = EarlyStopping(patience=early_stopping_patience, min_delta=0.01, mode='max')
    
    print("✅ Models initialized")
    
    # ========================================
    # TRAINING LOOP
    # ========================================
    print("\n" + "="*70)
    print(f"TRAINING ({epochs} EPOCHS)")
    print("="*70)
    
    best_quality = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        # ====================================
        # TRAIN
        # ====================================
        unet.train()
        style_proj.train()
        train_losses = {'total': [], 'mse': [], 'freq': []}
        
        # Show which loss mode we're in ***
        if epoch < 10:
            loss_mode = "MSE ONLY"
        else:
            loss_mode = f"MSE + {freq_loss_weight}×Freq"
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{epochs} [{loss_mode}]", leave=False)
        for batch in pbar:
            x = batch.to(DEVICE)
            
            with torch.no_grad():
                style = style_net(x)
                content = content_net(x)
            
            style_emb = style_proj(style).unsqueeze(1)
            conditioning = torch.cat([content, style_emb], dim=1)
            
            if use_cfg and np.random.random() < 0.1:
                conditioning = torch.zeros_like(conditioning)
            
            noise = torch.randn_like(x)
            t = torch.randint(0, 1000, (x.shape[0],), device=DEVICE).long()
            noisy_x = diffusion_scheduler.add_noise(x, noise, t)
            
            pred_noise = unet(noisy_x, t, conditioning)
            
            # *** FIX 4: MSE ONLY for first 10 epochs ***
            mse_loss = F.mse_loss(pred_noise, noise)
            
            if epoch < 10:
                # First 10 epochs: MSE only to stabilize
                loss = mse_loss
                freq_loss_val = torch.tensor(0.0)
            else:
                # After epoch 10: Add frequency loss with REDUCED weight
                freq_loss_val = compute_frequency_loss(pred_noise, noise)
                loss = mse_loss + freq_loss_weight * freq_loss_val
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
            optimizer.step()
            
            if use_ema:
                ema.update()
            
            train_losses['total'].append(loss.item())
            train_losses['mse'].append(mse_loss.item())
            train_losses['freq'].append(freq_loss_val.item())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_losses = {k: np.mean(v) for k, v in train_losses.items()}
        
        # ====================================
        # VALIDATION
        # ====================================
        if use_ema:
            ema.apply_shadow()
        
        unet.eval()
        style_proj.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(DEVICE)
                style = style_net(x)
                content = content_net(x)
                style_emb = style_proj(style).unsqueeze(1)
                conditioning = torch.cat([content, style_emb], dim=1)
                
                noise = torch.randn_like(x)
                t = torch.randint(0, 1000, (x.shape[0],), device=DEVICE).long()
                noisy_x = diffusion_scheduler.add_noise(x, noise, t)
                
                pred_noise = unet(noisy_x, t, conditioning)
                loss = F.mse_loss(pred_noise, noise)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        
        # ====================================
        # SAMPLE & DIAGNOSE
        # ====================================
        with torch.no_grad():
            style = style_net(val_sample_tensor)
            content = content_net(val_sample_tensor)
            style_emb = style_proj(style).unsqueeze(1)
            
            if use_cfg:
                cond_input = torch.cat([content, style_emb], dim=1)
                uncond_input = torch.zeros_like(cond_input)
            else:
                cond_input = torch.cat([content, style_emb], dim=1)
            
            test_scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="squaredcos_cap_v2",
                prediction_type="epsilon",
                clip_sample=False
            )
            test_scheduler.set_timesteps(50)
            
            latents = torch.randn_like(val_sample_tensor)
            
            for t in test_scheduler.timesteps:
                if use_cfg:
                    latent_input = torch.cat([latents] * 2)
                    t_input = torch.cat([t.unsqueeze(0).to(DEVICE)] * 2)
                    cond = torch.cat([cond_input, uncond_input])
                    
                    noise_pred = unet(latent_input, t_input, cond)
                    noise_cond, noise_uncond = noise_pred.chunk(2)
                    
                    # *** FIX 5: Use the REDUCED guidance_scale (now 1.5) ***
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                else:
                    noise_pred = unet(latents, t.unsqueeze(0).to(DEVICE), cond_input)
                
                latents = test_scheduler.step(noise_pred, t, latents).prev_sample
                
                # *** FIX 6: CRITICAL - Add clipping after each step ***
                latents = torch.clamp(latents, min=-4.0, max=4.0)
            
            generated_norm = latents.squeeze().cpu().numpy()
        
        original_denorm = val_sample_raw
        generated_denorm = generated_norm * norm_params['stds'][val_sample_idx] + norm_params['means'][val_sample_idx]
        
        metrics = diagnose_sample(original_denorm, generated_denorm, epoch, save_dir)
        
        # ====================================
        # LOGGING
        # ====================================
        lr_scheduler.step()
        
        ecg_quality = metrics['ecg_quality_score']
        
        # Show which loss mode
        if epoch < 10:
            loss_info = f"MSE:{avg_losses['mse']:.4f}"
        else:
            loss_info = f"MSE:{avg_losses['mse']:.4f}, Freq:{avg_losses['freq']:.4f}"
        
        print(f"\n📊 Epoch {epoch:2d}: "
              f"Train={avg_losses['total']:.4f} ({loss_info}) | "
              f"Val={avg_val_loss:.4f} | "
              f"ECG Quality={ecg_quality*100:.1f}% (noise ratio: {metrics['noise_ratio']*100:.1f}%)")
        
        # Save based on ECG quality
        if ecg_quality > best_quality:
            best_quality = ecg_quality
            best_epoch = epoch
            
            torch.save({
                'epoch': epoch,
                'unet_state_dict': unet.state_dict(),
                'style_proj_state_dict': style_proj.state_dict(),
                'metrics': metrics,
                'val_loss': avg_val_loss
            }, f"{save_dir}/best_model.pth")
            
            print(f"   ✅ New best! ECG Quality: {best_quality*100:.1f}%")
        
        if use_ema:
            ema.restore()
        
        # Early stopping
        if early_stopper(ecg_quality, epoch):
            break
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best ECG Quality: {best_quality*100:.1f}%")

# ============================================================================
# RUN WITH FIXED PARAMETERS
# ============================================================================

if __name__ == "__main__":
    train_counterfactual_model(
        dataset_path="./ecg_afib_data/X_combined.npy",
        weights_path="./best_model/best_model.pth",
        save_dir="./counterfactual_training_fixed",  # Different folder
        epochs=50,
        batch_size=16,
        use_ema=True,
        use_cfg=True,
        guidance_scale=1.5,       # ✅ FIXED: Was 3.0
        learning_rate=5e-5,
        early_stopping_patience=10,
        freq_loss_weight=0.05     # ✅ FIXED: Was 0.3
    )