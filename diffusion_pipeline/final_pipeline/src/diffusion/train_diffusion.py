"""
Enhanced Diffusion-Based Counterfactual ECG Generation
=======================================================

Key Improvements Over Previous Approaches:
1. Better noise scheduling (cosine + SDEdit partial denoising)
2. Content-style disentanglement with adversarial training
3. Classifier-free guidance for controllable generation
4. Post-processing to reduce high-frequency noise

Architecture:
- ContentEncoder: Captures class-invariant features (morphology, amplitude)
- StyleEncoder: Captures class-discriminative features (rhythm, P-waves)
- ConditionalUNet: Generates ECG conditioned on content + style + class
- DDIMScheduler: Efficient sampling with guidance

Training Strategy:
- Stage 1 (epochs 1-50): Train for high-quality reconstruction
- Stage 2 (epochs 51-100): Fine-tune for counterfactual generation


"""

import os
import sys
import time
import subprocess
import warnings
warnings.filterwarnings('ignore')

# GPU Selection
def get_best_gpu():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                idx, free = line.split(', ')
                gpus.append((int(idx), int(free)))
        gpus.sort(key=lambda x: x[1], reverse=True)
        return gpus[0][0] if gpus else 0
    except:
        return 0

selected_gpu = get_best_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)
print(f"Using GPU: {selected_gpu}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
import math
import sys
from datetime import datetime
from scipy.signal import savgol_filter

# Import shared models
from shared_models import load_classifier, ClassifierWrapper

# Command line argument for resuming
RESUME_FROM_STAGE1 = '--resume-stage1' in sys.argv

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
    DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
    MODEL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/enhanced_diffusion_cf'
    RESULTS_DIR = MODEL_DIR / 'results'
    
    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    
    # Data
    SIGNAL_LENGTH = 2500
    IN_CHANNELS = 1
    NUM_CLASSES = 2  # 0: Normal, 1: AFib
    
    # Encoders
    CONTENT_DIM = 256  # Class-invariant representation
    STYLE_DIM = 128    # Class-discriminative representation
    ENCODER_CHANNELS = 64
    
    # Diffusion
    DIFFUSION_TIMESTEPS = 1000
    BETA_SCHEDULE = 'cosine'
    SDEDIT_STRENGTH = 0.6  # Partial denoising: use 60% of timesteps
    
    # UNet
    MODEL_CHANNELS = 64
    CHANNEL_MULT = (1, 2, 4, 8)
    NUM_RES_BLOCKS = 2
    ATTENTION_RESOLUTIONS = (2, 3)
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    STAGE1_EPOCHS = 50  # Reconstruction
    STAGE2_EPOCHS = 50  # Counterfactual fine-tuning
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Loss weights - Stage 1
    RECON_WEIGHT = 1.0
    STYLE_CLASS_WEIGHT = 0.5
    CONTENT_INVARIANCE_WEIGHT = 0.1
    KL_WEIGHT = 0.01
    
    # Loss weights - Stage 2
    FLIP_WEIGHT = 1.0
    SIMILARITY_WEIGHT = 0.3
    
    # Classifier-free guidance
    CFG_SCALE = 3.0  # Guidance strength
    DROPOUT_PROB = 0.1  # Probability of unconditional training
    
    # Evaluation
    EVAL_INTERVAL = 10
    NUM_EVAL_SAMPLES = 50
    
    # Noise reduction
    SAVGOL_WINDOW = 11  # Savitzky-Golay filter window
    SAVGOL_POLY = 3     # Polynomial order

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
print(f"Device: {Config.DEVICE}")

# ============================================================================
# Load Data
# ============================================================================

print("\n" + "="*60)
print("Loading Data")
print("="*60)

train_data = np.load(Config.DATA_DIR / 'train_data.npz')
train_signals = torch.tensor(train_data['X'], dtype=torch.float32)
train_labels = torch.tensor(train_data['y'], dtype=torch.long)

val_data = np.load(Config.DATA_DIR / 'val_data.npz')
val_signals = torch.tensor(val_data['X'], dtype=torch.float32)
val_labels = torch.tensor(val_data['y'], dtype=torch.long)

if train_signals.dim() == 2:
    train_signals = train_signals.unsqueeze(1)
if val_signals.dim() == 2:
    val_signals = val_signals.unsqueeze(1)

# Separate by class for style transfer
normal_mask = train_labels == 0
afib_mask = train_labels == 1
normal_signals = train_signals[normal_mask]
afib_signals = train_signals[afib_mask]

print(f"Total training: {len(train_signals)}")
print(f"Normal ECGs: {len(normal_signals)}")
print(f"AFib ECGs: {len(afib_signals)}")
print(f"Validation: {len(val_signals)}")

# DataLoader
train_dataset = TensorDataset(train_signals, train_labels)
train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, 
                          drop_last=True, num_workers=4, pin_memory=True)

# ============================================================================
# Content Encoder - Class-Invariant Features
# ============================================================================

class ContentEncoder(nn.Module):
    """
    Extracts class-invariant content from ECG.
    Should capture: heart rate, basic rhythm pattern, signal morphology
    Should NOT capture: P-wave presence/absence, RR regularity (class-specific)
    """
    def __init__(self, in_channels=1, hidden_dim=64, content_dim=256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Input: (B, 1, 2500)
            nn.Conv1d(in_channels, hidden_dim, 7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim * 4, hidden_dim * 8, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim * 8, hidden_dim * 8, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool1d(8),
        )
        
        self.flat_size = hidden_dim * 8 * 8
        self.fc_mu = nn.Linear(self.flat_size, content_dim)
        self.fc_logvar = nn.Linear(self.flat_size, content_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
            
        return z, mu, logvar

# ============================================================================
# Style Encoder - Class-Discriminative Features
# ============================================================================

class StyleEncoder(nn.Module):
    """
    Extracts class-discriminative style from ECG.
    Should capture: P-wave characteristics, RR interval regularity, fibrillatory waves
    These features determine Normal vs AFib classification.
    """
    def __init__(self, in_channels=1, hidden_dim=64, style_dim=128, num_classes=2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 7, stride=2, padding=3),
            nn.InstanceNorm1d(hidden_dim),  # Instance norm for style
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.fc_style = nn.Linear(hidden_dim * 4, style_dim)
        
        # Style should be predictive of class
        self.classifier = nn.Linear(style_dim, num_classes)
        
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        style = self.fc_style(h)
        class_logits = self.classifier(style)
        return style, class_logits

# ============================================================================
# Conditional Diffusion UNet
# ============================================================================

def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class ConditionalResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)  # FiLM: scale and shift
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        # FiLM conditioning
        scale, shift = self.cond_proj(cond).chunk(2, dim=1)
        h = h * (1 + scale[:, :, None]) + shift[:, :, None]
        h = F.silu(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)

class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5
        
    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum('bhcl,bhck->bhlk', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhlk,bhck->bhcl', attn, v).reshape(B, C, L)
        return x + self.proj(out)

class ConditionalUNet(nn.Module):
    """
    UNet conditioned on:
    - Timestep embedding
    - Content embedding (class-invariant)
    - Style embedding (class-discriminative)
    - Target class (for classifier-free guidance)
    """
    def __init__(self, in_ch=1, model_ch=64, content_dim=256, style_dim=128, num_classes=2):
        super().__init__()
        
        time_dim = model_ch * 4
        cond_dim = time_dim  # Combined conditioning dimension
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Content and style projection
        self.content_proj = nn.Linear(content_dim, time_dim)
        self.style_proj = nn.Linear(style_dim, time_dim)
        
        # Class embedding (for CFG)
        self.class_embed = nn.Embedding(num_classes + 1, time_dim)  # +1 for unconditional
        
        # Input
        self.input_conv = nn.Conv1d(in_ch, model_ch, 3, padding=1)
        
        # Encoder
        self.down1 = ConditionalResBlock(model_ch, model_ch, cond_dim)
        self.down2 = ConditionalResBlock(model_ch, model_ch * 2, cond_dim)
        self.down3 = ConditionalResBlock(model_ch * 2, model_ch * 4, cond_dim)
        self.down4 = ConditionalResBlock(model_ch * 4, model_ch * 8, cond_dim)
        
        self.downsample1 = nn.Conv1d(model_ch, model_ch, 3, stride=2, padding=1)
        self.downsample2 = nn.Conv1d(model_ch * 2, model_ch * 2, 3, stride=2, padding=1)
        self.downsample3 = nn.Conv1d(model_ch * 4, model_ch * 4, 3, stride=2, padding=1)
        self.downsample4 = nn.Conv1d(model_ch * 8, model_ch * 8, 3, stride=2, padding=1)
        
        # Middle
        self.mid1 = ConditionalResBlock(model_ch * 8, model_ch * 8, cond_dim)
        self.mid_attn = SelfAttention1D(model_ch * 8)
        self.mid2 = ConditionalResBlock(model_ch * 8, model_ch * 8, cond_dim)
        
        # Decoder
        self.up4 = ConditionalResBlock(model_ch * 16, model_ch * 8, cond_dim)
        self.up3 = ConditionalResBlock(model_ch * 8, model_ch * 4, cond_dim)
        self.up2 = ConditionalResBlock(model_ch * 4, model_ch * 2, cond_dim)
        self.up1 = ConditionalResBlock(model_ch * 2, model_ch, cond_dim)
        
        self.upsample4 = nn.ConvTranspose1d(model_ch * 8, model_ch * 8, 4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose1d(model_ch * 8, model_ch * 4, 4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose1d(model_ch * 4, model_ch * 2, 4, stride=2, padding=1)
        self.upsample1 = nn.ConvTranspose1d(model_ch * 2, model_ch, 4, stride=2, padding=1)
        
        # Output
        self.out_norm = nn.GroupNorm(32, model_ch)
        self.out_conv = nn.Conv1d(model_ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        
    def forward(self, x, t, content, style, class_label=None):
        # Get embeddings
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        c_emb = self.content_proj(content)
        s_emb = self.style_proj(style)
        
        # Class embedding (for CFG)
        if class_label is not None:
            class_emb = self.class_embed(class_label)
        else:
            class_emb = 0.0
        
        # Combined conditioning
        cond = t_emb + c_emb + s_emb + class_emb
        
        # Encoder
        h = self.input_conv(x)
        h1 = self.down1(h, cond)
        h = self.downsample1(h1)
        
        h2 = self.down2(h, cond)
        h = self.downsample2(h2)
        
        h3 = self.down3(h, cond)
        h = self.downsample3(h3)
        
        h4 = self.down4(h, cond)
        h = self.downsample4(h4)
        
        # Middle
        h = self.mid1(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)
        
        # Decoder with skip connections
        h = self.upsample4(h)
        h = self._match_size(h, h4)
        h = self.up4(torch.cat([h, h4], dim=1), cond)
        
        h = self.upsample3(h)
        h = self._match_size(h, h3)
        h = self.up3(torch.cat([h, h3], dim=1), cond)
        
        h = self.upsample2(h)
        h = self._match_size(h, h2)
        h = self.up2(torch.cat([h, h2], dim=1), cond)
        
        h = self.upsample1(h)
        h = self._match_size(h, h1)
        h = self.up1(torch.cat([h, h1], dim=1), cond)
        
        # Output
        h = F.silu(self.out_norm(h))
        h = self._match_size(h, x)
        return self.out_conv(h)
    
    def _match_size(self, x, target):
        if x.size(-1) != target.size(-1):
            diff = target.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[:, :, :target.size(-1)]
        return x

# ============================================================================
# DDIM Scheduler with SDEdit Support
# ============================================================================

class DDIMScheduler:
    def __init__(self, num_timesteps=1000, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Cosine schedule
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps, device=device) / num_timesteps
        alphas_cumprod = torch.cos((t + 0.008) / 1.008 * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        self.betas = torch.clamp(betas, 0.0001, 0.9999)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """Add noise to clean signal."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (self.sqrt_alphas_cumprod[t][:, None, None] * x_start + 
                self.sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise)
    
    def sdedit_sample(self, model, content, style, class_label, x_orig, strength=0.6, num_steps=50, cfg_scale=3.0):
        """
        SDEdit-style sampling: Start from noisy original signal.
        
        Args:
            model: UNet model
            content: Content embedding
            style: Style embedding  
            class_label: Target class
            x_orig: Original signal
            strength: Noise strength (0=no change, 1=full generation)
            num_steps: Denoising steps
            cfg_scale: Classifier-free guidance scale
        """
        device = x_orig.device
        
        # Determine starting timestep based on strength
        start_timestep = int(self.num_timesteps * strength)
        
        # Add noise to original signal
        t_start = torch.full((x_orig.size(0),), start_timestep, device=device, dtype=torch.long)
        noise = torch.randn_like(x_orig)
        x = self.q_sample(x_orig, t_start, noise)
        
        # Denoise from start_timestep to 0
        step_size = max(1, start_timestep // num_steps)
        timesteps = list(range(0, start_timestep, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((x.size(0),), t, device=device, dtype=torch.long)
            
            # Classifier-free guidance
            if cfg_scale > 1.0:
                # Conditional prediction
                pred_cond = model(x, t_batch, content, style, class_label)
                # Unconditional prediction (use class = num_classes)
                uncond_label = torch.full_like(class_label, Config.NUM_CLASSES)
                pred_uncond = model(x, t_batch, content, style, uncond_label)
                # Guided prediction
                pred_noise = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred_noise = model(x, t_batch, content, style, class_label)
            
            # Predict x0
            alpha_t = self.alphas_cumprod[t]
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise
            else:
                x = pred_x0
                
        return x
    
    @torch.no_grad()
    def sample(self, model, content, style, class_label, shape, num_steps=50, cfg_scale=3.0):
        """Standard DDIM sampling from pure noise."""
        device = content.device
        x = torch.randn(shape, device=device)
        
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Classifier-free guidance
            if cfg_scale > 1.0:
                pred_cond = model(x, t_batch, content, style, class_label)
                uncond_label = torch.full_like(class_label, Config.NUM_CLASSES)
                pred_uncond = model(x, t_batch, content, style, uncond_label)
                pred_noise = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
            else:
                pred_noise = model(x, t_batch, content, style, class_label)
            
            alpha_t = self.alphas_cumprod[t]
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                x = torch.sqrt(alpha_prev) * pred_x0 + torch.sqrt(1 - alpha_prev) * pred_noise
            else:
                x = pred_x0
                
        return x

# ============================================================================
# Post-Processing: Noise Reduction
# ============================================================================

def reduce_noise(signal, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay filter to reduce high-frequency noise.
    
    Args:
        signal: ECG signal (B, C, L) tensor
        window_length: Filter window length (must be odd)
        polyorder: Polynomial order
        
    Returns:
        Smoothed signal
    """
    if isinstance(signal, torch.Tensor):
        signal_np = signal.cpu().numpy()
        device = signal.device
    else:
        signal_np = signal
        device = None
    
    smoothed = np.zeros_like(signal_np)
    for i in range(signal_np.shape[0]):
        for j in range(signal_np.shape[1]):
            smoothed[i, j] = savgol_filter(signal_np[i, j], window_length, polyorder)
    
    if device is not None:
        return torch.tensor(smoothed, device=device, dtype=torch.float32)
    return smoothed

# ============================================================================
# Initialize Models
# ============================================================================

print("\n" + "="*60)
print("Initializing Models")
print("="*60)

content_encoder = ContentEncoder(
    in_channels=Config.IN_CHANNELS,
    hidden_dim=Config.ENCODER_CHANNELS,
    content_dim=Config.CONTENT_DIM
).to(Config.DEVICE)

style_encoder = StyleEncoder(
    in_channels=Config.IN_CHANNELS,
    hidden_dim=Config.ENCODER_CHANNELS,
    style_dim=Config.STYLE_DIM,
    num_classes=Config.NUM_CLASSES
).to(Config.DEVICE)

unet = ConditionalUNet(
    in_ch=Config.IN_CHANNELS,
    model_ch=Config.MODEL_CHANNELS,
    content_dim=Config.CONTENT_DIM,
    style_dim=Config.STYLE_DIM,
    num_classes=Config.NUM_CLASSES
).to(Config.DEVICE)

scheduler = DDIMScheduler(Config.DIFFUSION_TIMESTEPS, Config.DEVICE)

# Load classifier for evaluation
classifier = load_classifier(Config.DEVICE)
classifier_wrapper = ClassifierWrapper(classifier).to(Config.DEVICE)

content_params = sum(p.numel() for p in content_encoder.parameters())
style_params = sum(p.numel() for p in style_encoder.parameters())
unet_params = sum(p.numel() for p in unet.parameters())
total_params = content_params + style_params + unet_params

print(f"Content Encoder: {content_params:,} params")
print(f"Style Encoder: {style_params:,} params")
print(f"UNet: {unet_params:,} params")
print(f"Total: {total_params:,} params (~{total_params/1e6:.1f}M)")

# Optimizers
all_params = list(content_encoder.parameters()) + list(style_encoder.parameters()) + list(unet.parameters())
optimizer = torch.optim.AdamW(all_params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

# ============================================================================
# Training Functions
# ============================================================================

def train_epoch_stage1(epoch):
    """Stage 1: Train for high-quality reconstruction."""
    content_encoder.train()
    style_encoder.train()
    unet.train()
    
    total_loss = 0
    total_recon = 0
    total_style = 0
    total_kl = 0
    
    pbar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch}")
    
    for signals, labels in pbar:
        signals = signals.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        # Encode content and style
        content, content_mu, content_logvar = content_encoder(signals)
        style, style_logits = style_encoder(signals)
        
        # Random class dropout for CFG
        if np.random.rand() < Config.DROPOUT_PROB:
            class_label = torch.full_like(labels, Config.NUM_CLASSES)  # Unconditional
        else:
            class_label = labels
        
        # Diffusion forward pass
        t = torch.randint(0, Config.DIFFUSION_TIMESTEPS, (signals.size(0),), device=Config.DEVICE)
        noise = torch.randn_like(signals)
        noisy = scheduler.q_sample(signals, t, noise)
        
        # Predict noise
        pred_noise = unet(noisy, t, content, style, class_label)
        
        # Losses
        recon_loss = F.mse_loss(pred_noise, noise)
        style_loss = F.cross_entropy(style_logits, labels)
        kl_loss = -0.5 * torch.mean(1 + content_logvar - content_mu.pow(2) - content_logvar.exp())
        
        loss = (Config.RECON_WEIGHT * recon_loss + 
                Config.STYLE_CLASS_WEIGHT * style_loss + 
                Config.KL_WEIGHT * kl_loss)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_style += style_loss.item()
        total_kl += kl_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'style': f'{style_loss.item():.4f}'
        })
    
    n = len(train_loader)
    return total_loss/n, total_recon/n, total_style/n, total_kl/n

def train_epoch_stage2(epoch):
    """Stage 2: Fine-tune for counterfactual generation with flip loss."""
    content_encoder.train()
    style_encoder.train()
    unet.train()
    
    total_loss = 0
    total_recon = 0
    total_flip = 0
    total_sim = 0
    
    pbar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch}")
    
    for signals, labels in pbar:
        signals = signals.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        # Encode content and style
        content, content_mu, content_logvar = content_encoder(signals)
        style, style_logits = style_encoder(signals)
        
        # Target class (flip)
        target_class = 1 - labels
        
        # Random class dropout for CFG
        if np.random.rand() < Config.DROPOUT_PROB:
            class_label = torch.full_like(labels, Config.NUM_CLASSES)
        else:
            class_label = labels
        
        # Diffusion reconstruction loss
        t = torch.randint(0, Config.DIFFUSION_TIMESTEPS, (signals.size(0),), device=Config.DEVICE)
        noise = torch.randn_like(signals)
        noisy = scheduler.q_sample(signals, t, noise)
        pred_noise = unet(noisy, t, content, style, class_label)
        recon_loss = F.mse_loss(pred_noise, noise)
        
        # Generate counterfactuals
        with torch.no_grad():
            # Use SDEdit to generate from partial noise
            cf = scheduler.sdedit_sample(
                unet, content, style, target_class, 
                signals, strength=Config.SDEDIT_STRENGTH, 
                num_steps=20, cfg_scale=Config.CFG_SCALE
            )
        
        # Flip loss: counterfactual should be classified as target
        cf_logits = classifier_wrapper(cf)
        cf_prob = torch.softmax(cf_logits, dim=1)
        flip_loss = F.cross_entropy(cf_logits, target_class)
        
        # Similarity loss: counterfactual should be similar to original
        sim_loss = F.mse_loss(cf, signals)
        
        loss = (Config.RECON_WEIGHT * recon_loss + 
                Config.FLIP_WEIGHT * flip_loss +
                Config.SIMILARITY_WEIGHT * sim_loss)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_flip += flip_loss.item()
        total_sim += sim_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'flip': f'{flip_loss.item():.4f}',
            'sim': f'{sim_loss.item():.4f}'
        })
    
    n = len(train_loader)
    return total_loss/n, total_recon/n, total_flip/n, total_sim/n

@torch.no_grad()
def evaluate(epoch, stage=1):
    """Evaluate reconstruction and counterfactual generation."""
    content_encoder.eval()
    style_encoder.eval()
    unet.eval()
    
    # Test reconstruction
    test_idx = np.random.choice(len(val_signals), min(Config.NUM_EVAL_SAMPLES, len(val_signals)), replace=False)
    test_signals = val_signals[test_idx].to(Config.DEVICE)
    test_labels = val_labels[test_idx].to(Config.DEVICE)
    
    # Encode
    content, _, _ = content_encoder(test_signals)
    style, _ = style_encoder(test_signals)
    
    # Reconstruct using same class
    shape = (len(test_signals), Config.IN_CHANNELS, Config.SIGNAL_LENGTH)
    generated = scheduler.sample(unet, content, style, test_labels, shape, num_steps=50, cfg_scale=1.0)
    
    # Reduce noise
    generated_smooth = reduce_noise(generated, Config.SAVGOL_WINDOW, Config.SAVGOL_POLY)
    
    # Compute reconstruction error
    recon_corr = np.mean([np.corrcoef(test_signals[i, 0].cpu(), generated_smooth[i, 0].cpu())[0, 1] 
                          for i in range(len(test_signals))])
    recon_mse = F.mse_loss(generated_smooth, test_signals).item()
    
    # Visualize reconstruction
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for i in range(min(3, len(test_signals))):
        # Original
        axes[i, 0].plot(test_signals[i, 0].cpu().numpy(), color='blue', linewidth=0.8)
        axes[i, 0].set_title(f'Original (Class {test_labels[i].item()})')
        axes[i, 0].set_ylim([-2, 2])
        
        # Reconstructed
        axes[i, 1].plot(generated_smooth[i, 0].cpu().numpy(), color='green', linewidth=0.8)
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].set_ylim([-2, 2])
        
        # Overlay
        axes[i, 2].plot(test_signals[i, 0].cpu().numpy(), color='blue', alpha=0.7, label='Original')
        axes[i, 2].plot(generated_smooth[i, 0].cpu().numpy(), color='green', alpha=0.7, label='Reconstructed')
        axes[i, 2].set_title(f'Corr: {np.corrcoef(test_signals[i, 0].cpu(), generated_smooth[i, 0].cpu())[0, 1]:.3f}')
        axes[i, 2].set_ylim([-2, 2])
        axes[i, 2].legend()
    
    plt.suptitle(f'Epoch {epoch} - Reconstruction (Corr: {recon_corr:.3f}, MSE: {recon_mse:.4f})')
    plt.tight_layout()
    plt.savefig(Config.RESULTS_DIR / f'epoch_{epoch:03d}_reconstruction.png', dpi=150)
    plt.close()
    
    # Generate counterfactuals (Stage 2)
    if stage == 2:
        normal_idx = (test_labels == 0).nonzero(as_tuple=True)[0]
        afib_idx = (test_labels == 1).nonzero(as_tuple=True)[0]
        
        if len(normal_idx) > 0 and len(afib_idx) > 0:
            # Normal → AFib
            n_samples = min(5, len(normal_idx))
            normal_samples = test_signals[normal_idx[:n_samples]]
            normal_content, _, _ = content_encoder(normal_samples)
            normal_style, _ = style_encoder(normal_samples)
            target_afib = torch.ones(n_samples, dtype=torch.long, device=Config.DEVICE)
            
            cf_n2a = scheduler.sdedit_sample(
                unet, normal_content, normal_style, target_afib,
                normal_samples, strength=Config.SDEDIT_STRENGTH,
                num_steps=50, cfg_scale=Config.CFG_SCALE
            )
            cf_n2a = reduce_noise(cf_n2a, Config.SAVGOL_WINDOW, Config.SAVGOL_POLY)
            
            # Check flip rate
            cf_logits_n2a = classifier_wrapper(cf_n2a)
            cf_pred = torch.argmax(cf_logits_n2a, dim=1)
            flip_rate_n2a = (cf_pred == 1).float().mean().item()
            
            # AFib → Normal
            a_samples = min(5, len(afib_idx))
            afib_samples = test_signals[afib_idx[:a_samples]]
            afib_content, _, _ = content_encoder(afib_samples)
            afib_style, _ = style_encoder(afib_samples)
            target_normal = torch.zeros(a_samples, dtype=torch.long, device=Config.DEVICE)
            
            cf_a2n = scheduler.sdedit_sample(
                unet, afib_content, afib_style, target_normal,
                afib_samples, strength=Config.SDEDIT_STRENGTH,
                num_steps=50, cfg_scale=Config.CFG_SCALE
            )
            cf_a2n = reduce_noise(cf_a2n, Config.SAVGOL_WINDOW, Config.SAVGOL_POLY)
            
            cf_logits_a2n = classifier_wrapper(cf_a2n)
            cf_pred = torch.argmax(cf_logits_a2n, dim=1)
            flip_rate_a2n = (cf_pred == 0).float().mean().item()
            
            # Visualize counterfactuals
            fig, axes = plt.subplots(2, min(5, n_samples), figsize=(15, 6))
            if min(5, n_samples) == 1:
                axes = axes.reshape(-1, 1)
            
            for i in range(min(5, n_samples)):
                # Normal → AFib
                axes[0, i].plot(normal_samples[i, 0].cpu().numpy(), color='blue', alpha=0.7, label='Original')
                axes[0, i].plot(cf_n2a[i, 0].cpu().numpy(), color='red', alpha=0.7, label='CF')
                axes[0, i].set_title(f'N→A (flip: {cf_pred[i]==1})')
                axes[0, i].set_ylim([-2, 2])
                if i == 0:
                    axes[0, i].legend()
            
            for i in range(min(5, a_samples)):
                # AFib → Normal
                axes[1, i].plot(afib_samples[i, 0].cpu().numpy(), color='blue', alpha=0.7, label='Original')
                axes[1, i].plot(cf_a2n[i, 0].cpu().numpy(), color='green', alpha=0.7, label='CF')
                axes[1, i].set_title(f'A→N (flip: {cf_pred[i]==0})')
                axes[1, i].set_ylim([-2, 2])
                if i == 0:
                    axes[1, i].legend()
            
            plt.suptitle(f'Epoch {epoch} - Counterfactuals (N→A: {flip_rate_n2a:.1%}, A→N: {flip_rate_a2n:.1%})')
            plt.tight_layout()
            plt.savefig(Config.RESULTS_DIR / f'epoch_{epoch:03d}_counterfactual.png', dpi=150)
            plt.close()
            
            return recon_corr, flip_rate_n2a, flip_rate_a2n
    
    return recon_corr, 0.0, 0.0

# ============================================================================
# Training Loop
# ============================================================================

print("\n" + "="*60)
print("Starting Training")
print("="*60)

history = {
    'stage1': {'loss': [], 'recon': [], 'style': [], 'kl': [], 'eval_corr': []},
    'stage2': {'loss': [], 'recon': [], 'flip': [], 'sim': [], 'eval_corr': [], 
               'flip_n2a': [], 'flip_a2n': []}
}

start_time = time.time()

# Load Stage 1 checkpoint if resuming
if RESUME_FROM_STAGE1:
    print("\n" + "="*60)
    print("RESUMING FROM STAGE 1 CHECKPOINT")
    print("="*60)
    
    checkpoint_path = Config.MODEL_DIR / 'checkpoint_stage1_epoch_050.pth'
    if checkpoint_path.exists():
        print(f"Loading: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)
        content_encoder.load_state_dict(checkpoint['content_encoder'])
        style_encoder.load_state_dict(checkpoint['style_encoder'])
        unet.load_state_dict(checkpoint['unet'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        history = checkpoint.get('history', history)
        print("✓ Stage 1 checkpoint loaded successfully")
        print("Skipping Stage 1, proceeding directly to Stage 2...")
    else:
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        print("Running full training instead...")
        RESUME_FROM_STAGE1 = False

# Stage 1: Reconstruction
if not RESUME_FROM_STAGE1:
    print("\n" + "="*60)
    print("STAGE 1: High-Quality Reconstruction Training")
    print("="*60)
    
    for epoch in range(1, Config.STAGE1_EPOCHS + 1):
        loss, recon, style, kl = train_epoch_stage1(epoch)
        history['stage1']['loss'].append(loss)
        history['stage1']['recon'].append(recon)
        history['stage1']['style'].append(style)
        history['stage1']['kl'].append(kl)
        
        print(f"Epoch {epoch}/{Config.STAGE1_EPOCHS}: Loss={loss:.4f}, Recon={recon:.4f}, Style={style:.4f}, KL={kl:.4f}")
        
        if epoch % Config.EVAL_INTERVAL == 0:
            eval_corr, _, _ = evaluate(epoch, stage=1)
            history['stage1']['eval_corr'].append(eval_corr)
            print(f"  Evaluation Correlation: {eval_corr:.4f}")
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'stage': 1,
                'content_encoder': content_encoder.state_dict(),
                'style_encoder': style_encoder.state_dict(),
                'unet': unet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'history': history,
            }, Config.MODEL_DIR / f'checkpoint_stage1_epoch_{epoch:03d}.pth')

# Stage 2: Counterfactual Fine-tuning
print("\n" + "="*60)
print("STAGE 2: Counterfactual Generation Fine-tuning")
print("="*60)

for epoch in range(Config.STAGE1_EPOCHS + 1, Config.STAGE1_EPOCHS + Config.STAGE2_EPOCHS + 1):
    loss, recon, flip, sim = train_epoch_stage2(epoch)
    history['stage2']['loss'].append(loss)
    history['stage2']['recon'].append(recon)
    history['stage2']['flip'].append(flip)
    history['stage2']['sim'].append(sim)
    
    print(f"Epoch {epoch}/{Config.NUM_EPOCHS}: Loss={loss:.4f}, Recon={recon:.4f}, Flip={flip:.4f}, Sim={sim:.4f}")
    
    if epoch % Config.EVAL_INTERVAL == 0:
        eval_corr, flip_n2a, flip_a2n = evaluate(epoch, stage=2)
        history['stage2']['eval_corr'].append(eval_corr)
        history['stage2']['flip_n2a'].append(flip_n2a)
        history['stage2']['flip_a2n'].append(flip_a2n)
        print(f"  Evaluation: Corr={eval_corr:.4f}, N→A Flip={flip_n2a:.2%}, A→N Flip={flip_a2n:.2%}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'stage': 2,
            'content_encoder': content_encoder.state_dict(),
            'style_encoder': style_encoder.state_dict(),
            'unet': unet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history,
        }, Config.MODEL_DIR / f'checkpoint_stage2_epoch_{epoch:03d}.pth')

training_time = time.time() - start_time

# Save final model
torch.save({
    'content_encoder': content_encoder.state_dict(),
    'style_encoder': style_encoder.state_dict(),
    'unet': unet.state_dict(),
    'config': {
        'content_dim': Config.CONTENT_DIM,
        'style_dim': Config.STYLE_DIM,
        'model_channels': Config.MODEL_CHANNELS,
        'num_classes': Config.NUM_CLASSES,
    },
    'history': history,
    'training_time_seconds': training_time,
    'total_epochs': Config.NUM_EPOCHS,
}, Config.MODEL_DIR / 'final_model.pth')

# Save metadata
metadata = {
    'model_name': 'Enhanced Diffusion Counterfactual Generator',
    'total_parameters': total_params,
    'content_encoder_params': content_params,
    'style_encoder_params': style_params,
    'unet_params': unet_params,
    'training_time_hours': training_time / 3600,
    'total_epochs': Config.NUM_EPOCHS,
    'stage1_epochs': Config.STAGE1_EPOCHS,
    'stage2_epochs': Config.STAGE2_EPOCHS,
    'final_recon_correlation': history['stage1']['eval_corr'][-1] if history['stage1']['eval_corr'] else 0,
    'final_flip_n2a': history['stage2']['flip_n2a'][-1] if history['stage2']['flip_a2n'] else 0,
    'final_flip_a2n': history['stage2']['flip_a2n'][-1] if history['stage2']['flip_a2n'] else 0,
}

with open(Config.MODEL_DIR / 'model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "="*60)
print("Training Complete!")
print(f"Total Training Time: {training_time/3600:.2f} hours")
print(f"Models saved to: {Config.MODEL_DIR}")
print("="*60)
