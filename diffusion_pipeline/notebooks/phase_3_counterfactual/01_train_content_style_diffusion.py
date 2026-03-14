"""
Phase 3: Content-Style Disentangled Diffusion for ECG Counterfactual Generation
================================================================================

Goal: Generate counterfactuals that flip classifier predictions while being realistic.

Architecture:
- Content Encoder: Extracts class-invariant features (heart rate, basic rhythm)
- Style Encoder: Extracts class-discriminative features (P-wave, RR regularity)
- Conditional Diffusion: Generates ECG conditioned on content + style

Training:
1. Reconstruction: Content(x) + Style(x) -> x (should match original)
2. Style Swap: Content(x_normal) + Style(x_afib) -> counterfactual

Validation:
- Counterfactual must flip classifier prediction
- Must be visually realistic and clinically meaningful
"""

import os
import sys
import time
import subprocess

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
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
    DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
    PHASE2_MODEL = PROJECT_ROOT / 'models/phase2_diffusion/diffusion_v2/best.pth'
    MODEL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual'
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
    
    # UNet (from Phase 2)
    MODEL_CHANNELS = 64
    CHANNEL_MULT = (1, 2, 4, 8)
    NUM_RES_BLOCKS = 2
    ATTENTION_RESOLUTIONS = (2, 3)
    
    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    
    # Loss weights
    RECON_WEIGHT = 1.0      # Reconstruction loss
    STYLE_WEIGHT = 0.5      # Style classification loss
    CONTENT_WEIGHT = 0.1    # Content invariance loss
    KL_WEIGHT = 0.01        # KL divergence for VAE-style encoding
    
    # Evaluation
    EVAL_INTERVAL = 10
    NUM_EVAL_SAMPLES = 50

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
# Content Encoder - Extracts class-invariant features
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
            
            # Use adaptive pooling to get fixed size output
            nn.AdaptiveAvgPool1d(8),
        )
        
        # Fixed flattened size after adaptive pooling
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
# Style Encoder - Extracts class-discriminative features
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
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, cond):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
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
    """
    def __init__(self, in_ch=1, model_ch=64, content_dim=256, style_dim=128):
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
        
        # Decoder - input channels = upsample output + skip connection
        self.up4 = ConditionalResBlock(model_ch * 16, model_ch * 8, cond_dim)  # 8+8=16 -> 8
        self.up3 = ConditionalResBlock(model_ch * 8, model_ch * 4, cond_dim)   # 4+4=8 -> 4
        self.up2 = ConditionalResBlock(model_ch * 4, model_ch * 2, cond_dim)   # 2+2=4 -> 2
        self.up1 = ConditionalResBlock(model_ch * 2, model_ch, cond_dim)       # 1+1=2 -> 1
        
        # Upsample layers take output from previous up block
        self.upsample4 = nn.ConvTranspose1d(model_ch * 8, model_ch * 8, 4, stride=2, padding=1)
        self.upsample3 = nn.ConvTranspose1d(model_ch * 8, model_ch * 4, 4, stride=2, padding=1)  # up4 outputs 8*ch
        self.upsample2 = nn.ConvTranspose1d(model_ch * 4, model_ch * 2, 4, stride=2, padding=1)  # up3 outputs 4*ch
        self.upsample1 = nn.ConvTranspose1d(model_ch * 2, model_ch, 4, stride=2, padding=1)      # up2 outputs 2*ch
        
        # Output
        self.out_norm = nn.GroupNorm(32, model_ch)
        self.out_conv = nn.Conv1d(model_ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        
    def forward(self, x, t, content, style):
        # Get embeddings
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        c_emb = self.content_proj(content)
        s_emb = self.style_proj(style)
        
        # Combined conditioning
        cond = t_emb + c_emb + s_emb
        
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
# DDIM Scheduler
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
        if noise is None:
            noise = torch.randn_like(x_start)
        return (self.sqrt_alphas_cumprod[t][:, None, None] * x_start + 
                self.sqrt_one_minus_alphas_cumprod[t][:, None, None] * noise)
    
    @torch.no_grad()
    def sample(self, model, content, style, shape, num_steps=50):
        device = content.device
        x = torch.randn(shape, device=device)
        
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch, content, style)
            
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
    style_dim=Config.STYLE_DIM
).to(Config.DEVICE)

scheduler = DDIMScheduler(Config.DIFFUSION_TIMESTEPS, Config.DEVICE)

print(f"Content Encoder: {sum(p.numel() for p in content_encoder.parameters()):,} params")
print(f"Style Encoder: {sum(p.numel() for p in style_encoder.parameters()):,} params")
print(f"UNet: {sum(p.numel() for p in unet.parameters()):,} params")

# Optimizers
all_params = list(content_encoder.parameters()) + list(style_encoder.parameters()) + list(unet.parameters())
optimizer = torch.optim.AdamW(all_params, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

# ============================================================================
# Training Loop
# ============================================================================

print("\n" + "="*60)
print("Starting Training")
print("="*60)

def train_epoch(epoch):
    content_encoder.train()
    style_encoder.train()
    unet.train()
    
    total_loss = 0
    total_recon = 0
    total_style = 0
    total_kl = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for signals, labels in pbar:
        signals = signals.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        # Encode content and style
        content, content_mu, content_logvar = content_encoder(signals)
        style, style_logits = style_encoder(signals)
        
        # Diffusion forward pass
        t = torch.randint(0, Config.DIFFUSION_TIMESTEPS, (signals.size(0),), device=Config.DEVICE)
        noise = torch.randn_like(signals)
        noisy = scheduler.q_sample(signals, t, noise)
        
        # Predict noise
        pred_noise = unet(noisy, t, content, style)
        
        # Losses
        recon_loss = F.mse_loss(pred_noise, noise)
        style_loss = F.cross_entropy(style_logits, labels)
        kl_loss = -0.5 * torch.mean(1 + content_logvar - content_mu.pow(2) - content_logvar.exp())
        
        loss = (Config.RECON_WEIGHT * recon_loss + 
                Config.STYLE_WEIGHT * style_loss + 
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

@torch.no_grad()
def evaluate(epoch):
    content_encoder.eval()
    style_encoder.eval()
    unet.eval()
    
    # Test reconstruction: Content(x) + Style(x) -> x
    test_idx = np.random.choice(len(val_signals), Config.NUM_EVAL_SAMPLES, replace=False)
    test_signals = val_signals[test_idx].to(Config.DEVICE)
    test_labels = val_labels[test_idx].to(Config.DEVICE)
    
    # Encode
    content, _, _ = content_encoder(test_signals)
    style, _ = style_encoder(test_signals)
    
    # Generate
    shape = (len(test_signals), Config.IN_CHANNELS, Config.SIGNAL_LENGTH)
    generated = scheduler.sample(unet, content, style, shape, num_steps=50)
    
    # Compute reconstruction error
    recon_error = F.mse_loss(generated, test_signals).item()
    
    # Test style transfer
    normal_idx = (test_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (test_labels == 1).nonzero(as_tuple=True)[0]
    
    if len(normal_idx) > 0 and len(afib_idx) > 0:
        # Get normal content + afib style
        normal_content, _, _ = content_encoder(test_signals[normal_idx[:5]])
        afib_style, _ = style_encoder(test_signals[afib_idx[:5]])
        
        # Generate counterfactual (normal -> afib)
        cf_shape = (5, Config.IN_CHANNELS, Config.SIGNAL_LENGTH)
        counterfactual = scheduler.sample(unet, normal_content, afib_style, cf_shape, num_steps=50)
    else:
        counterfactual = None
    
    # Visualize
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i in range(3):
        # Original
        axes[i, 0].plot(test_signals[i, 0].cpu().numpy(), color='blue', linewidth=0.8)
        axes[i, 0].set_title(f'Original (Class {test_labels[i].item()})')
        axes[i, 0].set_ylim([-2, 2])
        
        # Reconstructed
        axes[i, 1].plot(generated[i, 0].cpu().numpy(), color='green', linewidth=0.8)
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].set_ylim([-2, 2])
        
        # Overlay
        axes[i, 2].plot(test_signals[i, 0].cpu().numpy(), color='blue', alpha=0.7, label='Original')
        axes[i, 2].plot(generated[i, 0].cpu().numpy(), color='green', alpha=0.7, label='Reconstructed')
        axes[i, 2].set_title('Overlay')
        axes[i, 2].set_ylim([-2, 2])
        axes[i, 2].legend()
    
    plt.suptitle(f'Epoch {epoch} - Reconstruction Error: {recon_error:.4f}')
    plt.tight_layout()
    plt.savefig(Config.RESULTS_DIR / f'epoch_{epoch:03d}_reconstruction.png', dpi=150)
    plt.close()
    
    # Counterfactual visualization
    if counterfactual is not None:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        for i in range(min(3, len(normal_idx))):
            # Original normal
            axes[0, i].plot(test_signals[normal_idx[i], 0].cpu().numpy(), color='blue', linewidth=0.8)
            axes[0, i].set_title(f'Original Normal')
            axes[0, i].set_ylim([-2, 2])
            
            # Counterfactual (should look like AFib)
            if i < len(counterfactual):
                axes[1, i].plot(counterfactual[i, 0].cpu().numpy(), color='red', linewidth=0.8)
                axes[1, i].set_title('Counterfactual (Normal→AFib)')
                axes[1, i].set_ylim([-2, 2])
        
        plt.suptitle(f'Epoch {epoch} - Counterfactual Generation')
        plt.tight_layout()
        plt.savefig(Config.RESULTS_DIR / f'epoch_{epoch:03d}_counterfactual.png', dpi=150)
        plt.close()
    
    return recon_error

# Training
history = {'loss': [], 'recon': [], 'style': [], 'kl': [], 'eval_recon': []}

for epoch in range(1, Config.NUM_EPOCHS + 1):
    loss, recon, style, kl = train_epoch(epoch)
    history['loss'].append(loss)
    history['recon'].append(recon)
    history['style'].append(style)
    history['kl'].append(kl)
    
    print(f"Epoch {epoch}: Loss={loss:.4f}, Recon={recon:.4f}, Style={style:.4f}, KL={kl:.4f}")
    
    if epoch % Config.EVAL_INTERVAL == 0:
        eval_recon = evaluate(epoch)
        history['eval_recon'].append(eval_recon)
        print(f"  Evaluation Reconstruction Error: {eval_recon:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'content_encoder': content_encoder.state_dict(),
            'style_encoder': style_encoder.state_dict(),
            'unet': unet.state_dict(),
            'optimizer': optimizer.state_dict(),
            'history': history,
        }, Config.MODEL_DIR / f'checkpoint_epoch_{epoch:03d}.pth')

# Save final model
torch.save({
    'content_encoder': content_encoder.state_dict(),
    'style_encoder': style_encoder.state_dict(),
    'unet': unet.state_dict(),
    'config': {
        'content_dim': Config.CONTENT_DIM,
        'style_dim': Config.STYLE_DIM,
        'model_channels': Config.MODEL_CHANNELS,
    }
}, Config.MODEL_DIR / 'final_model.pth')

print("\n" + "="*60)
print("Training Complete!")
print(f"Models saved to: {Config.MODEL_DIR}")
print("="*60)
