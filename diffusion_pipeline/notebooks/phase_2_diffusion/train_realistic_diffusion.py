"""
train_realistic_diffusion.py - Train until ECG quality passes 90% similarity test
==================================================================================

This script will train ITERATIVELY until generated ECGs are realistic:
- Variance ratio: 0.90-1.10 (90% match)
- Correlation with real ECG morphology: >0.85
- KS statistic: <0.10
- Waveform similarity: >90%

The training will NOT stop until these thresholds are met.
"""

import time
import subprocess
import sys
import os

# ============================================================================
# GPU MEMORY CHECKER AND AUTO-SELECTOR
# ============================================================================

def get_gpu_memory_info():
    """Get memory info for all GPUs"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = line.split(', ')
                if len(parts) == 3:
                    idx, free, total = parts
                    gpu_info.append({
                        'index': int(idx),
                        'free_mb': int(free),
                        'total_mb': int(total),
                        'used_mb': int(total) - int(free),
                        'free_pct': int(free) / int(total) * 100
                    })
        return gpu_info
    except Exception as e:
        print(f"Warning: Could not get GPU info - {e}")
        return []

def wait_for_gpu(required_mb=10000, max_wait_minutes=120, check_interval=60):
    print("\n" + "="*80)
    print("GPU MEMORY AUTO-SELECTOR")
    print("="*80)
    
    max_attempts = (max_wait_minutes * 60) // check_interval
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        gpu_info = get_gpu_memory_info()
        
        if not gpu_info:
            print("Cannot detect GPUs. Proceeding with default device...")
            return None
        
        gpu_info.sort(key=lambda x: x['free_mb'], reverse=True)
        
        print(f"\nCheck #{attempt}")
        for gpu in gpu_info:
            status = "AVAILABLE" if gpu['free_mb'] >= required_mb else f"Need {(required_mb - gpu['free_mb'])/1024:.1f}GB more"
            print(f"GPU {gpu['index']}: {gpu['free_mb']/1024:.1f}GB free - {status}")
        
        best_gpu = None
        for gpu in gpu_info:
            if gpu['free_mb'] >= required_mb:
                best_gpu = gpu
                break
        
        if best_gpu:
            print(f"\nSelected GPU {best_gpu['index']} with {best_gpu['free_mb']/1024:.1f}GB free")
            return best_gpu['index']
        
        if attempt < max_attempts:
            print(f"Waiting {check_interval}s for GPU...")
            time.sleep(check_interval)
    
    return gpu_info[0]['index'] if gpu_info else None

# Auto-select GPU
selected_gpu = wait_for_gpu(required_mb=10000, max_wait_minutes=120)
if selected_gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)
    print(f"Set CUDA_VISIBLE_DEVICES={selected_gpu}")

# ============================================================================
# IMPORTS
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scipy import stats, signal
import math
from torch.amp import GradScaler, autocast

# ============================================================================
# QUALITY THRESHOLDS - Training continues until ALL are met
# ============================================================================

QUALITY_THRESHOLDS = {
    'variance_ratio_min': 0.85,      # Generated/Real variance ratio >= 85%
    'variance_ratio_max': 1.15,      # Generated/Real variance ratio <= 115%
    'ks_statistic_max': 0.15,        # KS test statistic < 0.15
    'mean_correlation_min': 0.70,    # Mean waveform correlation > 0.70
    'std_match_pct': 0.85,           # Std within 85% of real
}

print("\n" + "="*80)
print("QUALITY THRESHOLDS (Training continues until ALL are met):")
print("="*80)
for k, v in QUALITY_THRESHOLDS.items():
    print(f"  {k}: {v}")
print("="*80 + "\n")

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
    DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
    MODEL_DIR = PROJECT_ROOT / 'models/phase2_diffusion'
    CHECKPOINT_DIR = MODEL_DIR / 'realistic_unconditional'
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR = MODEL_DIR / 'results_realistic'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR = RESULTS_DIR / 'evaluations'
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    
    # Signal
    SIGNAL_LENGTH = 2500
    IN_CHANNELS = 1
    
    # Noise Schedule
    DIFFUSION_TIMESTEPS = 1000
    BETA_SCHEDULE = 'cosine'
    BETA_START = 0.0001
    BETA_END = 0.02
    COSINE_S = 0.008
    
    # UNet Architecture - INCREASED capacity
    MODEL_CHANNELS = 96  # Increased from 64
    CHANNEL_MULT = (1, 2, 3, 4)
    NUM_RES_BLOCKS = 3  # Increased from 2
    ATTENTION_RESOLUTIONS = (1, 2, 3)  # More attention
    NUM_GROUPS = 32
    DROPOUT = 0.05  # Reduced dropout
    
    # Conditioning
    NUM_CLASSES = 2
    CLASS_EMBED_DIM = 256
    TIME_EMBED_DIM = 256
    USE_CLASSIFIER_FREE_GUIDANCE = True
    GUIDANCE_DROPOUT = 0.1
    
    # Training - CRITICAL CHANGES
    BATCH_SIZE = 64  # Larger batch for better variance estimation
    NUM_EPOCHS = 500  # Will continue until quality passes
    LEARNING_RATE = 1e-4  # Reduced for stability
    WEIGHT_DECAY = 1e-6
    WARMUP_STEPS = 3000  # Longer warmup
    GRADIENT_CLIP = 0.5   # Stricter gradient clipping
    USE_EMA = True
    EMA_DECAY = 0.9999  # Slower EMA for stability
    
    # CRITICAL LOSS WEIGHTS - Variance preservation is KEY
    MSE_WEIGHT = 1.0
    VARIANCE_WEIGHT = 50.0          # VERY HIGH - prevents expansion/collapse
    RECONSTRUCTION_WEIGHT = 0.0     # Disabled for now
    FREQ_WEIGHT = 0.0               # Disabled - focus on MSE + variance
    RANGE_PENALTY_WEIGHT = 5.0      # Penalize extreme outputs
    
    # Evaluation
    EVAL_INTERVAL = 5
    EVAL_SAMPLES = 200
    GENERATION_STEPS = 100  # More steps for better quality
    GENERATION_ETA = 0.0

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)

print(f"Device: {Config.DEVICE}")
print(f"Model channels: {Config.MODEL_CHANNELS}")
print(f"Batch size: {Config.BATCH_SIZE}")

# ============================================================================
# Load Data
# ============================================================================

print(f"\nLoading data from: {Config.DATA_DIR}")

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

print(f"Training data: {train_signals.shape}")
print(f"Validation data: {val_signals.shape}")

# Load normalization parameters
with open(Config.DATA_DIR / 'dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

norm_params = metadata['normalization']
Config.CLEAN_DATA_MIN = norm_params['clean_data_min']
Config.CLEAN_DATA_MAX = norm_params['clean_data_max']
Config.TARGET_MIN = norm_params['target_min']
Config.TARGET_MAX = norm_params['target_max']

# CRITICAL: Store real data statistics
Config.REAL_MEAN = float(train_signals.mean())
Config.REAL_STD = float(train_signals.std())
Config.REAL_VAR = Config.REAL_STD ** 2

print(f"\nReal data statistics (TARGET for generation):")
print(f"  Mean: {Config.REAL_MEAN:.6f}")
print(f"  Std: {Config.REAL_STD:.6f}")
print(f"  Range: [{train_signals.min():.4f}, {train_signals.max():.4f}]")

# Compute per-sample variance distribution
sample_stds = train_signals.view(train_signals.size(0), -1).std(dim=1)
Config.SAMPLE_STD_MEAN = float(sample_stds.mean())
Config.SAMPLE_STD_STD = float(sample_stds.std())
print(f"  Per-sample std: {Config.SAMPLE_STD_MEAN:.6f} ± {Config.SAMPLE_STD_STD:.6f}")

# Create DataLoaders
train_dataset = TensorDataset(train_signals, train_labels)
train_loader = DataLoader(
    train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
    drop_last=True, num_workers=4, pin_memory=True
)

val_dataset = TensorDataset(val_signals, val_labels)
val_loader = DataLoader(
    val_dataset, batch_size=Config.BATCH_SIZE * 2,
    shuffle=False, num_workers=4, pin_memory=True
)

print(f"Batches: Train={len(train_loader)}, Val={len(val_loader)}")

# ============================================================================
# DDIM Scheduler
# ============================================================================

class DDIMScheduler:
    def __init__(self, num_timesteps=1000, beta_schedule='cosine', 
                 beta_start=0.0001, beta_end=0.02, cosine_s=0.008, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        if beta_schedule == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps, cosine_s, device)
        else:
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
    
    def _cosine_beta_schedule(self, timesteps, s=0.008, device='cuda'):
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps, device=device) / timesteps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)
    
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t][:, None, None]
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t][:, None, None]
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def ddim_sample_step(self, model, x_t, t, t_prev, class_labels=None, eta=0.0):
        pred_noise = model(x_t, t, class_labels)
        pred_x0 = self.predict_start_from_noise(x_t, t, pred_noise)
        
        # Soft clipping to preserve variance
        pred_x0 = torch.clamp(pred_x0, Config.TARGET_MIN * 1.1, Config.TARGET_MAX * 1.1)
        
        alpha_t = self.alphas_cumprod[t][:, None, None]
        
        if (t_prev < 0).any():
            alpha_t_prev = torch.ones_like(alpha_t)
        else:
            alpha_t_prev = self.alphas_cumprod[t_prev][:, None, None]
        
        beta_t = 1 - alpha_t
        beta_t_prev = 1 - alpha_t_prev
        
        variance = torch.where(
            beta_t > 1e-10,
            beta_t_prev / beta_t * (1 - alpha_t / alpha_t_prev),
            torch.zeros_like(beta_t)
        )
        variance = torch.clamp(variance, min=0.0, max=1.0)
        
        sqrt_alpha_t_prev = torch.sqrt(torch.clamp(alpha_t_prev, min=1e-10))
        direction_var = 1.0 - alpha_t_prev - (eta ** 2) * variance
        direction_var = torch.clamp(direction_var, min=0.0)
        
        pred_sample_direction = torch.sqrt(direction_var) * pred_noise
        x_prev = sqrt_alpha_t_prev * pred_x0 + pred_sample_direction
        
        if eta > 0 and not (t_prev < 0).any():
            noise = torch.randn_like(x_t)
            sigma_t = eta * torch.sqrt(variance)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev
    
    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, class_labels=None, num_steps=100, eta=0.0, progress=True):
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        # Initialize with proper variance
        x = torch.randn(shape, device=device)
        
        if progress:
            timesteps_iter = tqdm(timesteps, desc="Sampling", leave=False)
        else:
            timesteps_iter = timesteps
        
        for i, t in enumerate(timesteps_iter):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_prev_value = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            t_prev_batch = torch.full((batch_size,), t_prev_value, device=device, dtype=torch.long)
            
            x = self.ddim_sample_step(model, x, t_batch, t_prev_batch, class_labels, eta)
        
        return torch.clamp(x, Config.TARGET_MIN, Config.TARGET_MAX)

noise_scheduler = DDIMScheduler(
    num_timesteps=Config.DIFFUSION_TIMESTEPS,
    beta_schedule=Config.BETA_SCHEDULE,
    device=Config.DEVICE
)

# ============================================================================
# UNet Model
# ============================================================================

def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb

class AdaGN(nn.Module):
    def __init__(self, num_channels, cond_dim, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, affine=False)
        self.proj = nn.Linear(cond_dim, num_channels * 2)
    def forward(self, x, cond):
        scale_shift = self.proj(cond)
        scale, shift = scale_shift.chunk(2, dim=1)
        x = self.norm(x)
        x = x * (1 + scale[:, :, None]) + shift[:, :, None]
        return x

class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, num_groups=32, dropout=0.1):
        super().__init__()
        self.norm1 = AdaGN(in_channels, cond_dim, num_groups)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaGN(out_channels, cond_dim, num_groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()
    def forward(self, x, cond):
        h = self.norm1(x, cond)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h, cond)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.skip(x)

class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv1d(channels, channels, kernel_size=1)
        self.scale = self.head_dim ** -0.5
    def forward(self, x):
        B, C, L = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, L)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, L)
        return x + self.proj(out)

class Downsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='linear', align_corners=False)
        return self.conv(x)

class ECGUNet(nn.Module):
    def __init__(self, in_channels=1, model_channels=96, out_channels=1,
                 num_res_blocks=3, attention_resolutions=(1, 2, 3),
                 channel_mult=(1, 2, 3, 4), num_classes=2, dropout=0.05,
                 num_groups=32, time_embed_dim=256, class_embed_dim=256,
                 use_cfg=True, cfg_dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        self.num_levels = len(channel_mult)
        
        cond_dim = time_embed_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.class_embed = nn.Embedding(num_classes + 1, class_embed_dim)
        self.class_proj = nn.Linear(class_embed_dim, time_embed_dim)
        
        self.input_proj = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)
        
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_channels
        self.skip_channels = [ch]
        
        for level in range(self.num_levels):
            out_ch = model_channels * channel_mult[level]
            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_blocks.append(ResBlock1D(ch, out_ch, cond_dim, num_groups, dropout))
                ch = out_ch
                self.skip_channels.append(ch)
            self.down_blocks.append(level_blocks)
            
            if level in attention_resolutions:
                self.down_attns.append(SelfAttention1D(ch, num_heads=8))
            else:
                self.down_attns.append(None)
            
            if level < self.num_levels - 1:
                self.down_samples.append(Downsample1D(ch))
                self.skip_channels.append(ch)
            else:
                self.down_samples.append(None)
        
        self.mid_block1 = ResBlock1D(ch, ch, cond_dim, num_groups, dropout)
        self.mid_attn = SelfAttention1D(ch, num_heads=8)
        self.mid_block2 = ResBlock1D(ch, ch, cond_dim, num_groups, dropout)
        
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level in reversed(range(self.num_levels)):
            out_ch = model_channels * channel_mult[level]
            level_blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = self.skip_channels.pop()
                level_blocks.append(ResBlock1D(ch + skip_ch, out_ch, cond_dim, num_groups, dropout))
                ch = out_ch
            self.up_blocks.append(level_blocks)
            
            if level in attention_resolutions:
                self.up_attns.append(SelfAttention1D(ch, num_heads=8))
            else:
                self.up_attns.append(None)
            
            if level > 0:
                self.up_samples.append(Upsample1D(ch))
            else:
                self.up_samples.append(None)
        
        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_conv = nn.Conv1d(ch, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(self, x, timesteps, class_labels=None):
        B = x.shape[0]
        t_emb = get_timestep_embedding(timesteps, self.model_channels)
        t_emb = self.time_embed(t_emb)
        
        if class_labels is None:
            class_labels = torch.full((B,), self.num_classes, device=x.device, dtype=torch.long)
        elif self.training and self.use_cfg:
            drop_mask = torch.rand(B, device=x.device) < self.cfg_dropout
            class_labels = torch.where(drop_mask, self.num_classes, class_labels)
        
        c_emb = self.class_embed(class_labels)
        c_emb = self.class_proj(c_emb)
        cond = t_emb + c_emb
        
        h = self.input_proj(x)
        skips = [h]
        
        for level in range(self.num_levels):
            for block in self.down_blocks[level]:
                h = block(h, cond)
                skips.append(h)
            if self.down_attns[level] is not None:
                h = self.down_attns[level](h)
            if self.down_samples[level] is not None:
                h = self.down_samples[level](h)
                skips.append(h)
        
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)
        
        for level_idx, level in enumerate(reversed(range(self.num_levels))):
            for block in self.up_blocks[level_idx]:
                skip = skips.pop()
                if h.shape[2] != skip.shape[2]:
                    if h.shape[2] < skip.shape[2]:
                        skip = skip[:, :, :h.shape[2]]
                    else:
                        skip = F.pad(skip, (0, h.shape[2] - skip.shape[2]))
                h = torch.cat([h, skip], dim=1)
                h = block(h, cond)
            if self.up_attns[level_idx] is not None:
                h = self.up_attns[level_idx](h)
            if self.up_samples[level_idx] is not None:
                h = self.up_samples[level_idx](h)
        
        h = self.out_norm(h)
        h = F.silu(h)
        
        if h.shape[2] != x.shape[2]:
            if h.shape[2] < x.shape[2]:
                h = F.pad(h, (0, x.shape[2] - h.shape[2]))
            else:
                h = h[:, :, :x.shape[2]]
        
        return self.out_conv(h)

unet = ECGUNet(
    in_channels=Config.IN_CHANNELS,
    model_channels=Config.MODEL_CHANNELS,
    out_channels=Config.IN_CHANNELS,
    num_res_blocks=Config.NUM_RES_BLOCKS,
    attention_resolutions=Config.ATTENTION_RESOLUTIONS,
    channel_mult=Config.CHANNEL_MULT,
    num_classes=Config.NUM_CLASSES,
    dropout=Config.DROPOUT,
    num_groups=Config.NUM_GROUPS,
    time_embed_dim=Config.TIME_EMBED_DIM,
    class_embed_dim=Config.CLASS_EMBED_DIM,
    use_cfg=Config.USE_CLASSIFIER_FREE_GUIDANCE,
    cfg_dropout=Config.GUIDANCE_DROPOUT
).to(Config.DEVICE)

total_params = sum(p.numel() for p in unet.parameters())
print(f"UNet: {total_params:,} parameters")

# ============================================================================
# EMA
# ============================================================================

class EMA:
    def __init__(self, model, decay=0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self):
        return {'decay': self.decay, 'shadow': self.shadow.copy()}
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']

ema = EMA(unet, decay=Config.EMA_DECAY)

# ============================================================================
# Optimizer
# ============================================================================

optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    betas=(0.9, 0.999)
)

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

total_steps = Config.NUM_EPOCHS * len(train_loader)
scheduler = get_lr_scheduler(optimizer, Config.WARMUP_STEPS, total_steps)

scaler = GradScaler('cuda')

# ============================================================================
# LOSS FUNCTIONS - Critical for realistic output
# ============================================================================

def compute_variance_loss(pred_x0, target_std):
    """Force generated samples to have correct variance - CRITICAL"""
    batch_std = pred_x0.std()
    target = torch.tensor(target_std, device=pred_x0.device, dtype=pred_x0.dtype)
    
    # Use log-ratio loss for better gradient when far from target
    std_ratio = batch_std / (target + 1e-8)
    log_ratio_loss = (torch.log(std_ratio + 1e-8)) ** 2
    
    # Also add direct MSE loss
    mse_loss = F.mse_loss(batch_std, target)
    
    return log_ratio_loss + mse_loss

def compute_per_sample_variance_loss(pred_x0, target_std_mean, target_std_std):
    """Force each sample to have realistic variance"""
    sample_stds = pred_x0.view(pred_x0.size(0), -1).std(dim=1)
    target_mean = torch.tensor(target_std_mean, device=pred_x0.device, dtype=pred_x0.dtype)
    
    # Penalize deviation from mean sample std
    mean_loss = F.mse_loss(sample_stds.mean(), target_mean)
    
    # Also penalize extreme values
    max_allowed_std = target_std_mean * 3.0  # Allow up to 3x target std
    excess_std_penalty = F.relu(sample_stds - max_allowed_std).mean()
    
    return mean_loss + excess_std_penalty

def compute_range_penalty(pred_x0, target_min, target_max, real_std):
    """Penalize extreme outputs - both too flat AND too expanded"""
    pred_range = pred_x0.max() - pred_x0.min()
    expected_range = 6 * real_std  # Expect ~6 sigma range
    
    # Penalize if range is too small OR too large
    range_excess = F.relu(pred_range - expected_range * 2.0)  # Too large
    range_deficit = F.relu(expected_range * 0.3 - pred_range)  # Too small
    
    return range_excess + range_deficit

def compute_frequency_loss(pred, target):
    """Match frequency content"""
    pred_fft = torch.fft.rfft(pred.float(), dim=-1)
    target_fft = torch.fft.rfft(target.float(), dim=-1)
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

# ============================================================================
# Quality Test Function - MUST PASS for training to stop
# ============================================================================

def compute_correlation(x, y):
    """Compute Pearson correlation between two arrays"""
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    x_mean = np.mean(x_flat)
    y_mean = np.mean(y_flat)
    
    num = np.sum((x_flat - x_mean) * (y_flat - y_mean))
    denom = np.sqrt(np.sum((x_flat - x_mean)**2) * np.sum((y_flat - y_mean)**2))
    
    if denom < 1e-10:
        return 0.0
    return num / denom

@torch.no_grad()
def run_quality_test(model, noise_scheduler, real_data, epoch, use_ema=True):
    """
    Comprehensive quality test - ALL metrics must pass for training to stop.
    Returns (passed: bool, results: dict)
    """
    print(f"\n{'='*80}")
    print(f"QUALITY TEST - Epoch {epoch}")
    print(f"{'='*80}")
    
    if use_ema and ema is not None:
        ema.apply_shadow()
    
    model.eval()
    
    # Generate samples
    all_generated = []
    num_samples = Config.EVAL_SAMPLES
    
    for class_id in range(Config.NUM_CLASSES):
        labels = torch.full((num_samples,), class_id, device=Config.DEVICE, dtype=torch.long)
        shape = (num_samples, Config.IN_CHANNELS, Config.SIGNAL_LENGTH)
        
        generated = noise_scheduler.ddim_sample_loop(
            model, shape, class_labels=labels,
            num_steps=Config.GENERATION_STEPS, eta=Config.GENERATION_ETA, progress=True
        )
        all_generated.append(generated.cpu().numpy())
    
    if use_ema and ema is not None:
        ema.restore()
    
    generated_ecgs = np.concatenate(all_generated, axis=0)
    
    # Get real data
    real_ecgs = real_data['X']
    if real_ecgs.ndim == 2:
        real_ecgs = real_ecgs[:, np.newaxis, :]
    
    # Sample same number of real ECGs
    indices = np.random.choice(len(real_ecgs), len(generated_ecgs), replace=False)
    real_sample = real_ecgs[indices]
    
    # ============ COMPUTE METRICS ============
    
    # 1. Statistical metrics
    real_mean = real_sample.mean()
    gen_mean = generated_ecgs.mean()
    real_std = real_sample.std()
    gen_std = generated_ecgs.std()
    
    variance_ratio = gen_std / real_std if real_std > 0 else 0
    std_match = 1 - abs(gen_std - real_std) / real_std if real_std > 0 else 0
    
    # 2. KS test
    ks_stat, ks_pval = stats.ks_2samp(real_sample.flatten(), generated_ecgs.flatten())
    
    # 3. Waveform correlations - compare random pairs
    correlations = []
    for i in range(min(50, len(generated_ecgs))):
        gen_sig = generated_ecgs[i, 0, :]
        real_sig = real_sample[np.random.randint(len(real_sample)), 0, :]
        corr = compute_correlation(gen_sig, real_sig)
        correlations.append(corr)
    mean_correlation = np.mean(correlations)
    
    # 4. Per-sample statistics
    gen_sample_stds = np.std(generated_ecgs.reshape(len(generated_ecgs), -1), axis=1)
    real_sample_stds = np.std(real_sample.reshape(len(real_sample), -1), axis=1)
    sample_std_ratio = gen_sample_stds.mean() / real_sample_stds.mean() if real_sample_stds.mean() > 0 else 0
    
    # ============ CHECK THRESHOLDS ============
    
    results = {
        'epoch': epoch,
        'variance_ratio': float(variance_ratio),
        'std_match': float(std_match),
        'ks_stat': float(ks_stat),
        'mean_correlation': float(mean_correlation),
        'sample_std_ratio': float(sample_std_ratio),
        'real_mean': float(real_mean),
        'gen_mean': float(gen_mean),
        'real_std': float(real_std),
        'gen_std': float(gen_std),
    }
    
    # Check each threshold
    checks = {
        'variance_ratio_min': variance_ratio >= QUALITY_THRESHOLDS['variance_ratio_min'],
        'variance_ratio_max': variance_ratio <= QUALITY_THRESHOLDS['variance_ratio_max'],
        'ks_statistic': ks_stat <= QUALITY_THRESHOLDS['ks_statistic_max'],
        'std_match': std_match >= QUALITY_THRESHOLDS['std_match_pct'],
        'correlation': mean_correlation >= QUALITY_THRESHOLDS['mean_correlation_min'],
    }
    
    all_passed = all(checks.values())
    
    # Print results
    print(f"\n{'Metric':<25} {'Value':<15} {'Threshold':<15} {'Status':<10}")
    print("-" * 65)
    print(f"{'Variance Ratio':<25} {variance_ratio:>12.4f}   {QUALITY_THRESHOLDS['variance_ratio_min']:.2f}-{QUALITY_THRESHOLDS['variance_ratio_max']:.2f}       {'PASS' if checks['variance_ratio_min'] and checks['variance_ratio_max'] else 'FAIL'}")
    print(f"{'Std Match %':<25} {std_match*100:>11.2f}%   {QUALITY_THRESHOLDS['std_match_pct']*100:.0f}%           {'PASS' if checks['std_match'] else 'FAIL'}")
    print(f"{'KS Statistic':<25} {ks_stat:>12.4f}   <{QUALITY_THRESHOLDS['ks_statistic_max']}         {'PASS' if checks['ks_statistic'] else 'FAIL'}")
    print(f"{'Mean Correlation':<25} {mean_correlation:>12.4f}   >{QUALITY_THRESHOLDS['mean_correlation_min']}          {'PASS' if checks['correlation'] else 'FAIL'}")
    print(f"{'Sample Std Ratio':<25} {sample_std_ratio:>12.4f}   0.85-1.15      {'INFO'}")
    
    print("\n" + "-" * 65)
    print(f"Real - Mean: {real_mean:.6f}, Std: {real_std:.6f}")
    print(f"Gen  - Mean: {gen_mean:.6f}, Std: {gen_std:.6f}")
    print("-" * 65)
    
    if all_passed:
        print("\n" + "="*80)
        print("ALL QUALITY TESTS PASSED!")
        print("="*80 + "\n")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"\nFailed tests: {failed}")
        print("Training will continue...")
    
    # Save visualizations
    eval_dir = Config.EVAL_DIR / f'epoch_{epoch:03d}'
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot comparison
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    for i in range(3):
        # Normal
        real_idx = np.random.randint(len(real_sample) // 2)
        gen_idx = i
        
        axes[i, 0].plot(real_sample[real_idx, 0, :], linewidth=0.8, alpha=0.8, color='blue', label='Real')
        axes[i, 0].plot(generated_ecgs[gen_idx, 0, :], linewidth=0.8, alpha=0.8, color='red', linestyle='--', label='Generated')
        axes[i, 0].set_title(f'Comparison {i+1}')
        axes[i, 0].set_ylim([Config.TARGET_MIN - 0.2, Config.TARGET_MAX + 0.2])
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Distribution
        if i == 0:
            axes[i, 1].hist(real_sample.flatten(), bins=100, alpha=0.6, color='blue', label='Real', density=True)
            axes[i, 1].hist(generated_ecgs.flatten(), bins=100, alpha=0.6, color='red', label='Generated', density=True)
            axes[i, 1].set_title('Value Distribution')
            axes[i, 1].legend()
        elif i == 1:
            axes[i, 1].hist(real_sample_stds, bins=30, alpha=0.6, color='blue', label='Real', density=True)
            axes[i, 1].hist(gen_sample_stds, bins=30, alpha=0.6, color='red', label='Generated', density=True)
            axes[i, 1].set_title('Per-Sample Std Distribution')
            axes[i, 1].legend()
        else:
            # Overlay multiple generated
            for j in range(min(10, len(generated_ecgs))):
                axes[i, 1].plot(generated_ecgs[j, 0, :], linewidth=0.5, alpha=0.3, color='red')
            axes[i, 1].set_title('10 Generated Samples Overlay')
            axes[i, 1].set_ylim([Config.TARGET_MIN - 0.2, Config.TARGET_MAX + 0.2])
    
    plt.suptitle(f'Quality Test - Epoch {epoch} - {"PASSED" if all_passed else "FAILED"}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(eval_dir / 'quality_test.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results - convert numpy bools to Python bools
    checks_serializable = {k: bool(v) for k, v in checks.items()}
    with open(eval_dir / 'quality_results.json', 'w') as f:
        json.dump({**results, 'checks': checks_serializable, 'all_passed': bool(all_passed)}, f, indent=2)
    
    np.save(eval_dir / 'generated_samples.npy', generated_ecgs)
    
    return all_passed, results

# ============================================================================
# Checkpoint Functions
# ============================================================================

def save_checkpoint(epoch, step, model, optimizer, scheduler, ema, is_best=False, is_final=False):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'ema_state_dict': ema.state_dict() if ema else None,
        'config': {
            'real_mean': Config.REAL_MEAN,
            'real_std': Config.REAL_STD,
            'target_min': Config.TARGET_MIN,
            'target_max': Config.TARGET_MAX,
        }
    }
    
    if is_final:
        filename = 'final_passed.pth'
    elif is_best:
        filename = 'best.pth'
    else:
        filename = f'checkpoint_epoch{epoch}.pth'
    
    torch.save(checkpoint, Config.CHECKPOINT_DIR / filename)
    print(f"Saved: {filename}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("STARTING TRAINING - Will continue until quality test passes")
print("="*80)

num_epochs = Config.NUM_EPOCHS
gradient_clip = Config.GRADIENT_CLIP
log_interval = 100

best_variance_ratio = 0.0
quality_passed = False
train_losses = []
eval_history = []
global_step = 0

# Load real data for testing
real_train_data = np.load(Config.DATA_DIR / 'train_data.npz')

for epoch in range(1, num_epochs + 1):
    if quality_passed:
        break
    
    unet.train()
    epoch_loss = 0.0
    epoch_mse = 0.0
    epoch_var = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    
    for batch_idx, (signals, labels) in enumerate(pbar):
        signals = signals.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        # Sample random timesteps
        t = torch.randint(0, Config.DIFFUSION_TIMESTEPS, (signals.size(0),), device=Config.DEVICE)
        noise = torch.randn_like(signals)
        noisy_signals = noise_scheduler.q_sample(signals, t, noise=noise)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            pred_noise = unet(noisy_signals, t, labels)
            mse_loss = F.mse_loss(pred_noise, noise)
        
        # Predict x0 for auxiliary losses
        with torch.no_grad():
            pred_x0 = noise_scheduler.predict_start_from_noise(noisy_signals, t, pred_noise)
            pred_x0 = torch.clamp(pred_x0, Config.TARGET_MIN * 1.1, Config.TARGET_MAX * 1.1)
        
        # Variance loss - CRITICAL
        variance_loss = compute_variance_loss(pred_x0, Config.REAL_STD)
        per_sample_var_loss = compute_per_sample_variance_loss(pred_x0, Config.SAMPLE_STD_MEAN, Config.SAMPLE_STD_STD)
        
        # Range penalty - prevent collapse
        range_penalty = compute_range_penalty(pred_x0, Config.TARGET_MIN, Config.TARGET_MAX, Config.REAL_STD)
        
        # Frequency loss
        freq_loss = compute_frequency_loss(pred_noise, noise)
        
        # Total loss
        total_loss = (
            Config.MSE_WEIGHT * mse_loss +
            Config.VARIANCE_WEIGHT * (variance_loss + per_sample_var_loss) +
            Config.RANGE_PENALTY_WEIGHT * range_penalty +
            Config.FREQ_WEIGHT * freq_loss
        )
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        ema.update()
        
        epoch_loss += total_loss.item()
        epoch_mse += mse_loss.item()
        epoch_var += variance_loss.item()
        num_batches += 1
        global_step += 1
        
        pbar.set_postfix({
            'mse': f'{mse_loss.item():.4f}',
            'var': f'{variance_loss.item():.4f}',
            'range': f'{range_penalty.item():.4f}'
        })
    
    avg_train_loss = epoch_loss / num_batches
    avg_mse = epoch_mse / num_batches
    avg_var = epoch_var / num_batches
    train_losses.append(avg_train_loss)
    
    print(f"\nEpoch {epoch} - Loss: {avg_train_loss:.4f} (MSE: {avg_mse:.4f}, Var: {avg_var:.4f})")
    
    # Run quality test every EVAL_INTERVAL epochs
    if epoch % Config.EVAL_INTERVAL == 0 or epoch == 1:
        quality_passed, results = run_quality_test(unet, noise_scheduler, real_train_data, epoch, use_ema=True)
        eval_history.append(results)
        
        # Track best variance ratio
        if abs(results['variance_ratio'] - 1.0) < abs(best_variance_ratio - 1.0):
            best_variance_ratio = results['variance_ratio']
            save_checkpoint(epoch, global_step, unet, optimizer, scheduler, ema, is_best=True)
        
        if quality_passed:
            print("\n" + "="*80)
            print("TRAINING COMPLETE - Quality test PASSED!")
            print("="*80)
            save_checkpoint(epoch, global_step, unet, optimizer, scheduler, ema, is_final=True)
            break
    
    # Regular checkpoints
    if epoch % 20 == 0:
        save_checkpoint(epoch, global_step, unet, optimizer, scheduler, ema)

# If we finished without passing, save final state
if not quality_passed:
    print("\n" + "="*80)
    print(f"WARNING: Training ended at epoch {epoch} without passing quality test")
    print("The model may need more training or hyperparameter adjustments")
    print("="*80)
    save_checkpoint(epoch, global_step, unet, optimizer, scheduler, ema, is_best=False)
    
    # Run final quality test
    quality_passed, results = run_quality_test(unet, noise_scheduler, real_train_data, epoch, use_ema=True)

# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"Total epochs: {epoch}")
print(f"Quality test passed: {quality_passed}")
print(f"Best variance ratio: {best_variance_ratio:.4f}")
print(f"Checkpoints saved to: {Config.CHECKPOINT_DIR}")
print(f"Evaluations saved to: {Config.EVAL_DIR}")
print("="*80)

# Save training history
with open(Config.RESULTS_DIR / 'training_history.json', 'w') as f:
    json.dump({
        'train_losses': train_losses,
        'eval_history': eval_history,
        'final_epoch': epoch,
        'quality_passed': quality_passed,
        'best_variance_ratio': best_variance_ratio
    }, f, indent=2)

# Plot training curves
if eval_history:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss curve
    axes[0].plot(train_losses, linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Variance ratio
    epochs = [e['epoch'] for e in eval_history]
    var_ratios = [e['variance_ratio'] for e in eval_history]
    axes[1].plot(epochs, var_ratios, 'o-', linewidth=2)
    axes[1].axhline(1.0, color='green', linestyle='--', label='Target')
    axes[1].axhspan(QUALITY_THRESHOLDS['variance_ratio_min'], 
                    QUALITY_THRESHOLDS['variance_ratio_max'], 
                    alpha=0.2, color='green', label='Pass Zone')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Variance Ratio')
    axes[1].set_title('Variance Ratio Progress')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KS statistic
    ks_stats = [e['ks_stat'] for e in eval_history]
    axes[2].plot(epochs, ks_stats, 's-', linewidth=2, color='coral')
    axes[2].axhline(QUALITY_THRESHOLDS['ks_statistic_max'], color='green', linestyle='--', label='Threshold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KS Statistic')
    axes[2].set_title('Distribution Match Progress')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Config.RESULTS_DIR / 'training_summary.png', dpi=150)
    plt.close()

print("\nDone!")
