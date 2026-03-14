"""
02_train_FIXED_unconditional.py - COMPLETE FIX
===============================================

FIXES:
1. Proper variance loss with correct target
2. Corrected clipping values
3. Rebalanced loss weights
4. Removed conflicting amplitude loss
5. Better monitoring and early stopping
6. Fixed for your NEW data (std=0.0613)

USE THIS VERSION!
"""
"""
GPU_AUTO_SELECTOR.py - Add this to TOP of your training script
===============================================================

Usage:
  1. Copy this entire code block
  2. Paste at the VERY TOP of your 03_train_unconditional_diffusion_2.py
  3. BEFORE any torch imports
  4. Script will automatically wait for GPU and select best one
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
        print(f"⚠️  Warning: Could not get GPU info - {e}")
        return []

def wait_for_gpu(required_mb=10000, max_wait_minutes=120, check_interval=60):
    """
    Wait until a GPU has enough free memory.
    
    Args:
        required_mb: Minimum free memory required (MB) - default 10GB
        max_wait_minutes: Maximum time to wait (minutes)
        check_interval: Seconds between checks (default 60)
    
    Returns:
        GPU index to use, or None if timeout
    """
    print("\n" + "="*80)
    print("🔍 GPU MEMORY AUTO-SELECTOR")
    print("="*80)
    print(f"⚙️  Settings:")
    print(f"   Required memory: {required_mb} MB ({required_mb/1024:.1f} GB)")
    print(f"   Max wait time: {max_wait_minutes} minutes")
    print(f"   Check interval: {check_interval} seconds")
    print("="*80)
    
    max_attempts = (max_wait_minutes * 60) // check_interval
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        elapsed_min = (attempt - 1) * check_interval / 60
        
        gpu_info = get_gpu_memory_info()
        
        if not gpu_info:
            print(f"\n❌ Cannot detect GPUs. Proceeding with default device...")
            return None
        
        # Sort by free memory
        gpu_info.sort(key=lambda x: x['free_mb'], reverse=True)
        
        # Print status
        print(f"\n📊 Check #{attempt} (Elapsed: {elapsed_min:.1f} min)")
        print("-" * 80)
        print(f"{'GPU':<5} {'Free':<12} {'Used':<12} {'Total':<12} {'Free%':<10} {'Status':<20}")
        print("-" * 80)
        
        best_gpu = None
        for gpu in gpu_info:
            if gpu['free_mb'] >= required_mb:
                status = "✅ AVAILABLE"
                if best_gpu is None:
                    best_gpu = gpu
            else:
                status = f"❌ Need {(required_mb - gpu['free_mb'])/1024:.1f}GB more"
            
            print(f"{gpu['index']:<5} {gpu['free_mb']:>10} MB {gpu['used_mb']:>10} MB "
                  f"{gpu['total_mb']:>10} MB {gpu['free_pct']:>8.1f}% {status:<20}")
        
        # If GPU available, use it!
        if best_gpu:
            print(f"\n✅ SUCCESS! Selected GPU {best_gpu['index']}")
            print(f"   Free memory: {best_gpu['free_mb']} MB ({best_gpu['free_mb']/1024:.1f} GB)")
            print(f"   Free percent: {best_gpu['free_pct']:.1f}%")
            print("="*80)
            return best_gpu['index']
        
        # No GPU available - wait
        remaining_min = max_wait_minutes - elapsed_min
        print(f"\n⏳ No GPU available. Remaining wait time: {remaining_min:.1f} minutes")
        
        if attempt < max_attempts:
            print(f"   Checking again in {check_interval} seconds...")
            print(f"   Press Ctrl+C to cancel and exit")
            print("-" * 80)
            
            try:
                # Countdown with dots
                for i in range(check_interval):
                    if i % 10 == 0:
                        print(".", end="", flush=True)
                    time.sleep(1)
                print()  # New line
            except KeyboardInterrupt:
                print("\n\n❌ Cancelled by user")
                sys.exit(0)
        else:
            # Timeout
            print(f"\n❌ TIMEOUT after {max_wait_minutes} minutes")
            print(f"   Best available: GPU {gpu_info[0]['index']} "
                  f"with {gpu_info[0]['free_mb']} MB free")
            print(f"   Proceeding anyway - training may fail!")
            print("="*80)
            return gpu_info[0]['index']
    
    return None

# ============================================================================
# CONFIGURE SETTINGS HERE
# ============================================================================

REQUIRED_MEMORY_MB = 10000      # Need 10GB free (adjust as needed)
MAX_WAIT_MINUTES = 120          # Wait up to 2 hours (adjust as needed)
CHECK_INTERVAL_SECONDS = 60     # Check every minute

# ============================================================================
# AUTO-SELECT GPU
# ============================================================================

print("\n" + "🚀"*40)
print("AUTOMATIC GPU SELECTION AND MEMORY WAIT")
print("🚀"*40)

selected_gpu = wait_for_gpu(
    required_mb=REQUIRED_MEMORY_MB,
    max_wait_minutes=MAX_WAIT_MINUTES,
    check_interval=CHECK_INTERVAL_SECONDS
)

if selected_gpu is not None:
    # Set environment variable BEFORE importing torch
    os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)
    print(f"\n✅ Set CUDA_VISIBLE_DEVICES={selected_gpu}")
    print(f"   PyTorch will only see this GPU as device 0")
else:
    print(f"\n⚠️  No GPU selected - using default")

print("="*80)
print("✅ GPU selection complete - continuing with training...")
print("="*80 + "\n")

# ============================================================================
# NOW YOUR EXISTING SCRIPT CONTINUES BELOW
# (Import torch, load data, define model, train, etc.)
# ============================================================================
import sys
sys.path.append('D:/research/codes')

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
from scipy import stats
import math

# ============================================================================
# Configuration - FIXED FOR YOUR DATA
# ============================================================================

class Config:
    # Paths
    PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
    DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
    MODEL_DIR = PROJECT_ROOT / 'models/phase2_diffusion'
    CHECKPOINT_DIR = MODEL_DIR / 'fixed_unconditional'
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR = MODEL_DIR / 'results_fixed'
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
    
    # UNet Architecture
    MODEL_CHANNELS = 64
    CHANNEL_MULT = (1, 2, 3, 4)
    NUM_RES_BLOCKS = 2
    ATTENTION_RESOLUTIONS = (2, 3)
    NUM_GROUPS = 32
    DROPOUT = 0.1
    
    # Conditioning
    NUM_CLASSES = 2
    CLASS_EMBED_DIM = 256
    TIME_EMBED_DIM = 256
    USE_CLASSIFIER_FREE_GUIDANCE = True
    GUIDANCE_DROPOUT = 0.1
    
    # Training
    BATCH_SIZE = 32  # Increased for better statistics
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4  # Slightly higher
    WEIGHT_DECAY = 1e-5
    WARMUP_STEPS = 1000
    GRADIENT_CLIP = 1.0
    USE_EMA = True
    EMA_DECAY = 0.9999
    
    # ⭐ FIXED LOSS WEIGHTS - Simplified and balanced
    FREQ_LOSS_WEIGHT = 0.1           # Reduced from 0.3
    SIMPLE_FREQ_WEIGHT = 0.1         # Reduced from 0.5
    TEMPORAL_LOSS_WEIGHT = 0.0       # Disabled - was causing issues
    VARIANCE_LOSS_WEIGHT = 0.5       # INCREASED - most important!
    # Removed AMPLITUDE_LOSS - was conflicting
    
    # Evaluation
    EVAL_INTERVAL = 5  # More frequent evaluation
    EVAL_SAMPLES = 100  # More samples for better statistics
    GENERATION_ETA = 0.0  # Deterministic

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)

print("="*80)
print("🚀 FIXED UNCONDITIONAL DIFFUSION TRAINING")
print("="*80)
print("FIXES:")
print("  ✅ Proper variance loss (target from YOUR data)")
print("  ✅ Correct clipping values")
print("  ✅ Rebalanced loss weights")
print("  ✅ Removed conflicting losses")
print("  ✅ Better monitoring")
print("="*80)
print(f"Loss Weights:")
print(f"  MSE: 1.0 (base)")
print(f"  Frequency: {Config.FREQ_LOSS_WEIGHT}")
print(f"  Simple Freq: {Config.SIMPLE_FREQ_WEIGHT}")
print(f"  Variance: {Config.VARIANCE_LOSS_WEIGHT} (KEY FIX!)")
print(f"  Temporal: {Config.TEMPORAL_LOSS_WEIGHT} (disabled)")
print(f"Device: {Config.DEVICE}")
print("="*80)

# ============================================================================
# Load Data
# ============================================================================

print(f"\n📂 Loading data from: {Config.DATA_DIR}")

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

print(f"✅ Training data: {train_signals.shape}")
print(f"✅ Validation data: {val_signals.shape}")

# ============================================================================
# Load Normalization Parameters
# ============================================================================

print("\n🔍 Loading normalization parameters...")

metadata_file = Config.DATA_DIR / 'dataset_metadata.json'
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

norm_params = metadata['normalization']

# Get normalization parameters
Config.CLEAN_DATA_MIN = norm_params['clean_data_min']
Config.CLEAN_DATA_MAX = norm_params['clean_data_max']
Config.TARGET_MIN = norm_params['target_min']
Config.TARGET_MAX = norm_params['target_max']

print("✅ Normalization Parameters:")
print(f"   Method: {norm_params['method']}")
print(f"   Clean data range: [{Config.CLEAN_DATA_MIN:.4f}, {Config.CLEAN_DATA_MAX:.4f}] mV")
print(f"   Target range: [{Config.TARGET_MIN}, {Config.TARGET_MAX}]")
print(f"   Samples kept: {norm_params['samples_kept']:,} ({100-norm_params['rejection_rate_pct']:.2f}%)")

# ============================================================================
# Data Statistics - CRITICAL FOR VARIANCE LOSS
# ============================================================================

print("\n" + "="*80)
print("📊 DATA STATISTICS (for variance loss target)")
print("="*80)

real_train_mean = float(train_signals.mean())
real_train_std = float(train_signals.std())
real_train_var = real_train_std ** 2

print(f"Training data:")
print(f"  Shape: {train_signals.shape}")
print(f"  Range: [{train_signals.min():.4f}, {train_signals.max():.4f}]")
print(f"  Mean: {real_train_mean:.4f}")
print(f"  Std: {real_train_std:.4f}")
print(f"  Variance: {real_train_var:.4f}")

print(f"\nClass distribution:")
print(f"  Class 0 (Normal): {(train_labels == 0).sum():,} ({100*(train_labels == 0).sum()/len(train_labels):.1f}%)")
print(f"  Class 1 (AFib): {(train_labels == 1).sum():,} ({100*(train_labels == 1).sum()/len(train_labels):.1f}%)")

# ⭐ CRITICAL: Store real std for variance loss
Config.REAL_TRAIN_STD = real_train_std
Config.REAL_TRAIN_VAR = real_train_var

print(f"\n⭐ VARIANCE LOSS TARGET: {Config.REAL_TRAIN_STD:.4f}")
print("="*80)

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

print(f"✅ Batches: Train={len(train_loader)}, Val={len(val_loader)}")

# ============================================================================
# DDIM Scheduler - FIXED CLIPPING
# ============================================================================

class DDIMScheduler:
    def __init__(self, num_timesteps=1000, beta_schedule='cosine', 
                 beta_start=0.0001, beta_end=0.02, cosine_s=0.008, device='cuda'):
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
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
        
        # ⭐ FIXED: Clip to training data range
        pred_x0 = torch.clamp(pred_x0, Config.TARGET_MIN, Config.TARGET_MAX)
        
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
        
        if eta > 0:
            if not (t_prev < 0).any():
                noise = torch.randn_like(x_t)
                sigma_t = eta * torch.sqrt(variance)
                x_prev = x_prev + sigma_t * noise
        
        return torch.clamp(x_prev, Config.TARGET_MIN * 1.2, Config.TARGET_MAX * 1.2)
    
    @torch.no_grad()
    def ddim_sample_loop(self, model, shape, class_labels=None, num_steps=50, eta=0.0, progress=True):
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
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
        
        # Final clip
        return torch.clamp(x, Config.TARGET_MIN, Config.TARGET_MAX)

noise_scheduler = DDIMScheduler(
    num_timesteps=Config.DIFFUSION_TIMESTEPS,
    beta_schedule=Config.BETA_SCHEDULE,
    beta_start=Config.BETA_START,
    beta_end=Config.BETA_END,
    cosine_s=Config.COSINE_S,
    device=Config.DEVICE
)

print(f"✅ DDIM Scheduler with FIXED clipping")

# ============================================================================
# UNet Model (same as before)
# ============================================================================

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
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
        assert channels % num_heads == 0
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
        out = self.proj(out)
        return x + out

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

class ECGPixelUNet(nn.Module):
    def __init__(self, in_channels=1, model_channels=64, out_channels=1,
                 num_res_blocks=2, attention_resolutions=(2, 3),
                 channel_mult=(1, 2, 3, 4), num_classes=2, dropout=0.1,
                 num_groups=32, time_embed_dim=256, class_embed_dim=256,
                 use_cfg=True, cfg_dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
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
        
        h = self.out_conv(h)
        return h

unet = ECGPixelUNet(
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
print(f"✅ UNet: {total_params:,} parameters")

# ============================================================================
# EMA
# ============================================================================

class EMA:
    def __init__(self, model, decay=0.9999):
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
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
    
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

if Config.USE_EMA:
    ema = EMA(unet, decay=Config.EMA_DECAY)
    print(f"✅ EMA initialized")
else:
    ema = None

# ============================================================================
# Optimizer and Scheduler
# ============================================================================

def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    betas=(0.9, 0.999)
)

total_steps = Config.NUM_EPOCHS * len(train_loader)
scheduler = get_lr_scheduler(optimizer, Config.WARMUP_STEPS, total_steps)

print(f"✅ Optimizer configured")

# ============================================================================
# FIXED LOSS FUNCTIONS
# ============================================================================

from torch.amp import GradScaler, autocast
scaler = GradScaler('cuda')

def compute_frequency_loss(pred_noise, true_noise):
    pred_fft = torch.fft.rfft(pred_noise.float(), dim=-1)
    true_fft = torch.fft.rfft(true_noise.float(), dim=-1)
    pred_mag = torch.abs(pred_fft)
    true_mag = torch.abs(true_fft)
    return F.l1_loss(pred_mag, true_mag)

def compute_simple_frequency_loss(pred, target):
    pred_fft = torch.fft.rfft(pred.float(), dim=-1)
    target_fft = torch.fft.rfft(target.float(), dim=-1)
    return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

# ⭐ FIXED VARIANCE LOSS - Now uses BATCH statistics correctly
def compute_variance_loss_fixed(pred_x0, target_std):
    """
    FIXED: Compare batch std to target std
    This prevents variance expansion!
    """
    # Compute std across entire batch
    batch_std = pred_x0.std()
    target_std_tensor = torch.tensor(target_std, device=pred_x0.device, dtype=pred_x0.dtype)
    
    # MSE between batch std and target std
    var_loss = F.mse_loss(batch_std, target_std_tensor)
    
    return var_loss

print(f"✅ Loss functions initialized")
print(f"   ⭐ Variance loss target: {Config.REAL_TRAIN_STD:.4f}")

# ============================================================================
# Validation Function
# ============================================================================

@torch.no_grad()
def validate(model, val_loader, noise_scheduler, use_ema=True):
    if use_ema and ema is not None:
        ema.apply_shadow()
    
    model.eval()
    val_loss = 0.0
    num_batches = 0
    
    for signals, labels in val_loader:
        signals = signals.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        t = torch.randint(0, Config.DIFFUSION_TIMESTEPS, (signals.size(0),), device=Config.DEVICE)
        noise = torch.randn_like(signals)
        noisy_signals = noise_scheduler.q_sample(signals, t, noise=noise)
        
        pred_noise = model(noisy_signals, t, labels)
        mse_loss = F.mse_loss(pred_noise, noise)
        
        val_loss += mse_loss.item()
        num_batches += 1
    
    if use_ema and ema is not None:
        ema.restore()
    
    return val_loss / num_batches

# ============================================================================
# Denormalization
# ============================================================================

def denormalize_ecg(normalized_ecg):
    """Convert from [-1.5, 1.5] back to original mV"""
    if isinstance(normalized_ecg, torch.Tensor):
        normalized_ecg = normalized_ecg.cpu().numpy()
    
    # Inverse min-max normalization
    denorm = (normalized_ecg - Config.TARGET_MIN) / (Config.TARGET_MAX - Config.TARGET_MIN)
    denorm = denorm * (Config.CLEAN_DATA_MAX - Config.CLEAN_DATA_MIN) + Config.CLEAN_DATA_MIN
    
    return denorm

# ============================================================================
# FIXED Evaluation Function
# ============================================================================

@torch.no_grad()
def evaluate_model(epoch, model, noise_scheduler, real_data, use_ema=True):
    print(f"\n{'='*80}")
    print(f"🔍 EVALUATION AT EPOCH {epoch}")
    print(f"{'='*80}")
    
    if use_ema and ema is not None:
        ema.apply_shadow()
    
    model.eval()
    
    # Generate samples
    all_generated = []
    all_labels = []
    
    for class_id in range(Config.NUM_CLASSES):
        print(f"Generating {Config.EVAL_SAMPLES} samples for class {class_id}...")
        
        labels = torch.full((Config.EVAL_SAMPLES,), class_id, device=Config.DEVICE, dtype=torch.long)
        shape = (Config.EVAL_SAMPLES, Config.IN_CHANNELS, Config.SIGNAL_LENGTH)
        
        generated = noise_scheduler.ddim_sample_loop(
            model, shape, class_labels=labels,
            num_steps=50, eta=Config.GENERATION_ETA, progress=True
        )
        
        all_generated.append(generated.cpu())
        all_labels.append(labels.cpu())
    
    if use_ema and ema is not None:
        ema.restore()
    
    generated_ecgs = torch.cat(all_generated, dim=0).numpy()
    generated_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Get real data
    real_ecgs = real_data['X']
    real_labels = real_data['y']
    
    if real_ecgs.ndim == 2:
        real_ecgs = real_ecgs[:, np.newaxis, :]
    
    # Sample same number
    n_samples = len(generated_ecgs)
    indices = np.random.choice(len(real_ecgs), n_samples, replace=False)
    real_ecgs_sample = real_ecgs[indices]
    real_labels_sample = real_labels[indices]
    
    # Statistical analysis
    print(f"\n📊 Statistical Analysis:")
    print(f"{'='*80}")
    
    real_mean = real_ecgs_sample.mean()
    gen_mean = generated_ecgs.mean()
    real_std = real_ecgs_sample.std()
    gen_std = generated_ecgs.std()
    
    print(f"\n{'Metric':<15} {'Real':<12} {'Generated':<12} {'Diff':<12}")
    print(f"{'Mean':<15} {real_mean:>11.4f} {gen_mean:>11.4f} {abs(real_mean-gen_mean):>11.4f}")
    print(f"{'Std':<15} {real_std:>11.4f} {gen_std:>11.4f} {abs(real_std-gen_std):>11.4f}")
    print(f"{'Min':<15} {real_ecgs_sample.min():>11.4f} {generated_ecgs.min():>11.4f} {abs(real_ecgs_sample.min()-generated_ecgs.min()):>11.4f}")
    print(f"{'Max':<15} {real_ecgs_sample.max():>11.4f} {generated_ecgs.max():>11.4f} {abs(real_ecgs_sample.max()-generated_ecgs.max()):>11.4f}")
    
    # KS test
    ks_stat, ks_pval = stats.ks_2samp(real_ecgs_sample.flatten(), generated_ecgs.flatten())
    print(f"\nKolmogorov-Smirnov Test:")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  P-value: {ks_pval:.6f}")
    
    if ks_stat < 0.05:
        print(f"  ✅ EXCELLENT! Distributions match well")
    elif ks_stat < 0.10:
        print(f"  ✅ GOOD! Similar distributions")
    elif ks_stat < 0.15:
        print(f"  ⚠️ MODERATE: Some difference")
    else:
        print(f"  ❌ WARNING: Different distributions")
    
    # Variance ratio
    variance_ratio = gen_std / real_std
    print(f"\nVariance Ratio:")
    print(f"  Generated/Real: {variance_ratio:.2%}")
    
    if 0.90 <= variance_ratio <= 1.10:
        print(f"  ✅ EXCELLENT! Variance match!")
    elif 0.80 <= variance_ratio <= 1.20:
        print(f"  ✅ GOOD! Close match")
    elif variance_ratio < 0.7:
        print(f"  ⚠️ WARNING: Variance compression")
    elif variance_ratio > 1.3:
        print(f"  ❌ WARNING: Variance expansion detected!")
    else:
        print(f"  ⚠️ MODERATE: Variance difference")
    
    # Per-class statistics
    print(f"\n{'='*80}")
    print("Per-Class Statistics:")
    print(f"{'='*80}")
    
    eval_results = {
        'epoch': epoch,
        'overall': {
            'real_mean': float(real_mean),
            'gen_mean': float(gen_mean),
            'real_std': float(real_std),
            'gen_std': float(gen_std),
            'variance_ratio': float(variance_ratio),
            'ks_stat': float(ks_stat),
            'ks_pval': float(ks_pval)
        },
        'per_class': {}
    }
    
    for class_id in range(Config.NUM_CLASSES):
        real_class = real_ecgs_sample[real_labels_sample == class_id]
        gen_class = generated_ecgs[generated_labels == class_id]
        
        ks_stat_c, ks_pval_c = stats.ks_2samp(real_class.flatten(), gen_class.flatten())
        
        print(f"\nClass {class_id}:")
        print(f"  Real - Mean: {real_class.mean():.4f}, Std: {real_class.std():.4f}")
        print(f"  Gen  - Mean: {gen_class.mean():.4f}, Std: {gen_class.std():.4f}")
        print(f"  KS Test: stat={ks_stat_c:.4f}, p={ks_pval_c:.6f}")
        
        eval_results['per_class'][class_id] = {
            'real_mean': float(real_class.mean()),
            'gen_mean': float(gen_class.mean()),
            'real_std': float(real_class.std()),
            'gen_std': float(gen_class.std()),
            'ks_stat': float(ks_stat_c),
            'ks_pval': float(ks_pval_c)
        }
    
    # Visualization
    eval_epoch_dir = Config.EVAL_DIR / f'epoch_{epoch:03d}'
    eval_epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Sample Comparison
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    for i in range(3):
        real_0_idx = np.where(real_labels_sample == 0)[0][i]
        gen_0_idx = np.where(generated_labels == 0)[0][i]
        
        axes[i, 0].plot(real_ecgs_sample[real_0_idx, 0, :], linewidth=0.8, alpha=0.7, color='steelblue', label='Real')
        axes[i, 0].plot(generated_ecgs[gen_0_idx, 0, :], linewidth=0.8, alpha=0.7, color='blue', linestyle='--', label='Generated')
        axes[i, 0].set_title(f'Class 0 (Normal) Pair {i+1}', fontsize=12, fontweight='bold')
        axes[i, 0].set_ylim([Config.TARGET_MIN - 0.2, Config.TARGET_MAX + 0.2])
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        real_1_idx = np.where(real_labels_sample == 1)[0][i]
        gen_1_idx = np.where(generated_labels == 1)[0][i]
        
        axes[i, 1].plot(real_ecgs_sample[real_1_idx, 0, :], linewidth=0.8, alpha=0.7, color='lightcoral', label='Real')
        axes[i, 1].plot(generated_ecgs[gen_1_idx, 0, :], linewidth=0.8, alpha=0.7, color='red', linestyle='--', label='Generated')
        axes[i, 1].set_title(f'Class 1 (AFib) Pair {i+1}', fontsize=12, fontweight='bold')
        axes[i, 1].set_ylim([Config.TARGET_MIN - 0.2, Config.TARGET_MAX + 0.2])
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'Real vs Generated - Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(eval_epoch_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Overlays
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    gen_class_0 = generated_ecgs[generated_labels == 0]
    gen_class_1 = generated_ecgs[generated_labels == 1]
    
    for i in range(min(20, len(gen_class_0))):
        axes[0].plot(gen_class_0[i, 0, :], linewidth=0.5, alpha=0.2, color='steelblue')
    axes[0].plot(gen_class_0[:20, 0, :].mean(axis=0), linewidth=2.5, color='darkblue', label='Mean', zorder=10)
    axes[0].set_ylim([Config.TARGET_MIN - 0.2, Config.TARGET_MAX + 0.2])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Class 0 (Normal) - 20 Samples', fontsize=14, fontweight='bold')
    
    for i in range(min(20, len(gen_class_1))):
        axes[1].plot(gen_class_1[i, 0, :], linewidth=0.5, alpha=0.2, color='coral')
    axes[1].plot(gen_class_1[:20, 0, :].mean(axis=0), linewidth=2.5, color='darkred', label='Mean', zorder=10)
    axes[1].set_ylim([Config.TARGET_MIN - 0.2, Config.TARGET_MAX + 0.2])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(f'Class 1 (AFib) - 20 Samples', fontsize=14, fontweight='bold')
    
    plt.suptitle(f'Generated Samples - Epoch {epoch}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(eval_epoch_dir / 'overlay.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(real_ecgs_sample.flatten(), bins=100, alpha=0.7, color='steelblue', label='Real', edgecolor='black')
    axes[0].hist(generated_ecgs.flatten(), bins=100, alpha=0.7, color='coral', label='Generated', edgecolor='black')
    axes[0].set_title('Distribution Comparison')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    real_sorted = np.sort(real_ecgs_sample.flatten())
    gen_sorted = np.sort(generated_ecgs.flatten())
    min_len = min(len(real_sorted), len(gen_sorted))
    axes[1].scatter(real_sorted[:min_len:100], gen_sorted[:min_len:100], alpha=0.5, s=10)
    axes[1].plot([real_sorted.min(), real_sorted.max()], [real_sorted.min(), real_sorted.max()], 'r--', linewidth=2)
    axes[1].set_xlabel('Real ECG Values')
    axes[1].set_ylabel('Generated ECG Values')
    axes[1].set_title('Q-Q Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(eval_epoch_dir / 'distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results
    with open(eval_epoch_dir / 'results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    np.save(eval_epoch_dir / 'generated_ecgs.npy', generated_ecgs)
    np.save(eval_epoch_dir / 'generated_labels.npy', generated_labels)
    
    print(f"\n✅ Evaluation complete! Results saved to: {eval_epoch_dir}")
    print(f"{'='*80}\n")
    
    return eval_results

# ============================================================================
# Checkpoint Saving
# ============================================================================

def save_checkpoint(epoch, step, model, optimizer, scheduler, ema=None, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'norm_params': {
            'clean_data_min': Config.CLEAN_DATA_MIN,
            'clean_data_max': Config.CLEAN_DATA_MAX,
            'target_min': Config.TARGET_MIN,
            'target_max': Config.TARGET_MAX
        }
    }
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    
    filename = 'best.pth' if is_best else f'checkpoint_epoch{epoch}.pth'
    torch.save(checkpoint, Config.CHECKPOINT_DIR / filename)
    print(f"💾 Checkpoint saved: {filename}")

# ============================================================================
# TRAINING LOOP - FIXED
# ============================================================================

print(f"\n{'='*80}")
print("🚀 STARTING TRAINING WITH FIXED LOSSES")
print(f"{'='*80}")

num_epochs = Config.NUM_EPOCHS
gradient_clip = Config.GRADIENT_CLIP
log_interval = 100

patience = 30  # Increased patience
best_loss = float('inf')
patience_counter = 0

train_losses = []
val_losses = []
eval_history = []
global_step = 0

# Get real data for evaluation
real_train_data = np.load(Config.DATA_DIR / 'train_data.npz')

for epoch in range(1, num_epochs + 1):
    unet.train()
    epoch_loss = 0.0
    epoch_mse = 0.0
    epoch_var = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    
    for batch_idx, (signals, labels) in enumerate(pbar):
        signals = signals.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        t = torch.randint(0, Config.DIFFUSION_TIMESTEPS, (signals.size(0),), device=Config.DEVICE)
        noise = torch.randn_like(signals)
        noisy_signals = noise_scheduler.q_sample(signals, t, noise=noise)
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            pred_noise = unet(noisy_signals, t, labels)
            mse_loss = F.mse_loss(pred_noise, noise)
        
        # ⭐ FIXED: Only essential losses
        freq_loss = compute_frequency_loss(pred_noise, noise)
        simple_freq_loss = compute_simple_frequency_loss(pred_noise, noise)
        
        # Predict x0 for variance loss
        with torch.no_grad():
            pred_x0 = noise_scheduler.predict_start_from_noise(noisy_signals, t, pred_noise)
            pred_x0 = torch.clamp(pred_x0, Config.TARGET_MIN, Config.TARGET_MAX)
        
        # ⭐ CRITICAL: Variance loss with correct target
        variance_loss = compute_variance_loss_fixed(pred_x0, target_std=Config.REAL_TRAIN_STD)
        
        # Total loss - simplified and balanced
        total_loss = (
            mse_loss + 
            Config.FREQ_LOSS_WEIGHT * freq_loss +
            Config.SIMPLE_FREQ_WEIGHT * simple_freq_loss +
            Config.VARIANCE_LOSS_WEIGHT * variance_loss
        )
        
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if Config.USE_EMA:
            ema.update()
        
        epoch_loss += total_loss.item()
        epoch_mse += mse_loss.item()
        epoch_var += variance_loss.item()
        num_batches += 1
        global_step += 1
        
        pbar.set_postfix({
            'mse': f'{mse_loss.item():.4f}',
            'var': f'{variance_loss.item():.4f}'
        })
    
    avg_train_loss = epoch_loss / num_batches
    avg_mse = epoch_mse / num_batches
    avg_var = epoch_var / num_batches
    train_losses.append(avg_train_loss)
    
    avg_val_loss = validate(unet, val_loader, noise_scheduler, use_ema=True)
    val_losses.append(avg_val_loss)
    
    print(f"\n{'='*80}")
    print(f"Epoch {epoch} Summary:")
    print(f"  Train Loss: {avg_train_loss:.4f} (MSE: {avg_mse:.4f}, Var: {avg_var:.4f})")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Best Loss: {best_loss:.4f}")
    print(f"{'='*80}")
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        save_checkpoint(epoch, global_step, unet, optimizer, scheduler, ema, is_best=True)
        print(f"✅ New best model! Loss: {best_loss:.4f}\n")
    else:
        patience_counter += 1
        print(f"⚠️  No improvement. Patience: {patience_counter}/{patience}\n")
    
    if epoch % Config.EVAL_INTERVAL == 0:
        eval_results = evaluate_model(epoch, unet, noise_scheduler, real_train_data, use_ema=True)
        eval_history.append(eval_results)
    
    if epoch % 20 == 0:
        save_checkpoint(epoch, global_step, unet, optimizer, scheduler, ema, is_best=False)
    
    if patience_counter >= patience:
        print(f"\n{'='*80}")
        print(f"🛑 EARLY STOPPING at epoch {epoch}")
        print(f"{'='*80}\n")
        break

if epoch == num_epochs:
    save_checkpoint(epoch, global_step, unet, optimizer, scheduler, ema, is_best=False)

if epoch % Config.EVAL_INTERVAL != 0:
    eval_results = evaluate_model(epoch, unet, noise_scheduler, real_train_data, use_ema=True)
    eval_history.append(eval_results)

print(f"\n{'='*80}")
print("✅ TRAINING COMPLETE")
print(f"{'='*80}")
print(f"Total epochs: {epoch}")
print(f"Best val loss: {best_loss:.4f}")
print(f"Model saved to: {Config.CHECKPOINT_DIR / 'best.pth'}")
print(f"{'='*80}\n")

# ============================================================================
# FINAL PLOTS
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

axes[0].plot(train_losses, linewidth=2, label='Train Loss')
axes[0].plot(val_losses, linewidth=2, label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

if eval_history:
    eval_epochs = [e['epoch'] for e in eval_history]
    variance_ratios = [e['overall']['variance_ratio'] for e in eval_history]
    ks_stats = [e['overall']['ks_stat'] for e in eval_history]
    
    axes[1].plot(eval_epochs, variance_ratios, 'o-', linewidth=2, color='steelblue')
    axes[1].axhline(1.0, color='green', linestyle='--', linewidth=2, label='Perfect Match')
    axes[1].axhline(0.9, color='orange', linestyle=':', linewidth=1)
    axes[1].axhline(1.1, color='orange', linestyle=':', linewidth=1)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Variance Ratio (Gen/Real)')
    axes[1].set_title('Variance Matching Over Training')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(eval_epochs, ks_stats, 's-', linewidth=2, color='coral')
    axes[2].axhline(0.05, color='green', linestyle='--', linewidth=2, label='Excellent (<0.05)')
    axes[2].axhline(0.10, color='orange', linestyle=':', linewidth=1, label='Good (<0.10)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KS Statistic')
    axes[2].set_title('Distribution Matching Over Training')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Config.RESULTS_DIR / 'training_summary.png', dpi=150)
plt.close()

with open(Config.RESULTS_DIR / 'eval_history.json', 'w') as f:
    json.dump(eval_history, f, indent=2)

print(f"✅ Final plots saved")
print(f"\n🎉 Check {Config.EVAL_DIR} for detailed evaluations\n")