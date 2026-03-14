"""
Phase 3: Classifier-Free Guided Counterfactual Generation
==========================================================

The Phase 2 model is already class-conditional with CFG support.
We can directly generate counterfactuals by:
1. Starting from the original ECG (add partial noise)
2. Conditioning on the OPPOSITE class during denoising
3. Using high guidance scale to push towards target class

Test Cases:
- Content(Original) + Style(Original) → Original ECG (condition on original class)
- Content(Original) + Style(Other Class) → Counterfactual (condition on opposite class)
"""

import os
import sys
import subprocess

# Auto-select GPU
def get_free_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        gpu_info = [(int(l.split(',')[0]), int(l.split(',')[1])) for l in lines if l.strip()]
        if gpu_info:
            best_gpu = max(gpu_info, key=lambda x: x[1])
            return str(best_gpu[0])
    except:
        pass
    return '0'

os.environ['CUDA_VISIBLE_DEVICES'] = get_free_gpu()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import signal
import math

# Add models path to import pre-trained classifier
sys.path.insert(0, '/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')
from model_architecture import AFibResLSTM, ModelConfig

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
PHASE2_MODEL_DIR = PROJECT_ROOT / 'models/phase2_diffusion/diffusion_v2'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/guided_counterfactual'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pre-trained classifier path (from 05_train_model.ipynb)
PRETRAINED_CLASSIFIER_PATH = PROJECT_ROOT / 'models/afib_reslstm_final.pth'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Data normalization constants
REAL_MEAN = -0.00396
REAL_STD = 0.14716

# ============================================================================
# Phase 2 Model Architecture (exact copy from train_diffusion_v2.py)
# ============================================================================

def get_timestep_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, num_groups=32, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.cond_proj = nn.Linear(cond_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
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

class Downsample1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, stride=2, padding=1)
    def forward(self, x):
        return self.conv(x)

class Upsample1D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=1)
    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='linear', align_corners=False))

class ECGUNet(nn.Module):
    def __init__(self, in_ch=1, model_ch=64, out_ch=1, num_res_blocks=2,
                 attn_resolutions=(2, 3), ch_mult=(1, 2, 4, 8), num_classes=2,
                 dropout=0.1, num_groups=32, time_dim=256, class_dim=256,
                 use_cfg=True, cfg_dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        
        # Embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(model_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, class_dim)
        self.class_proj = nn.Linear(class_dim, time_dim)
        
        # Input
        self.input_proj = nn.Conv1d(in_ch, model_ch, 3, padding=1)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        ch = model_ch
        self.skip_chs = [ch]
        
        for level, mult in enumerate(ch_mult):
            out_ch = model_ch * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock1D(ch, out_ch, time_dim, num_groups, dropout))
                ch = out_ch
                self.skip_chs.append(ch)
            self.down_blocks.append(blocks)
            self.down_attns.append(SelfAttention1D(ch) if level in attn_resolutions else None)
            self.down_samples.append(Downsample1D(ch) if level < len(ch_mult) - 1 else None)
            if level < len(ch_mult) - 1:
                self.skip_chs.append(ch)
        
        # Middle
        self.mid_block1 = ResBlock1D(ch, ch, time_dim, num_groups, dropout)
        self.mid_attn = SelfAttention1D(ch)
        self.mid_block2 = ResBlock1D(ch, ch, time_dim, num_groups, dropout)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for level in reversed(range(len(ch_mult))):
            out_ch = model_ch * ch_mult[level]
            blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = self.skip_chs.pop()
                blocks.append(ResBlock1D(ch + skip_ch, out_ch, time_dim, num_groups, dropout))
                ch = out_ch
            self.up_blocks.append(blocks)
            self.up_attns.append(SelfAttention1D(ch) if level in attn_resolutions else None)
            self.up_samples.append(Upsample1D(ch) if level > 0 else None)
        
        # Output
        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_conv = nn.Conv1d(ch, in_ch, 3, padding=1)
    
    def forward(self, x, t, class_labels=None):
        B = x.shape[0]
        
        # Time embedding
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        
        # Class embedding
        if class_labels is None:
            class_labels = torch.full((B,), self.num_classes, device=x.device, dtype=torch.long)
        
        c_emb = self.class_proj(self.class_embed(class_labels))
        cond = t_emb + c_emb
        
        # Encoder
        h = self.input_proj(x)
        skips = [h]
        
        for blocks, attn, down in zip(self.down_blocks, self.down_attns, self.down_samples):
            for block in blocks:
                h = block(h, cond)
                skips.append(h)
            if attn:
                h = attn(h)
            if down:
                h = down(h)
                skips.append(h)
        
        # Middle
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)
        
        # Decoder
        for blocks, attn, up in zip(self.up_blocks, self.up_attns, self.up_samples):
            for block in blocks:
                skip = skips.pop()
                if h.shape[2] != skip.shape[2]:
                    skip = skip[:, :, :h.shape[2]] if skip.shape[2] > h.shape[2] else F.pad(skip, (0, h.shape[2] - skip.shape[2]))
                h = block(torch.cat([h, skip], dim=1), cond)
            if attn:
                h = attn(h)
            if up:
                h = up(h)
        
        # Output
        h = F.silu(self.out_norm(h))
        if h.shape[2] != x.shape[2]:
            h = h[:, :, :x.shape[2]] if h.shape[2] > x.shape[2] else F.pad(h, (0, x.shape[2] - h.shape[2]))
        return self.out_conv(h)

# ============================================================================
# DDIM Scheduler with Classifier-Free Guidance
# ============================================================================

class CFGDDIMScheduler:
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
    
    def add_noise(self, x0, t):
        """Add noise to x0 at timestep t."""
        alpha_t = self.alphas_cumprod[t]
        if alpha_t.dim() == 0:
            alpha_t = alpha_t.unsqueeze(0)
        noise = torch.randn_like(x0)
        x_t = torch.sqrt(alpha_t)[:, None, None] * x0 + torch.sqrt(1 - alpha_t)[:, None, None] * noise
        return x_t, noise
    
    @torch.no_grad()
    def sample_with_cfg(self, model, x0, target_class, noise_level=0.5, num_steps=50, guidance_scale=7.5):
        """
        Generate counterfactual using Classifier-Free Guidance.
        
        Args:
            model: Class-conditional diffusion model
            x0: Original ECG
            target_class: Target class (0=Normal, 1=AFib)
            noise_level: How much noise to add (0-1)
            num_steps: Number of denoising steps
            guidance_scale: CFG scale (higher = stronger conditioning)
        """
        device = x0.device
        B = x0.shape[0]
        
        # Add noise to original
        t_start = int(noise_level * self.num_timesteps)
        if t_start == 0:
            return x0.clone()
        
        t_batch = torch.full((B,), t_start, device=device, dtype=torch.long)
        x_noisy, _ = self.add_noise(x0, t_batch)
        
        # Denoise with CFG
        step_size = max(1, t_start // num_steps)
        timesteps = list(range(0, t_start, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        target_labels = torch.full((B,), target_class, device=device, dtype=torch.long)
        uncond_labels = torch.full((B,), 2, device=device, dtype=torch.long)  # 2 = unconditional
        
        x = x_noisy
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Conditional prediction
            noise_cond = model(x, t_batch, target_labels)
            
            # Unconditional prediction
            noise_uncond = model(x, t_batch, uncond_labels)
            
            # CFG: noise = uncond + scale * (cond - uncond)
            pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            
            # DDIM step
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
    
    def sample_with_classifier_guidance(self, model, classifier, x0, target_class, 
                                         noise_level=0.5, num_steps=50, guidance_scale=100.0):
        """
        Generate counterfactual using CLASSIFIER GRADIENT guidance.
        
        Uses gradients from the classifier to push generation towards target class.
        This is different from CFG - it uses an external classifier.
        """
        device = x0.device
        B = x0.shape[0]
        
        # Add noise to original
        t_start = int(noise_level * self.num_timesteps)
        if t_start == 0:
            return x0.clone()
        
        t_batch = torch.full((B,), t_start, device=device, dtype=torch.long)
        x_noisy, _ = self.add_noise(x0, t_batch)
        
        # Denoise with classifier guidance
        step_size = max(1, t_start // num_steps)
        timesteps = list(range(0, t_start, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        uncond_labels = torch.full((B,), 2, device=device, dtype=torch.long)
        
        x = x_noisy.clone()
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            x = x.detach().requires_grad_(True)
            
            # Get noise prediction
            with torch.no_grad():
                pred_noise = model(x, t_batch, uncond_labels)
            
            # Predict x0
            alpha_t = self.alphas_cumprod[t]
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            
            # Get classifier gradient on predicted x0
            # Need to temporarily set classifier to train mode for LSTM backward
            classifier.train()
            pred_x0_grad = pred_x0.detach().requires_grad_(True)
            logits = classifier(pred_x0_grad)
            log_probs = F.log_softmax(logits, dim=1)
            target_log_prob = log_probs[:, target_class].sum()
            target_log_prob.backward()
            classifier_grad = pred_x0_grad.grad
            classifier.eval()
            
            # Apply classifier guidance
            pred_x0_guided = pred_x0 + guidance_scale * classifier_grad
            pred_x0_guided = torch.clamp(pred_x0_guided, -3, 3)
            
            # DDIM step
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                x = torch.sqrt(alpha_prev) * pred_x0_guided + torch.sqrt(1 - alpha_prev) * pred_noise
            else:
                x = pred_x0_guided
            
            x = x.detach()
        
        return x

# ============================================================================
# ECG Classifier for Validation (Using Pre-trained AFibResLSTM)
# ============================================================================

def load_pretrained_classifier():
    """Load the pre-trained AFibResLSTM classifier from 05_train_model.ipynb."""
    print("\nLoading pre-trained AFibResLSTM classifier...")
    
    if not PRETRAINED_CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Pre-trained classifier not found at {PRETRAINED_CLASSIFIER_PATH}")
    
    # Create model with default config
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(DEVICE)
    
    # Load pre-trained weights
    checkpoint = torch.load(PRETRAINED_CLASSIFIER_PATH, map_location=DEVICE)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    
    classifier.eval()
    print(f"  Loaded from: {PRETRAINED_CLASSIFIER_PATH}")
    
    return classifier

class ClassifierWrapper(nn.Module):
    """Wrapper to make AFibResLSTM interface consistent with simple classifier.
    
    IMPORTANT: The classifier was trained with per-sample z-score normalization,
    so we need to apply the same normalization here.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # Apply per-sample z-score normalization (as used in training)
        # x shape: (batch, 1, seq_len)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x_normalized = (x - mean) / std
        
        # AFibResLSTM returns (logits, attention_weights)
        logits, _ = self.model(x_normalized)
        return logits

# ============================================================================
# Visualization
# ============================================================================

def to_millivolts(normalized_signal):
    return normalized_signal * REAL_STD + REAL_MEAN

def analyze_rr_intervals(ecg_signal, fs=500):
    ecg = ecg_signal.flatten()
    ecg_norm = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-8)
    height = 0.5
    distance = int(0.3 * fs)
    peaks, _ = signal.find_peaks(ecg_norm, height=height, distance=distance)
    
    if len(peaks) < 2:
        return {'rr_mean': 0, 'rr_std': 0, 'rr_irregularity': 0, 'num_beats': 0}
    
    rr_intervals = np.diff(peaks) / fs * 1000
    return {
        'rr_mean': np.mean(rr_intervals),
        'rr_std': np.std(rr_intervals),
        'rr_irregularity': np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0,
        'num_beats': len(peaks)
    }

# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("CLASSIFIER-FREE GUIDED COUNTERFACTUAL GENERATION")
    print("="*70)
    
    # Load Phase 2 model
    print("\nLoading Phase 2 diffusion model...")
    model_path = PHASE2_MODEL_DIR / 'best.pth'
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return
    
    # Create model with same config as training
    unet = ECGUNet(
        in_ch=1,
        model_ch=64,
        out_ch=1,
        num_res_blocks=2,
        attn_resolutions=(2, 3),
        ch_mult=(1, 2, 4, 8),
        num_classes=2,
        dropout=0.1,
        num_groups=32,
        time_dim=256,
        class_dim=256,
        use_cfg=True,
        cfg_dropout=0.1
    ).to(DEVICE)
    
    checkpoint = torch.load(model_path, map_location=DEVICE)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    print(f"  Loaded from: {model_path}")
    
    # Load pre-trained classifier (AFibResLSTM from 05_train_model.ipynb)
    raw_classifier = load_pretrained_classifier()
    classifier = ClassifierWrapper(raw_classifier)
    classifier.eval()
    
    # Load data
    val_data = np.load(DATA_DIR / 'val_data.npz')
    val_signals = torch.tensor(val_data['X'], dtype=torch.float32)
    val_labels = torch.tensor(val_data['y'], dtype=torch.long)
    
    if val_signals.dim() == 2:
        val_signals = val_signals.unsqueeze(1)
    
    # Validate classifier
    print("\nValidating pre-trained classifier...")
    with torch.no_grad():
        test_batch = val_signals[:200].to(DEVICE)
        test_labels = val_labels[:200].to(DEVICE)
        preds = classifier(test_batch).argmax(dim=1)
        acc = (preds == test_labels).float().mean().item() * 100
        print(f"  Validation accuracy: {acc:.1f}%")
    
    if acc < 70:
        print(f"  WARNING: Classifier accuracy is lower than expected ({acc:.1f}%), but using pre-trained model anyway.")
    else:
        print(f"  Classifier is working well with {acc:.1f}% accuracy.")
    
    # Initialize scheduler
    scheduler = CFGDDIMScheduler(1000, DEVICE)
    
    # Separate classes
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    
    print(f"\nNormal samples: {len(normal_idx)}")
    print(f"AFib samples: {len(afib_idx)}")
    
    time_axis = np.arange(val_signals.shape[-1]) / 500
    
    # ========================================================================
    # TEST 1: Reconstruction (condition on original class)
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 1: RECONSTRUCTION")
    print("Content(Original) + Style(Original) → Should match Original")
    print("="*70)
    
    num_test = 10
    noise_level = 0.02  # Minimal noise for best reconstruction
    
    with torch.no_grad():
        # Mix of normal and afib
        test_idx = torch.cat([normal_idx[:5], afib_idx[:5]])
        test_signals = val_signals[test_idx].to(DEVICE)
        test_labels_actual = val_labels[test_idx]
        
        # Reconstruct by conditioning on the SAME class
        reconstructed = []
        for i in range(num_test):
            orig_class = test_labels_actual[i].item()
            recon = scheduler.sample_with_cfg(
                unet, test_signals[i:i+1], target_class=orig_class,
                noise_level=noise_level, num_steps=50, guidance_scale=3.0
            )
            reconstructed.append(recon)
        reconstructed = torch.cat(reconstructed, dim=0)
        
        # Metrics
        mse = F.mse_loss(reconstructed, test_signals).item()
        
        correlations = []
        for i in range(num_test):
            orig = test_signals[i, 0].cpu().numpy()
            recon = reconstructed[i, 0].cpu().numpy()
            corr, _ = pearsonr(orig, recon)
            correlations.append(corr)
        
        mean_corr = np.mean(correlations)
        
        print(f"\nNoise level: {noise_level}")
        print(f"MSE: {mse:.6f}")
        print(f"Mean Correlation: {mean_corr:.4f}")
        print(f"Correlations: {[f'{c:.3f}' for c in correlations]}")
        
        recon_pass = mean_corr > 0.7
        print(f"\n{'PASS' if recon_pass else 'FAIL'}: Correlation > 0.7")
    
    # Visualization
    fig, axes = plt.subplots(5, 2, figsize=(18, 20))
    fig.suptitle(f'TEST 1: Reconstruction (noise={noise_level}) - Clinical Range (mV)', 
                 fontsize=16, fontweight='bold')
    
    for i in range(5):
        orig_mv = to_millivolts(test_signals[i, 0].cpu().numpy())
        recon_mv = to_millivolts(reconstructed[i, 0].cpu().numpy())
        
        axes[i, 0].plot(time_axis, orig_mv, 'b-', linewidth=1.2, alpha=0.8, label='Original')
        axes[i, 0].plot(time_axis, recon_mv, 'g-', linewidth=1.2, alpha=0.7, label='Reconstructed')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_title(f'Sample {i+1} (Class={test_labels_actual[i].item()}): Corr={correlations[i]:.3f}')
        
        axes[i, 1].plot(time_axis[:1000], orig_mv[:1000], 'b-', linewidth=1.5, alpha=0.8, label='Original')
        axes[i, 1].plot(time_axis[:1000], recon_mv[:1000], 'g-', linewidth=1.5, alpha=0.7, label='Reconstructed')
        axes[i, 1].set_xlabel('Time (s)')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_title(f'Zoomed (0-2s)')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test1_reconstruction.png', dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'test1_reconstruction.png'}")
    
    # ========================================================================
    # TEST 2: Counterfactual Normal → AFib
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 2: COUNTERFACTUAL (Normal → AFib)")
    print("="*70)
    
    num_cf = min(20, len(normal_idx))
    guidance_scale = 7.5
    noise_level = 0.5
    
    normal_ecgs = val_signals[normal_idx[:num_cf]].to(DEVICE)
    
    print(f"\nGenerating counterfactuals (guidance={guidance_scale}, noise={noise_level})...")
    
    with torch.no_grad():
        # Generate counterfactual by conditioning on AFib (class 1)
        cf_normal_to_afib = scheduler.sample_with_cfg(
            unet, normal_ecgs, target_class=1,
            noise_level=noise_level, num_steps=50, guidance_scale=guidance_scale
        )
        
        # Classify
        orig_logits = classifier(normal_ecgs)
        cf_logits = classifier(cf_normal_to_afib)
        
        orig_preds = orig_logits.argmax(dim=1)
        cf_preds = cf_logits.argmax(dim=1)
        
        orig_probs = F.softmax(orig_logits, dim=1)
        cf_probs = F.softmax(cf_logits, dim=1)
    
    originally_normal = (orig_preds == 0).sum().item()
    flipped_to_afib = ((orig_preds == 0) & (cf_preds == 1)).sum().item()
    
    print(f"\nOriginal predictions: {orig_preds.cpu().numpy()}")
    print(f"Counterfactual predictions: {cf_preds.cpu().numpy()}")
    print(f"\nOriginally Normal: {originally_normal}/{num_cf}")
    print(f"Flipped to AFib: {flipped_to_afib}/{max(1,originally_normal)} ({100*flipped_to_afib/max(1,originally_normal):.1f}%)")
    print(f"\nMean AFib prob: Original={orig_probs[:,1].mean():.3f}, CF={cf_probs[:,1].mean():.3f}")
    
    # Visualization
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    fig.suptitle(f'TEST 2: Normal→AFib (guidance={guidance_scale}, noise={noise_level}) - mV', 
                 fontsize=16, fontweight='bold')
    
    for i in range(5):
        orig_mv = to_millivolts(normal_ecgs[i, 0].cpu().numpy())
        cf_mv = to_millivolts(cf_normal_to_afib[i, 0].cpu().numpy())
        
        axes[i, 0].plot(time_axis, orig_mv, 'b-', linewidth=1)
        axes[i, 0].set_title(f'Original (P(AFib)={orig_probs[i,1]:.2f})')
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(time_axis, cf_mv, 'r-', linewidth=1)
        axes[i, 1].set_title(f'Counterfactual (P(AFib)={cf_probs[i,1]:.2f})')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(time_axis, orig_mv, 'b-', linewidth=1, alpha=0.7, label='Original')
        axes[i, 2].plot(time_axis, cf_mv, 'r-', linewidth=1, alpha=0.7, label='Counterfactual')
        axes[i, 2].fill_between(time_axis, orig_mv, cf_mv, alpha=0.3, color='purple')
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].set_title(f'Overlay ({orig_preds[i].item()}→{cf_preds[i].item()})')
        axes[i, 2].set_ylabel('mV')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test2_normal_to_afib.png', dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'test2_normal_to_afib.png'}")
    
    # ========================================================================
    # TEST 3: Counterfactual AFib → Normal (using Classifier Gradient Guidance)
    # ========================================================================
    print("\n" + "="*70)
    print("TEST 3: COUNTERFACTUAL (AFib → Normal)")
    print("Using CLASSIFIER GRADIENT GUIDANCE")
    print("="*70)
    
    afib_ecgs = val_signals[afib_idx[:num_cf]].to(DEVICE)
    
    # Use classifier gradient guidance - push towards Normal using classifier gradients
    classifier_guidance_scale = 50.0
    afib_to_normal_noise = 0.5
    
    print(f"\nGenerating counterfactuals using classifier gradient guidance...")
    print(f"  (scale={classifier_guidance_scale}, noise={afib_to_normal_noise})")
    
    # Need to use the raw classifier for gradient computation
    cf_afib_to_normal = scheduler.sample_with_classifier_guidance(
        unet, classifier, afib_ecgs, target_class=0,  # 0=Normal
        noise_level=afib_to_normal_noise, num_steps=50, guidance_scale=classifier_guidance_scale
    )
    
    with torch.no_grad():
        orig_logits = classifier(afib_ecgs)
        cf_logits = classifier(cf_afib_to_normal)
        
        orig_preds = orig_logits.argmax(dim=1)
        cf_preds = cf_logits.argmax(dim=1)
        
        orig_probs = F.softmax(orig_logits, dim=1)
        cf_probs = F.softmax(cf_logits, dim=1)
    
    originally_afib = (orig_preds == 1).sum().item()
    flipped_to_normal = ((orig_preds == 1) & (cf_preds == 0)).sum().item()
    
    print(f"\nOriginal predictions: {orig_preds.cpu().numpy()}")
    print(f"Counterfactual predictions: {cf_preds.cpu().numpy()}")
    print(f"\nOriginally AFib: {originally_afib}/{num_cf}")
    print(f"Flipped to Normal: {flipped_to_normal}/{max(1,originally_afib)} ({100*flipped_to_normal/max(1,originally_afib):.1f}%)")
    print(f"\nMean Normal prob: Original={orig_probs[:,0].mean():.3f}, CF={cf_probs[:,0].mean():.3f}")
    
    # Visualization
    fig, axes = plt.subplots(5, 3, figsize=(20, 20))
    fig.suptitle(f'TEST 3: AFib→Normal (Classifier Gradient, scale={classifier_guidance_scale}, noise={afib_to_normal_noise}) - mV', 
                 fontsize=16, fontweight='bold')
    
    for i in range(5):
        orig_mv = to_millivolts(afib_ecgs[i, 0].cpu().numpy())
        cf_mv = to_millivolts(cf_afib_to_normal[i, 0].cpu().numpy())
        
        axes[i, 0].plot(time_axis, orig_mv, 'r-', linewidth=1)
        axes[i, 0].set_title(f'Original AFib (P(Normal)={orig_probs[i,0]:.2f})')
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(time_axis, cf_mv, 'b-', linewidth=1)
        axes[i, 1].set_title(f'Counterfactual (P(Normal)={cf_probs[i,0]:.2f})')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(time_axis, orig_mv, 'r-', linewidth=1, alpha=0.7, label='Original AFib')
        axes[i, 2].plot(time_axis, cf_mv, 'b-', linewidth=1, alpha=0.7, label='Counterfactual')
        axes[i, 2].fill_between(time_axis, orig_mv, cf_mv, alpha=0.3, color='purple')
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].set_title(f'Overlay ({orig_preds[i].item()}→{cf_preds[i].item()})')
        axes[i, 2].set_ylabel('mV')
        axes[i, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'test3_afib_to_normal.png', dpi=150)
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'test3_afib_to_normal.png'}")
    
    # ========================================================================
    # Clinical Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("CLINICAL ANALYSIS: RR Intervals")
    print("="*70)
    
    print("\nNormal → AFib (expect: increased RR irregularity):")
    for i in range(min(5, num_cf)):
        orig_rr = analyze_rr_intervals(normal_ecgs[i, 0].cpu().numpy())
        cf_rr = analyze_rr_intervals(cf_normal_to_afib[i, 0].cpu().numpy())
        change = cf_rr['rr_irregularity'] - orig_rr['rr_irregularity']
        print(f"  Sample {i+1}: {orig_rr['rr_irregularity']:.3f} → {cf_rr['rr_irregularity']:.3f} ({'+' if change > 0 else ''}{change:.3f})")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nReconstruction: Mean Corr = {mean_corr:.4f}")
    print(f"Normal→AFib Flip Rate: {100*flipped_to_afib/max(1,originally_normal):.1f}%")
    print(f"AFib→Normal Flip Rate: {100*flipped_to_normal/max(1,originally_afib):.1f}%")
    print(f"\nResults: {RESULTS_DIR}")
    print("="*70)

if __name__ == '__main__':
    main()
