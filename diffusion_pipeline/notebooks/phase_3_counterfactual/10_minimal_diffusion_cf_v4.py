"""
Phase 3: Minimal Perturbation Diffusion Counterfactual V4
==========================================================

Key insight: Previous version changed too much of the signal.
This version uses GUIDED DIFFUSION with RECONSTRUCTION LOSS to ensure:
1. Counterfactuals stay close to original (high similarity)
2. Only class-discriminative features change
3. Classifier predicts target class

Approach: 
- Start from original signal (not noise)
- Add minimal noise and denoise with classifier guidance
- Use SDEdit approach: noisy_original → guided_denoising → counterfactual
"""

import os
import sys
import subprocess

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
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from scipy import signal as scipy_signal
import math
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, '/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')
from model_architecture import AFibResLSTM, ModelConfig

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
PHASE2_MODEL = PROJECT_ROOT / 'models/phase2_diffusion/diffusion_v2/best.pth'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/minimal_diffusion_v4'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

REAL_MEAN = -0.00396
REAL_STD = 0.14716
SIGNAL_LENGTH = 2500
FS = 500


# ============================================================================
# Phase 2 Model Architecture (from train_diffusion_v2.py)
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
    """Class-conditional UNet for diffusion."""
    def __init__(self, in_ch=1, model_ch=64, num_res_blocks=2,
                 attn_resolutions=(2, 3), ch_mult=(1, 2, 4, 8), num_classes=2,
                 dropout=0.1, num_groups=32, time_dim=256, class_dim=256):
        super().__init__()
        self.num_classes = num_classes
        
        # Embeddings
        self.time_embed = nn.Sequential(
            nn.Linear(model_ch, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_embed = nn.Embedding(num_classes + 1, class_dim)  # +1 for unconditional
        self.class_proj = nn.Linear(class_dim, time_dim)
        
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
            for _ in range(num_res_blocks + 1):
                skip_ch = self.skip_chs.pop()
                blocks.append(ResBlock1D(ch + skip_ch, out_ch, time_dim, num_groups, dropout))
                ch = out_ch
            self.up_blocks.append(blocks)
            self.up_attns.append(SelfAttention1D(ch) if level in attn_resolutions else None)
            self.up_samples.append(Upsample1D(ch) if level > 0 else None)
        
        self.out_norm = nn.GroupNorm(num_groups, ch)
        self.out_conv = nn.Conv1d(ch, in_ch, 3, padding=1)
    
    def forward(self, x, t, class_labels=None):
        B = x.shape[0]
        
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        
        if class_labels is None:
            class_labels = torch.full((B,), self.num_classes, device=x.device, dtype=torch.long)
        
        c_emb = self.class_proj(self.class_embed(class_labels))
        cond = t_emb + c_emb
        
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
        
        h = self.mid_block1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_block2(h, cond)
        
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
        
        h = F.silu(self.out_norm(h))
        if h.shape[2] != x.shape[2]:
            h = h[:, :, :x.shape[2]] if h.shape[2] > x.shape[2] else F.pad(h, (0, x.shape[2] - h.shape[2]))
        return self.out_conv(h)


# ============================================================================
# Classifier-Guided DDIM with Minimal Perturbation
# ============================================================================

class MinimalPerturbationDDIM:
    """
    SDEdit-style approach with classifier guidance:
    1. Add small amount of noise to original
    2. Denoise with classifier guidance toward target class
    3. Result preserves most of original while changing class features
    """
    
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
        noise = torch.randn_like(x0)
        alpha_t = self.alphas_cumprod[t]
        if alpha_t.dim() == 0:
            alpha_t = alpha_t.view(1)
        x_t = alpha_t.sqrt()[:, None, None] * x0 + (1 - alpha_t).sqrt()[:, None, None] * noise
        return x_t, noise
    
    def generate_counterfactual_cfg(self, model, x0, target_class, 
                                     noise_level=0.15, num_steps=30, 
                                     guidance_scale=5.0, preservation_weight=0.3):
        """
        Generate counterfactual using Classifier-Free Guidance (CFG).
        
        Key: Use LOW noise_level (0.1-0.3) to preserve original signal.
        
        Args:
            x0: Original signal
            target_class: Target class (0=Normal, 1=AFib)
            noise_level: Fraction of noise schedule to start from (lower = closer to original)
            guidance_scale: CFG strength
            preservation_weight: Blend factor to preserve original (0-1)
        """
        device = x0.device
        B = x0.shape[0]
        
        # Start from noisy original (not pure noise!)
        t_start = int(noise_level * self.num_timesteps)
        if t_start < 1:
            t_start = 1
        
        t_batch = torch.full((B,), t_start, device=device, dtype=torch.long)
        x, initial_noise = self.add_noise(x0, t_batch)
        
        # DDIM steps
        step_size = max(1, t_start // num_steps)
        timesteps = list(range(0, t_start, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        target_labels = torch.full((B,), target_class, device=device, dtype=torch.long)
        uncond_labels = torch.full((B,), 2, device=device, dtype=torch.long)  # Unconditional
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            with torch.no_grad():
                # CFG: Interpolate between conditional and unconditional
                noise_cond = model(x, t_batch, target_labels)
                noise_uncond = model(x, t_batch, uncond_labels)
                pred_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            
            # Predict x0
            alpha_t = self.alphas_cumprod[t]
            pred_x0 = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            
            # Preservation: Blend predicted x0 with original x0
            pred_x0 = (1 - preservation_weight) * pred_x0 + preservation_weight * x0
            
            # DDIM step
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                x = alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * pred_noise
            else:
                x = pred_x0
        
        return x
    
    def generate_counterfactual_classifier_guided(self, model, classifier, x0, target_class,
                                                    noise_level=0.2, num_steps=40,
                                                    classifier_scale=100.0, 
                                                    preservation_weight=0.3):
        """
        Generate counterfactual using external classifier gradient guidance.
        
        The gradient from classifier pushes signal toward target class.
        """
        device = x0.device
        B = x0.shape[0]
        
        t_start = int(noise_level * self.num_timesteps)
        if t_start < 1:
            t_start = 1
        
        t_batch = torch.full((B,), t_start, device=device, dtype=torch.long)
        x, _ = self.add_noise(x0, t_batch)
        
        step_size = max(1, t_start // num_steps)
        timesteps = list(range(0, t_start, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            x = x.detach().requires_grad_(True)
            
            # Get unconditional noise prediction
            with torch.no_grad():
                pred_noise = model(x, t_batch, None)
            
            # Predict x0
            alpha_t = self.alphas_cumprod[t]
            pred_x0 = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            
            # Get classifier gradient on predicted x0
            pred_x0_grad = pred_x0.detach().requires_grad_(True)
            
            # Classifier needs training mode for LSTM backward
            classifier.train()
            logits = classifier(pred_x0_grad)
            log_probs = F.log_softmax(logits, dim=1)
            target_log_prob = log_probs[:, target_class].sum()
            target_log_prob.backward()
            classifier_grad = pred_x0_grad.grad
            classifier.eval()
            
            # Apply classifier guidance
            pred_x0_guided = pred_x0 + classifier_scale * classifier_grad
            pred_x0_guided = torch.clamp(pred_x0_guided, -3, 3)
            
            # Preservation: Blend with original
            pred_x0_guided = (1 - preservation_weight) * pred_x0_guided + preservation_weight * x0
            
            # DDIM step
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                x = alpha_prev.sqrt() * pred_x0_guided + (1 - alpha_prev).sqrt() * pred_noise
            else:
                x = pred_x0_guided
            
            x = x.detach()
        
        return x


# ============================================================================
# Classifier
# ============================================================================

def load_classifier():
    classifier_path = PROJECT_ROOT / 'models/afib_reslstm_final.pth'
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(DEVICE)
    checkpoint = torch.load(classifier_path, map_location=DEVICE)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    return classifier


class ClassifierWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
    
    def forward(self, x):
        self.model.train()  # Required for LSTM backward
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        logits, _ = self.model(x_norm)
        return logits


# ============================================================================
# Clinical Feature Analysis
# ============================================================================

def detect_r_peaks(signal, fs=500):
    signal = signal.flatten()
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    peaks, props = scipy_signal.find_peaks(signal_norm, height=0.3, distance=int(0.3*fs), prominence=0.2)
    return peaks, props

def compute_rr_features(signal, fs=500):
    peaks, _ = detect_r_peaks(signal, fs)
    if len(peaks) < 2:
        return {'rr_mean': 0, 'rr_std': 0, 'rr_cv': 0, 'hr': 0, 'num_beats': 0}
    rr = np.diff(peaks) / fs * 1000  # ms
    return {
        'rr_mean': np.mean(rr),
        'rr_std': np.std(rr),
        'rr_cv': np.std(rr) / (np.mean(rr) + 1e-8),
        'hr': 60000 / np.mean(rr) if np.mean(rr) > 0 else 0,
        'num_beats': len(peaks)
    }

def check_signal_validity(signal, fs=500):
    signal = signal.flatten()
    amp_range = np.max(signal) - np.min(signal)
    peaks, _ = detect_r_peaks(signal, fs)
    
    if len(peaks) >= 2:
        rr_mean = np.mean(np.diff(peaks)) / fs
        hr = 60 / rr_mean if rr_mean > 0 else 0
    else:
        hr = 0
    
    is_valid = True
    if amp_range < 0.01 or amp_range > 10:
        is_valid = False
    if len(peaks) < 2:
        is_valid = False
    if hr > 0 and (hr < 30 or hr > 200):
        is_valid = False
    
    return {'valid': is_valid, 'hr': hr, 'num_peaks': len(peaks)}


# ============================================================================
# Visualization
# ============================================================================

def create_comparison_figure(original, counterfactual, orig_class, target_class,
                              orig_prob, cf_prob, sample_id, save_path):
    """Create detailed comparison figure."""
    
    orig_np = original.cpu().numpy().flatten()
    cf_np = counterfactual.cpu().numpy().flatten()
    
    # Convert to mV for display
    orig_mv = orig_np * REAL_STD + REAL_MEAN
    cf_mv = cf_np * REAL_STD + REAL_MEAN
    diff_mv = cf_mv - orig_mv
    
    time_ms = np.arange(len(orig_np)) / FS
    
    # Compute features
    orig_feat = compute_rr_features(orig_np, FS)
    cf_feat = compute_rr_features(cf_np, FS)
    
    corr, _ = pearsonr(orig_np, cf_np)
    
    orig_peaks, _ = detect_r_peaks(orig_np, FS)
    cf_peaks, _ = detect_r_peaks(cf_np, FS)
    
    # Create figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[1.5, 1.5, 1, 1, 1])
    
    class_names = ['Normal', 'AFib']
    
    # Row 1: Full signal overlay
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.plot(time_ms, orig_mv, 'b-', lw=1, alpha=0.8, label=f'Original ({class_names[orig_class]})')
    ax_full.plot(time_ms, cf_mv, 'r-', lw=1, alpha=0.7, label=f'Counterfactual (→{class_names[target_class]})')
    ax_full.set_xlabel('Time (s)', fontsize=11)
    ax_full.set_ylabel('Amplitude (mV)', fontsize=11)
    ax_full.set_title(f'Sample {sample_id}: Full Signal (5s) | Correlation: {corr:.4f}', fontsize=13, fontweight='bold')
    ax_full.legend(loc='upper right')
    ax_full.grid(True, alpha=0.3)
    ax_full.set_xlim([0, 5])
    
    # Row 2: Three zoomed segments (1s each)
    for col, (start, end) in enumerate([(0, 500), (1000, 1500), (2000, 2500)]):
        ax = fig.add_subplot(gs[1, col])
        t = time_ms[start:end]
        ax.plot(t, orig_mv[start:end], 'b-', lw=1.5, label='Original')
        ax.plot(t, cf_mv[start:end], 'r-', lw=1.5, alpha=0.8, label='CF')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('mV')
        ax.set_title(f'{t[0]:.1f}s - {t[-1]:.1f}s')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Difference signal
    ax_diff = fig.add_subplot(gs[1, 3])
    ax_diff.plot(time_ms, diff_mv, 'purple', lw=0.8)
    ax_diff.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax_diff.fill_between(time_ms, 0, diff_mv, alpha=0.3, color='purple')
    ax_diff.set_xlabel('Time (s)')
    ax_diff.set_ylabel('Δ mV')
    ax_diff.set_title(f'Difference (MSE: {np.mean(diff_mv**2):.6f})')
    ax_diff.grid(True, alpha=0.3)
    
    # Row 3: R-R interval comparison
    ax_rr1 = fig.add_subplot(gs[2, 0])
    if len(orig_peaks) > 1:
        orig_rr = np.diff(orig_peaks) / FS * 1000
        ax_rr1.bar(range(len(orig_rr)), orig_rr, color='blue', alpha=0.7)
        ax_rr1.axhline(y=np.mean(orig_rr), color='darkblue', linestyle='--', lw=2)
        ax_rr1.set_title(f'Original RR\nCV={orig_feat["rr_cv"]:.3f}')
    ax_rr1.set_xlabel('Beat #')
    ax_rr1.set_ylabel('RR (ms)')
    ax_rr1.grid(True, alpha=0.3)
    
    ax_rr2 = fig.add_subplot(gs[2, 1])
    if len(cf_peaks) > 1:
        cf_rr = np.diff(cf_peaks) / FS * 1000
        ax_rr2.bar(range(len(cf_rr)), cf_rr, color='red', alpha=0.7)
        ax_rr2.axhline(y=np.mean(cf_rr), color='darkred', linestyle='--', lw=2)
        ax_rr2.set_title(f'CF RR\nCV={cf_feat["rr_cv"]:.3f}')
    ax_rr2.set_xlabel('Beat #')
    ax_rr2.set_ylabel('RR (ms)')
    ax_rr2.grid(True, alpha=0.3)
    
    # Poincare plot
    ax_poincare = fig.add_subplot(gs[2, 2])
    if len(orig_peaks) > 2:
        orig_rr = np.diff(orig_peaks) / FS * 1000
        ax_poincare.scatter(orig_rr[:-1], orig_rr[1:], c='blue', alpha=0.6, s=60, label='Orig')
    if len(cf_peaks) > 2:
        cf_rr = np.diff(cf_peaks) / FS * 1000
        ax_poincare.scatter(cf_rr[:-1], cf_rr[1:], c='red', alpha=0.6, s=60, label='CF')
    ax_poincare.plot([200, 1500], [200, 1500], 'k--', alpha=0.3)
    ax_poincare.set_xlabel('RR_n (ms)')
    ax_poincare.set_ylabel('RR_{n+1} (ms)')
    ax_poincare.set_title('Poincaré Plot')
    ax_poincare.legend()
    ax_poincare.grid(True, alpha=0.3)
    
    # Power spectrum
    ax_psd = fig.add_subplot(gs[2, 3])
    f_orig, psd_orig = scipy_signal.welch(orig_np, fs=FS, nperseg=256)
    f_cf, psd_cf = scipy_signal.welch(cf_np, fs=FS, nperseg=256)
    ax_psd.semilogy(f_orig[:40], psd_orig[:40], 'b-', lw=1.5, label='Orig')
    ax_psd.semilogy(f_cf[:40], psd_cf[:40], 'r-', lw=1.5, label='CF')
    ax_psd.set_xlabel('Freq (Hz)')
    ax_psd.set_ylabel('PSD')
    ax_psd.set_title('Power Spectrum')
    ax_psd.legend()
    ax_psd.grid(True, alpha=0.3)
    
    # Row 4: R-peak aligned beats
    ax_beats = fig.add_subplot(gs[3, :2])
    window = int(0.35 * FS)  # 350ms window around peak
    
    # Plot aligned beats
    for idx, p in enumerate(orig_peaks[:6]):
        if p - window >= 0 and p + window < len(orig_np):
            beat = orig_mv[p-window:p+window]
            t_beat = np.arange(len(beat)) / FS * 1000 - window / FS * 1000
            ax_beats.plot(t_beat, beat, 'b-', alpha=0.5, lw=1)
    for idx, p in enumerate(cf_peaks[:6]):
        if p - window >= 0 and p + window < len(cf_np):
            beat = cf_mv[p-window:p+window]
            t_beat = np.arange(len(beat)) / FS * 1000 - window / FS * 1000
            ax_beats.plot(t_beat, beat, 'r-', alpha=0.5, lw=1)
    
    ax_beats.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax_beats.set_xlabel('Time from R-peak (ms)')
    ax_beats.set_ylabel('mV')
    ax_beats.set_title('Aligned Beats (Blue=Orig, Red=CF)')
    ax_beats.grid(True, alpha=0.3)
    
    # Classification probabilities
    ax_prob = fig.add_subplot(gs[3, 2:])
    categories = ['Orig\nP(N)', 'Orig\nP(AF)', 'CF\nP(N)', 'CF\nP(AF)']
    
    orig_pn = 1 - orig_prob if target_class == 1 else orig_prob
    orig_pa = orig_prob if target_class == 1 else 1 - orig_prob
    cf_pn = 1 - cf_prob if target_class == 1 else cf_prob
    cf_pa = cf_prob if target_class == 1 else 1 - cf_prob
    
    probs = [orig_pn, orig_pa, cf_pn, cf_pa]
    colors = ['lightblue', 'lightcoral', 'blue', 'red']
    bars = ax_prob.bar(categories, probs, color=colors, edgecolor='black')
    ax_prob.axhline(y=0.5, color='gray', linestyle='--', lw=2)
    ax_prob.set_ylabel('Probability')
    ax_prob.set_title('Classifier Output', fontweight='bold')
    ax_prob.set_ylim([0, 1])
    for bar, p in zip(bars, probs):
        ax_prob.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{p:.3f}', ha='center', fontweight='bold')
    
    # Row 5: Clinical summary
    ax_summary = fig.add_subplot(gs[4, :])
    ax_summary.axis('off')
    
    rr_cv_change = cf_feat['rr_cv'] - orig_feat['rr_cv']
    expected_direction = 'INCREASE' if target_class == 1 else 'DECREASE'
    actual_direction = 'INCREASED' if rr_cv_change > 0 else 'DECREASED'
    correct_direction = (target_class == 1 and rr_cv_change > 0) or (target_class == 0 and rr_cv_change < 0)
    
    summary = f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                    CLINICAL SUMMARY                                        ║
    ╠═══════════════════════════════════════════════════════════════════════════════════════════╣
    ║  Signal Correlation:     {corr:.4f}  {'✓ HIGH (>0.7)' if corr > 0.7 else '✗ LOW (<0.7)':35}                      ║
    ║  Classifier Flip:        P({class_names[target_class]}) {orig_prob if target_class==1 else 1-orig_prob:.3f} → {cf_prob if target_class==1 else 1-cf_prob:.3f}  {'✓ FLIPPED' if (cf_prob if target_class==1 else 1-cf_prob) > 0.5 else '✗ NOT FLIPPED':30}   ║
    ║  RR CV Change:           {orig_feat['rr_cv']:.3f} → {cf_feat['rr_cv']:.3f} ({'+' if rr_cv_change > 0 else ''}{rr_cv_change:.3f})  Expected: {expected_direction} | Actual: {actual_direction} {'✓' if correct_direction else '✗'}   ║
    ║  Heart Rate:             {orig_feat['hr']:.1f} → {cf_feat['hr']:.1f} bpm                                                    ║
    ║  Valid Signal:           {'✓ YES' if check_signal_validity(cf_np, FS)['valid'] else '✗ NO':50}                             ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════════╝
    """
    ax_summary.text(0.02, 0.5, summary, fontfamily='monospace', fontsize=10, va='center', transform=ax_summary.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'correlation': corr,
        'orig_rr_cv': orig_feat['rr_cv'],
        'cf_rr_cv': cf_feat['rr_cv'],
        'rr_cv_change': rr_cv_change,
        'correct_direction': correct_direction,
        'valid': check_signal_validity(cf_np, FS)['valid'],
        'flipped': (cf_prob > 0.5) if target_class == 1 else (cf_prob < 0.5),
        'orig_prob': orig_prob,
        'cf_prob': cf_prob
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("MINIMAL PERTURBATION DIFFUSION COUNTERFACTUAL V4")
    print("="*70)
    print(f"Results: {RESULTS_DIR}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load diffusion model
    print("\nLoading Phase 2 diffusion model...")
    if not PHASE2_MODEL.exists():
        print(f"ERROR: Model not found at {PHASE2_MODEL}")
        return
    
    unet = ECGUNet(
        in_ch=1, model_ch=64, num_res_blocks=2,
        attn_resolutions=(2, 3), ch_mult=(1, 2, 4, 8),
        num_classes=2, dropout=0.1
    ).to(DEVICE)
    
    checkpoint = torch.load(PHASE2_MODEL, map_location=DEVICE)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    print(f"  Loaded from: {PHASE2_MODEL}")
    
    # Load classifier
    print("\nLoading classifier...")
    raw_classifier = load_classifier()
    classifier = ClassifierWrapper(raw_classifier)
    
    # Verify classifier
    val_data = np.load(DATA_DIR / 'val_data.npz')
    val_signals = torch.tensor(val_data['X'], dtype=torch.float32)
    val_labels = torch.tensor(val_data['y'], dtype=torch.long)
    if val_signals.dim() == 2:
        val_signals = val_signals.unsqueeze(1)
    
    with torch.no_grad():
        test_batch = val_signals[:200].to(DEVICE)
        test_labels_batch = val_labels[:200].to(DEVICE)
        preds = classifier(test_batch).argmax(dim=1)
        acc = (preds == test_labels_batch).float().mean().item() * 100
        print(f"  Classifier accuracy on val: {acc:.1f}%")
    
    # Initialize scheduler
    scheduler = MinimalPerturbationDDIM(num_timesteps=1000, device=DEVICE)
    
    # Separate by class
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    print(f"\nVal data: {len(normal_idx)} Normal, {len(afib_idx)} AFib")
    
    # ========================================================================
    # Hyperparameter sweep to find optimal settings
    # ========================================================================
    print("\n" + "="*70)
    print("FINDING OPTIMAL HYPERPARAMETERS")
    print("="*70)
    
    # Test different noise levels and preservation weights
    test_configs = [
        # (noise_level, preservation_weight, guidance_scale)
        (0.10, 0.5, 5.0),
        (0.15, 0.4, 5.0),
        (0.20, 0.3, 5.0),
        (0.25, 0.2, 7.5),
        (0.30, 0.2, 7.5),
    ]
    
    best_config = None
    best_score = 0
    
    print("\nTesting configurations on 5 samples each direction...")
    
    for noise_level, preservation, guidance in test_configs:
        n2a_flips = 0
        n2a_corrs = []
        a2n_flips = 0
        a2n_corrs = []
        
        # Test Normal -> AFib
        for i in range(5):
            x = val_signals[normal_idx[i]:normal_idx[i]+1].to(DEVICE)
            with torch.no_grad():
                orig_prob = F.softmax(classifier(x), dim=1)[0, 1].item()
            
            cf = scheduler.generate_counterfactual_cfg(
                unet, x, target_class=1,
                noise_level=noise_level, num_steps=30,
                guidance_scale=guidance, preservation_weight=preservation
            )
            
            with torch.no_grad():
                cf_prob = F.softmax(classifier(cf), dim=1)[0, 1].item()
            
            if cf_prob > 0.5:
                n2a_flips += 1
            
            corr, _ = pearsonr(x[0, 0].cpu().numpy(), cf[0, 0].cpu().numpy())
            n2a_corrs.append(corr)
        
        # Test AFib -> Normal
        for i in range(5):
            x = val_signals[afib_idx[i]:afib_idx[i]+1].to(DEVICE)
            with torch.no_grad():
                orig_prob = F.softmax(classifier(x), dim=1)[0, 0].item()
            
            cf = scheduler.generate_counterfactual_cfg(
                unet, x, target_class=0,
                noise_level=noise_level, num_steps=30,
                guidance_scale=guidance, preservation_weight=preservation
            )
            
            with torch.no_grad():
                cf_prob = F.softmax(classifier(cf), dim=1)[0, 0].item()
            
            if cf_prob > 0.5:
                a2n_flips += 1
            
            corr, _ = pearsonr(x[0, 0].cpu().numpy(), cf[0, 0].cpu().numpy())
            a2n_corrs.append(corr)
        
        flip_rate = (n2a_flips + a2n_flips) / 10
        mean_corr = np.mean(n2a_corrs + a2n_corrs)
        
        # Combined score: balance between flip rate and correlation
        score = flip_rate * 0.5 + (mean_corr if mean_corr > 0 else 0) * 0.5
        
        print(f"  noise={noise_level:.2f}, pres={preservation:.1f}, guide={guidance:.1f} -> "
              f"flip={flip_rate*100:.0f}%, corr={mean_corr:.3f}, score={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_config = (noise_level, preservation, guidance)
    
    print(f"\nBest config: noise={best_config[0]}, preservation={best_config[1]}, guidance={best_config[2]}")
    
    # ========================================================================
    # Generate counterfactuals with best config
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING COUNTERFACTUALS WITH OPTIMAL CONFIG")
    print("="*70)
    
    noise_level, preservation_weight, guidance_scale = best_config
    num_samples = 10
    
    results = {
        'config': {'noise_level': noise_level, 'preservation': preservation_weight, 'guidance': guidance_scale},
        'normal_to_afib': [],
        'afib_to_normal': []
    }
    
    # Normal → AFib
    print("\n--- Normal → AFib ---")
    for i in tqdm(range(num_samples), desc="Normal→AFib"):
        x = val_signals[normal_idx[i]:normal_idx[i]+1].to(DEVICE)
        
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 1].item()
        
        cf = scheduler.generate_counterfactual_cfg(
            unet, x, target_class=1,
            noise_level=noise_level, num_steps=50,
            guidance_scale=guidance_scale, preservation_weight=preservation_weight
        )
        
        with torch.no_grad():
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 1].item()
        
        save_path = RESULTS_DIR / f'normal_to_afib_{i+1:02d}.png'
        result = create_comparison_figure(x[0, 0], cf[0, 0], 0, 1, orig_prob, cf_prob, i+1, save_path)
        results['normal_to_afib'].append(result)
        
        print(f"  #{i+1}: P(AFib) {orig_prob:.3f}→{cf_prob:.3f} | Corr: {result['correlation']:.3f} | "
              f"RR CV: {result['orig_rr_cv']:.3f}→{result['cf_rr_cv']:.3f}")
    
    # AFib → Normal
    print("\n--- AFib → Normal ---")
    for i in tqdm(range(num_samples), desc="AFib→Normal"):
        x = val_signals[afib_idx[i]:afib_idx[i]+1].to(DEVICE)
        
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 0].item()
        
        cf = scheduler.generate_counterfactual_cfg(
            unet, x, target_class=0,
            noise_level=noise_level, num_steps=50,
            guidance_scale=guidance_scale, preservation_weight=preservation_weight
        )
        
        with torch.no_grad():
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 0].item()
        
        save_path = RESULTS_DIR / f'afib_to_normal_{i+1:02d}.png'
        result = create_comparison_figure(x[0, 0], cf[0, 0], 1, 0, orig_prob, cf_prob, i+1, save_path)
        results['afib_to_normal'].append(result)
        
        print(f"  #{i+1}: P(Normal) {orig_prob:.3f}→{cf_prob:.3f} | Corr: {result['correlation']:.3f} | "
              f"RR CV: {result['orig_rr_cv']:.3f}→{result['cf_rr_cv']:.3f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    n2a = results['normal_to_afib']
    a2n = results['afib_to_normal']
    
    n2a_valid = sum(1 for r in n2a if r['valid'])
    n2a_flip = sum(1 for r in n2a if r['flipped'])
    n2a_corr = np.mean([r['correlation'] for r in n2a])
    n2a_correct_rr = sum(1 for r in n2a if r['correct_direction'])
    
    a2n_valid = sum(1 for r in a2n if r['valid'])
    a2n_flip = sum(1 for r in a2n if r['flipped'])
    a2n_corr = np.mean([r['correlation'] for r in a2n])
    a2n_correct_rr = sum(1 for r in a2n if r['correct_direction'])
    
    print(f"\nNormal → AFib ({num_samples} samples):")
    print(f"  Valid signals:      {n2a_valid}/{num_samples} ({100*n2a_valid/num_samples:.0f}%)")
    print(f"  Flip rate:          {n2a_flip}/{num_samples} ({100*n2a_flip/num_samples:.0f}%)")
    print(f"  Mean correlation:   {n2a_corr:.4f}")
    print(f"  Correct RR change:  {n2a_correct_rr}/{num_samples} ({100*n2a_correct_rr/num_samples:.0f}%)")
    
    print(f"\nAFib → Normal ({num_samples} samples):")
    print(f"  Valid signals:      {a2n_valid}/{num_samples} ({100*a2n_valid/num_samples:.0f}%)")
    print(f"  Flip rate:          {a2n_flip}/{num_samples} ({100*a2n_flip/num_samples:.0f}%)")
    print(f"  Mean correlation:   {a2n_corr:.4f}")
    print(f"  Correct RR change:  {a2n_correct_rr}/{num_samples} ({100*a2n_correct_rr/num_samples:.0f}%)")
    
    # Save results
    summary = {
        'config': results['config'],
        'normal_to_afib': {
            'valid_rate': n2a_valid / num_samples,
            'flip_rate': n2a_flip / num_samples,
            'mean_correlation': float(n2a_corr),
            'correct_rr_direction': n2a_correct_rr / num_samples,
        },
        'afib_to_normal': {
            'valid_rate': a2n_valid / num_samples,
            'flip_rate': a2n_flip / num_samples,
            'mean_correlation': float(a2n_corr),
            'correct_rr_direction': a2n_correct_rr / num_samples,
        }
    }
    
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Test case summary
    print("\n" + "="*70)
    print("TEST CASE EVALUATION")
    print("="*70)
    
    # Test 1: Realistic signals
    all_valid = n2a_valid + a2n_valid
    print(f"\n1. Realistic signals (valid): {all_valid}/{2*num_samples} ({100*all_valid/(2*num_samples):.0f}%)")
    print(f"   {'✓ PASS' if all_valid >= 0.9 * 2 * num_samples else '✗ FAIL'}")
    
    # Test 2: Preserve original + only change class features
    corr_threshold = 0.7
    good_corr = sum(1 for r in n2a + a2n if r['correlation'] > corr_threshold)
    print(f"\n2. High similarity (corr > {corr_threshold}): {good_corr}/{2*num_samples} ({100*good_corr/(2*num_samples):.0f}%)")
    print(f"   Mean correlation: {(n2a_corr + a2n_corr) / 2:.4f}")
    print(f"   {'✓ PASS' if (n2a_corr + a2n_corr) / 2 > 0.7 else '✗ NEEDS IMPROVEMENT'}")
    
    # Test 3: Flip classifier
    all_flipped = n2a_flip + a2n_flip
    print(f"\n3. Classifier flip rate: {all_flipped}/{2*num_samples} ({100*all_flipped/(2*num_samples):.0f}%)")
    print(f"   {'✓ PASS' if all_flipped >= 0.8 * 2 * num_samples else '✗ FAIL'}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("="*70)


if __name__ == '__main__':
    main()
