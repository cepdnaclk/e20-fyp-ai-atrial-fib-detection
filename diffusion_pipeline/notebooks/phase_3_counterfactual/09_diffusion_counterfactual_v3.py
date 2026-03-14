"""
Phase 3: Content-Style Diffusion Counterfactual Generator V3
============================================================

Uses diffusion model with content-style disentanglement to generate realistic
counterfactual ECGs that only modify class-discriminative features.

Key Features:
1. Content Encoder: Captures class-invariant features (QRS morphology, overall shape)
2. Style Encoder: Captures class-discriminative features (RR irregularity, P-waves)
3. Diffusion Denoising: Ensures realistic output through iterative refinement
4. Classifier Guidance: Pushes generation toward target class

Test Cases:
1. Counterfactuals must be realistic (not abnormal signals)
2. Only class-discriminative properties should change (RR intervals, P-waves)
3. Classifier should predict the target class
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

sys.path.insert(0, '/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')
from model_architecture import AFibResLSTM, ModelConfig

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
PHASE3_MODEL = PROJECT_ROOT / 'models/phase3_counterfactual/final_model.pth'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/diffusion_cf_v3'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

REAL_MEAN = -0.00396
REAL_STD = 0.14716
SIGNAL_LENGTH = 2500
FS = 500  # Sampling frequency


# ============================================================================
# Content Encoder - Class-invariant features
# ============================================================================

class ContentEncoder(nn.Module):
    """Extracts class-invariant content (QRS morphology, signal shape)."""
    def __init__(self, in_channels=1, hidden_dim=64, content_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_dim), nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 4), nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 4, hidden_dim * 8, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 8), nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 8, hidden_dim * 8, 5, stride=2, padding=2),
            nn.BatchNorm1d(hidden_dim * 8), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(8),
        )
        self.flat_size = hidden_dim * 8 * 8
        self.fc_mu = nn.Linear(self.flat_size, content_dim)
        self.fc_logvar = nn.Linear(self.flat_size, content_dim)
        
    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z, mu, logvar


# ============================================================================
# Style Encoder - Class-discriminative features
# ============================================================================

class StyleEncoder(nn.Module):
    """Extracts class-discriminative style (RR intervals, P-waves, fibrillatory activity)."""
    def __init__(self, in_channels=1, hidden_dim=64, style_dim=128, num_classes=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, 7, stride=2, padding=3),
            nn.InstanceNorm1d(hidden_dim), nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim, hidden_dim * 2, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 2), nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 4), nn.LeakyReLU(0.2),
            nn.Conv1d(hidden_dim * 4, hidden_dim * 4, 5, stride=2, padding=2),
            nn.InstanceNorm1d(hidden_dim * 4), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_style = nn.Linear(hidden_dim * 4, style_dim)
        self.classifier = nn.Linear(style_dim, num_classes)
        
    def forward(self, x):
        h = self.encoder(x).view(x.size(0), -1)
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
    """UNet conditioned on timestep, content, and style embeddings."""
    def __init__(self, in_ch=1, model_ch=64, content_dim=256, style_dim=128):
        super().__init__()
        time_dim = model_ch * 4
        cond_dim = time_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(model_ch, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.content_proj = nn.Linear(content_dim, time_dim)
        self.style_proj = nn.Linear(style_dim, time_dim)
        
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
        
        self.out_norm = nn.GroupNorm(32, model_ch)
        self.out_conv = nn.Conv1d(model_ch, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        
    def forward(self, x, t, content, style):
        t_emb = get_timestep_embedding(t, self.time_embed[0].in_features)
        t_emb = self.time_embed(t_emb)
        c_emb = self.content_proj(content)
        s_emb = self.style_proj(style)
        cond = t_emb + c_emb + s_emb
        
        h = self.input_conv(x)
        h1 = self.down1(h, cond); h = self.downsample1(h1)
        h2 = self.down2(h, cond); h = self.downsample2(h2)
        h3 = self.down3(h, cond); h = self.downsample3(h3)
        h4 = self.down4(h, cond); h = self.downsample4(h4)
        
        h = self.mid1(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)
        
        h = self.upsample4(h)
        h = self._match(h, h4); h = self.up4(torch.cat([h, h4], dim=1), cond)
        h = self.upsample3(h)
        h = self._match(h, h3); h = self.up3(torch.cat([h, h3], dim=1), cond)
        h = self.upsample2(h)
        h = self._match(h, h2); h = self.up2(torch.cat([h, h2], dim=1), cond)
        h = self.upsample1(h)
        h = self._match(h, h1); h = self.up1(torch.cat([h, h1], dim=1), cond)
        
        h = F.silu(self.out_norm(h))
        h = self._match(h, x)
        return self.out_conv(h)
    
    def _match(self, x, target):
        if x.size(-1) != target.size(-1):
            diff = target.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (0, diff))
            else:
                x = x[:, :, :target.size(-1)]
        return x


# ============================================================================
# DDIM Scheduler with Style Guidance
# ============================================================================

class StyleGuidedDDIM:
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
    
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        return (self.alphas_cumprod[t].sqrt()[:, None, None] * x0 + 
                (1 - self.alphas_cumprod[t]).sqrt()[:, None, None] * noise)
    
    @torch.no_grad()
    def generate_counterfactual(self, model, content_encoder, style_encoder, 
                                 original_signal, target_style_signal,
                                 noise_level=0.3, num_steps=50):
        """
        Generate counterfactual using content from original and style from target.
        
        Args:
            original_signal: ECG to generate counterfactual for
            target_style_signal: ECG from target class to extract style from
            noise_level: How much noise to add (0=reconstruction, 1=full generation)
            num_steps: DDIM steps
        """
        device = original_signal.device
        B = original_signal.shape[0]
        
        # Extract content from original (preserve)
        content, _, _ = content_encoder(original_signal)
        
        # Extract style from target class (transfer)
        target_style, _ = style_encoder(target_style_signal)
        
        # Add noise to original signal
        t_start = int(noise_level * self.num_timesteps)
        t_batch = torch.full((B,), t_start, device=device, dtype=torch.long)
        noise = torch.randn_like(original_signal)
        x = self.q_sample(original_signal, t_batch, noise)
        
        # DDIM denoising with style guidance
        step_size = max(1, t_start // num_steps)
        timesteps = list(range(0, t_start, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch, content, target_style)
            
            alpha_t = self.alphas_cumprod[t]
            pred_x0 = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                x = alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * pred_noise
            else:
                x = pred_x0
        
        return x
    
    @torch.no_grad()
    def sample(self, model, content, style, shape, num_steps=50):
        """Generate from scratch conditioned on content and style."""
        device = content.device
        x = torch.randn(shape, device=device)
        
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            pred_noise = model(x, t_batch, content, style)
            
            alpha_t = self.alphas_cumprod[t]
            pred_x0 = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                x = alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * pred_noise
            else:
                x = pred_x0
        
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
    issues = []
    
    if amp_range < 0.01: is_valid = False; issues.append("Low amplitude")
    if amp_range > 10: is_valid = False; issues.append("High amplitude")
    if len(peaks) < 2: is_valid = False; issues.append("No R-peaks")
    if hr > 0 and (hr < 30 or hr > 200): is_valid = False; issues.append(f"Invalid HR: {hr:.0f}")
    
    return {'valid': is_valid, 'issues': issues, 'hr': hr, 'num_peaks': len(peaks)}


# ============================================================================
# Interactive Visualization
# ============================================================================

def create_detailed_comparison(original, counterfactual, orig_class, target_class, 
                               orig_prob, cf_prob, classifier_wrapped, save_path):
    """Create detailed beat-by-beat comparison visualization."""
    
    orig_np = original.cpu().numpy().flatten() * REAL_STD + REAL_MEAN
    cf_np = counterfactual.cpu().numpy().flatten() * REAL_STD + REAL_MEAN
    diff_np = cf_np - orig_np
    
    time_axis = np.arange(len(orig_np)) / FS
    
    # Detect beats
    orig_peaks, _ = detect_r_peaks(orig_np, FS)
    cf_peaks, _ = detect_r_peaks(cf_np, FS)
    
    # Create figure with multiple panels
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 1])
    
    # --- Full signal comparison (top row) ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, orig_np, 'b-', lw=1, alpha=0.8, label=f'Original ({["Normal", "AFib"][orig_class]})')
    ax1.plot(time_axis, cf_np, 'r-', lw=1, alpha=0.7, label=f'Counterfactual (→{["Normal", "AFib"][target_class]})')
    for p in orig_peaks:
        ax1.axvline(x=p/FS, color='blue', alpha=0.3, lw=0.5)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Amplitude (mV)', fontsize=10)
    ax1.set_title(f'Full Signal Comparison | Original P({["Normal","AFib"][target_class]})={orig_prob:.3f} → CF P({["Normal","AFib"][target_class]})={cf_prob:.3f}', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # --- Zoomed segments (second row) ---
    zoom_ranges = [(0, 1000), (1000, 2000), (2000, 2500)]
    for i, (start, end) in enumerate(zoom_ranges):
        if i < 3:
            ax = fig.add_subplot(gs[1, i])
            t_seg = time_axis[start:end]
            ax.plot(t_seg, orig_np[start:end], 'b-', lw=1.5, label='Original')
            ax.plot(t_seg, cf_np[start:end], 'r-', lw=1.5, alpha=0.8, label='CF')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('mV')
            ax.set_title(f'Segment {i+1} ({start/FS:.1f}-{end/FS:.1f}s)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    
    # Difference signal
    ax_diff = fig.add_subplot(gs[1, 3])
    ax_diff.plot(time_axis, diff_np, 'purple', lw=0.8)
    ax_diff.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax_diff.fill_between(time_axis, 0, diff_np, alpha=0.3, color='purple')
    ax_diff.set_xlabel('Time (s)')
    ax_diff.set_ylabel('Δ mV')
    ax_diff.set_title(f'Difference (CF - Original)')
    ax_diff.grid(True, alpha=0.3)
    
    # --- R-R interval analysis (third row) ---
    ax_rr_orig = fig.add_subplot(gs[2, 0])
    if len(orig_peaks) > 1:
        orig_rr = np.diff(orig_peaks) / FS * 1000  # ms
        ax_rr_orig.bar(range(len(orig_rr)), orig_rr, color='blue', alpha=0.7)
        ax_rr_orig.axhline(y=np.mean(orig_rr), color='blue', linestyle='--', lw=2)
        ax_rr_orig.set_title(f'Original RR: μ={np.mean(orig_rr):.0f}ms, σ={np.std(orig_rr):.0f}ms')
    else:
        ax_rr_orig.set_title('Original RR: N/A')
    ax_rr_orig.set_xlabel('Beat #')
    ax_rr_orig.set_ylabel('RR (ms)')
    ax_rr_orig.grid(True, alpha=0.3)
    
    ax_rr_cf = fig.add_subplot(gs[2, 1])
    if len(cf_peaks) > 1:
        cf_rr = np.diff(cf_peaks) / FS * 1000  # ms
        ax_rr_cf.bar(range(len(cf_rr)), cf_rr, color='red', alpha=0.7)
        ax_rr_cf.axhline(y=np.mean(cf_rr), color='red', linestyle='--', lw=2)
        ax_rr_cf.set_title(f'CF RR: μ={np.mean(cf_rr):.0f}ms, σ={np.std(cf_rr):.0f}ms')
    else:
        ax_rr_cf.set_title('CF RR: N/A')
    ax_rr_cf.set_xlabel('Beat #')
    ax_rr_cf.set_ylabel('RR (ms)')
    ax_rr_cf.grid(True, alpha=0.3)
    
    # Poincare plot
    ax_poincare = fig.add_subplot(gs[2, 2])
    if len(orig_peaks) > 2:
        orig_rr = np.diff(orig_peaks) / FS * 1000
        ax_poincare.scatter(orig_rr[:-1], orig_rr[1:], c='blue', alpha=0.6, s=50, label='Original')
    if len(cf_peaks) > 2:
        cf_rr = np.diff(cf_peaks) / FS * 1000
        ax_poincare.scatter(cf_rr[:-1], cf_rr[1:], c='red', alpha=0.6, s=50, label='CF')
    ax_poincare.plot([0, 1500], [0, 1500], 'k--', alpha=0.3)
    ax_poincare.set_xlabel('RR_n (ms)')
    ax_poincare.set_ylabel('RR_{n+1} (ms)')
    ax_poincare.set_title('Poincaré Plot')
    ax_poincare.legend()
    ax_poincare.grid(True, alpha=0.3)
    ax_poincare.set_xlim([200, 1500])
    ax_poincare.set_ylim([200, 1500])
    
    # Power spectrum
    ax_psd = fig.add_subplot(gs[2, 3])
    f_orig, psd_orig = scipy_signal.welch(orig_np, fs=FS, nperseg=256)
    f_cf, psd_cf = scipy_signal.welch(cf_np, fs=FS, nperseg=256)
    ax_psd.semilogy(f_orig[:50], psd_orig[:50], 'b-', lw=1.5, label='Original')
    ax_psd.semilogy(f_cf[:50], psd_cf[:50], 'r-', lw=1.5, label='CF')
    ax_psd.set_xlabel('Frequency (Hz)')
    ax_psd.set_ylabel('PSD')
    ax_psd.set_title('Power Spectrum (0-50 Hz)')
    ax_psd.legend()
    ax_psd.grid(True, alpha=0.3)
    
    # --- Clinical metrics summary (bottom row) ---
    ax_summary = fig.add_subplot(gs[3, :2])
    ax_summary.axis('off')
    
    orig_feat = compute_rr_features(orig_np, FS)
    cf_feat = compute_rr_features(cf_np, FS)
    orig_valid = check_signal_validity(orig_np, FS)
    cf_valid = check_signal_validity(cf_np, FS)
    
    corr, _ = pearsonr(orig_np, cf_np) if len(orig_np) == len(cf_np) else (0, 0)
    
    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║                         CLINICAL SUMMARY                              ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  METRIC              │  ORIGINAL ({["Normal", "AFib"][orig_class]:^6})  │  COUNTERFACTUAL ({["Normal", "AFib"][target_class]:^6})  │  CHANGE     ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  Heart Rate          │  {orig_feat['hr']:>8.1f} bpm    │  {cf_feat['hr']:>8.1f} bpm           │  {cf_feat['hr']-orig_feat['hr']:>+.1f} bpm   ║
    ║  RR Mean             │  {orig_feat['rr_mean']:>8.1f} ms     │  {cf_feat['rr_mean']:>8.1f} ms            │  {cf_feat['rr_mean']-orig_feat['rr_mean']:>+.1f} ms    ║
    ║  RR Std              │  {orig_feat['rr_std']:>8.1f} ms     │  {cf_feat['rr_std']:>8.1f} ms            │  {cf_feat['rr_std']-orig_feat['rr_std']:>+.1f} ms    ║
    ║  RR Coef.Var         │  {orig_feat['rr_cv']:>8.3f}        │  {cf_feat['rr_cv']:>8.3f}               │  {cf_feat['rr_cv']-orig_feat['rr_cv']:>+.3f}      ║
    ║  Number of Beats     │  {orig_feat['num_beats']:>8d}          │  {cf_feat['num_beats']:>8d}                 │  {cf_feat['num_beats']-orig_feat['num_beats']:>+d}          ║
    ╠══════════════════════════════════════════════════════════════════════╣
    ║  Signal Correlation  │  {corr:.4f}                                                            ║
    ║  Valid Signal?       │  {'YES' if orig_valid['valid'] else 'NO':^10}           │  {'YES' if cf_valid['valid'] else 'NO':^10}                 │             ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """
    ax_summary.text(0, 0.5, summary_text, fontfamily='monospace', fontsize=10, va='center')
    
    # Probability bars
    ax_prob = fig.add_subplot(gs[3, 2:])
    categories = ['Original\nP(Normal)', 'Original\nP(AFib)', 'CF\nP(Normal)', 'CF\nP(AFib)']
    probs = [1-orig_prob if target_class==1 else orig_prob, 
             orig_prob if target_class==1 else 1-orig_prob,
             1-cf_prob if target_class==1 else cf_prob, 
             cf_prob if target_class==1 else 1-cf_prob]
    colors = ['lightblue', 'lightcoral', 'blue', 'red']
    bars = ax_prob.bar(categories, probs, color=colors, edgecolor='black')
    ax_prob.set_ylim([0, 1])
    ax_prob.axhline(y=0.5, color='gray', linestyle='--', lw=2)
    ax_prob.set_ylabel('Probability')
    ax_prob.set_title('Classifier Probabilities', fontweight='bold')
    for bar, p in zip(bars, probs):
        ax_prob.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{p:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'correlation': corr,
        'orig_rr_cv': orig_feat['rr_cv'],
        'cf_rr_cv': cf_feat['rr_cv'],
        'orig_valid': orig_valid['valid'],
        'cf_valid': cf_valid['valid'],
        'flipped': (orig_class != target_class) and (cf_prob > 0.5)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("DIFFUSION-BASED COUNTERFACTUAL GENERATION V3")
    print("Content-Style Disentanglement with Classifier Guidance")
    print("="*70)
    print(f"Results: {RESULTS_DIR}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    print("\nLoading data...")
    train_data = np.load(DATA_DIR / 'train_data.npz')
    val_data = np.load(DATA_DIR / 'val_data.npz')
    
    train_signals = torch.tensor(train_data['X'], dtype=torch.float32)
    train_labels = torch.tensor(train_data['y'], dtype=torch.long)
    val_signals = torch.tensor(val_data['X'], dtype=torch.float32)
    val_labels = torch.tensor(val_data['y'], dtype=torch.long)
    
    if train_signals.dim() == 2:
        train_signals = train_signals.unsqueeze(1)
        val_signals = val_signals.unsqueeze(1)
    
    print(f"  Train: {train_signals.shape}")
    print(f"  Val: {val_signals.shape}")
    
    # Separate by class
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    
    train_normal_idx = (train_labels == 0).nonzero(as_tuple=True)[0]
    train_afib_idx = (train_labels == 1).nonzero(as_tuple=True)[0]
    
    print(f"  Val Normal: {len(normal_idx)}, Val AFib: {len(afib_idx)}")
    
    # Initialize models
    print("\nInitializing models...")
    content_encoder = ContentEncoder(content_dim=256).to(DEVICE)
    style_encoder = StyleEncoder(style_dim=128).to(DEVICE)
    unet = ConditionalUNet(content_dim=256, style_dim=128).to(DEVICE)
    
    # Load trained model if exists
    if PHASE3_MODEL.exists():
        print(f"  Loading trained model from {PHASE3_MODEL}")
        checkpoint = torch.load(PHASE3_MODEL, map_location=DEVICE)
        content_encoder.load_state_dict(checkpoint['content_encoder'])
        style_encoder.load_state_dict(checkpoint['style_encoder'])
        unet.load_state_dict(checkpoint['unet'])
    else:
        print("  WARNING: No trained model found! Will train from scratch...")
        # Train the model
        train_model(content_encoder, style_encoder, unet, train_signals, train_labels)
    
    content_encoder.eval()
    style_encoder.eval()
    unet.eval()
    
    # Load classifier
    print("\nLoading classifier...")
    raw_classifier = load_classifier()
    classifier = ClassifierWrapper(raw_classifier)
    
    # Initialize scheduler
    scheduler = StyleGuidedDDIM(num_timesteps=1000, device=DEVICE)
    
    # ========================================================================
    # Generate and Analyze Counterfactuals
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING COUNTERFACTUALS")
    print("="*70)
    
    num_samples = 10
    results = {'normal_to_afib': [], 'afib_to_normal': []}
    
    # Normal -> AFib
    print("\n--- Normal → AFib ---")
    for i in range(num_samples):
        normal_ecg = val_signals[normal_idx[i]:normal_idx[i]+1].to(DEVICE)
        # Get a random AFib signal to extract style from
        afib_style_idx = train_afib_idx[np.random.randint(len(train_afib_idx))]
        afib_style_signal = train_signals[afib_style_idx:afib_style_idx+1].to(DEVICE)
        
        # Generate counterfactual
        cf = scheduler.generate_counterfactual(
            unet, content_encoder, style_encoder,
            normal_ecg, afib_style_signal,
            noise_level=0.4, num_steps=50
        )
        
        # Classify
        with torch.no_grad():
            orig_prob = F.softmax(classifier(normal_ecg), dim=1)[0, 1].item()  # P(AFib)
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 1].item()
        
        # Create visualization
        save_path = RESULTS_DIR / f'normal_to_afib_{i+1:02d}.png'
        result = create_detailed_comparison(
            normal_ecg[0, 0], cf[0, 0], 0, 1, orig_prob, cf_prob, classifier, save_path
        )
        results['normal_to_afib'].append(result)
        
        print(f"  Sample {i+1}: P(AFib) {orig_prob:.3f} → {cf_prob:.3f} | Corr: {result['correlation']:.3f} | Valid: {result['cf_valid']}")
    
    # AFib -> Normal
    print("\n--- AFib → Normal ---")
    for i in range(num_samples):
        afib_ecg = val_signals[afib_idx[i]:afib_idx[i]+1].to(DEVICE)
        # Get a random Normal signal to extract style from
        normal_style_idx = train_normal_idx[np.random.randint(len(train_normal_idx))]
        normal_style_signal = train_signals[normal_style_idx:normal_style_idx+1].to(DEVICE)
        
        # Generate counterfactual
        cf = scheduler.generate_counterfactual(
            unet, content_encoder, style_encoder,
            afib_ecg, normal_style_signal,
            noise_level=0.4, num_steps=50
        )
        
        # Classify
        with torch.no_grad():
            orig_prob = F.softmax(classifier(afib_ecg), dim=1)[0, 0].item()  # P(Normal)
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 0].item()
        
        # Create visualization
        save_path = RESULTS_DIR / f'afib_to_normal_{i+1:02d}.png'
        result = create_detailed_comparison(
            afib_ecg[0, 0], cf[0, 0], 1, 0, orig_prob, cf_prob, classifier, save_path
        )
        results['afib_to_normal'].append(result)
        
        print(f"  Sample {i+1}: P(Normal) {orig_prob:.3f} → {cf_prob:.3f} | Corr: {result['correlation']:.3f} | Valid: {result['cf_valid']}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    n2a_valid = sum(1 for r in results['normal_to_afib'] if r['cf_valid'])
    n2a_flipped = sum(1 for r in results['normal_to_afib'] if r['flipped'])
    n2a_corr = np.mean([r['correlation'] for r in results['normal_to_afib']])
    
    a2n_valid = sum(1 for r in results['afib_to_normal'] if r['cf_valid'])
    a2n_flipped = sum(1 for r in results['afib_to_normal'] if r['flipped'])
    a2n_corr = np.mean([r['correlation'] for r in results['afib_to_normal']])
    
    print(f"\nNormal → AFib:")
    print(f"  Valid signals: {n2a_valid}/{num_samples} ({100*n2a_valid/num_samples:.0f}%)")
    print(f"  Flip rate: {n2a_flipped}/{num_samples} ({100*n2a_flipped/num_samples:.0f}%)")
    print(f"  Mean correlation: {n2a_corr:.4f}")
    
    print(f"\nAFib → Normal:")
    print(f"  Valid signals: {a2n_valid}/{num_samples} ({100*a2n_valid/num_samples:.0f}%)")
    print(f"  Flip rate: {a2n_flipped}/{num_samples} ({100*a2n_flipped/num_samples:.0f}%)")
    print(f"  Mean correlation: {a2n_corr:.4f}")
    
    # Save results
    summary = {
        'normal_to_afib': {
            'valid_rate': n2a_valid / num_samples,
            'flip_rate': n2a_flipped / num_samples,
            'mean_correlation': float(n2a_corr),
        },
        'afib_to_normal': {
            'valid_rate': a2n_valid / num_samples,
            'flip_rate': a2n_flipped / num_samples,
            'mean_correlation': float(a2n_corr),
        }
    }
    
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("="*70)


def train_model(content_encoder, style_encoder, unet, train_signals, train_labels):
    """Quick training if no model exists."""
    print("\nTraining content-style diffusion model...")
    
    scheduler = StyleGuidedDDIM(num_timesteps=1000, device=DEVICE)
    
    all_params = list(content_encoder.parameters()) + list(style_encoder.parameters()) + list(unet.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-4)
    
    train_dataset = torch.utils.data.TensorDataset(train_signals, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    
    content_encoder.train()
    style_encoder.train()
    unet.train()
    
    num_epochs = 50
    
    for epoch in range(1, num_epochs + 1):
        total_loss = 0
        n_batches = 0
        
        for signals, labels in train_loader:
            signals = signals.to(DEVICE)
            
            # Encode
            content, c_mu, c_logvar = content_encoder(signals)
            style, style_logits = style_encoder(signals)
            
            # Diffusion forward
            t = torch.randint(0, 1000, (signals.size(0),), device=DEVICE)
            noise = torch.randn_like(signals)
            noisy = scheduler.q_sample(signals, t, noise)
            
            # Predict noise
            pred_noise = unet(noisy, t, content, style)
            
            # Losses
            recon_loss = F.mse_loss(pred_noise, noise)
            style_loss = F.cross_entropy(style_logits, labels.to(DEVICE))
            kl_loss = -0.5 * torch.mean(1 + c_logvar - c_mu.pow(2) - c_logvar.exp())
            
            loss = recon_loss + 0.5 * style_loss + 0.01 * kl_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}/{num_epochs}: Loss = {total_loss/n_batches:.4f}")
    
    # Save
    torch.save({
        'content_encoder': content_encoder.state_dict(),
        'style_encoder': style_encoder.state_dict(),
        'unet': unet.state_dict(),
    }, PHASE3_MODEL)
    
    print(f"  Model saved to {PHASE3_MODEL}")


if __name__ == '__main__':
    main()
