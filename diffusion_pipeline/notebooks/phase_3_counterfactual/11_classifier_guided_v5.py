"""
Phase 3: Strong Classifier-Guided Diffusion Counterfactual V5
=============================================================

Key insight from V4:
- Low noise + high preservation = high similarity BUT very low flip rate
- Need to use CLASSIFIER GRADIENT GUIDANCE (not CFG) to directly push toward target

Approach:
1. Use classifier gradients during diffusion denoising
2. Higher noise level to allow more modification room
3. Iterative refinement: if not flipped, increase guidance strength
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

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
PHASE2_MODEL = PROJECT_ROOT / 'models/phase2_diffusion/diffusion_v2/best.pth'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/classifier_guided_v5'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

REAL_MEAN, REAL_STD = -0.00396, 0.14716
FS = 500


# ============================================================================
# Model architecture (same as Phase 2)
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
    def __init__(self, in_ch=1, model_ch=64, num_res_blocks=2,
                 attn_resolutions=(2, 3), ch_mult=(1, 2, 4, 8), num_classes=2,
                 dropout=0.1, num_groups=32, time_dim=256, class_dim=256):
        super().__init__()
        self.num_classes = num_classes
        self.time_embed = nn.Sequential(nn.Linear(model_ch, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.class_embed = nn.Embedding(num_classes + 1, class_dim)
        self.class_proj = nn.Linear(class_dim, time_dim)
        self.input_proj = nn.Conv1d(in_ch, model_ch, 3, padding=1)
        
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
        
        self.mid_block1 = ResBlock1D(ch, ch, time_dim, num_groups, dropout)
        self.mid_attn = SelfAttention1D(ch)
        self.mid_block2 = ResBlock1D(ch, ch, time_dim, num_groups, dropout)
        
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
# Classifier-Guided DDIM with adaptive guidance
# ============================================================================

class ClassifierGuidedDDIM:
    def __init__(self, num_timesteps=1000, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
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
    
    def generate_counterfactual(self, model, classifier, x0, target_class,
                                 noise_level=0.4, num_steps=50,
                                 classifier_scale=200.0, similarity_weight=0.3):
        """
        Generate counterfactual using CLASSIFIER GRADIENT during diffusion.
        
        This directly uses gradients from the external classifier to push
        the generated signal toward the target class.
        """
        device = x0.device
        B = x0.shape[0]
        
        t_start = int(noise_level * self.num_timesteps)
        t_batch = torch.full((B,), t_start, device=device, dtype=torch.long)
        x, _ = self.add_noise(x0, t_batch)
        
        step_size = max(1, t_start // num_steps)
        timesteps = list(range(0, t_start, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            x = x.detach().requires_grad_(True)
            
            # Get unconditional noise prediction from diffusion model
            with torch.no_grad():
                pred_noise = model(x, t_batch, None)
            
            # Predict x0
            alpha_t = self.alphas_cumprod[t]
            pred_x0 = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            pred_x0 = torch.clamp(pred_x0, -3, 3)
            
            # Get classifier gradient on predicted x0
            pred_x0_for_grad = pred_x0.detach().requires_grad_(True)
            
            classifier.train()  # LSTM requires training mode for backward
            logits = classifier(pred_x0_for_grad)
            log_probs = F.log_softmax(logits, dim=1)
            target_log_prob = log_probs[:, target_class].sum()
            target_log_prob.backward()
            grad = pred_x0_for_grad.grad
            classifier.eval()
            
            # Apply classifier guidance
            pred_x0_guided = pred_x0 + classifier_scale * grad
            pred_x0_guided = torch.clamp(pred_x0_guided, -3, 3)
            
            # Similarity regularization: blend with original
            pred_x0_guided = (1 - similarity_weight) * pred_x0_guided + similarity_weight * x0
            
            # DDIM step
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev]
                x = alpha_prev.sqrt() * pred_x0_guided + (1 - alpha_prev).sqrt() * pred_noise
            else:
                x = pred_x0_guided
            
            x = x.detach()
        
        return x
    
    def generate_counterfactual_adaptive(self, model, classifier, x0, target_class,
                                          max_iterations=5, initial_scale=100.0,
                                          noise_level=0.4, similarity_target=0.7):
        """
        Adaptive counterfactual generation: 
        - If not flipped, increase guidance
        - If too different from original, decrease guidance
        """
        device = x0.device
        
        best_cf = None
        best_score = -float('inf')
        
        for iteration in range(max_iterations):
            # Adjust parameters based on iteration
            scale = initial_scale * (1.5 ** iteration)  # Increase scale each iteration
            sim_weight = 0.3 - (0.05 * iteration)  # Decrease similarity weight
            sim_weight = max(0.1, sim_weight)
            
            cf = self.generate_counterfactual(
                model, classifier, x0, target_class,
                noise_level=noise_level, num_steps=50,
                classifier_scale=scale, similarity_weight=sim_weight
            )
            
            # Evaluate
            with torch.no_grad():
                prob = F.softmax(classifier(cf), dim=1)[0, target_class].item()
            
            corr, _ = pearsonr(x0[0, 0].cpu().numpy().flatten(), 
                               cf[0, 0].cpu().numpy().flatten())
            
            # Score: balance flip and similarity
            if prob > 0.5:  # Flipped
                score = prob + corr
            else:  # Not flipped yet
                score = prob * 0.5 + corr * 0.5
            
            if score > best_score:
                best_score = score
                best_cf = cf.clone()
            
            # If we achieved both goals, stop early
            if prob > 0.5 and corr > similarity_target:
                break
        
        return best_cf


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
        self.model.train()
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        logits, _ = self.model(x_norm)
        return logits


# ============================================================================
# Analysis functions
# ============================================================================

def detect_r_peaks(signal, fs=500):
    signal = signal.flatten()
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    peaks, _ = scipy_signal.find_peaks(signal_norm, height=0.3, distance=int(0.3*fs), prominence=0.2)
    return peaks

def compute_rr_features(signal, fs=500):
    peaks = detect_r_peaks(signal, fs)
    if len(peaks) < 2:
        return {'rr_cv': 0, 'hr': 0, 'num_beats': 0}
    rr = np.diff(peaks) / fs * 1000
    return {
        'rr_cv': np.std(rr) / (np.mean(rr) + 1e-8),
        'hr': 60000 / np.mean(rr) if np.mean(rr) > 0 else 0,
        'num_beats': len(peaks)
    }

def check_validity(signal, fs=500):
    signal = signal.flatten()
    amp = np.max(signal) - np.min(signal)
    peaks = detect_r_peaks(signal, fs)
    if len(peaks) >= 2:
        hr = 60 / (np.mean(np.diff(peaks)) / fs)
    else:
        hr = 0
    return (0.01 < amp < 10) and (len(peaks) >= 2) and (30 < hr < 200)


def create_visualization(orig, cf, orig_class, target_class, orig_prob, cf_prob, idx, save_path):
    """Create detailed comparison visualization."""
    orig_np = orig.cpu().numpy().flatten()
    cf_np = cf.cpu().numpy().flatten()
    
    # Convert to mV
    orig_mv = orig_np * REAL_STD + REAL_MEAN
    cf_mv = cf_np * REAL_STD + REAL_MEAN
    diff_mv = cf_mv - orig_mv
    
    time = np.arange(len(orig_np)) / FS
    
    corr, _ = pearsonr(orig_np, cf_np)
    orig_feat = compute_rr_features(orig_np, FS)
    cf_feat = compute_rr_features(cf_np, FS)
    
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1.5, 1, 1, 0.8])
    
    class_names = ['Normal', 'AFib']
    
    # Full signal
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.plot(time, orig_mv, 'b-', lw=1, alpha=0.8, label=f'Original ({class_names[orig_class]})')
    ax_full.plot(time, cf_mv, 'r-', lw=1, alpha=0.7, label=f'CF (→{class_names[target_class]})')
    ax_full.legend(loc='upper right', fontsize=11)
    ax_full.set_xlabel('Time (s)')
    ax_full.set_ylabel('Amplitude (mV)')
    ax_full.set_title(f'Sample {idx} | Corr: {corr:.4f} | P({class_names[target_class]}): {orig_prob:.3f} → {cf_prob:.3f}', 
                      fontsize=14, fontweight='bold')
    ax_full.grid(True, alpha=0.3)
    
    # Zoomed segments
    segments = [(0, 500), (1000, 1500), (2000, 2500)]
    for i, (s, e) in enumerate(segments):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(time[s:e], orig_mv[s:e], 'b-', lw=1.5, label='Orig')
        ax.plot(time[s:e], cf_mv[s:e], 'r-', lw=1.5, alpha=0.8, label='CF')
        ax.set_title(f'{time[s]:.1f}-{time[e-1]:.1f}s')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Difference
    ax_diff = fig.add_subplot(gs[1, 3])
    ax_diff.plot(time, diff_mv, 'purple', lw=0.8)
    ax_diff.fill_between(time, 0, diff_mv, alpha=0.3, color='purple')
    ax_diff.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax_diff.set_title(f'Difference | MSE: {np.mean(diff_mv**2):.6f}')
    ax_diff.grid(True, alpha=0.3)
    
    # R-R intervals
    orig_peaks = detect_r_peaks(orig_np, FS)
    cf_peaks = detect_r_peaks(cf_np, FS)
    
    ax_rr = fig.add_subplot(gs[2, 0:2])
    if len(orig_peaks) > 1:
        orig_rr = np.diff(orig_peaks) / FS * 1000
        ax_rr.bar(np.arange(len(orig_rr)) - 0.2, orig_rr, 0.4, color='blue', alpha=0.7, label='Orig')
    if len(cf_peaks) > 1:
        cf_rr = np.diff(cf_peaks) / FS * 1000
        ax_rr.bar(np.arange(len(cf_rr)) + 0.2, cf_rr, 0.4, color='red', alpha=0.7, label='CF')
    ax_rr.set_xlabel('Beat #')
    ax_rr.set_ylabel('RR (ms)')
    ax_rr.set_title(f'RR Intervals | CV: {orig_feat["rr_cv"]:.3f} → {cf_feat["rr_cv"]:.3f}')
    ax_rr.legend()
    ax_rr.grid(True, alpha=0.3)
    
    # Poincare
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
    ax_poincare.set_title('Poincaré')
    ax_poincare.legend()
    ax_poincare.grid(True, alpha=0.3)
    
    # Probabilities
    ax_prob = fig.add_subplot(gs[2, 3])
    labels = ['Orig P(N)', 'Orig P(AF)', 'CF P(N)', 'CF P(AF)']
    if target_class == 1:
        probs = [1-orig_prob, orig_prob, 1-cf_prob, cf_prob]
    else:
        probs = [orig_prob, 1-orig_prob, cf_prob, 1-cf_prob]
    colors = ['lightblue', 'lightcoral', 'blue', 'red']
    bars = ax_prob.bar(labels, probs, color=colors, edgecolor='black')
    ax_prob.axhline(0.5, color='gray', linestyle='--', lw=2)
    ax_prob.set_ylim([0, 1])
    for bar, p in zip(bars, probs):
        ax_prob.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{p:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax_prob.set_title('Classification')
    
    # Summary
    ax_summary = fig.add_subplot(gs[3, :])
    ax_summary.axis('off')
    
    rr_cv_change = cf_feat['rr_cv'] - orig_feat['rr_cv']
    expected = 'increase' if target_class == 1 else 'decrease'
    actual = 'increased' if rr_cv_change > 0 else 'decreased'
    correct_rr = (target_class == 1 and rr_cv_change > 0) or (target_class == 0 and rr_cv_change < 0)
    flipped = cf_prob > 0.5
    valid = check_validity(cf_np, FS)
    
    summary = f"""
    ╔════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  CORRELATION: {corr:.4f}  {'✓ HIGH' if corr > 0.7 else '✗ LOW':20}    CLASSIFIER FLIP: {'✓ YES' if flipped else '✗ NO':20}    VALID: {'✓' if valid else '✗'}  ║
    ║  RR CV: {orig_feat['rr_cv']:.3f} → {cf_feat['rr_cv']:.3f} ({'+' if rr_cv_change > 0 else ''}{rr_cv_change:.3f})  Expected: {expected}  Actual: {actual}  {'✓' if correct_rr else '✗'}                              ║
    ╚════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    ax_summary.text(0.02, 0.5, summary, fontfamily='monospace', fontsize=11, va='center', transform=ax_summary.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'corr': corr, 'flipped': flipped, 'valid': valid,
        'orig_rr_cv': orig_feat['rr_cv'], 'cf_rr_cv': cf_feat['rr_cv'],
        'correct_rr': correct_rr, 'orig_prob': orig_prob, 'cf_prob': cf_prob
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STRONG CLASSIFIER-GUIDED DIFFUSION COUNTERFACTUAL V5")
    print("="*70)
    print(f"Results: {RESULTS_DIR}")
    
    # Load models
    print("\nLoading diffusion model...")
    unet = ECGUNet(in_ch=1, model_ch=64, num_res_blocks=2, attn_resolutions=(2, 3), 
                   ch_mult=(1, 2, 4, 8), num_classes=2, dropout=0.1).to(DEVICE)
    checkpoint = torch.load(PHASE2_MODEL, map_location=DEVICE)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    
    print("Loading classifier...")
    raw_classifier = load_classifier()
    classifier = ClassifierWrapper(raw_classifier)
    
    # Load data
    val_data = np.load(DATA_DIR / 'val_data.npz')
    val_signals = torch.tensor(val_data['X'], dtype=torch.float32)
    val_labels = torch.tensor(val_data['y'], dtype=torch.long)
    if val_signals.dim() == 2:
        val_signals = val_signals.unsqueeze(1)
    
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    
    print(f"\nData: {len(normal_idx)} Normal, {len(afib_idx)} AFib")
    
    # Test classifier
    with torch.no_grad():
        test = val_signals[:200].to(DEVICE)
        preds = classifier(test).argmax(dim=1)
        acc = (preds == val_labels[:200].to(DEVICE)).float().mean().item() * 100
        print(f"Classifier accuracy: {acc:.1f}%")
    
    scheduler = ClassifierGuidedDDIM(1000, DEVICE)
    
    # ========================================================================
    # Find optimal hyperparameters
    # ========================================================================
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH")
    print("="*70)
    
    configs = [
        # (noise_level, classifier_scale, similarity_weight)
        (0.3, 100, 0.3),
        (0.4, 150, 0.25),
        (0.4, 200, 0.2),
        (0.5, 200, 0.2),
        (0.5, 300, 0.15),
    ]
    
    best_config = None
    best_score = 0
    
    for noise, scale, sim in configs:
        flips = 0
        corrs = []
        
        for i in range(3):
            x = val_signals[normal_idx[i]:normal_idx[i]+1].to(DEVICE)
            cf = scheduler.generate_counterfactual(
                unet, classifier, x, target_class=1,
                noise_level=noise, num_steps=50,
                classifier_scale=scale, similarity_weight=sim
            )
            with torch.no_grad():
                p = F.softmax(classifier(cf), dim=1)[0, 1].item()
            if p > 0.5:
                flips += 1
            c, _ = pearsonr(x[0,0].cpu().numpy(), cf[0,0].cpu().numpy())
            corrs.append(c)
        
        for i in range(3):
            x = val_signals[afib_idx[i]:afib_idx[i]+1].to(DEVICE)
            cf = scheduler.generate_counterfactual(
                unet, classifier, x, target_class=0,
                noise_level=noise, num_steps=50,
                classifier_scale=scale, similarity_weight=sim
            )
            with torch.no_grad():
                p = F.softmax(classifier(cf), dim=1)[0, 0].item()
            if p > 0.5:
                flips += 1
            c, _ = pearsonr(x[0,0].cpu().numpy(), cf[0,0].cpu().numpy())
            corrs.append(c)
        
        flip_rate = flips / 6
        mean_corr = np.mean(corrs)
        score = flip_rate * 0.6 + mean_corr * 0.4
        
        print(f"  noise={noise}, scale={scale}, sim={sim} -> flip={flip_rate*100:.0f}%, corr={mean_corr:.3f}, score={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_config = (noise, scale, sim)
    
    noise_level, classifier_scale, sim_weight = best_config
    print(f"\nBest: noise={noise_level}, scale={classifier_scale}, sim={sim_weight}")
    
    # ========================================================================
    # Generate counterfactuals
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING COUNTERFACTUALS")
    print("="*70)
    
    num_samples = 10
    results = {'normal_to_afib': [], 'afib_to_normal': []}
    
    # Normal → AFib
    print("\n--- Normal → AFib ---")
    for i in tqdm(range(num_samples), desc="N→AF"):
        x = val_signals[normal_idx[i]:normal_idx[i]+1].to(DEVICE)
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 1].item()
        
        cf = scheduler.generate_counterfactual_adaptive(
            unet, classifier, x, target_class=1,
            max_iterations=4, initial_scale=classifier_scale,
            noise_level=noise_level, similarity_target=0.7
        )
        
        with torch.no_grad():
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 1].item()
        
        save_path = RESULTS_DIR / f'n2a_{i+1:02d}.png'
        result = create_visualization(x[0,0], cf[0,0], 0, 1, orig_prob, cf_prob, i+1, save_path)
        results['normal_to_afib'].append(result)
        print(f"  #{i+1}: P(AF) {orig_prob:.3f}→{cf_prob:.3f} | Corr: {result['corr']:.3f}")
    
    # AFib → Normal
    print("\n--- AFib → Normal ---")
    for i in tqdm(range(num_samples), desc="AF→N"):
        x = val_signals[afib_idx[i]:afib_idx[i]+1].to(DEVICE)
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 0].item()
        
        cf = scheduler.generate_counterfactual_adaptive(
            unet, classifier, x, target_class=0,
            max_iterations=4, initial_scale=classifier_scale,
            noise_level=noise_level, similarity_target=0.7
        )
        
        with torch.no_grad():
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 0].item()
        
        save_path = RESULTS_DIR / f'a2n_{i+1:02d}.png'
        result = create_visualization(x[0,0], cf[0,0], 1, 0, orig_prob, cf_prob, i+1, save_path)
        results['afib_to_normal'].append(result)
        print(f"  #{i+1}: P(N) {orig_prob:.3f}→{cf_prob:.3f} | Corr: {result['corr']:.3f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    n2a = results['normal_to_afib']
    a2n = results['afib_to_normal']
    
    def summarize(data, name):
        valid = sum(r['valid'] for r in data)
        flipped = sum(r['flipped'] for r in data)
        corr = np.mean([r['corr'] for r in data])
        correct_rr = sum(r['correct_rr'] for r in data)
        print(f"\n{name}:")
        print(f"  Valid: {valid}/{len(data)} ({100*valid/len(data):.0f}%)")
        print(f"  Flipped: {flipped}/{len(data)} ({100*flipped/len(data):.0f}%)")
        print(f"  Correlation: {corr:.4f}")
        print(f"  Correct RR direction: {correct_rr}/{len(data)} ({100*correct_rr/len(data):.0f}%)")
        return valid, flipped, corr, correct_rr
    
    n2a_valid, n2a_flip, n2a_corr, n2a_rr = summarize(n2a, "Normal → AFib")
    a2n_valid, a2n_flip, a2n_corr, a2n_rr = summarize(a2n, "AFib → Normal")
    
    # Save
    summary = {
        'config': {'noise': noise_level, 'scale': classifier_scale, 'sim': sim_weight},
        'normal_to_afib': {'valid': n2a_valid/num_samples, 'flip': n2a_flip/num_samples, 'corr': float(n2a_corr), 'rr': n2a_rr/num_samples},
        'afib_to_normal': {'valid': a2n_valid/num_samples, 'flip': a2n_flip/num_samples, 'corr': float(a2n_corr), 'rr': a2n_rr/num_samples}
    }
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST CASE RESULTS")
    print("="*70)
    
    total = 2 * num_samples
    all_valid = n2a_valid + a2n_valid
    all_flip = n2a_flip + a2n_flip
    mean_corr = (n2a_corr + a2n_corr) / 2
    
    print(f"\n1. REALISTIC SIGNALS: {all_valid}/{total} ({100*all_valid/total:.0f}%) {'✓' if all_valid >= 0.9*total else '✗'}")
    print(f"2. HIGH SIMILARITY (corr): {mean_corr:.4f} {'✓' if mean_corr > 0.7 else '✗'}")
    print(f"3. CLASSIFIER FLIP: {all_flip}/{total} ({100*all_flip/total:.0f}%) {'✓' if all_flip >= 0.8*total else '✗'}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")

if __name__ == '__main__':
    main()
