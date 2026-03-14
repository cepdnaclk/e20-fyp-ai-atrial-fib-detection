"""
Phase 3: Hybrid Counterfactual Generator V8
============================================

Combines:
1. PHYSIOLOGICAL beat manipulation (rhythm changes, P-wave modification)
2. DIFFUSION refinement to make the signal more convincing to the classifier

The idea:
1. First apply physiological modifications (V7 approach)
2. Then use diffusion model to refine and make the signal more class-consistent
3. The diffusion step learns the class-specific signal characteristics beyond just rhythm
"""

import os
import sys
import subprocess
import math

def get_free_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        gpu_info = [(int(l.split(',')[0]), int(l.split(',')[1])) for l in lines if l.strip()]
        if gpu_info:
            return str(max(gpu_info, key=lambda x: x[1])[0])
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
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, '/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')
from model_architecture import AFibResLSTM, ModelConfig

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
PHASE2_MODEL = PROJECT_ROOT / 'models/phase2_diffusion/diffusion_v2/best.pth'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/hybrid_v8'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

REAL_MEAN, REAL_STD = -0.00396, 0.14716
FS = 500
SIGNAL_LENGTH = 2500


# ============================================================================
# Diffusion Model Architecture (simplified from Phase 2)
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
# Beat Detection and Manipulation (from V7)
# ============================================================================

def detect_r_peaks(signal, fs=500):
    signal = signal.flatten()
    nyq = fs / 2
    b, a = scipy_signal.butter(2, [5/nyq, 15/nyq], btype='band')
    filtered = scipy_signal.filtfilt(b, a, signal)
    diff = np.diff(filtered)
    squared = diff ** 2
    window = int(0.08 * fs)
    ma = np.convolve(squared, np.ones(window)/window, mode='same')
    peaks, _ = scipy_signal.find_peaks(ma, distance=int(0.3*fs), height=np.max(ma)*0.1)
    
    refined = []
    search = int(0.05 * fs)
    for p in peaks:
        start = max(0, p - search)
        end = min(len(signal), p + search)
        refined.append(start + np.argmax(signal[start:end]))
    return np.array(refined)

def segment_beats(signal, r_peaks, fs=500):
    signal = signal.flatten()
    pre = int(0.2 * fs)
    post = int(0.4 * fs)
    beats = []
    for r in r_peaks:
        if r - pre >= 0 and r + post <= len(signal):
            beats.append({'signal': signal[r-pre:r+post], 'r_position': pre, 'original_r': r})
    return beats

def remove_p_wave(beat, r_pos, fs=500):
    p_start = max(0, int(r_pos - 0.2 * fs))
    p_end = int(r_pos - 0.08 * fs)
    modified = beat.copy()
    if p_start < p_end < len(beat):
        modified[p_start:p_end] = np.linspace(beat[p_start], beat[p_end], p_end - p_start)
        modified[p_start:p_end] += np.random.randn(p_end - p_start) * 0.02
    return modified

def add_p_wave(beat, r_pos, fs=500):
    p_template = 0.1 * np.sin(np.linspace(0, np.pi, int(0.1 * fs)))
    p_center = int(r_pos - 0.16 * fs)
    p_start = p_center - len(p_template) // 2
    p_end = p_start + len(p_template)
    modified = beat.copy()
    if p_start >= 0 and p_end < r_pos:
        modified[p_start:p_end] += p_template * np.std(beat) * 0.5
    return modified

def reconstruct_signal(beats, rr_ms, fs=500, length=2500):
    output = np.zeros(length)
    rr_samples = (rr_ms / 1000 * fs).astype(int)
    r_positions = [100]
    for rr in rr_samples:
        next_r = r_positions[-1] + rr
        if next_r < length - 200:
            r_positions.append(int(next_r))
    
    for i, r_pos in enumerate(r_positions):
        beat = beats[i % len(beats)]
        start = r_pos - beat['r_position']
        end = start + len(beat['signal'])
        sig = beat['signal'].copy()
        
        if start < 0:
            sig = sig[-start:]
            start = 0
        if end > length:
            sig = sig[:length - start]
            end = length
        
        if end - start == len(sig):
            blend = min(20, len(sig) // 4)
            w = np.ones(len(sig))
            w[:blend] = np.linspace(0, 1, blend)
            w[-blend:] = np.linspace(1, 0, blend)
            output[start:end] = output[start:end] * (1 - w) + sig * w
    return output


# ============================================================================
# Diffusion-based Refinement
# ============================================================================

class DiffusionRefiner:
    def __init__(self, model, num_timesteps=1000, device='cuda'):
        self.model = model
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
    
    def refine(self, signal, target_class, noise_level=0.2, num_steps=30):
        """
        Refine a physiologically modified signal using diffusion.
        Add small amount of noise and denoise with target class conditioning.
        """
        x = signal.clone()
        B = x.shape[0]
        
        t_start = int(noise_level * self.num_timesteps)
        t_batch = torch.full((B,), t_start, device=self.device, dtype=torch.long)
        
        noise = torch.randn_like(x)
        alpha = self.alphas_cumprod[t_start]
        x_noisy = alpha.sqrt() * x + (1 - alpha).sqrt() * noise
        
        target_labels = torch.full((B,), target_class, device=self.device, dtype=torch.long)
        
        step_size = max(1, t_start // num_steps)
        timesteps = list(range(0, t_start, step_size))[:num_steps]
        timesteps = list(reversed(timesteps))
        
        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_batch = torch.full((B,), t, device=self.device, dtype=torch.long)
                pred_noise = self.model(x_noisy, t_batch, target_labels)
                
                alpha_t = self.alphas_cumprod[t]
                pred_x0 = (x_noisy - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
                pred_x0 = torch.clamp(pred_x0, -3, 3)
                
                # Blend with original to preserve physiological changes
                pred_x0 = 0.7 * pred_x0 + 0.3 * signal
                
                if i < len(timesteps) - 1:
                    t_prev = timesteps[i + 1]
                    alpha_prev = self.alphas_cumprod[t_prev]
                    x_noisy = alpha_prev.sqrt() * pred_x0 + (1 - alpha_prev).sqrt() * pred_noise
                else:
                    x_noisy = pred_x0
        
        return x_noisy


# ============================================================================
# Hybrid Counterfactual Generator
# ============================================================================

class HybridCounterfactualGenerator:
    def __init__(self, classifier, diffusion_model, device='cuda'):
        self.classifier = classifier
        self.refiner = DiffusionRefiner(diffusion_model, device=device)
        self.device = device
    
    def generate_normal_to_afib(self, signal):
        signal_np = signal.cpu().numpy().flatten()
        r_peaks = detect_r_peaks(signal_np, FS)
        
        if len(r_peaks) < 3:
            return signal
        
        beats = segment_beats(signal_np, r_peaks, FS)
        if len(beats) < 2:
            return signal
        
        # Modify beats: remove P-waves
        modified_beats = []
        for beat in beats:
            mod = remove_p_wave(beat['signal'], beat['r_position'], FS)
            modified_beats.append({'signal': mod, 'r_position': beat['r_position']})
        
        # Generate irregular RR intervals
        rr_stats = np.diff(r_peaks) / FS * 1000
        mean_rr = np.mean(rr_stats)
        cv_target = 0.15 + np.random.rand() * 0.1
        irregular_rr = np.random.normal(mean_rr, mean_rr * cv_target, len(r_peaks))
        irregular_rr = np.clip(irregular_rr, 300, 1500)
        
        # Reconstruct
        cf_np = reconstruct_signal(modified_beats, irregular_rr, FS, SIGNAL_LENGTH)
        
        # Add fibrillatory baseline
        t = np.linspace(0, 5, SIGNAL_LENGTH)
        fib = 0.03 * np.sin(2 * np.pi * 5 * t) + 0.02 * np.sin(2 * np.pi * 7 * t)
        cf_np += fib * np.std(signal_np)
        
        cf_tensor = torch.tensor(cf_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Refine with diffusion
        cf_refined = self.refiner.refine(cf_tensor, target_class=1, noise_level=0.15, num_steps=20)
        
        return cf_refined
    
    def generate_afib_to_normal(self, signal):
        signal_np = signal.cpu().numpy().flatten()
        r_peaks = detect_r_peaks(signal_np, FS)
        
        if len(r_peaks) < 3:
            return signal
        
        beats = segment_beats(signal_np, r_peaks, FS)
        if len(beats) < 2:
            return signal
        
        # Modify beats: smooth and add P-waves
        modified_beats = []
        for beat in beats:
            # Smooth to remove high-freq noise
            b, a = scipy_signal.butter(3, 40 / (FS/2), btype='low')
            smoothed = scipy_signal.filtfilt(b, a, beat['signal'])
            mod = add_p_wave(smoothed, beat['r_position'], FS)
            modified_beats.append({'signal': mod, 'r_position': beat['r_position']})
        
        # Generate regular RR intervals
        rr_stats = np.diff(r_peaks) / FS * 1000
        mean_rr = np.mean(rr_stats)
        cv_target = 0.02 + np.random.rand() * 0.02
        regular_rr = np.random.normal(mean_rr, mean_rr * cv_target, len(r_peaks))
        regular_rr = np.clip(regular_rr, 400, 1200)
        
        # Reconstruct
        cf_np = reconstruct_signal(modified_beats, regular_rr, FS, SIGNAL_LENGTH)
        
        cf_tensor = torch.tensor(cf_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # Refine with diffusion
        cf_refined = self.refiner.refine(cf_tensor, target_class=0, noise_level=0.15, num_steps=20)
        
        return cf_refined


# ============================================================================
# Analysis Functions
# ============================================================================

def compute_rr_cv(signal, fs=500):
    r_peaks = detect_r_peaks(signal.flatten(), fs)
    if len(r_peaks) < 2:
        return 0
    rr = np.diff(r_peaks) / fs * 1000
    return np.std(rr) / (np.mean(rr) + 1e-8)

def check_validity(signal, fs=500):
    signal = np.array(signal).flatten()
    amp = np.max(signal) - np.min(signal)
    r_peaks = detect_r_peaks(signal, fs)
    if len(r_peaks) >= 2:
        hr = 60 / (np.mean(np.diff(r_peaks)) / fs)
    else:
        hr = 0
    return (0.01 < amp < 10) and (len(r_peaks) >= 2) and (30 < hr < 200)


def create_visualization(orig, cf, orig_class, target_class, orig_prob, cf_prob, idx, save_path):
    orig_np = orig.cpu().numpy().flatten()
    cf_np = cf.cpu().numpy().flatten()
    time = np.arange(len(orig_np)) / FS
    
    orig_cv = compute_rr_cv(orig_np, FS)
    cf_cv = compute_rr_cv(cf_np, FS)
    
    corr, _ = pearsonr(orig_np, cf_np)
    
    fig = plt.figure(figsize=(24, 14))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1.5, 1, 1, 0.6])
    
    class_names = ['Normal', 'AFib']
    
    # Full signals
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(time, orig_np, 'b-', lw=1)
    ax1.set_title(f'ORIGINAL ({class_names[orig_class]}) | RR CV: {orig_cv:.3f}', fontweight='bold', color='blue')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Time (s)')
    
    ax2 = fig.add_subplot(gs[0, 2:])
    ax2.plot(time, cf_np, 'r-', lw=1)
    ax2.set_title(f'COUNTERFACTUAL (→{class_names[target_class]}) | RR CV: {cf_cv:.3f}', fontweight='bold', color='red')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    
    # Zoomed
    for col, (s, e) in enumerate([(0, 500), (1000, 1500)]):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(time[s:e], orig_np[s:e], 'b-', lw=1.5)
        ax.set_title(f'Orig {s/FS:.1f}-{e/FS:.1f}s')
        ax.grid(True, alpha=0.3)
        
        ax = fig.add_subplot(gs[1, col + 2])
        ax.plot(time[s:e], cf_np[s:e], 'r-', lw=1.5)
        ax.set_title(f'CF {s/FS:.1f}-{e/FS:.1f}s')
        ax.grid(True, alpha=0.3)
    
    # RR intervals
    orig_peaks = detect_r_peaks(orig_np, FS)
    cf_peaks = detect_r_peaks(cf_np, FS)
    
    ax_rr1 = fig.add_subplot(gs[2, :2])
    if len(orig_peaks) > 1:
        rr = np.diff(orig_peaks) / FS * 1000
        ax_rr1.bar(range(len(rr)), rr, color='blue', alpha=0.7)
        ax_rr1.axhline(np.mean(rr), color='darkblue', linestyle='--', lw=2)
    ax_rr1.set_xlabel('Beat #')
    ax_rr1.set_ylabel('RR (ms)')
    ax_rr1.set_title(f'Original RR (CV={orig_cv:.3f})')
    ax_rr1.grid(True, alpha=0.3)
    
    ax_rr2 = fig.add_subplot(gs[2, 2:])
    if len(cf_peaks) > 1:
        rr = np.diff(cf_peaks) / FS * 1000
        ax_rr2.bar(range(len(rr)), rr, color='red', alpha=0.7)
        ax_rr2.axhline(np.mean(rr), color='darkred', linestyle='--', lw=2)
    ax_rr2.set_xlabel('Beat #')
    ax_rr2.set_ylabel('RR (ms)')
    ax_rr2.set_title(f'Counterfactual RR (CV={cf_cv:.3f})')
    ax_rr2.grid(True, alpha=0.3)
    
    # Summary
    ax_sum = fig.add_subplot(gs[3, :])
    ax_sum.axis('off')
    
    cv_change = cf_cv - orig_cv
    expected = 'INCREASE' if target_class == 1 else 'DECREASE'
    actual = 'INCREASED' if cv_change > 0 else 'DECREASED'
    correct = (target_class == 1 and cv_change > 0) or (target_class == 0 and cv_change < 0)
    flipped = (cf_prob > 0.5) if target_class == 1 else (cf_prob < 0.5)
    valid = check_validity(cf_np, FS)
    
    summary = f"""
    SAMPLE {idx} | {class_names[orig_class]} → {class_names[target_class]}
    ═══════════════════════════════════════════════════════════════════════════════════════
    RR CV: {orig_cv:.3f} → {cf_cv:.3f} ({'+' if cv_change > 0 else ''}{cv_change:.3f})  |  Expected: {expected}  Actual: {actual}  {'✓' if correct else '✗'}
    CLASSIFIER: P({class_names[target_class]}) = {orig_prob:.3f} → {cf_prob:.3f}  {'✓ FLIPPED' if flipped else '✗ NOT FLIPPED'}  |  VALID: {'✓' if valid else '✗'}
    """
    ax_sum.text(0.02, 0.5, summary, fontfamily='monospace', fontsize=12, va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'corr': corr, 'flipped': flipped, 'valid': valid,
        'orig_cv': orig_cv, 'cf_cv': cf_cv, 'correct_rr': correct
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("HYBRID COUNTERFACTUAL GENERATOR V8")
    print("Physiological Modification + Diffusion Refinement")
    print("="*70)
    print(f"Results: {RESULTS_DIR}")
    
    # Load diffusion model
    print("\nLoading diffusion model...")
    unet = ECGUNet(in_ch=1, model_ch=64, num_res_blocks=2, attn_resolutions=(2, 3),
                   ch_mult=(1, 2, 4, 8), num_classes=2, dropout=0.1).to(DEVICE)
    checkpoint = torch.load(PHASE2_MODEL, map_location=DEVICE)
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()
    
    # Load classifier
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
    print(f"Data: {len(normal_idx)} Normal, {len(afib_idx)} AFib")
    
    generator = HybridCounterfactualGenerator(classifier, unet, DEVICE)
    
    # Generate
    print("\n" + "="*70)
    print("GENERATING COUNTERFACTUALS")
    print("="*70)
    
    num_samples = 10
    results = {'n2a': [], 'a2n': []}
    
    print("\n--- Normal → AFib ---")
    for i in tqdm(range(num_samples), desc="N→AF"):
        x = val_signals[normal_idx[i]:normal_idx[i]+1].to(DEVICE)
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 1].item()
        
        cf = generator.generate_normal_to_afib(x)
        
        with torch.no_grad():
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 1].item()
        
        save_path = RESULTS_DIR / f'n2a_{i+1:02d}.png'
        result = create_visualization(x[0,0], cf[0,0], 0, 1, orig_prob, cf_prob, i+1, save_path)
        results['n2a'].append(result)
        print(f"  #{i+1}: P(AF) {orig_prob:.3f}→{cf_prob:.3f} | RR CV: {result['orig_cv']:.3f}→{result['cf_cv']:.3f}")
    
    print("\n--- AFib → Normal ---")
    for i in tqdm(range(num_samples), desc="AF→N"):
        x = val_signals[afib_idx[i]:afib_idx[i]+1].to(DEVICE)
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 0].item()
        
        cf = generator.generate_afib_to_normal(x)
        
        with torch.no_grad():
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 0].item()
        
        save_path = RESULTS_DIR / f'a2n_{i+1:02d}.png'
        result = create_visualization(x[0,0], cf[0,0], 1, 0, orig_prob, cf_prob, i+1, save_path)
        results['a2n'].append(result)
        print(f"  #{i+1}: P(N) {orig_prob:.3f}→{cf_prob:.3f} | RR CV: {result['orig_cv']:.3f}→{result['cf_cv']:.3f}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    n2a = results['n2a']
    a2n = results['a2n']
    
    n2a_valid = sum(r['valid'] for r in n2a)
    n2a_flip = sum(r['flipped'] for r in n2a)
    n2a_rr = sum(r['correct_rr'] for r in n2a)
    
    a2n_valid = sum(r['valid'] for r in a2n)
    a2n_flip = sum(r['flipped'] for r in a2n)
    a2n_rr = sum(r['correct_rr'] for r in a2n)
    
    print(f"\nNormal → AFib: Valid={n2a_valid}/{num_samples}, Flip={n2a_flip}/{num_samples}, RR={n2a_rr}/{num_samples}")
    print(f"AFib → Normal: Valid={a2n_valid}/{num_samples}, Flip={a2n_flip}/{num_samples}, RR={a2n_rr}/{num_samples}")
    
    total = 2 * num_samples
    print(f"\nOVERALL: Valid={(n2a_valid+a2n_valid)}/{total}, Flip={(n2a_flip+a2n_flip)}/{total}, RR correct={(n2a_rr+a2n_rr)}/{total}")
    
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump({
            'n2a': {'valid': n2a_valid/num_samples, 'flip': n2a_flip/num_samples, 'rr': n2a_rr/num_samples},
            'a2n': {'valid': a2n_valid/num_samples, 'flip': a2n_flip/num_samples, 'rr': a2n_rr/num_samples}
        }, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
