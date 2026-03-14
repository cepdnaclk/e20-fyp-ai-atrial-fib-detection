"""
Phase 3: Physiologically Meaningful Counterfactual ECG Generator V7
===================================================================

KEY INSIGHT: Previous approaches added adversarial noise - that's NOT clinically meaningful!

What makes Normal vs AFib different (class-discriminative features):
1. R-R INTERVALS: Normal = regular, AFib = irregular (variable)
2. P-WAVES: Normal = present before QRS, AFib = absent/fibrillatory waves

This approach:
1. Detect individual heartbeats in the original signal
2. Extract beat templates (QRS-T complexes)
3. For Normal→AFib: Re-place beats with IRREGULAR timing + remove P-waves
4. For AFib→Normal: Re-place beats with REGULAR timing + add P-wave templates

This preserves the QRS morphology while changing ONLY the rhythm - which is the 
actual clinical difference between Normal and AFib!
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
from scipy.interpolate import interp1d
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, '/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')
from model_architecture import AFibResLSTM, ModelConfig

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/beat_manipulation_v7'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

REAL_MEAN, REAL_STD = -0.00396, 0.14716
FS = 500  # Sampling frequency
SIGNAL_LENGTH = 2500  # 5 seconds


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
# Beat Detection and Segmentation
# ============================================================================

def detect_r_peaks(signal, fs=500):
    """Detect R-peaks using Pan-Tompkins like approach."""
    signal = signal.flatten()
    
    # Bandpass filter (5-15 Hz for QRS)
    nyq = fs / 2
    low = 5 / nyq
    high = 15 / nyq
    b, a = scipy_signal.butter(2, [low, high], btype='band')
    filtered = scipy_signal.filtfilt(b, a, signal)
    
    # Differentiate
    diff = np.diff(filtered)
    
    # Square
    squared = diff ** 2
    
    # Moving average
    window = int(0.08 * fs)  # 80ms window
    ma = np.convolve(squared, np.ones(window)/window, mode='same')
    
    # Find peaks
    min_distance = int(0.3 * fs)  # Minimum 300ms between beats (max 200 BPM)
    peaks, properties = scipy_signal.find_peaks(ma, distance=min_distance, height=np.max(ma)*0.1)
    
    # Refine peak positions using original signal
    refined_peaks = []
    search_window = int(0.05 * fs)  # 50ms search window
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(signal), peak + search_window)
        local_max = start + np.argmax(signal[start:end])
        refined_peaks.append(local_max)
    
    return np.array(refined_peaks)


def segment_beats(signal, r_peaks, fs=500, pre_r=0.2, post_r=0.4):
    """
    Segment signal into individual beats.
    
    Args:
        signal: ECG signal
        r_peaks: R-peak locations
        pre_r: Seconds before R-peak to include (for P-wave)
        post_r: Seconds after R-peak to include (for T-wave)
    
    Returns:
        List of (beat_signal, r_position_in_beat) tuples
    """
    signal = signal.flatten()
    pre_samples = int(pre_r * fs)
    post_samples = int(post_r * fs)
    beat_length = pre_samples + post_samples
    
    beats = []
    for r_peak in r_peaks:
        start = r_peak - pre_samples
        end = r_peak + post_samples
        
        if start < 0 or end > len(signal):
            continue
        
        beat = signal[start:end]
        beats.append({
            'signal': beat,
            'r_position': pre_samples,
            'original_r_peak': r_peak
        })
    
    return beats


def get_p_wave_template(fs=500):
    """Generate a typical P-wave template."""
    # P-wave is typically ~100ms duration, small positive deflection
    duration_ms = 100
    duration_samples = int(duration_ms / 1000 * fs)
    
    t = np.linspace(0, np.pi, duration_samples)
    p_wave = 0.1 * np.sin(t)  # Small positive hump
    
    return p_wave


def remove_p_wave(beat, r_position, fs=500):
    """Remove P-wave from a beat (for Normal→AFib)."""
    # P-wave is typically 120-200ms before R-peak
    p_start = int(r_position - 0.2 * fs)
    p_end = int(r_position - 0.08 * fs)
    
    if p_start < 0:
        p_start = 0
    
    # Replace P-wave region with interpolated baseline
    beat_modified = beat.copy()
    
    if p_start < p_end and p_end < len(beat):
        # Get values at boundaries
        start_val = beat[p_start]
        end_val = beat[p_end]
        
        # Linear interpolation
        beat_modified[p_start:p_end] = np.linspace(start_val, end_val, p_end - p_start)
        
        # Add small fibrillatory oscillations (characteristic of AFib)
        noise = np.random.randn(p_end - p_start) * 0.02
        beat_modified[p_start:p_end] += noise
    
    return beat_modified


def add_p_wave(beat, r_position, fs=500):
    """Add P-wave to a beat (for AFib→Normal)."""
    # P-wave should be 160-200ms before R-peak
    p_template = get_p_wave_template(fs)
    
    # Position P-wave center at 160ms before R
    p_center = int(r_position - 0.16 * fs)
    p_start = p_center - len(p_template) // 2
    p_end = p_start + len(p_template)
    
    beat_modified = beat.copy()
    
    if p_start >= 0 and p_end < r_position:
        # Add P-wave (scaled to match signal amplitude)
        signal_amplitude = np.std(beat)
        beat_modified[p_start:p_end] += p_template * signal_amplitude * 0.5
    
    return beat_modified


# ============================================================================
# Rhythm Modification
# ============================================================================

def compute_rr_variability(r_peaks, fs=500):
    """Compute R-R interval statistics."""
    if len(r_peaks) < 2:
        return {'mean': 0, 'std': 0, 'cv': 0}
    
    rr_intervals = np.diff(r_peaks) / fs * 1000  # Convert to ms
    
    return {
        'mean': np.mean(rr_intervals),
        'std': np.std(rr_intervals),
        'cv': np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-8)
    }


def generate_irregular_rr_intervals(mean_rr_ms, num_intervals, cv_target=0.15):
    """
    Generate irregular R-R intervals for AFib.
    
    AFib typically has CV (coefficient of variation) of 10-20% or higher.
    Normal rhythm has CV < 5%.
    """
    # Generate intervals with random variation
    std = mean_rr_ms * cv_target
    intervals = np.random.normal(mean_rr_ms, std, num_intervals)
    
    # Ensure physiologically valid (300-1500ms, i.e., 40-200 BPM)
    intervals = np.clip(intervals, 300, 1500)
    
    return intervals


def generate_regular_rr_intervals(mean_rr_ms, num_intervals, cv_target=0.02):
    """
    Generate regular R-R intervals for Normal rhythm.
    
    Normal sinus rhythm has very low RR variability (CV < 5%).
    """
    # Small respiratory sinus arrhythmia variation
    std = mean_rr_ms * cv_target
    intervals = np.random.normal(mean_rr_ms, std, num_intervals)
    
    # Ensure physiologically valid
    intervals = np.clip(intervals, 400, 1200)
    
    return intervals


def reconstruct_signal_from_beats(beats, rr_intervals_ms, fs=500, target_length=2500):
    """
    Reconstruct ECG signal by placing beats at specified R-R intervals.
    
    Args:
        beats: List of beat dictionaries with 'signal' and 'r_position'
        rr_intervals_ms: Desired R-R intervals in milliseconds
        fs: Sampling frequency
        target_length: Target signal length in samples
    
    Returns:
        Reconstructed signal
    """
    # Initialize output signal
    output = np.zeros(target_length)
    
    # Convert RR intervals to samples
    rr_samples = (rr_intervals_ms / 1000 * fs).astype(int)
    
    # Calculate R-peak positions
    r_positions = [100]  # Start first R-peak at 100 samples
    for rr in rr_samples:
        next_r = r_positions[-1] + rr
        if next_r < target_length - 200:  # Leave room for T-wave
            r_positions.append(int(next_r))
    
    # Place beats at calculated positions
    for i, r_pos in enumerate(r_positions):
        beat_idx = i % len(beats)  # Cycle through available beats
        beat = beats[beat_idx]
        
        beat_signal = beat['signal']
        beat_r_pos = beat['r_position']
        
        # Calculate where to place this beat
        start = r_pos - beat_r_pos
        end = start + len(beat_signal)
        
        if start < 0:
            beat_signal = beat_signal[-start:]
            start = 0
        if end > target_length:
            beat_signal = beat_signal[:target_length - start]
            end = target_length
        
        # Blend with existing signal (avoid sharp transitions)
        if end - start == len(beat_signal):
            # Create blending weights
            blend_len = min(20, len(beat_signal) // 4)
            weights = np.ones(len(beat_signal))
            weights[:blend_len] = np.linspace(0, 1, blend_len)
            weights[-blend_len:] = np.linspace(1, 0, blend_len)
            
            output[start:end] = output[start:end] * (1 - weights) + beat_signal * weights
    
    return output


# ============================================================================
# Counterfactual Generator
# ============================================================================

class PhysiologicalCounterfactualGenerator:
    """
    Generate counterfactual ECGs by modifying rhythm characteristics.
    
    Normal → AFib: Make RR intervals irregular, remove P-waves
    AFib → Normal: Make RR intervals regular, add P-waves
    """
    
    def __init__(self, classifier, device='cuda'):
        self.classifier = classifier
        self.device = device
    
    def generate_normal_to_afib(self, signal):
        """
        Convert Normal ECG to AFib by:
        1. Making R-R intervals irregular (increased CV)
        2. Removing P-waves
        3. Adding subtle fibrillatory oscillations
        """
        signal_np = signal.cpu().numpy().flatten()
        
        # Detect R-peaks
        r_peaks = detect_r_peaks(signal_np, FS)
        
        if len(r_peaks) < 3:
            return signal  # Can't modify if too few beats
        
        # Segment into beats
        beats = segment_beats(signal_np, r_peaks, FS)
        
        if len(beats) < 2:
            return signal
        
        # Get current RR statistics
        rr_stats = compute_rr_variability(r_peaks, FS)
        mean_rr = rr_stats['mean']
        
        # Modify beats: remove P-waves and add fibrillatory baseline
        modified_beats = []
        for beat in beats:
            # Remove P-wave
            modified_signal = remove_p_wave(beat['signal'], beat['r_position'], FS)
            modified_beats.append({
                'signal': modified_signal,
                'r_position': beat['r_position']
            })
        
        # Generate irregular RR intervals (AFib-like)
        num_intervals = len(r_peaks)
        irregular_rr = generate_irregular_rr_intervals(
            mean_rr, num_intervals, 
            cv_target=0.15 + np.random.rand() * 0.1  # CV between 0.15-0.25
        )
        
        # Reconstruct signal with irregular timing
        counterfactual = reconstruct_signal_from_beats(
            modified_beats, irregular_rr, FS, SIGNAL_LENGTH
        )
        
        # Add subtle baseline wander and fibrillatory waves
        t = np.linspace(0, 5, SIGNAL_LENGTH)
        fibrillatory = 0.02 * np.sin(2 * np.pi * 5 * t + np.random.rand() * 2 * np.pi)
        fibrillatory += 0.01 * np.sin(2 * np.pi * 7 * t + np.random.rand() * 2 * np.pi)
        counterfactual += fibrillatory * np.std(signal_np)
        
        # Convert back to tensor
        cf_tensor = torch.tensor(counterfactual, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return cf_tensor
    
    def generate_afib_to_normal(self, signal):
        """
        Convert AFib ECG to Normal by:
        1. Regularizing R-R intervals (low CV)
        2. Adding clear P-waves
        3. Removing fibrillatory baseline
        """
        signal_np = signal.cpu().numpy().flatten()
        
        # Detect R-peaks
        r_peaks = detect_r_peaks(signal_np, FS)
        
        if len(r_peaks) < 3:
            return signal
        
        # Segment into beats
        beats = segment_beats(signal_np, r_peaks, FS)
        
        if len(beats) < 2:
            return signal
        
        # Get current RR statistics
        rr_stats = compute_rr_variability(r_peaks, FS)
        mean_rr = rr_stats['mean']
        
        # Smooth the beats to remove fibrillatory waves
        # and add P-waves
        modified_beats = []
        for beat in beats:
            # Smooth the signal slightly to remove high-freq noise
            b, a = scipy_signal.butter(3, 40 / (FS/2), btype='low')
            smoothed = scipy_signal.filtfilt(b, a, beat['signal'])
            
            # Add P-wave
            modified_signal = add_p_wave(smoothed, beat['r_position'], FS)
            
            modified_beats.append({
                'signal': modified_signal,
                'r_position': beat['r_position']
            })
        
        # Generate regular RR intervals (Normal sinus rhythm)
        num_intervals = len(r_peaks)
        regular_rr = generate_regular_rr_intervals(
            mean_rr, num_intervals,
            cv_target=0.02 + np.random.rand() * 0.02  # CV between 0.02-0.04
        )
        
        # Reconstruct signal with regular timing
        counterfactual = reconstruct_signal_from_beats(
            modified_beats, regular_rr, FS, SIGNAL_LENGTH
        )
        
        # Convert back to tensor
        cf_tensor = torch.tensor(counterfactual, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        return cf_tensor


# ============================================================================
# Analysis and Visualization
# ============================================================================

def compute_features(signal, fs=500):
    """Compute clinical features."""
    signal = np.array(signal).flatten()
    r_peaks = detect_r_peaks(signal, fs)
    rr_stats = compute_rr_variability(r_peaks, fs)
    
    return {
        'num_beats': len(r_peaks),
        'rr_mean_ms': rr_stats['mean'],
        'rr_std_ms': rr_stats['std'],
        'rr_cv': rr_stats['cv'],
        'hr_bpm': 60000 / rr_stats['mean'] if rr_stats['mean'] > 0 else 0
    }


def check_validity(signal, fs=500):
    signal = np.array(signal).flatten()
    amp = np.max(signal) - np.min(signal)
    r_peaks = detect_r_peaks(signal, fs)
    
    if len(r_peaks) >= 2:
        hr = 60 / (np.mean(np.diff(r_peaks)) / fs)
    else:
        hr = 0
    
    valid = (0.01 < amp < 10) and (len(r_peaks) >= 2) and (30 < hr < 200)
    return valid


def create_visualization(orig, cf, orig_class, target_class, orig_prob, cf_prob, idx, save_path):
    """Create detailed visualization of counterfactual."""
    orig_np = orig.cpu().numpy().flatten()
    cf_np = cf.cpu().numpy().flatten()
    
    time = np.arange(len(orig_np)) / FS
    
    orig_feat = compute_features(orig_np, FS)
    cf_feat = compute_features(cf_np, FS)
    
    # Calculate correlation
    corr, _ = pearsonr(orig_np, cf_np)
    
    # Detect R-peaks for visualization
    orig_peaks = detect_r_peaks(orig_np, FS)
    cf_peaks = detect_r_peaks(cf_np, FS)
    
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[1.5, 1, 1, 1, 0.8])
    
    class_names = ['Normal', 'AFib']
    
    # Row 1: Full signals (separate, not overlaid)
    ax_orig = fig.add_subplot(gs[0, :2])
    ax_orig.plot(time, orig_np, 'b-', lw=1)
    for p in orig_peaks:
        ax_orig.axvline(x=p/FS, color='blue', alpha=0.3, lw=0.5)
    ax_orig.set_xlabel('Time (s)')
    ax_orig.set_ylabel('Amplitude')
    ax_orig.set_title(f'ORIGINAL ({class_names[orig_class]}) | HR: {orig_feat["hr_bpm"]:.0f} BPM | RR CV: {orig_feat["rr_cv"]:.3f}', 
                      fontsize=12, fontweight='bold', color='blue')
    ax_orig.grid(True, alpha=0.3)
    
    ax_cf = fig.add_subplot(gs[0, 2:])
    ax_cf.plot(time, cf_np, 'r-', lw=1)
    for p in cf_peaks:
        ax_cf.axvline(x=p/FS, color='red', alpha=0.3, lw=0.5)
    ax_cf.set_xlabel('Time (s)')
    ax_cf.set_ylabel('Amplitude')
    ax_cf.set_title(f'COUNTERFACTUAL (→{class_names[target_class]}) | HR: {cf_feat["hr_bpm"]:.0f} BPM | RR CV: {cf_feat["rr_cv"]:.3f}', 
                    fontsize=12, fontweight='bold', color='red')
    ax_cf.grid(True, alpha=0.3)
    
    # Row 2: Zoomed segments side by side
    for col, (start, end) in enumerate([(0, 500), (1000, 1500)]):
        # Original
        ax = fig.add_subplot(gs[1, col])
        ax.plot(time[start:end], orig_np[start:end], 'b-', lw=1.5)
        ax.set_title(f'Original {time[start]:.1f}-{time[end-1]:.1f}s')
        ax.grid(True, alpha=0.3)
        
        # Counterfactual
        ax = fig.add_subplot(gs[1, col + 2])
        ax.plot(time[start:end], cf_np[start:end], 'r-', lw=1.5)
        ax.set_title(f'CF {time[start]:.1f}-{time[end-1]:.1f}s')
        ax.grid(True, alpha=0.3)
    
    # Row 3: R-R interval comparison
    ax_rr_orig = fig.add_subplot(gs[2, 0:2])
    if len(orig_peaks) > 1:
        orig_rr = np.diff(orig_peaks) / FS * 1000
        ax_rr_orig.bar(range(len(orig_rr)), orig_rr, color='blue', alpha=0.7)
        ax_rr_orig.axhline(y=np.mean(orig_rr), color='darkblue', linestyle='--', lw=2, 
                          label=f'Mean: {np.mean(orig_rr):.0f}ms')
        ax_rr_orig.fill_between(range(len(orig_rr)), 
                                np.mean(orig_rr) - np.std(orig_rr),
                                np.mean(orig_rr) + np.std(orig_rr),
                                alpha=0.2, color='blue')
    ax_rr_orig.set_xlabel('Beat #')
    ax_rr_orig.set_ylabel('RR Interval (ms)')
    ax_rr_orig.set_title(f'Original RR Intervals | CV = {orig_feat["rr_cv"]:.3f}')
    ax_rr_orig.legend()
    ax_rr_orig.grid(True, alpha=0.3)
    
    ax_rr_cf = fig.add_subplot(gs[2, 2:])
    if len(cf_peaks) > 1:
        cf_rr = np.diff(cf_peaks) / FS * 1000
        ax_rr_cf.bar(range(len(cf_rr)), cf_rr, color='red', alpha=0.7)
        ax_rr_cf.axhline(y=np.mean(cf_rr), color='darkred', linestyle='--', lw=2,
                        label=f'Mean: {np.mean(cf_rr):.0f}ms')
        ax_rr_cf.fill_between(range(len(cf_rr)),
                              np.mean(cf_rr) - np.std(cf_rr),
                              np.mean(cf_rr) + np.std(cf_rr),
                              alpha=0.2, color='red')
    ax_rr_cf.set_xlabel('Beat #')
    ax_rr_cf.set_ylabel('RR Interval (ms)')
    ax_rr_cf.set_title(f'Counterfactual RR Intervals | CV = {cf_feat["rr_cv"]:.3f}')
    ax_rr_cf.legend()
    ax_rr_cf.grid(True, alpha=0.3)
    
    # Row 4: Poincaré plots
    ax_poin_orig = fig.add_subplot(gs[3, 0:2])
    if len(orig_peaks) > 2:
        orig_rr = np.diff(orig_peaks) / FS * 1000
        ax_poin_orig.scatter(orig_rr[:-1], orig_rr[1:], c='blue', alpha=0.7, s=80)
        # Add identity line
        lim_min, lim_max = min(orig_rr) * 0.9, max(orig_rr) * 1.1
        ax_poin_orig.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3)
    ax_poin_orig.set_xlabel('RR_n (ms)')
    ax_poin_orig.set_ylabel('RR_{n+1} (ms)')
    ax_poin_orig.set_title(f'Original Poincaré Plot')
    ax_poin_orig.set_aspect('equal')
    ax_poin_orig.grid(True, alpha=0.3)
    
    ax_poin_cf = fig.add_subplot(gs[3, 2:])
    if len(cf_peaks) > 2:
        cf_rr = np.diff(cf_peaks) / FS * 1000
        ax_poin_cf.scatter(cf_rr[:-1], cf_rr[1:], c='red', alpha=0.7, s=80)
        lim_min, lim_max = min(cf_rr) * 0.9, max(cf_rr) * 1.1
        ax_poin_cf.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.3)
    ax_poin_cf.set_xlabel('RR_n (ms)')
    ax_poin_cf.set_ylabel('RR_{n+1} (ms)')
    ax_poin_cf.set_title(f'Counterfactual Poincaré Plot')
    ax_poin_cf.set_aspect('equal')
    ax_poin_cf.grid(True, alpha=0.3)
    
    # Row 5: Summary
    ax_summary = fig.add_subplot(gs[4, :])
    ax_summary.axis('off')
    
    rr_cv_change = cf_feat['rr_cv'] - orig_feat['rr_cv']
    expected_cv = 'INCREASE' if target_class == 1 else 'DECREASE'
    actual_cv = 'INCREASED' if rr_cv_change > 0 else 'DECREASED'
    correct_cv = (target_class == 1 and rr_cv_change > 0) or (target_class == 0 and rr_cv_change < 0)
    
    flipped = (cf_prob > 0.5) if target_class == 1 else (cf_prob < 0.5)
    valid = check_validity(cf_np, FS)
    
    summary = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  SAMPLE {idx}                                                                                                                        ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  TRANSFORMATION: {class_names[orig_class]} → {class_names[target_class]}                                                                                                  ║
    ║                                                                                                                                   ║
    ║  RR CV (Rhythm Irregularity):  {orig_feat['rr_cv']:.3f} → {cf_feat['rr_cv']:.3f}  ({'+' if rr_cv_change > 0 else ''}{rr_cv_change:.3f})                                                              ║
    ║      Expected: {expected_cv}  |  Actual: {actual_cv}  {'✓ CORRECT' if correct_cv else '✗ WRONG':15}                                                                  ║
    ║                                                                                                                                   ║
    ║  CLASSIFIER:  P({class_names[target_class]}) = {orig_prob:.3f} → {cf_prob:.3f}  {'✓ FLIPPED' if flipped else '✗ NOT FLIPPED':15}                                                              ║
    ║  VALID ECG:   {'✓ YES' if valid else '✗ NO':10}                                                                                                           ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    ax_summary.text(0.02, 0.5, summary, fontfamily='monospace', fontsize=11, va='center', transform=ax_summary.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'corr': corr,
        'flipped': flipped,
        'valid': valid,
        'orig_rr_cv': orig_feat['rr_cv'],
        'cf_rr_cv': cf_feat['rr_cv'],
        'correct_rr': correct_cv,
        'orig_prob': orig_prob,
        'cf_prob': cf_prob
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PHYSIOLOGICAL COUNTERFACTUAL ECG GENERATOR V7")
    print("Beat-Level Rhythm Modification")
    print("="*70)
    print(f"Results: {RESULTS_DIR}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load classifier
    print("\nLoading classifier...")
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
    
    # Test classifier
    with torch.no_grad():
        test = val_signals[:200].to(DEVICE)
        preds = classifier(test).argmax(dim=1)
        acc = (preds == val_labels[:200].to(DEVICE)).float().mean().item() * 100
        print(f"Classifier accuracy: {acc:.1f}%")
    
    generator = PhysiologicalCounterfactualGenerator(classifier, DEVICE)
    
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
    print("Making rhythm IRREGULAR, removing P-waves...")
    
    for i in tqdm(range(num_samples), desc="N→AF"):
        x = val_signals[normal_idx[i]:normal_idx[i]+1].to(DEVICE)
        
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 1].item()
        
        cf = generator.generate_normal_to_afib(x)
        
        with torch.no_grad():
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 1].item()
        
        save_path = RESULTS_DIR / f'n2a_{i+1:02d}.png'
        result = create_visualization(x[0, 0], cf[0, 0], 0, 1, orig_prob, cf_prob, i+1, save_path)
        results['normal_to_afib'].append(result)
        
        print(f"  #{i+1}: P(AF) {orig_prob:.3f}→{cf_prob:.3f} | RR CV: {result['orig_rr_cv']:.3f}→{result['cf_rr_cv']:.3f}")
    
    # AFib → Normal
    print("\n--- AFib → Normal ---")
    print("Making rhythm REGULAR, adding P-waves...")
    
    for i in tqdm(range(num_samples), desc="AF→N"):
        x = val_signals[afib_idx[i]:afib_idx[i]+1].to(DEVICE)
        
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 0].item()
        
        cf = generator.generate_afib_to_normal(x)
        
        with torch.no_grad():
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 0].item()
        
        save_path = RESULTS_DIR / f'a2n_{i+1:02d}.png'
        result = create_visualization(x[0, 0], cf[0, 0], 1, 0, orig_prob, cf_prob, i+1, save_path)
        results['afib_to_normal'].append(result)
        
        print(f"  #{i+1}: P(N) {orig_prob:.3f}→{cf_prob:.3f} | RR CV: {result['orig_rr_cv']:.3f}→{result['cf_rr_cv']:.3f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    n2a = results['normal_to_afib']
    a2n = results['afib_to_normal']
    
    def summarize(data, name, direction):
        valid = sum(r['valid'] for r in data)
        flipped = sum(r['flipped'] for r in data)
        correct_rr = sum(r['correct_rr'] for r in data)
        
        avg_orig_cv = np.mean([r['orig_rr_cv'] for r in data])
        avg_cf_cv = np.mean([r['cf_rr_cv'] for r in data])
        
        print(f"\n{name}:")
        print(f"  Valid signals: {valid}/{len(data)} ({100*valid/len(data):.0f}%)")
        print(f"  Flipped: {flipped}/{len(data)} ({100*flipped/len(data):.0f}%)")
        print(f"  Correct RR change: {correct_rr}/{len(data)} ({100*correct_rr/len(data):.0f}%)")
        print(f"  RR CV: {avg_orig_cv:.3f} → {avg_cf_cv:.3f} ({direction} expected)")
        
        return valid, flipped, correct_rr
    
    n2a_valid, n2a_flip, n2a_rr = summarize(n2a, "Normal → AFib", "INCREASE")
    a2n_valid, a2n_flip, a2n_rr = summarize(a2n, "AFib → Normal", "DECREASE")
    
    # Save results
    summary = {
        'normal_to_afib': {
            'valid': n2a_valid / num_samples,
            'flip': n2a_flip / num_samples,
            'correct_rr': n2a_rr / num_samples
        },
        'afib_to_normal': {
            'valid': a2n_valid / num_samples,
            'flip': a2n_flip / num_samples,
            'correct_rr': a2n_rr / num_samples
        }
    }
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST CASE EVALUATION")
    print("="*70)
    
    total = 2 * num_samples
    all_valid = n2a_valid + a2n_valid
    all_flip = n2a_flip + a2n_flip
    all_correct_rr = n2a_rr + a2n_rr
    
    print(f"\n1. REALISTIC SIGNALS: {all_valid}/{total} ({100*all_valid/total:.0f}%) {'✓' if all_valid >= 0.8*total else '✗'}")
    print(f"2. CORRECT RR DIRECTION: {all_correct_rr}/{total} ({100*all_correct_rr/total:.0f}%) {'✓' if all_correct_rr >= 0.8*total else '✗'}")
    print(f"3. CLASSIFIER FLIP: {all_flip}/{total} ({100*all_flip/total:.0f}%) {'✓' if all_flip >= 0.5*total else '✗'}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
