"""
evaluate_model.py - Comprehensive evaluation and visualization of the trained diffusion model
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy import stats
import math

# Configuration
PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
MODEL_DIR = PROJECT_ROOT / 'models/phase2_diffusion/diffusion_v2'
RESULTS_DIR = PROJECT_ROOT / 'models/phase2_diffusion/results_v2'
EVAL_OUTPUT = RESULTS_DIR / 'final_evaluation'
EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Load data
print("Loading data...")
train_data = np.load(DATA_DIR / 'train_data.npz')
train_signals = train_data['X']
train_labels = train_data['y']

if train_signals.ndim == 2:
    train_signals = train_signals[:, np.newaxis, :]

with open(DATA_DIR / 'dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

REAL_MEAN = float(np.mean(train_signals))
REAL_STD = float(np.std(train_signals))
TARGET_MIN = metadata['normalization']['target_min']
TARGET_MAX = metadata['normalization']['target_max']

print(f"Real data: {train_signals.shape}")
print(f"Real mean: {REAL_MEAN:.6f}, std: {REAL_STD:.6f}")

# Load generated samples from latest epoch
eval_dirs = sorted(EVAL_OUTPUT.parent.glob('evaluations/epoch_*'))
if eval_dirs:
    latest_eval = eval_dirs[-1]
    print(f"Loading from: {latest_eval}")
    
    normalized_samples = np.load(latest_eval / 'normalized_samples.npy')
    raw_samples = np.load(latest_eval / 'raw_samples.npy')
    
    with open(latest_eval / 'results.json', 'r') as f:
        results = json.load(f)
else:
    print("No evaluation data found. Please run training first.")
    exit(1)

print(f"Generated samples: {normalized_samples.shape}")
print(f"Results: {json.dumps(results, indent=2)}")

# ============================================================================
# COMPREHENSIVE VISUALIZATION
# ============================================================================

print("\nCreating visualizations...")

# 1. Side-by-side ECG comparison
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
fig.suptitle('Real vs Generated ECG Comparison', fontsize=16, fontweight='bold')

for i in range(5):
    real_idx = np.random.randint(len(train_signals))
    gen_idx = i
    
    # Real ECG
    axes[i, 0].plot(train_signals[real_idx, 0, :], color='blue', linewidth=0.8)
    axes[i, 0].set_title(f'Real ECG #{real_idx}', fontsize=12)
    axes[i, 0].set_ylim([TARGET_MIN - 0.3, TARGET_MAX + 0.3])
    axes[i, 0].set_xlabel('Sample')
    axes[i, 0].set_ylabel('Amplitude')
    axes[i, 0].grid(True, alpha=0.3)
    
    # Generated ECG
    axes[i, 1].plot(normalized_samples[gen_idx, 0, :], color='red', linewidth=0.8)
    axes[i, 1].set_title(f'Generated ECG #{gen_idx}', fontsize=12)
    axes[i, 1].set_ylim([TARGET_MIN - 0.3, TARGET_MAX + 0.3])
    axes[i, 1].set_xlabel('Sample')
    axes[i, 1].set_ylabel('Amplitude')
    axes[i, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(EVAL_OUTPUT / 'ecg_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: ecg_comparison.png")

# 2. Distribution comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Statistical Distribution Comparison', fontsize=16, fontweight='bold')

# Flatten for distribution analysis
real_flat = train_signals.flatten()
gen_flat = normalized_samples.flatten()

# Histogram comparison
axes[0, 0].hist(real_flat, bins=100, alpha=0.7, density=True, label='Real', color='blue')
axes[0, 0].hist(gen_flat, bins=100, alpha=0.7, density=True, label='Generated', color='red')
axes[0, 0].set_title('Value Distribution (Normalized)')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Q-Q plot
real_quantiles = np.percentile(real_flat, np.linspace(0, 100, 100))
gen_quantiles = np.percentile(gen_flat, np.linspace(0, 100, 100))
axes[0, 1].scatter(real_quantiles, gen_quantiles, alpha=0.5, s=20)
axes[0, 1].plot([real_quantiles.min(), real_quantiles.max()], 
                [real_quantiles.min(), real_quantiles.max()], 'r--', linewidth=2, label='Perfect match')
axes[0, 1].set_title('Q-Q Plot (Real vs Generated)')
axes[0, 1].set_xlabel('Real Quantiles')
axes[0, 1].set_ylabel('Generated Quantiles')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Per-sample statistics
real_sample_stds = train_signals.reshape(train_signals.shape[0], -1).std(axis=1)
gen_sample_stds = normalized_samples.reshape(normalized_samples.shape[0], -1).std(axis=1)

axes[1, 0].hist(real_sample_stds, bins=50, alpha=0.7, density=True, label='Real', color='blue')
axes[1, 0].hist(gen_sample_stds, bins=50, alpha=0.7, density=True, label='Generated', color='red')
axes[1, 0].set_title('Per-Sample Standard Deviation Distribution')
axes[1, 0].set_xlabel('Sample Std')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Per-sample means
real_sample_means = train_signals.reshape(train_signals.shape[0], -1).mean(axis=1)
gen_sample_means = normalized_samples.reshape(normalized_samples.shape[0], -1).mean(axis=1)

axes[1, 1].hist(real_sample_means, bins=50, alpha=0.7, density=True, label='Real', color='blue')
axes[1, 1].hist(gen_sample_means, bins=50, alpha=0.7, density=True, label='Generated', color='red')
axes[1, 1].set_title('Per-Sample Mean Distribution')
axes[1, 1].set_xlabel('Sample Mean')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(EVAL_OUTPUT / 'distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: distribution_comparison.png")

# 3. Frequency domain analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Frequency Domain Analysis', fontsize=16, fontweight='bold')

# Compute average power spectrum
def compute_power_spectrum(signals):
    ffts = np.abs(np.fft.rfft(signals, axis=-1))
    return np.mean(ffts ** 2, axis=(0, 1))

real_spectrum = compute_power_spectrum(train_signals[:1000])
gen_spectrum = compute_power_spectrum(normalized_samples)

freqs = np.fft.rfftfreq(train_signals.shape[-1])

axes[0, 0].semilogy(freqs, real_spectrum, label='Real', alpha=0.8, color='blue')
axes[0, 0].semilogy(freqs, gen_spectrum, label='Generated', alpha=0.8, color='red')
axes[0, 0].set_title('Average Power Spectrum')
axes[0, 0].set_xlabel('Normalized Frequency')
axes[0, 0].set_ylabel('Power (log scale)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Linear scale (low frequencies)
axes[0, 1].plot(freqs[:100], real_spectrum[:100], label='Real', alpha=0.8, color='blue')
axes[0, 1].plot(freqs[:100], gen_spectrum[:100], label='Generated', alpha=0.8, color='red')
axes[0, 1].set_title('Power Spectrum (Low Frequencies)')
axes[0, 1].set_xlabel('Normalized Frequency')
axes[0, 1].set_ylabel('Power')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Individual sample FFT comparison
real_fft = np.abs(np.fft.rfft(train_signals[0, 0, :]))
gen_fft = np.abs(np.fft.rfft(normalized_samples[0, 0, :]))

axes[1, 0].plot(freqs, real_fft, label='Real Sample', alpha=0.8, color='blue')
axes[1, 0].plot(freqs, gen_fft, label='Generated Sample', alpha=0.8, color='red')
axes[1, 0].set_title('Single Sample FFT Comparison')
axes[1, 0].set_xlabel('Normalized Frequency')
axes[1, 0].set_ylabel('Magnitude')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Spectral correlation
axes[1, 1].scatter(real_spectrum[:500], gen_spectrum[:500], alpha=0.5, s=10)
axes[1, 1].set_title('Spectral Correlation')
axes[1, 1].set_xlabel('Real Power')
axes[1, 1].set_ylabel('Generated Power')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(EVAL_OUTPUT / 'frequency_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: frequency_analysis.png")

# 4. Summary metrics
fig, ax = plt.subplots(figsize=(10, 8))

metrics = {
    'Variance Ratio': results['variance_ratio'],
    'KS Statistic': results['ks_stat'],
    'Real Mean': results['real_mean'],
    'Gen Mean': results['gen_mean'],
    'Real Std': results['real_std'],
    'Gen Std': results['gen_std'],
}

thresholds = {
    'Variance Ratio': (0.90, 1.10),
    'KS Statistic': (0, 0.10),
}

# Create bar chart
names = list(metrics.keys())
values = list(metrics.values())
colors = ['green' if 'Ratio' in n or 'KS' in n else 'steelblue' for n in names]

bars = ax.barh(names, values, color=colors, alpha=0.7)
ax.set_xlabel('Value')
ax.set_title('Model Quality Metrics', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
            va='center', fontsize=10)

plt.tight_layout()
plt.savefig(EVAL_OUTPUT / 'metrics_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: metrics_summary.png")

# 5. Grid of generated samples
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle('Grid of Generated ECG Samples', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < len(normalized_samples):
        ax.plot(normalized_samples[i, 0, :], color='red', linewidth=0.6)
        ax.set_ylim([TARGET_MIN - 0.3, TARGET_MAX + 0.3])
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig(EVAL_OUTPUT / 'generated_samples_grid.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: generated_samples_grid.png")

# ============================================================================
# FINAL QUALITY REPORT
# ============================================================================

print("\n" + "=" * 80)
print("FINAL QUALITY REPORT")
print("=" * 80)

print(f"\nEpoch: {results['epoch']}")
print(f"\n--- Key Metrics ---")
print(f"Variance Ratio:  {results['variance_ratio']:.4f}  (Target: 0.90-1.10) {'PASS' if 0.90 <= results['variance_ratio'] <= 1.10 else 'FAIL'}")
print(f"KS Statistic:    {results['ks_stat']:.4f}  (Target: < 0.10) {'PASS' if results['ks_stat'] < 0.10 else 'FAIL'}")

print(f"\n--- Statistics Comparison ---")
print(f"{'Metric':<20} {'Real':<15} {'Generated':<15} {'Match %':<10}")
print("-" * 60)
print(f"{'Mean':<20} {results['real_mean']:<15.6f} {results['gen_mean']:<15.6f} {100 * (1 - abs(results['real_mean'] - results['gen_mean']) / (abs(results['real_mean']) + 1e-8)):.1f}%")
print(f"{'Std':<20} {results['real_std']:<15.6f} {results['gen_std']:<15.6f} {100 * min(results['real_std'], results['gen_std']) / max(results['real_std'], results['gen_std']):.1f}%")

# Compute additional metrics
ks_pass = results['ks_stat'] < 0.10
var_pass = 0.90 <= results['variance_ratio'] <= 1.10

overall_similarity = 100 * (1 - results['ks_stat']) * min(1.0, 1.0 / abs(results['variance_ratio'] - 1.0 + 1.0))

print(f"\n--- Overall Assessment ---")
print(f"Distribution Similarity (1 - KS): {100 * (1 - results['ks_stat']):.1f}%")
print(f"Variance Match: {100 * min(results['real_std'], results['gen_std']) / max(results['real_std'], results['gen_std']):.1f}%")

if ks_pass and var_pass:
    print(f"\n*** MODEL PASSED 90% SIMILARITY THRESHOLD ***")
    print(f"The generated ECG data is statistically similar to real ECG data.")
else:
    print(f"\nModel needs more training.")

print(f"\nVisualizations saved to: {EVAL_OUTPUT}")
print("=" * 80)

# Save final report
report = {
    'epoch': results['epoch'],
    'variance_ratio': results['variance_ratio'],
    'ks_statistic': results['ks_stat'],
    'distribution_similarity_pct': float(100 * (1 - results['ks_stat'])),
    'variance_match_pct': float(100 * min(results['real_std'], results['gen_std']) / max(results['real_std'], results['gen_std'])),
    'real_mean': results['real_mean'],
    'real_std': results['real_std'],
    'gen_mean': results['gen_mean'],
    'gen_std': results['gen_std'],
    'passed_90_similarity': bool(ks_pass and var_pass),
}

with open(EVAL_OUTPUT / 'final_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\nDone!")
