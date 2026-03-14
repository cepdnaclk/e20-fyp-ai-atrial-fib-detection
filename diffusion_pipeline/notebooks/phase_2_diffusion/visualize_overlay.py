"""
visualize_overlay.py - Overlay visualization with denormalized ECG values
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json

# Configuration
PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
RESULTS_DIR = PROJECT_ROOT / 'models/phase2_diffusion/results_v2'
EVAL_OUTPUT = RESULTS_DIR / 'final_evaluation'
EVAL_OUTPUT.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
train_data = np.load(DATA_DIR / 'train_data.npz')
train_signals = train_data['X']

if train_signals.ndim == 2:
    train_signals = train_signals[:, np.newaxis, :]

with open(DATA_DIR / 'dataset_metadata.json', 'r') as f:
    metadata = json.load(f)

# Get normalization parameters
norm_params = metadata['normalization']
CLEAN_MIN = norm_params['clean_data_min']
CLEAN_MAX = norm_params['clean_data_max']
TARGET_MIN = norm_params['target_min']
TARGET_MAX = norm_params['target_max']

print(f"Normalization: [{CLEAN_MIN:.4f}, {CLEAN_MAX:.4f}] -> [{TARGET_MIN:.4f}, {TARGET_MAX:.4f}]")

# Function to denormalize to original mV scale
def denormalize(signals, target_min=TARGET_MIN, target_max=TARGET_MAX, 
                original_min=CLEAN_MIN, original_max=CLEAN_MAX):
    """Convert from normalized [-1.5, 1.5] back to original mV scale"""
    # Reverse the normalization
    normalized_01 = (signals - target_min) / (target_max - target_min)
    original = normalized_01 * (original_max - original_min) + original_min
    return original

# Also create a visual-friendly scale (millivolts typical ECG range)
def to_millivolts(signals):
    """Scale to typical ECG millivolt range (-2 to +2 mV)"""
    # Normalize to 0-1 first
    sig_min, sig_max = signals.min(), signals.max()
    normalized = (signals - sig_min) / (sig_max - sig_min + 1e-8)
    # Scale to -2 to +2 mV range (typical ECG)
    return normalized * 4 - 2

# Load generated samples
eval_dirs = sorted(EVAL_OUTPUT.parent.glob('evaluations/epoch_*'))
if eval_dirs:
    latest_eval = eval_dirs[-1]
    print(f"Loading from: {latest_eval}")
    normalized_samples = np.load(latest_eval / 'normalized_samples.npy')
else:
    print("No evaluation data found!")
    exit(1)

print(f"Real data shape: {train_signals.shape}")
print(f"Generated shape: {normalized_samples.shape}")

# Denormalize both
real_denorm = denormalize(train_signals)
gen_denorm = denormalize(normalized_samples)

# Convert to millivolts for visual display
real_mv = to_millivolts(train_signals)
gen_mv = to_millivolts(normalized_samples)

print(f"\nDenormalized ranges:")
print(f"  Real: [{real_denorm.min():.4f}, {real_denorm.max():.4f}]")
print(f"  Generated: [{gen_denorm.min():.4f}, {gen_denorm.max():.4f}]")
print(f"\nMillivolt ranges:")
print(f"  Real: [{real_mv.min():.4f}, {real_mv.max():.4f}] mV")
print(f"  Generated: [{gen_mv.min():.4f}, {gen_mv.max():.4f}] mV")

# Time axis (assuming 500 Hz sampling rate, 5 second recording)
SAMPLING_RATE = 500  # Hz
time_axis = np.arange(train_signals.shape[-1]) / SAMPLING_RATE

# ============================================================================
# 1. OVERLAY COMPARISON (Normalized)
# ============================================================================
print("\nCreating overlay visualizations...")

fig, axes = plt.subplots(5, 1, figsize=(16, 20))
fig.suptitle('ECG Overlay Comparison (Normalized Scale)', fontsize=16, fontweight='bold')

for i in range(5):
    real_idx = np.random.randint(len(train_signals))
    gen_idx = i
    
    ax = axes[i]
    ax.plot(time_axis, train_signals[real_idx, 0, :], color='blue', linewidth=1.0, 
            alpha=0.8, label=f'Real ECG #{real_idx}')
    ax.plot(time_axis, normalized_samples[gen_idx, 0, :], color='red', linewidth=1.0, 
            alpha=0.7, linestyle='-', label=f'Generated ECG #{gen_idx}')
    
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Amplitude (normalized)', fontsize=11)
    ax.set_title(f'Overlay {i+1}: Real vs Generated', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, time_axis[-1]])

plt.tight_layout()
plt.savefig(EVAL_OUTPUT / 'overlay_normalized.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: overlay_normalized.png")

# ============================================================================
# 2. OVERLAY COMPARISON (Millivolt Scale - Visual Friendly)
# ============================================================================

fig, axes = plt.subplots(5, 1, figsize=(16, 20))
fig.suptitle('ECG Overlay Comparison (Millivolt Scale - Visual Range)', fontsize=16, fontweight='bold')

for i in range(5):
    real_idx = np.random.randint(len(train_signals))
    gen_idx = i
    
    ax = axes[i]
    ax.plot(time_axis, real_mv[real_idx, 0, :], color='blue', linewidth=1.0, 
            alpha=0.8, label=f'Real ECG #{real_idx}')
    ax.plot(time_axis, gen_mv[gen_idx, 0, :], color='red', linewidth=1.0, 
            alpha=0.7, linestyle='-', label=f'Generated ECG #{gen_idx}')
    
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Amplitude (mV)', fontsize=11)
    ax.set_title(f'Overlay {i+1}: Real vs Generated', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_ylim([-2.5, 2.5])  # Standard ECG viewing range

plt.tight_layout()
plt.savefig(EVAL_OUTPUT / 'overlay_millivolts.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: overlay_millivolts.png")

# ============================================================================
# 3. SIDE-BY-SIDE with Millivolt Scale
# ============================================================================

fig, axes = plt.subplots(5, 2, figsize=(18, 20))
fig.suptitle('Real vs Generated ECG (Millivolt Scale)', fontsize=16, fontweight='bold')

for i in range(5):
    real_idx = np.random.randint(len(train_signals))
    gen_idx = i
    
    # Real ECG
    axes[i, 0].plot(time_axis, real_mv[real_idx, 0, :], color='blue', linewidth=0.9)
    axes[i, 0].set_title(f'Real ECG #{real_idx}', fontsize=12, color='blue')
    axes[i, 0].set_ylim([-2.5, 2.5])
    axes[i, 0].set_xlabel('Time (seconds)')
    axes[i, 0].set_ylabel('Amplitude (mV)')
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Generated ECG
    axes[i, 1].plot(time_axis, gen_mv[gen_idx, 0, :], color='red', linewidth=0.9)
    axes[i, 1].set_title(f'Generated ECG #{gen_idx}', fontsize=12, color='red')
    axes[i, 1].set_ylim([-2.5, 2.5])
    axes[i, 1].set_xlabel('Time (seconds)')
    axes[i, 1].set_ylabel('Amplitude (mV)')
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(EVAL_OUTPUT / 'sidebyside_millivolts.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: sidebyside_millivolts.png")

# ============================================================================
# 4. ZOOMED OVERLAY (2 seconds)
# ============================================================================

fig, axes = plt.subplots(4, 1, figsize=(16, 16))
fig.suptitle('Zoomed ECG Overlay (2 second window - Millivolt Scale)', fontsize=16, fontweight='bold')

zoom_samples = int(2.0 * SAMPLING_RATE)  # 2 seconds
zoom_time = time_axis[:zoom_samples]

for i in range(4):
    real_idx = np.random.randint(len(train_signals))
    gen_idx = i
    
    ax = axes[i]
    ax.plot(zoom_time, real_mv[real_idx, 0, :zoom_samples], color='blue', linewidth=1.2, 
            alpha=0.9, label=f'Real ECG #{real_idx}')
    ax.plot(zoom_time, gen_mv[gen_idx, 0, :zoom_samples], color='red', linewidth=1.2, 
            alpha=0.8, linestyle='-', label=f'Generated ECG #{gen_idx}')
    
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Amplitude (mV)', fontsize=11)
    ax.set_title(f'Zoomed Overlay {i+1}', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2.0])
    ax.set_ylim([-2.5, 2.5])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(EVAL_OUTPUT / 'overlay_zoomed_millivolts.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: overlay_zoomed_millivolts.png")

# ============================================================================
# 5. GRID of Generated Samples (Millivolt Scale)
# ============================================================================

fig, axes = plt.subplots(4, 4, figsize=(20, 16))
fig.suptitle('Generated ECG Samples (Millivolt Scale)', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < len(gen_mv):
        ax.plot(time_axis, gen_mv[i, 0, :], color='red', linewidth=0.7)
        ax.set_ylim([-2.5, 2.5])
        ax.set_title(f'Sample {i+1}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('mV', fontsize=8)

plt.tight_layout()
plt.savefig(EVAL_OUTPUT / 'generated_grid_millivolts.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: generated_grid_millivolts.png")

# ============================================================================
# 6. COMBINED OVERLAY (Multiple on same plot)
# ============================================================================

fig, axes = plt.subplots(2, 1, figsize=(18, 12))

# Top: Multiple real ECGs overlaid
ax = axes[0]
ax.set_title('Multiple Real ECGs Overlaid (Millivolt Scale)', fontsize=14, fontweight='bold')
for i in range(10):
    idx = np.random.randint(len(train_signals))
    ax.plot(time_axis, real_mv[idx, 0, :], linewidth=0.6, alpha=0.5)
ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Amplitude (mV)', fontsize=11)
ax.set_ylim([-2.5, 2.5])
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Bottom: Multiple generated ECGs overlaid
ax = axes[1]
ax.set_title('Multiple Generated ECGs Overlaid (Millivolt Scale)', fontsize=14, fontweight='bold')
for i in range(min(10, len(gen_mv))):
    ax.plot(time_axis, gen_mv[i, 0, :], linewidth=0.6, alpha=0.5, color='red')
ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Amplitude (mV)', fontsize=11)
ax.set_ylim([-2.5, 2.5])
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(EVAL_OUTPUT / 'multiple_overlay_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: multiple_overlay_comparison.png")

print(f"\n{'='*60}")
print("All visualizations saved to:")
print(f"  {EVAL_OUTPUT}")
print(f"{'='*60}")
print("\nFiles created:")
print("  - overlay_normalized.png (normalized scale overlay)")
print("  - overlay_millivolts.png (mV scale overlay)")
print("  - sidebyside_millivolts.png (side-by-side mV scale)")
print("  - overlay_zoomed_millivolts.png (2-second zoomed view)")
print("  - generated_grid_millivolts.png (16 samples grid)")
print("  - multiple_overlay_comparison.png (10 samples overlaid)")
print("\nDone!")
