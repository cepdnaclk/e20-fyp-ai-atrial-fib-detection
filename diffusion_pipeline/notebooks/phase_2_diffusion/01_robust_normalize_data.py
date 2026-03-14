# 01_robust_normalize_data_FINAL.py
# FINAL VERSION: Robust outlier clipping + Standard Z-score normalization

import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

print("="*70)
print("🔧 HYBRID NORMALIZATION: Robust Clipping + Standard Z-score")
print("="*70)

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'

TRAIN_INPUT = DATA_DIR / 'train_data.npz'
VAL_INPUT = DATA_DIR / 'val_data.npz'

TRAIN_OUTPUT = DATA_DIR / 'train_data_robust_normalized.npz'
VAL_OUTPUT = DATA_DIR / 'val_data_robust_normalized.npz'

VIZ_DIR = PROJECT_ROOT / 'models/phase2_diffusion/results/normalization_comparison'
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# FINAL: Hybrid Robust + Standard Normalization
# ============================================================================

def hybrid_normalize(data, clip_sigma=5.0):
    """
    FINAL SOLUTION: Robust outlier clipping + Standard Z-score
    
    Step 1: Use robust statistics (median, IQR) to identify outliers
    Step 2: Clip outliers based on robust bounds
    Step 3: Apply standard Z-score (mean=0, std=1) on clipped data
    
    This gives us:
    - Outlier resistance (from robust statistics)
    - Perfect Gaussian normalization (from standard Z-score)
    
    Args:
        data: [N, C, L] array
        clip_sigma: Clip based on robust sigma (default: 5)
    
    Returns:
        normalized: Data with mean=0, std=1
        stats: Dictionary of normalization parameters
    """
    # Flatten for global statistics
    data_flat = data.flatten()
    
    # STEP 1: Compute robust statistics for outlier detection
    median = np.median(data_flat)
    q75 = np.percentile(data_flat, 75)
    q25 = np.percentile(data_flat, 25)
    iqr = q75 - q25
    robust_std = iqr / 1.349
    
    if robust_std < 1e-6:
        robust_std = 1.0
    
    # STEP 2: Clip outliers using robust bounds
    lower_bound = median - clip_sigma * robust_std
    upper_bound = median + clip_sigma * robust_std
    
    data_clipped = np.clip(data, lower_bound, upper_bound)
    
    print(f"   Outlier clipping:")
    print(f"     Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
    print(f"     Clipped: {(data != data_clipped).sum()} / {data.size} values ({(data != data_clipped).sum()/data.size*100:.2f}%)")
    
    # STEP 3: Apply standard Z-score on clipped data
    mean = data_clipped.mean()
    std = data_clipped.std()
    
    if std < 1e-6:
        std = 1.0
    
    normalized = (data_clipped - mean) / std
    
    # Final safety clip (should be rare after robust clipping)
    normalized = np.clip(normalized, -10.0, 10.0)
    
    stats = {
        'mean': float(mean),
        'std': float(std),
        'median': float(median),
        'robust_std': float(robust_std),
        'lower_bound': float(lower_bound),
        'upper_bound': float(upper_bound)
    }
    
    return normalized, stats

# ============================================================================
# Process Training Data
# ============================================================================

print(f"\n📂 Loading training data: {TRAIN_INPUT}")
train_data = np.load(TRAIN_INPUT)
train_X = train_data['X']
train_y = train_data['y']

print(f"   Shape: {train_X.shape}")

if train_X.ndim == 2:
    train_X = train_X[:, np.newaxis, :]
    print(f"   Reshaped to: {train_X.shape}")

# Statistics BEFORE
print(f"\n📊 BEFORE Normalization:")
print(f"   Range: [{train_X.min():.4f}, {train_X.max():.4f}]")
print(f"   Mean: {train_X.mean():.4f}")
print(f"   Std: {train_X.std():.4f}")
print(f"   Median: {np.median(train_X):.4f}")

# Apply hybrid normalization
print(f"\n🔧 Applying hybrid normalization...")
train_X_normalized, train_stats = hybrid_normalize(train_X, clip_sigma=5.0)

# Statistics AFTER
print(f"\n📊 AFTER Normalization:")
print(f"   Range: [{train_X_normalized.min():.4f}, {train_X_normalized.max():.4f}]")
print(f"   Mean: {train_X_normalized.mean():.4f}  (target: 0.00)")
print(f"   Std: {train_X_normalized.std():.4f}  (target: 1.00)")
print(f"   Median: {np.median(train_X_normalized):.4f}")

print(f"\n🔢 Normalization Parameters:")
for key, value in train_stats.items():
    print(f"   {key}: {value:.4f}")

# Validation
mean_ok = abs(train_X_normalized.mean()) < 0.01
std_ok = abs(train_X_normalized.std() - 1.0) < 0.01

if mean_ok and std_ok:
    print(f"\n✅ PERFECT NORMALIZATION!")
    print(f"   Mean: {train_X_normalized.mean():.6f} ≈ 0 ✅")
    print(f"   Std: {train_X_normalized.std():.6f} ≈ 1 ✅")
elif abs(train_X_normalized.mean()) < 0.05 and abs(train_X_normalized.std() - 1.0) < 0.05:
    print(f"\n✅ GOOD NORMALIZATION (within tolerance)")
    print(f"   Mean: {train_X_normalized.mean():.6f} ≈ 0")
    print(f"   Std: {train_X_normalized.std():.6f} ≈ 1")
else:
    print(f"\n⚠️  ACCEPTABLE but not perfect")
    print(f"   Mean: {train_X_normalized.mean():.6f}")
    print(f"   Std: {train_X_normalized.std():.6f}")

# Save
print(f"\n💾 Saving normalized training data: {TRAIN_OUTPUT}")
np.savez_compressed(
    TRAIN_OUTPUT,
    X=train_X_normalized,
    y=train_y,
    **train_stats  # Save all normalization parameters
)
print("   ✅ Saved!")

# ============================================================================
# Process Validation Data (using SAME parameters)
# ============================================================================

print(f"\n📂 Loading validation data: {VAL_INPUT}")
val_data = np.load(VAL_INPUT)
val_X = val_data['X']
val_y = val_data['y']

if val_X.ndim == 2:
    val_X = val_X[:, np.newaxis, :]

print(f"\n🔧 Normalizing validation data using TRAINING parameters...")

# Apply same clipping bounds
val_X_clipped = np.clip(val_X, train_stats['lower_bound'], train_stats['upper_bound'])

# Apply same standardization
val_X_normalized = (val_X_clipped - train_stats['mean']) / train_stats['std']
val_X_normalized = np.clip(val_X_normalized, -10.0, 10.0)

print(f"\n📊 Validation Stats AFTER Normalization:")
print(f"   Range: [{val_X_normalized.min():.4f}, {val_X_normalized.max():.4f}]")
print(f"   Mean: {val_X_normalized.mean():.4f}")
print(f"   Std: {val_X_normalized.std():.4f}")

# Save
print(f"\n💾 Saving normalized validation data: {VAL_OUTPUT}")
np.savez_compressed(
    VAL_OUTPUT,
    X=val_X_normalized,
    y=val_y,
    **train_stats  # Same parameters as training
)
print("   ✅ Saved!")

# ============================================================================
# Visualization
# ============================================================================

print(f"\n📊 Creating visualizations...")

np.random.seed(42)
sample_indices = np.random.choice(len(train_X), size=6, replace=False)

fig, axes = plt.subplots(6, 2, figsize=(16, 18))

for i, idx in enumerate(sample_indices):
    # Before
    axes[i, 0].plot(train_X[idx, 0, :], linewidth=0.5, color='steelblue')
    axes[i, 0].set_title(f'Sample {idx} - BEFORE')
    axes[i, 0].set_ylim([-2, 2])
    axes[i, 0].grid(True, alpha=0.3)
    axes[i, 0].axhline(y=train_X.mean(), color='blue', linestyle='--', alpha=0.7, 
                       label=f'Mean={train_X.mean():.2f}')
    axes[i, 0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
    axes[i, 0].legend(fontsize=8)
    
    # After
    axes[i, 1].plot(train_X_normalized[idx, 0, :], linewidth=0.5, color='darkgreen')
    axes[i, 1].set_title(f'Sample {idx} - AFTER')
    axes[i, 1].set_ylim([-4, 4])
    axes[i, 1].grid(True, alpha=0.3)
    axes[i, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=2,
                       label=f'Mean={train_X_normalized.mean():.4f}')
    axes[i, 1].axhline(y=-1, color='orange', linestyle=':', alpha=0.5)
    axes[i, 1].axhline(y=1, color='orange', linestyle=':', alpha=0.5)
    axes[i, 1].legend(fontsize=8)

plt.suptitle('Hybrid Normalization: Robust Clipping + Standard Z-score', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(VIZ_DIR / 'normalization_final.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: {VIZ_DIR / 'normalization_final.png'}")

# Distribution comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Before
axes[0].hist(train_X.flatten(), bins=100, alpha=0.7, color='steelblue', edgecolor='black')
axes[0].set_title('BEFORE Normalization')
axes[0].set_xlabel('Value (mV)')
axes[0].set_ylabel('Frequency')
axes[0].axvline(x=train_X.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean={train_X.mean():.3f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# After (histogram)
axes[1].hist(train_X_normalized.flatten(), bins=100, alpha=0.7, color='darkgreen', 
             edgecolor='black', density=True)
axes[1].set_title('AFTER Normalization')
axes[1].set_xlabel('Normalized Value (σ)')
axes[1].set_ylabel('Density')

# Overlay N(0,1) reference
x_ref = np.linspace(-4, 4, 100)
y_ref = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_ref**2)
axes[1].plot(x_ref, y_ref, 'r--', linewidth=2, label='N(0,1) reference')

axes[1].axvline(x=0, color='blue', linestyle='--', linewidth=2, alpha=0.5, 
                label=f'Mean={train_X_normalized.mean():.4f}')
axes[1].axvline(x=-1, color='orange', linestyle=':', alpha=0.5)
axes[1].axvline(x=1, color='orange', linestyle=':', alpha=0.5)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Q-Q plot against N(0,1)
from scipy import stats as sp_stats
axes[2].set_title('Q-Q Plot vs N(0,1)')
sp_stats.probplot(train_X_normalized.flatten()[::100], dist="norm", plot=axes[2])
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(VIZ_DIR / 'distribution_final.png', dpi=150, bbox_inches='tight')
print(f"   ✅ Saved: {VIZ_DIR / 'distribution_final.png'}")

# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*70}")
print("📈 FINAL NORMALIZATION SUMMARY")
print(f"{'='*70}")

print(f"\nTraining Set:")
print(f"  Samples: {len(train_X)}")
print(f"  Original   - Mean: {train_X.mean():.4f}, Std: {train_X.std():.4f}")
print(f"  Normalized - Mean: {train_X_normalized.mean():.6f}, Std: {train_X_normalized.std():.6f}")

print(f"\nValidation Set:")
print(f"  Samples: {len(val_X)}")
print(f"  Normalized - Mean: {val_X_normalized.mean():.6f}, Std: {val_X_normalized.std():.6f}")

print(f"\nNormalization Parameters (for denormalization):")
print(f"  Mean: {train_stats['mean']:.4f}")
print(f"  Std: {train_stats['std']:.4f}")

print(f"\n{'='*70}")
if mean_ok and std_ok:
    print("✅✅ PERFECT NORMALIZATION! Ready for training! ✅✅")
elif abs(train_X_normalized.mean()) < 0.05 and abs(train_X_normalized.std() - 1.0) < 0.05:
    print("✅ GOOD NORMALIZATION! Ready for training! ✅")
else:
    print("✅ ACCEPTABLE NORMALIZATION (good enough for diffusion)")
print(f"{'='*70}")

print(f"\nFiles saved to:")
print(f"  Training: {TRAIN_OUTPUT}")
print(f"  Validation: {VAL_OUTPUT}")
print(f"  Visualizations: {VIZ_DIR}")

print(f"\n🚀 Next step: Update training script and retrain!")
print(f"{'='*70}")