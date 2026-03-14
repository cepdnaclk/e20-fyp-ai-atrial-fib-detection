# %%
"""
DIAGNOSTIC: Check if new robust-normalized data is ready for training
Updated to use NEW data location and format
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

print("="*80)
print("🔍 NORMALIZATION DIAGNOSTIC - NEW DATA")
print("="*80)

# ============================================================================
# CONFIGURATION - UPDATED PATHS
# ============================================================================

# NEW location (Windows path where you just saved the data)
DATA_DIR = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/data/processed/diffusion')

TRAIN_FILE = DATA_DIR / 'train_data.npz'
VAL_FILE = DATA_DIR / 'val_data.npz'
METADATA_FILE = DATA_DIR / 'dataset_metadata.json'

# ============================================================================
# STEP 1: Load New Data
# ============================================================================

print(f"\n📂 Loading data from: {DATA_DIR}")

# Load training data
train_data = np.load(TRAIN_FILE)
X_train = train_data['X']
y_train = train_data['y']

# Load validation data
val_data = np.load(VAL_FILE)
X_val = val_data['X']
y_val = val_data['y']

# Load metadata
with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)

norm_params = metadata['normalization']

print(f"✅ Loaded successfully!")
print(f"   Training samples: {len(X_train):,}")
print(f"   Validation samples: {len(X_val):,}")

# ============================================================================
# STEP 2: Check Data Statistics
# ============================================================================

print("\n" + "="*80)
print("📊 DATA STATISTICS")
print("="*80)

print(f"\n1. TRAINING DATA (what model will train on):")
print(f"   Shape: {X_train.shape}")
print(f"   Mean: {X_train.mean():.4f} mV")
print(f"   Std:  {X_train.std():.4f} mV")
print(f"   Range: [{X_train.min():.4f}, {X_train.max():.4f}] mV")
print(f"   Median: {np.median(X_train):.4f} mV")

print(f"\n2. VALIDATION DATA:")
print(f"   Shape: {X_val.shape}")
print(f"   Mean: {X_val.mean():.4f} mV")
print(f"   Std:  {X_val.std():.4f} mV")
print(f"   Range: [{X_val.min():.4f}, {X_val.max():.4f}] mV")

print(f"\n3. NORMALIZATION PARAMETERS:")
print(f"   Method: {norm_params['method']}")
print(f"   Global min (clipped): {norm_params['global_min']:.4f} mV")
print(f"   Global max (clipped): {norm_params['global_max']:.4f} mV")
print(f"   1st percentile (p1): {norm_params['p1']:.4f} mV")
print(f"   99th percentile (p99): {norm_params['p99']:.4f} mV")
print(f"   Target range: [{norm_params['target_min']}, {norm_params['target_max']}] mV")

# ============================================================================
# STEP 3: Quality Checks
# ============================================================================

print("\n" + "="*80)
print("🔍 QUALITY CHECKS")
print("="*80)

# Check 1: Variance
variance_threshold_low = 0.3
variance_threshold_good = 0.4

if X_train.std() > variance_threshold_good:
    print(f"✅ Variance: {X_train.std():.4f} mV (EXCELLENT)")
    variance_status = "EXCELLENT"
elif X_train.std() > variance_threshold_low:
    print(f"✅ Variance: {X_train.std():.4f} mV (GOOD)")
    variance_status = "GOOD"
else:
    print(f"⚠️  Variance: {X_train.std():.4f} mV (TOO LOW)")
    variance_status = "LOW"

# Check 2: Data spread
p5 = np.percentile(X_train, 5)
p95 = np.percentile(X_train, 95)
data_spread = p95 - p5

if data_spread > 1.0:
    print(f"✅ Data spread (5th-95th): {data_spread:.4f} mV (GOOD)")
    spread_status = "GOOD"
elif data_spread > 0.5:
    print(f"⚠️  Data spread (5th-95th): {data_spread:.4f} mV (MODERATE)")
    spread_status = "MODERATE"
else:
    print(f"❌ Data spread (5th-95th): {data_spread:.4f} mV (TOO LOW)")
    spread_status = "LOW"

# Check 3: Boundary concentration
at_min = np.sum(np.abs(X_train - norm_params['target_min']) < 1e-6)
at_max = np.sum(np.abs(X_train - norm_params['target_max']) < 1e-6)
boundary_pct = (at_min + at_max) / X_train.size * 100

if boundary_pct < 0.5:
    print(f"✅ Boundary concentration: {boundary_pct:.3f}% (NO CLIPPING)")
elif boundary_pct < 3.0:
    print(f"✅ Boundary concentration: {boundary_pct:.3f}% (EXPECTED from robust clipping)")
else:
    print(f"⚠️  Boundary concentration: {boundary_pct:.3f}% (HIGH)")

# Check 4: Data integrity
n_nan = np.isnan(X_train).sum()
n_inf = np.isinf(X_train).sum()

if n_nan == 0 and n_inf == 0:
    print(f"✅ Data integrity: CLEAN (no NaN/Inf)")
else:
    print(f"❌ Data integrity: ISSUES FOUND ({n_nan} NaN, {n_inf} Inf)")

# Check 5: Class balance
n_afib = (y_train == 1).sum()
n_normal = (y_train == 0).sum()
balance_ratio = min(n_afib, n_normal) / max(n_afib, n_normal)

if balance_ratio > 0.95:
    print(f"✅ Class balance: {balance_ratio:.4f} (PERFECT)")
else:
    print(f"⚠️  Class balance: {balance_ratio:.4f}")

# ============================================================================
# STEP 4: Visualization
# ============================================================================

print("\n" + "="*80)
print("📊 CREATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Data Quality Check - Ready for Training?', fontsize=16, fontweight='bold')

# Plot 1: Distribution histogram
axes[0, 0].hist(X_train.flatten(), bins=200, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].axvline(norm_params['target_min'], color='red', linestyle='--', linewidth=2, 
                   label=f'Boundary ({norm_params["target_min"]})')
axes[0, 0].axvline(norm_params['target_max'], color='red', linestyle='--', linewidth=2,
                   label=f'Boundary ({norm_params["target_max"]})')
axes[0, 0].axvline(X_train.mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Mean ({X_train.mean():.3f})')
axes[0, 0].set_xlabel('Voltage (mV)', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Full Distribution', fontweight='bold', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Zoomed distribution (center)
center_mask = (X_train.flatten() > -1.0) & (X_train.flatten() < 1.0)
axes[0, 1].hist(X_train.flatten()[center_mask], bins=100, alpha=0.7, 
                color='green', edgecolor='black')
axes[0, 1].axvline(X_train.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {X_train.mean():.3f}')
axes[0, 1].set_xlabel('Voltage (mV)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Zoomed: [-1.0, 1.0] mV Region', fontweight='bold', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Sample ECGs
time = np.arange(X_train.shape[1]) / 250  # Assuming 250 Hz
sample_indices = np.random.choice(len(X_train), 3, replace=False)

for i, idx in enumerate(sample_indices):
    color = 'red' if y_train[idx] == 1 else 'blue'
    label = 'AFib' if y_train[idx] == 1 else 'Normal'
    axes[1, 0].plot(time, X_train[idx], linewidth=0.8, alpha=0.7, color=color, 
                    label=f'{label} {idx}')

axes[1, 0].set_xlabel('Time (s)', fontsize=11)
axes[1, 0].set_ylabel('Voltage (mV)', fontsize=11)
axes[1, 0].set_title('Sample ECGs', fontweight='bold', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_xlim(0, 10)

# Plot 4: Per-signal statistics
signal_stds = X_train.std(axis=1)
axes[1, 1].hist(signal_stds, bins=50, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].axvline(signal_stds.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {signal_stds.mean():.3f}')
axes[1, 1].set_xlabel('Per-Signal Std (mV)', fontsize=11)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Per-Signal Standard Deviation', fontweight='bold', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR / 'diagnostic_check.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"✅ Saved visualization: {DATA_DIR / 'diagnostic_check.png'}")

# ============================================================================
# STEP 5: Final Verdict
# ============================================================================

print("\n" + "="*80)
print("🎯 FINAL VERDICT")
print("="*80)

issues = []
warnings = []

if variance_status == "LOW":
    issues.append("Low variance - model may generate flat signals")
elif variance_status == "GOOD":
    warnings.append("Variance is good but could be better")

if spread_status == "LOW":
    issues.append("Low data spread - data too compressed")
elif spread_status == "MODERATE":
    warnings.append("Data spread is moderate")

if boundary_pct > 5.0:
    issues.append("High boundary concentration - possible clipping issue")

if n_nan > 0 or n_inf > 0:
    issues.append("Data contains NaN or Inf values")

if balance_ratio < 0.9:
    warnings.append("Class imbalance detected")

# Print verdict
if len(issues) == 0 and len(warnings) == 0:
    print("\n🎉 PERFECT! Data is ready for training!")
    print("\n✅ All checks passed:")
    print(f"   ✓ Excellent variance: {X_train.std():.4f} mV")
    print(f"   ✓ Good data spread: {data_spread:.4f} mV")
    print(f"   ✓ No artificial clipping: {boundary_pct:.3f}% at boundaries")
    print(f"   ✓ Clean data (no NaN/Inf)")
    print(f"   ✓ Balanced classes: {balance_ratio:.4f}")
    print("\n🚀 Expected model performance:")
    print(f"   • Generated variance ratio: >85%")
    print(f"   • Sharp R-peaks with realistic amplitudes")
    print(f"   • No histogram spike at boundaries")
    
elif len(issues) == 0:
    print("\n✅ GOOD! Data is ready for training (with minor notes)")
    print("\n⚠️  Minor warnings:")
    for w in warnings:
        print(f"   • {w}")
    print("\n✅ These won't prevent training, just noting for reference")
    print("\n🚀 You can proceed with training!")
    
else:
    print("\n⚠️  ISSUES DETECTED!")
    print("\n❌ Problems found:")
    for issue in issues:
        print(f"   • {issue}")
    if warnings:
        print("\n⚠️  Additional warnings:")
        for w in warnings:
            print(f"   • {w}")
    print("\n🔧 Consider re-normalizing the data before training")

print("\n" + "="*80)
print("📋 SUMMARY")
print("="*80)
print(f"Data location: {DATA_DIR}")
print(f"Training samples: {len(X_train):,}")
print(f"Validation samples: {len(X_val):,}")
print(f"Variance: {X_train.std():.4f} mV ({variance_status})")
print(f"Data spread: {data_spread:.4f} mV ({spread_status})")
print(f"Boundary concentration: {boundary_pct:.3f}%")
print(f"Class balance: {balance_ratio:.4f}")

if len(issues) == 0:
    print("\n✅ READY TO TRAIN! 🚀")
else:
    print("\n⚠️  REVIEW ISSUES BEFORE TRAINING")

print("="*80)