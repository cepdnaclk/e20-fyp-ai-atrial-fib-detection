"""
Assemble Counterfactual Dataset
================================
Merges batch .npz files into unified datasets for:
1. counterfactual_test_data.npz - For three-way evaluation (X, y format)
2. Full metrics data with all metadata
"""
import numpy as np
from pathlib import Path

RESULTS_DIR = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_3_counterfactual/results')
CF_DIR = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models/phase3_counterfactual/generated_counterfactuals')
CF_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Assembling Counterfactual Dataset")
print("=" * 60)

# Collect all batch files
n2a_files = sorted(RESULTS_DIR.glob('batch_Normal_to_AFib_*.npz'))
a2n_files = sorted(RESULTS_DIR.glob('batch_AFib_to_Normal_*.npz'))

print(f"Found {len(n2a_files)} Normal→AFib batch files")
print(f"Found {len(a2n_files)} AFib→Normal batch files")

all_cfs = []
all_originals = []
all_target_labels = []
all_original_labels = []
all_val_scores = []
all_attempts = []

for batch_file in sorted(n2a_files + a2n_files):
    data = np.load(batch_file)
    all_cfs.append(data['counterfactuals'])
    all_originals.append(data['originals'])
    all_target_labels.append(data['target_labels'])
    all_original_labels.append(data['original_labels'])
    all_val_scores.append(data['validation_scores'])
    all_attempts.append(data['attempts'])
    print(f"  Loaded {batch_file.name}: {len(data['counterfactuals'])} samples")

# Concatenate
cfs = np.concatenate(all_cfs, axis=0)
originals = np.concatenate(all_originals, axis=0)
target_labels = np.concatenate(all_target_labels, axis=0)
original_labels = np.concatenate(all_original_labels, axis=0)
val_scores = np.concatenate(all_val_scores, axis=0)
attempts = np.concatenate(all_attempts, axis=0)

print(f"\nTotal assembled:")
print(f"  Counterfactuals: {cfs.shape}")
print(f"  Originals: {originals.shape}")
print(f"  Target labels: {target_labels.shape} (Normal={np.sum(target_labels==0)}, AFib={np.sum(target_labels==1)})")

# Save format 1: For three-way evaluation (X=counterfactuals, y=target_labels)
# The target_labels are what the counterfactuals SHOULD be classified as
np.savez(
    CF_DIR / 'counterfactual_test_data.npz',
    X=cfs,
    y=target_labels
)
print(f"\n✓ Saved counterfactual_test_data.npz to {CF_DIR}")
print(f"  X shape: {cfs.shape}, y shape: {target_labels.shape}")

# Save format 2: Full metadata for enhanced metrics
np.savez(
    RESULTS_DIR / 'counterfactual_full_data.npz',
    counterfactuals=cfs,
    originals=originals,
    target_labels=target_labels,
    original_labels=original_labels,
    validation_scores=val_scores,
    attempts=attempts
)
print(f"✓ Saved counterfactual_full_data.npz to {RESULTS_DIR}")

# Summary statistics
n2a_mask = original_labels == 0  # Normal → AFib
a2n_mask = original_labels == 1  # AFib → Normal

print(f"\nDataset Summary:")
print(f"  Normal→AFib: {n2a_mask.sum()} samples")
print(f"  AFib→Normal: {a2n_mask.sum()} samples")
print(f"  Mean plausibility: {val_scores.mean():.3f}")
print(f"  Mean attempts: {attempts.mean():.2f}")
print(f"  Plausibility > 0.7: {(val_scores > 0.7).mean():.1%}")

print("\n✓ Data assembly complete!")
