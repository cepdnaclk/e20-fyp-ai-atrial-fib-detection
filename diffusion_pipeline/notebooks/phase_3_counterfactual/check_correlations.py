#!/usr/bin/env python3
"""Check correlation distribution of existing CFs"""
import numpy as np
from scipy.stats import pearsonr
from pathlib import Path

CF_DIR = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_3_counterfactual/results')
batch_files = list(CF_DIR.glob('batch_Normal_to_AFib_*.npz'))[:10]

all_corrs = []
for bf in batch_files:
    data = np.load(bf)
    cfs = data['counterfactuals']
    originals = data['originals']
    
    for i in range(min(50, len(cfs))):
        try:
            corr, _ = pearsonr(originals[i], cfs[i])
            all_corrs.append(corr)
        except:
            pass

all_corrs = np.array(all_corrs)
print(f"Correlation statistics from {len(all_corrs)} CFs:")
print(f"  Mean: {all_corrs.mean():.3f}")
print(f"  Median: {np.median(all_corrs):.3f}")
print(f"  Std: {all_corrs.std():.3f}")
print(f"  Min: {all_corrs.min():.3f}, Max: {all_corrs.max():.3f}")
print(f"\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95]:
    print(f"  {p}th: {np.percentile(all_corrs, p):.3f}")

print(f"\nPassing rates at different thresholds:")
for thresh in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
    passing = (all_corrs > thresh).sum()
    print(f"  > {thresh}: {passing}/{len(all_corrs)} = {100*passing/len(all_corrs):.1f}%")
