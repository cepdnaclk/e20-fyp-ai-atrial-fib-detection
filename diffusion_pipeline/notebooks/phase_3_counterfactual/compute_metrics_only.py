"""
Compute metrics from saved batch files.
Run this after generation is complete.
"""
import os
import sys
import gc
import json
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
sys.path.append(str(Path(__file__).parent.parent))

from shared_models import load_classifier, ClassifierWrapper

# Config
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULTS_DIR = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_3_counterfactual/results')

# Load classifier
classifier = load_classifier(DEVICE)
classifier_wrapper = ClassifierWrapper(classifier).to(DEVICE)
classifier_wrapper.eval()
print("✓ Classifier loaded")

def compute_metrics_from_files(batch_files, direction_name):
    """Compute metrics from saved batch files."""
    flip_rates = []
    correlations = []
    mses = []
    plausibility_scores = []
    attempts_list = []
    total_samples = 0
    
    for batch_file in sorted(batch_files):
        data = np.load(batch_file)
        
        cfs = torch.FloatTensor(data['counterfactuals']).unsqueeze(1)
        originals = data['originals']
        target_labels = torch.LongTensor(data['target_labels'])
        batch_size = len(cfs)
        total_samples += batch_size
        
        # Compute flip rate in sub-batches of 100
        with torch.no_grad():
            batch_preds = []
            for i in range(0, len(cfs), 100):
                batch_cf = cfs[i:i+100].to(DEVICE)
                logits = classifier_wrapper(batch_cf)  # Returns only logits
                preds = torch.argmax(logits, dim=1)
                batch_preds.append(preds.cpu())
                del batch_cf, logits, preds
                torch.cuda.empty_cache()
            
            all_preds = torch.cat(batch_preds)
            flip_rate = (all_preds == target_labels).float().mean().item()
            flip_rates.append((flip_rate, batch_size))
        
        # Compute similarities
        for cf, orig in zip(data['counterfactuals'], originals):
            corr, _ = pearsonr(orig, cf)
            correlations.append(corr)
            mses.append(np.mean((orig - cf) ** 2))
        
        plausibility_scores.extend(data['validation_scores'])
        attempts_list.extend(data['attempts'])
        
        del data, cfs, originals, target_labels, all_preds
        gc.collect()
        
        print(f"  Processed {batch_file.name} (flip: {flip_rate:.2%})", flush=True)
    
    # Weighted average flip rate
    total_flipped = sum(rate * count for rate, count in flip_rates)
    overall_flip_rate = total_flipped / total_samples
    
    metrics = {
        'direction': direction_name,
        'total_samples': total_samples,
        'flip_rate': overall_flip_rate,
        'mean_correlation': float(np.mean(correlations)),
        'std_correlation': float(np.std(correlations)),
        'mean_mse': float(np.mean(mses)),
        'mean_plausibility': float(np.mean(plausibility_scores)),
        'mean_attempts': float(np.mean(attempts_list))
    }
    
    print(f"\n{direction_name}:")
    print(f"  Total Samples: {metrics['total_samples']}")
    print(f"  Flip Rate: {metrics['flip_rate']:.2%}")
    print(f"  Correlation: {metrics['mean_correlation']:.3f} ± {metrics['std_correlation']:.3f}")
    print(f"  MSE: {metrics['mean_mse']:.4f}")
    print(f"  Plausibility: {metrics['mean_plausibility']:.3f}")
    print(f"  Mean Attempts: {metrics['mean_attempts']:.2f}")
    
    return metrics

# Find batch files
n2a_files = sorted(RESULTS_DIR.glob('batch_Normal_to_AFib_*.npz'))
a2n_files = sorted(RESULTS_DIR.glob('batch_AFib_to_Normal_*.npz'))

print(f"\nFound {len(n2a_files)} Normal→AFib batch files")
print(f"Found {len(a2n_files)} AFib→Normal batch files")

print("\n" + "="*60)
print("Computing Metrics")
print("="*60)

metrics_n2a = compute_metrics_from_files(n2a_files, "Normal → AFib")
metrics_a2n = compute_metrics_from_files(a2n_files, "AFib → Normal")

overall = {
    'normal_to_afib': metrics_n2a,
    'afib_to_normal': metrics_a2n,
    'overall_flip_rate': (metrics_n2a['flip_rate'] + metrics_a2n['flip_rate']) / 2,
    'overall_correlation': (metrics_n2a['mean_correlation'] + metrics_a2n['mean_correlation']) / 2,
    'overall_plausibility': (metrics_n2a['mean_plausibility'] + metrics_a2n['mean_plausibility']) / 2,
}

with open(RESULTS_DIR / 'metrics.json', 'w') as f:
    json.dump(overall, f, indent=2)

print("\n" + "="*60)
print("METRICS COMPUTATION COMPLETE!")
print("="*60)
print(f"✓ Overall Flip Rate: {overall['overall_flip_rate']:.2%}")
print(f"✓ Overall Correlation: {overall['overall_correlation']:.3f}")
print(f"✓ Overall Plausibility: {overall['overall_plausibility']:.3f}")
print(f"✓ Saved to {RESULTS_DIR / 'metrics.json'}")
