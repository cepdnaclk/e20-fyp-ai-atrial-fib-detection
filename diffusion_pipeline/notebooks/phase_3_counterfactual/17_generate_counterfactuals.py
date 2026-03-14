"""
Generate Counterfactual ECG Dataset
====================================

End-to-end pipeline to generate large-scale validated counterfactual ECG dataset.

Steps:
1. Load trained diffusion model
2. Load test data
3. Generate counterfactuals (Normal → AFib, AFib → Normal)
4. Apply plausibility validation
5. Regenerate failed samples (up to 3 attempts)
6. Compute comprehensive metrics
7. Save outputs and visualizations

Author: Phase 3 Counterfactual Generation
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Import models and validator
from plausibility_validator import PlausibilityValidator
from shared_models import load_classifier, ClassifierWrapper

# Import model architectures
sys.path.insert(0, str(Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_3_counterfactual')))
# We'll import classes from 16_enhanced_diffusion_cf.py by exec-ing relevant parts


# ============================================================================
# Configuration
# ============================================================================

class Config:
    PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
    DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
    MODEL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/enhanced_diffusion_cf'
    OUTPUT_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/generated_counterfactuals'
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model params
    SIGNAL_LENGTH = 2500
    CONTENT_DIM = 256
    STYLE_DIM = 128
    MODEL_CHANNELS = 64
    NUM_CLASSES = 2
    
    # Generation params
    SDEDIT_STRENGTH = 0.6
    CFG_SCALE = 3.0
    NUM_STEPS = 50
    MAX_REGENERATION_ATTEMPTS = 3
    MIN_PLAUSIBILITY_SCORE = 0.7
    
    # Savgol filter
    SAVGOL_WINDOW = 11
    SAVGOL_POLY = 3
    
    # Batch processing
    BATCH_SIZE = 32

print(f"Device: {Config.DEVICE}")

# ============================================================================
# Load Model Architectures (import from 16_)
# ============================================================================

# Import necessary classes
from diffusion_models import ContentEncoder, StyleEncoder, ConditionalUNet, DDIMScheduler

# ============================================================================
# Load Trained Models
# ============================================================================

print("\n" + "="*60)
print("Loading Trained Models")
print("="*60)

checkpoint = torch.load(Config.MODEL_DIR / 'final_model.pth', map_location=Config.DEVICE, weights_only=False)

content_encoder = ContentEncoder(
    in_channels=1,
    hidden_dim=64,
    content_dim=Config.CONTENT_DIM
).to(Config.DEVICE)
content_encoder.load_state_dict(checkpoint['content_encoder'])
content_encoder.eval()

style_encoder = StyleEncoder(
    in_channels=1,
    hidden_dim=64,
    style_dim=Config.STYLE_DIM,
    num_classes=Config.NUM_CLASSES
).to(Config.DEVICE)
style_encoder.load_state_dict(checkpoint['style_encoder'])
style_encoder.eval()

unet = ConditionalUNet(
    in_ch=1,
    model_ch=Config.MODEL_CHANNELS,
    content_dim=Config.CONTENT_DIM,
    style_dim=Config.STYLE_DIM,
    num_classes=Config.NUM_CLASSES
).to(Config.DEVICE)
unet.load_state_dict(checkpoint['unet'])
unet.eval()

scheduler = DDIMScheduler(1000, Config.DEVICE)

print("✓ Models loaded successfully")

# Load classifier
classifier = load_classifier(Config.DEVICE)
classifier_wrapper = ClassifierWrapper(classifier).to(Config.DEVICE)
print("✓ Classifier loaded")

# Load validator
validator = PlausibilityValidator()
print("✓ Plausibility validator initialized")

# ============================================================================
# Load Test Data
# ============================================================================

print("\n" + "="*60)
print("Loading Test Data")
print("="*60)

test_data = np.load(Config.DATA_DIR / 'test_data.npz')
test_signals = torch.tensor(test_data['X'], dtype=torch.float32)
test_labels = torch.tensor(test_data['y'], dtype=torch.long)

if test_signals.dim() == 2:
    test_signals = test_signals.unsqueeze(1)

# Separate by class
normal_mask = test_labels == 0
afib_mask = test_labels == 1
normal_signals = test_signals[normal_mask]
afib_signals = test_signals[afib_mask]

print(f"Total test samples: {len(test_signals)}")
print(f"Normal: {len(normal_signals)}")
print(f"AFib: {len(afib_signals)}")

# ============================================================================
# Noise Reduction Function
# ============================================================================

def reduce_noise(signal, window_length=11, polyorder=3):
    """Apply Savitzky-Golay filter to reduce high-frequency noise."""
    if isinstance(signal, torch.Tensor):
        signal_np = signal.cpu().numpy()
        device = signal.device
    else:
        signal_np = signal
        device = None
    
    smoothed = np.zeros_like(signal_np)
    for i in range(signal_np.shape[0]):
        for j in range(signal_np.shape[1]):
            smoothed[i, j] = savgol_filter(signal_np[i, j], window_length, polyorder)
    
    if device is not None:
        return torch.tensor(smoothed, device=device, dtype=torch.float32)
    return smoothed

# ============================================================================
# Counterfactual Generation with Validation
# ============================================================================

@torch.no_grad()
def generate_counterfactual_with_validation(original_signal, original_class, target_class,
                                           validator, max_attempts=3):
    """
    Generate counterfactual with plausibility validation and regeneration.
    
    Args:
        original_signal: Input ECG (1, 2500) tensor
        original_class: Original class label (0 or 1)
        target_class: Target class label (0 or 1)  
        validator: PlausibilityValidator instance
        max_attempts: Maximum regeneration attempts
        
    Returns:
        tuple: (counterfactual_signal, validation_result, attempt_count)
    """
    original_signal = original_signal.unsqueeze(0).to(Config.DEVICE)
    target_class_tensor = torch.tensor([target_class], dtype=torch.long, device=Config.DEVICE)
    
    for attempt in range(max_attempts):
        # Encode original
        content, _, _ = content_encoder(original_signal)
        style, _ = style_encoder(original_signal)
        
        # Generate counterfactual
        cf = scheduler.sdedit_sample(
            unet, content, style, target_class_tensor,
            original_signal, strength=Config.SDEDIT_STRENGTH,
            num_steps=Config.NUM_STEPS, cfg_scale=Config.CFG_SCALE
        )
        
        # Reduce noise
        cf = reduce_noise(cf, Config.SAVGOL_WINDOW, Config.SAVGOL_POLY)
        
        # Validate
        result = validator.validate(
            cf[0], 
            original_ecg=original_signal[0],
            target_class=target_class,
            original_class=original_class
        )
        
        if result['valid'] and result['score'] >= Config.MIN_PLAUSIBILITY_SCORE:
            return cf[0], result, attempt + 1
    
    # If all attempts failed, return best one
    return cf[0], result, max_attempts

@torch.no_grad()
def generate_counterfactuals_batch(signals, original_class, target_class, validator):
    """Generate counterfactuals for a batch of signals."""
    results = {
        'counterfactuals': [],
        'validation_results': [],
        'attempts': [],
        'original_signals': [],
        'original_labels': [],
        'target_labels': [],
    }
    
    total = len(signals)
    start_time = time.time()
    
    for i, signal in enumerate(signals):
        cf, val_result, attempts = generate_counterfactual_with_validation(
            signal, original_class, target_class, validator
        )
        
        results['counterfactuals'].append(cf.cpu())
        results['validation_results'].append(val_result)
        results['attempts'].append(attempts)
        results['original_signals'].append(signal.cpu())
        results['original_labels'].append(original_class)
        results['target_labels'].append(target_class)
        
        # Print progress every 100 samples
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta_seconds = (total - (i + 1)) / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            pct = 100 * (i + 1) / total
            print(f"  [{i+1}/{total}] {pct:.1f}% | {rate:.2f} samples/s | ETA: {eta_minutes:.1f} min", flush=True)
    
    return results

# ============================================================================
# Generate Counterfactuals
# ============================================================================

print("\n" + "="*60)
print("Generating Counterfactuals")
print("="*60)

# Check if we have previously generated counterfactuals
import pickle
intermediate_file = Config.OUTPUT_DIR / 'counterfactuals_generated.pkl'

if intermediate_file.exists():
    print(f"\n✓ Found existing generated counterfactuals: {intermediate_file}")
    print("  Loading previously generated results...")
    
    with open(intermediate_file, 'rb') as f:
        saved_data = pickle.load(f)
        n2a_results = saved_data['n2a_results']
        a2n_results = saved_data['a2n_results  ']
    
    print(f"  Loaded Normal→AFib: {len(n2a_results['counterfactuals'])} samples")
    print(f"  Loaded AFib→Normal: {len(a2n_results['counterfactuals'])} samples")
    print("  Skipping regeneration, will compute metrics...")
else:
    print("\n  No existing results found. Generating from scratch...")
    
    # Normal → AFib
    print("\n1. Normal → AFib")
    n2a_results = generate_counterfactuals_batch(normal_signals, 0, 1, validator)
    
    # AFib → Normal
    print("\n2. AFib → Normal")
    a2n_results = generate_counterfactuals_batch(afib_signals, 1, 0, validator)

# ============================================================================
# Save Generated Counterfactuals (before metrics to avoid data loss)
# ============================================================================

print("\n" + "="*60)
print("Saving Generated Counterfactuals")
print("="*60)

# Save intermediate results dict with PyTorch tensors
import pickle
intermediate_file = Config.OUTPUT_DIR / 'counterfactuals_generated.pkl'
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(intermediate_file, 'wb') as f:
    pickle.dump({
        'n2a_results': n2a_results,
        'a2n_results': a2n_results
    }, f)
print(f"✓ Saved generated counterfactuals to {intermediate_file}")
print(f"  Normal→AFib: {len(n2a_results['counterfactuals'])} samples")
print(f"  AFib→Normal: {len(a2n_results['counterfactuals'])} samples")

# ============================================================================
# Compute Metrics
# ============================================================================

print("\n" + "="*60)
print("Computing Metrics")
print("="*60)

def compute_metrics(results, direction_name):
    """Compute comprehensive metrics for generated counterfactuals."""
    cfs = torch.stack(results['counterfactuals'])
    originals = torch.stack(results['original_signals'])
    target_labels = torch.tensor(results['target_labels'], dtype=torch.long)
    
    # 1. Classifier flip rate (process in batches to avoid OOM)
    batch_size = 100
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(cfs), batch_size):
            batch_cfs = cfs[i:i+batch_size].to(Config.DEVICE)
            cf_logits, _ = classifier_wrapper(batch_cfs)
            cf_preds = torch.argmax(cf_logits, dim=1)
            all_preds.append(cf_preds.cpu())
            
            # Clear GPU memory
            del batch_cfs, cf_logits, cf_preds
            torch.cuda.empty_cache()
    
    all_preds = torch.cat(all_preds)
    flip_rate = (all_preds == target_labels).float().mean().item()
    
    # 2. Signal similarity
    correlations = []
    mses = []
    for i in range(len(cfs)):
        orig = originals[i, 0].numpy()
        cf = cfs[i, 0].numpy()
        corr, _ = pearsonr(orig, cf)
        correlations.append(corr)
        mses.append(np.mean((orig - cf) ** 2))
    
    mean_corr = np.mean(correlations)
    mean_mse = np.mean(mses)
    
    # 3. Plausibility statistics
    plausibility_stats = validator.compute_plausibility_stats(results['validation_results'])
    
    # 4. Attempt statistics
    mean_attempts = np.mean(results['attempts'])
    max_attempts = np.max(results['attempts'])
    
    metrics = {
        'direction': direction_name,
        'total_samples': len(cfs),
        'flip_rate': flip_rate,
        'mean_correlation': mean_corr,
        'std_correlation': np.std(correlations),
        'mean_mse': mean_mse,
        'plausibility': plausibility_stats,
        'mean_attempts': mean_attempts,
        'max_attempts': max_attempts,
    }
    
    print(f"\n{direction_name}:")
    print(f"  Samples: {len(cfs)}")
    print(f"  Flip Rate: {flip_rate:.2%}")
    print(f"  Correlation: {mean_corr:.3f} ± {np.std(correlations):.3f}")
    print(f"  MSE: {mean_mse:.4f}")
    print(f"  Plausibility Score: {plausibility_stats.get('mean_score', 0):.3f}")
    print(f"  Valid Rate: {plausibility_stats.get('valid_rate', 0):.2%}")
    if 'rr_direction_correctness' in plausibility_stats:
        print(f"  RR Direction Correct: {plausibility_stats['rr_direction_correctness']:.2%}")
    print(f"  Mean Attempts: {mean_attempts:.2f}")
    
    return metrics

metrics_n2a = compute_metrics(n2a_results, "Normal → AFib")
metrics_a2n = compute_metrics(a2n_results, "AFib → Normal")

# Overall metrics
overall_flip_rate = (metrics_n2a['flip_rate'] + metrics_a2n['flip_rate']) / 2
overall_corr = (metrics_n2a['mean_correlation'] + metrics_a2n['mean_correlation']) / 2

print(f"\n{'='*60}")
print("Overall Performance")
print(f"{'='*60}")
print(f"Average Flip Rate: {overall_flip_rate:.2%}")
print(f"Average Correlation: {overall_corr:.3f}")

# ============================================================================
# Save Outputs
# ============================================================================

print("\n" + "="*60)
print("Saving Outputs")
print("="*60)

# Combine all counterfactuals
all_cfs = torch.cat([
    torch.stack(n2a_results['counterfactuals']),
    torch.stack(a2n_results['counterfactuals'])
], dim=0)

all_cf_labels = torch.tensor(
    n2a_results['target_labels'] + a2n_results['target_labels'],
    dtype=torch.long
)

# Save as npz
np.savez(
    Config.OUTPUT_DIR / 'counterfactual_test_data.npz',
    X=all_cfs.numpy(),
    y=all_cf_labels.numpy()
)
print(f"✓ Saved counterfactual dataset: {len(all_cfs)} samples")

# Save metrics
all_metrics = {
    'normal_to_afib': metrics_n2a,
    'afib_to_normal': metrics_a2n,
    'overall': {
        'average_flip_rate': overall_flip_rate,
        'average_correlation': overall_corr,
        'total_samples': len(all_cfs),
    },
    'generation_params': {
        'sdedit_strength': Config.SDEDIT_STRENGTH,
        'cfg_scale': Config.CFG_SCALE,
        'num_steps': Config.NUM_STEPS,
        'min_plausibility_score': Config.MIN_PLAUSIBILITY_SCORE,
        'max_attempts': Config.MAX_REGENERATION_ATTEMPTS,
    }
}

with open(Config.OUTPUT_DIR / 'counterfactual_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)
print("✓ Saved metrics")

# ============================================================================
# Visualizations
# ============================================================================

print("\n" + "="*60)
print("Creating Visualizations")
print("="*60)

# 1. Sample counterfactuals grid
fig, axes = plt.subplots(4, 5, figsize=(20, 12))
axes = axes.flatten()

for i in range(10):
    # Normal → AFib
    idx = i
    axes[i].plot(n2a_results['original_signals'][idx][0].numpy(), 'b-', alpha=0.7, linewidth=0.8, label='Original')
    axes[i].plot(n2a_results['counterfactuals'][idx][0].numpy(), 'r-', alpha=0.7, linewidth=0.8, label='CF')
    axes[i].set_title(f'N→A #{i+1}\nScore: {n2a_results["validation_results"][idx]["score"]:.2f}', fontsize=9)
    axes[i].set_ylim([-2, 2])
    if i == 0:
        axes[i].legend(fontsize=8)

for i in range(10):
    # AFib → Normal
    idx = i
    axes[i+10].plot(a2n_results['original_signals'][idx][0].numpy(), 'b-', alpha=0.7, linewidth=0.8, label='Original')
    axes[i+10].plot(a2n_results['counterfactuals'][idx][0].numpy(), 'g-', alpha=0.7, linewidth=0.8, label='CF')
    axes[i+10].set_title(f'A→N #{i+1}\nScore: {a2n_results["validation_results"][idx]["score"]:.2f}', fontsize=9)
    axes[i+10].set_ylim([-2, 2])
    if i == 0:
        axes[i+10].legend(fontsize=8)

plt.suptitle(f'Sample Counterfactuals\nN→A Flip: {metrics_n2a["flip_rate"]:.1%}, A→N Flip: {metrics_a2n["flip_rate"]:.1%}', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(Config.OUTPUT_DIR / 'counterfactual_samples.png', dpi=200)
plt.close()
print("✓ Sample grid saved")

# 2. Plausibility score distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

n2a_scores = [r['score'] for r in n2a_results['validation_results']]
a2n_scores = [r['score'] for r in a2n_results['validation_results']]

axes[0].hist(n2a_scores, bins=30, alpha=0.7, color='red', edgecolor='black')
axes[0].axvline(Config.MIN_PLAUSIBILITY_SCORE, color='k', linestyle='--', label='Threshold')
axes[0].set_xlabel('Plausibility Score')
axes[0].set_ylabel('Count')
axes[0].set_title(f'Normal → AFib\nMean: {np.mean(n2a_scores):.3f}')
axes[0].legend()

axes[1].hist(a2n_scores, bins=30, alpha=0.7, color='green', edgecolor='black')
axes[1].axvline(Config.MIN_PLAUSIBILITY_SCORE, color='k', linestyle='--', label='Threshold')
axes[1].set_xlabel('Plausibility Score')
axes[1].set_ylabel('Count')
axes[1].set_title(f'AFib → Normal\nMean: {np.mean(a2n_scores):.3f}')
axes[1].legend()

plt.suptitle('Plausibility Score Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(Config.OUTPUT_DIR / 'plausibility_scores.png', dpi=150)
plt.close()
print("✓ Score distribution saved")

# 3. Correlation distribution
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

n2a_corrs = [pearsonr(n2a_results['original_signals'][i][0].numpy(), 
                      n2a_results['counterfactuals'][i][0].numpy())[0] 
             for i in range(len(n2a_results['counterfactuals']))]
a2n_corrs = [pearsonr(a2n_results['original_signals'][i][0].numpy(),
                      a2n_results['counterfactuals'][i][0].numpy())[0]
             for i in range(len(a2n_results['counterfactuals']))]

ax.hist(n2a_corrs, bins=30, alpha=0.6, color='red', label=f'N→A (μ={np.mean(n2a_corrs):.3f})', edgecolor='black')
ax.hist(a2n_corrs, bins=30, alpha=0.6, color='green', label=f'A→N (μ={np.mean(a2n_corrs):.3f})', edgecolor='black')
ax.set_xlabel('Correlation with Original')
ax.set_ylabel('Count')
ax.set_title('Signal Similarity Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Config.OUTPUT_DIR / 'correlation_distribution.png', dpi=150)
plt.close()
print("✓ Correlation distribution saved")

print("\n" + "="*60)
print("Counterfactual Generation Complete!")
print("="*60)
print(f"Output directory: {Config.OUTPUT_DIR}")
print(f"Generated {len(all_cfs)} validated counterfactuals")
print(f"Overall flip rate: {overall_flip_rate:.2%}")
print(f"Overall correlation: {overall_corr:.3f}")
print("="*60)
