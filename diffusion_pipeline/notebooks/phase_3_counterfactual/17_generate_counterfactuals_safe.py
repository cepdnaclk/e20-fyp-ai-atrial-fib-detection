"""
Memory-Efficient Counterfactual Generation Script
==================================================

Processes counterfactuals in small batches to prevent RAM overflow.
- Batch size: 500 samples
- Saves immediately after each batch
- Clears memory between batches
- Single process only

Author: Phase 3 Counterfactual Generation (Memory-Safe Version)
"""

import os
import sys
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import custom modules
from diffusion_models import ContentEncoder, StyleEncoder, ConditionalUNet, DDIMScheduler
from shared_models import load_classifier, ClassifierWrapper
from plausibility_validator import PlausibilityValidator

# ============================================================================
# Configuration
# ============================================================================

class Config:
    # Paths
    BASE_DIR = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/notebooks/phase_3_counterfactual')
    MODEL_DIR = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models/phase3_counterfactual/enhanced_diffusion_cf')
    OUTPUT_DIR = BASE_DIR / 'results'
    PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
    DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
    
    # Model parameters
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONTENT_DIM = 256  # Must match checkpoint
    STYLE_DIM = 128    # Must match checkpoint
    MODEL_CHANNELS = 64  # Must match checkpoint (base channels, scales to 512)
    NUM_CLASSES = 2
    
    # Generation parameters (must match original working script)
    BATCH_SIZE = 500  # CRITICAL: Process 500 samples at a time to control memory
    NUM_INFERENCE_STEPS = 50
    GUIDANCE_SCALE = 3.0  # CFG scale from original
    SDEDIT_STRENGTH = 0.6  # SDEdit strength from original
    MAX_VALIDATION_ATTEMPTS = 3
    MIN_PLAUSIBILITY_SCORE = 0.7  # From original
    
    # Signal processing
    SAVGOL_WINDOW = 11
    SAVGOL_POLY = 3

print(f"Device: {Config.DEVICE}")
print(f"Batch size: {Config.BATCH_SIZE} (memory-safe)")

# Create output directory
Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Load Models
# ============================================================================

print("\n" + "="*60)
print("Loading Trained Models")
print("="*60)

checkpoint = torch.load(
    Config.MODEL_DIR / 'final_model.pth',
    map_location=Config.DEVICE,
    weights_only=False
)

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

scheduler = DDIMScheduler(num_timesteps=1000)

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
X_test = torch.FloatTensor(test_data['X'])
y_test = torch.LongTensor(test_data['y'])

# Add channel dimension if needed (N, L) -> (N, 1, L)
if X_test.dim() == 2:
    X_test = X_test.unsqueeze(1)

# Separate by class
normal_signals = X_test[y_test == 0]
afib_signals = X_test[y_test == 1]

print(f"Total test samples: {len(X_test)}")
print(f"Normal: {len(normal_signals)}")
print(f"AFib: {len(afib_signals)}")

# ============================================================================
# Helper Functions
# ============================================================================

def reduce_noise(signal, window_length=11, polyorder=3):
    """Apply Savitzky-Golay filter to reduce noise."""
    return torch.FloatTensor(
        savgol_filter(signal.cpu().numpy(), window_length, polyorder, axis=-1)
    )

@torch.no_grad()
def generate_single_counterfactual(signal, original_class, target_class):
    """Generate counterfactual for a single signal using SDEdit."""
    signal_device = signal.unsqueeze(0).to(Config.DEVICE)
    target_class_tensor = torch.tensor([target_class], dtype=torch.long, device=Config.DEVICE)
    
    # Encode original (returns tuple: z, mu, logvar)
    content, _, _ = content_encoder(signal_device)
    style, _ = style_encoder(signal_device)
    
    # Generate via SDEdit
    cf = scheduler.sdedit_sample(
        unet, content, style, target_class_tensor,
        signal_device, strength=0.6,
        num_steps=Config.NUM_INFERENCE_STEPS, cfg_scale=Config.GUIDANCE_SCALE
    )
    
    return cf[0]

def generate_with_validation(signal, original_class, target_class, validator):
    """Generate with plausibility validation and retries."""
    for attempt in range(Config.MAX_VALIDATION_ATTEMPTS):
        cf = generate_single_counterfactual(signal, original_class, target_class)
        cf = reduce_noise(cf, Config.SAVGOL_WINDOW, Config.SAVGOL_POLY)
        
        result = validator.validate(
            cf[0],
            original_ecg=signal[0],
            target_class=target_class,
            original_class=original_class
        )
        
        if result['valid'] and result['score'] >= Config.MIN_PLAUSIBILITY_SCORE:
            return cf[0].cpu(), result, attempt + 1
    
    return cf[0].cpu(), result, Config.MAX_VALIDATION_ATTEMPTS

# ============================================================================
# Memory-Efficient Batch Generation
# ============================================================================

def save_batch(batch_data, batch_idx, direction):
    """Save a batch to disk and clear memory."""
    filename = Config.OUTPUT_DIR / f'batch_{direction}_{batch_idx:04d}.npz'
    np.savez_compressed(
        filename,
        counterfactuals=batch_data['counterfactuals'],
        originals=batch_data['originals'],
        target_labels=batch_data['target_labels'],
        original_labels=batch_data['original_labels'],
        validation_scores=batch_data['validation_scores'],
        attempts=batch_data['attempts']
    )
    return filename

def generate_batched(signals, original_class, target_class, direction_name):
    """Generate counterfactuals in memory-safe batches."""
    print(f"\n{direction_name}")
    print(f"Processing {len(signals)} samples in batches of {Config.BATCH_SIZE}")
    
    total_samples = len(signals)
    num_batches = (total_samples + Config.BATCH_SIZE - 1) // Config.BATCH_SIZE
    batch_files = []
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * Config.BATCH_SIZE
        end_idx = min(start_idx + Config.BATCH_SIZE, total_samples)
        batch_signals = signals[start_idx:end_idx]
        
        print(f"\n  Batch {batch_idx + 1}/{num_batches} [{start_idx}-{end_idx}]")
        
        batch_data = {
            'counterfactuals': [],
            'originals': [],
            'target_labels': [],
            'original_labels': [],
            'validation_scores': [],
            'attempts': []
        }
        
        batch_start_time = time.time()
        
        for i, signal in enumerate(batch_signals):
            cf, val_result, attempts = generate_with_validation(
                signal, original_class, target_class, validator
            )
            
            batch_data['counterfactuals'].append(cf.numpy())
            batch_data['originals'].append(signal[0].numpy())
            batch_data['target_labels'].append(target_class)
            batch_data['original_labels'].append(original_class)
            batch_data['validation_scores'].append(val_result['score'])
            batch_data['attempts'].append(attempts)
            
            # Progress within batch
            if (i + 1) % 100 == 0 or (i + 1) == len(batch_signals):
                elapsed = time.time() - batch_start_time
                rate = (i + 1) / elapsed
                pct = 100 * (i + 1) / len(batch_signals)
                global_pct = 100 * (start_idx + i + 1) / total_samples
                print(f"    [{i+1}/{len(batch_signals)}] {pct:.1f}% batch | {global_pct:.1f}% total | {rate:.2f} samples/s", flush=True)
        
        # Convert to arrays
        for key in batch_data:
            batch_data[key] = np.array(batch_data[key])
        
        # Save batch
        batch_file = save_batch(batch_data, batch_idx, direction_name.replace(' → ', '_to_').replace(' ', '_'))
        batch_files.append(batch_file)
        print(f"  ✓ Saved batch to {batch_file.name}")
        
        # CRITICAL: Clear memory
        del batch_data, batch_signals
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  ✓ Memory cleared")
    
    return batch_files

# ============================================================================
# Generate Counterfactuals
# ============================================================================

print("\n" + "="*60)
print("Generating Counterfactuals (Memory-Safe Batched)")
print("="*60)

# Normal → AFib
n2a_files = generate_batched(normal_signals, 0, 1, "Normal → AFib")

# AFib → Normal
a2n_files = generate_batched(afib_signals, 1, 0, "AFib → Normal")

# ============================================================================
# Compute Metrics from Batches
# ============================================================================

print("\n" + "="*60)
print("Computing Metrics from Batches")
print("="*60)

def compute_metrics_from_files(batch_files, direction_name):
    """Compute metrics from saved batch files."""
    flip_rates = []
    correlations = []
    mses = []
    plausibility_scores = []
    attempts_list = []
    
    for batch_file in batch_files:
        data = np.load(batch_file)
        
        # Load batch
        cfs = torch.FloatTensor(data['counterfactuals']).unsqueeze(1)
        originals = torch.FloatTensor(data['originals'])
        target_labels = torch.LongTensor(data['target_labels'])
        
        # Compute flip rate for this batch
        with torch.no_grad():
            batch_preds = []
            for i in range(0, len(cfs), 100):
                batch_cf = cfs[i:i+100].to(Config.DEVICE)
                logits, _ = classifier_wrapper(batch_cf)
                preds = torch.argmax(logits, dim=1)
                batch_preds.append(preds.cpu())
                del batch_cf, logits, preds
                torch.cuda.empty_cache()
            
            all_preds = torch.cat(batch_preds)
            flip_rate = (all_preds == target_labels).float().mean().item()
            flip_rates.append(flip_rate)
        
        # Compute similarities
        for cf, orig in zip(data['counterfactuals'], data['originals']):
            corr, _ = pearsonr(orig, cf)
            correlations.append(corr)
            mses.append(np.mean((orig - cf) ** 2))
        
        plausibility_scores.extend(data['validation_scores'])
        attempts_list.extend(data['attempts'])
        
        # Clear memory
        del data, cfs, originals, target_labels, all_preds
        gc.collect()
    
    metrics = {
        'direction': direction_name,
        'flip_rate': np.mean(flip_rates),
        'mean_correlation': np.mean(correlations),
        'std_correlation': np.std(correlations),
        'mean_mse': np.mean(mses),
        'mean_plausibility': np.mean(plausibility_scores),
        'mean_attempts': np.mean(attempts_list)
    }
    
    print(f"\n{direction_name}:")
    print(f"  Flip Rate: {metrics['flip_rate']:.2%}")
    print(f"  Correlation: {metrics['mean_correlation']:.3f} ± {metrics['std_correlation']:.3f}")
    print(f"  MSE: {metrics['mean_mse']:.4f}")
    print(f"  Plausibility: {metrics['mean_plausibility']:.3f}")
    print(f"  Mean Attempts: {metrics['mean_attempts']:.2f}")
    
    return metrics

metrics_n2a = compute_metrics_from_files(n2a_files, "Normal → AFib")
metrics_a2n = compute_metrics_from_files(a2n_files, "AFib → Normal")

# Save metrics
with open(Config.OUTPUT_DIR / 'metrics.json', 'w') as f:
    json.dump({
        'normal_to_afib': metrics_n2a,
        'afib_to_normal': metrics_a2n,
        'overall_flip_rate': (metrics_n2a['flip_rate'] + metrics_a2n['flip_rate']) / 2
    }, f, indent=2)

print("\n" + "="*60)
print("COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"✓ Generated {len(normal_signals) + len(afib_signals)} counterfactuals")
print(f"✓ Saved in {len(n2a_files) + len(a2n_files)} batch files")
print(f"✓ Metrics saved to metrics.json")
print(f"✓ Memory-safe processing complete")
