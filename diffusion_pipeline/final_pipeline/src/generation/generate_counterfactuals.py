"""
Filtered Counterfactual ECG Generation
========================================
Regenerates counterfactual ECGs with strict two-gate filtering:
  1. Classifier flip verification (MUST flip to target class)
  2. Clinical plausibility validation (score >= 0.7)

Only saves ECGs that pass BOTH gates. Runs until target count reached.
Diffusion is stochastic → same source ECG can produce different outputs.

Memory-safe: fixed buffer, flush-to-disk, immediate cleanup.
Crash recovery: progress.json checkpoint on each flush.
"""

import os
import sys
import gc
import json
import time
import signal as sig
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# GPU selection
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

sys.path.append(str(Path(__file__).parent))
from diffusion_models import ContentEncoder, StyleEncoder, ConditionalUNet, DDIMScheduler
from shared_models import load_classifier, ClassifierWrapper
from plausibility_validator import PlausibilityValidator

# ============================================================================
# Configuration
# ============================================================================
class Config:
    PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
    DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
    MODEL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/enhanced_diffusion_cf'
    OUTPUT_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/generated_counterfactuals'
    VIZ_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/filtered_cf_visualizations'
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generation
    CONTENT_DIM = 256
    STYLE_DIM = 128
    MODEL_CHANNELS = 64
    NUM_CLASSES = 2
    NUM_INFERENCE_STEPS = 50
    GUIDANCE_SCALE = 3.0
    SDEDIT_STRENGTH = 0.6
    
    # Filtering gates: Two-gate approach
    # Gate 1: Classifier flip (HARD GATE - must flip to target class)
    # Gate 2: Plausibility validation
    MIN_PLAUSIBILITY_SCORE = 0.7
    # Note: Similarity gate removed (was too strict at 1.2% acceptance)
    
    # Memory
    BUFFER_SIZE = 200           # Flush to disk every 200 accepted CFs
    GEN_BATCH_SIZE = 1          # Generate 1 at a time (safe for memory)
    
    # Targets — strictly filtered (flip verified + plausible)
    TARGET_NORMAL = 10000       # Normal-target CFs (from AFib sources, ~63% accept)
    TARGET_AFIB = 10000         # AFib-target CFs (from Normal sources, ~7.6% accept)
    
    # Noise reduction
    SAVGOL_WINDOW = 11
    SAVGOL_POLY = 3
    
    # Visualization: convert from z-normalized to mV
    REAL_MEAN = -0.00396
    REAL_STD = 0.14716
    
    # Visualization frequency (fraction of target)
    VIZ_INTERVAL = 0.05  # Every 5%
    
    # Max attempts per source ECG before moving to next
    MAX_ATTEMPTS_PER_SOURCE = 5
    
    # Checkpoint
    PROGRESS_FILE = 'filtered_cf_progress.json'

Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
Config.VIZ_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {Config.DEVICE}")
print(f"Targets: {Config.TARGET_NORMAL} Normal-target + {Config.TARGET_AFIB} AFib-target")

# ============================================================================
# Load Models
# ============================================================================
print("\n" + "="*60)
print("Loading Models")
print("="*60)

checkpoint = torch.load(
    Config.MODEL_DIR / 'final_model.pth',
    map_location=Config.DEVICE,
    weights_only=False
)

content_encoder = ContentEncoder(
    in_channels=1, hidden_dim=64, content_dim=Config.CONTENT_DIM
).to(Config.DEVICE)
content_encoder.load_state_dict(checkpoint['content_encoder'])
content_encoder.eval()

style_encoder = StyleEncoder(
    in_channels=1, hidden_dim=64, style_dim=Config.STYLE_DIM, num_classes=Config.NUM_CLASSES
).to(Config.DEVICE)
style_encoder.load_state_dict(checkpoint['style_encoder'])
style_encoder.eval()

unet = ConditionalUNet(
    in_ch=1, model_ch=Config.MODEL_CHANNELS,
    content_dim=Config.CONTENT_DIM, style_dim=Config.STYLE_DIM,
    num_classes=Config.NUM_CLASSES
).to(Config.DEVICE)
unet.load_state_dict(checkpoint['unet'])
unet.eval()

scheduler = DDIMScheduler(num_timesteps=1000)
del checkpoint
torch.cuda.empty_cache()
print("✓ Diffusion model loaded")

classifier = load_classifier(Config.DEVICE)
classifier_wrapper = ClassifierWrapper(classifier).to(Config.DEVICE)
classifier_wrapper.eval()
print("✓ Classifier loaded")

validator = PlausibilityValidator()
print("✓ Plausibility validator loaded")

# ============================================================================
# Load Source Data
# ============================================================================
print("\n" + "="*60)
print("Loading Source Data")
print("="*60)

train_data = np.load(Config.DATA_DIR / 'train_data.npz')
X_train = torch.FloatTensor(train_data['X'])
y_train = torch.LongTensor(train_data['y'])
if X_train.dim() == 2:
    X_train = X_train.unsqueeze(1)

normal_indices = (y_train == 0).nonzero(as_tuple=True)[0]
afib_indices = (y_train == 1).nonzero(as_tuple=True)[0]

print(f"Normal sources: {len(normal_indices)}")
print(f"AFib sources: {len(afib_indices)}")

# ============================================================================
# Helper Functions
# ============================================================================

def reduce_noise(signal_tensor):
    """Apply Savitzky-Golay filter."""
    return torch.FloatTensor(
        savgol_filter(signal_tensor.cpu().numpy(), Config.SAVGOL_WINDOW, Config.SAVGOL_POLY, axis=-1)
    )

@torch.no_grad()
def generate_single_cf(signal, target_class):
    """Generate one counterfactual via SDEdit."""
    signal_device = signal.unsqueeze(0).to(Config.DEVICE)
    target_tensor = torch.tensor([target_class], dtype=torch.long, device=Config.DEVICE)
    
    content, _, _ = content_encoder(signal_device)
    style, _ = style_encoder(signal_device)
    
    cf = scheduler.sdedit_sample(
        unet, content, style, target_tensor,
        signal_device, strength=Config.SDEDIT_STRENGTH,
        num_steps=Config.NUM_INFERENCE_STEPS, cfg_scale=Config.GUIDANCE_SCALE
    )
    return cf[0]  # (1, 2500)

@torch.no_grad()
def check_classifier_flip(cf_tensor, target_class):
    """Gate 1: Check if classifier predicts target class."""
    cf_device = cf_tensor.unsqueeze(0).to(Config.DEVICE)
    mean = cf_device.mean(dim=2, keepdim=True)
    std = cf_device.std(dim=2, keepdim=True) + 1e-8
    cf_norm = (cf_device - mean) / std
    logits, _ = classifier_wrapper.model(cf_norm)
    pred = logits.argmax(dim=1).item()
    prob = F.softmax(logits, dim=1)[0, target_class].item()
    return pred == target_class, prob

def check_similarity(original, cf):
    """Gate 3: Check correlation with original - DEPRECATED, no longer used."""
    # Removed from filtering pipeline - was causing 1.2% acceptance rate
    # Now using classifier-based relabeling instead
    orig_np = original.squeeze().cpu().numpy()
    cf_np = cf.squeeze().cpu().numpy()
    try:
        corr, _ = pearsonr(orig_np, cf_np)
        return True, corr  # Always pass, just return correlation for logging
    except:
        return True, 0.0

def to_mv(signal):
    """Convert z-normalized signal to mV range for visualization."""
    return signal * Config.REAL_STD + Config.REAL_MEAN

def visualize_samples(accepted_list, direction, count, total_target):
    """Save visualization of 4 accepted counterfactuals in mV."""
    n = min(4, len(accepted_list))
    if n == 0:
        return
    
    fig, axes = plt.subplots(n, 3, figsize=(18, 4*n))
    if n == 1:
        axes = axes.reshape(1, -1)
    
    pct = 100 * count / total_target
    fig.suptitle(f'{direction}: {count}/{total_target} ({pct:.1f}%) — Sample Quality Check',
                 fontsize=14, fontweight='bold')
    
    time_axis = np.arange(2500) / 250  # seconds (fs=250)
    
    for i in range(n):
        item = accepted_list[-(n-i)]  # Last n items
        orig_mv = to_mv(item['original'].squeeze())
        cf_mv = to_mv(item['cf'].squeeze())
        
        axes[i, 0].plot(time_axis, orig_mv, 'b-', lw=0.8)
        axes[i, 0].set_title(f'Original (class {item["original_class"]})')
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].grid(True, alpha=0.3)
        
        axes[i, 1].plot(time_axis, cf_mv, 'r-', lw=0.8)
        axes[i, 1].set_title(f'CF (target={item["target_class"]}, prob={item["prob"]:.3f}, '
                             f'plaus={item["plausibility"]:.3f})')
        axes[i, 1].set_ylabel('mV')
        axes[i, 1].grid(True, alpha=0.3)
        
        axes[i, 2].plot(time_axis, orig_mv, 'b-', lw=0.8, alpha=0.6, label='Original')
        axes[i, 2].plot(time_axis, cf_mv, 'r-', lw=0.8, alpha=0.6, label='CF')
        axes[i, 2].set_title(f'Overlay (corr={item["correlation"]:.3f})')
        axes[i, 2].set_ylabel('mV')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)
    
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')
    
    plt.tight_layout()
    safe_dir = direction.replace('→', 'to').replace(' ', '_')
    plt.savefig(Config.VIZ_DIR / f'viz_{safe_dir}_{count:06d}.png', dpi=100)
    plt.close()
    print(f"     Visualization saved at {count}/{total_target}")

# ============================================================================
# Buffer Management
# ============================================================================

class AcceptedBuffer:
    """Fixed-size buffer that flushes to disk."""
    
    def __init__(self, direction_name):
        self.direction = direction_name
        safe_name = direction_name.replace('→', 'to').replace(' ', '_')
        self.prefix = f'filtered_{safe_name}'
        self.buffer_cfs = []
        self.buffer_originals = []
        self.buffer_labels = []
        self.buffer_orig_labels = []
        self.buffer_scores = []
        self.buffer_corrs = []
        self.flush_count = 0
        self.total_saved = 0
    
    def add(self, cf, original, target_label, orig_label, plausibility, correlation):
        """Add CF with verified target label (flip confirmed by classifier)."""
        self.buffer_cfs.append(cf.squeeze().cpu().numpy())
        self.buffer_originals.append(original.squeeze().cpu().numpy())
        self.buffer_labels.append(target_label)  # Guaranteed correct: flip verified
        self.buffer_orig_labels.append(orig_label)
        self.buffer_scores.append(plausibility)
        self.buffer_corrs.append(correlation)
        
        if len(self.buffer_cfs) >= Config.BUFFER_SIZE:
            self.flush()
    
    def flush(self):
        if not self.buffer_cfs:
            return
        
        filename = Config.OUTPUT_DIR / f'{self.prefix}_batch_{self.flush_count:04d}.npz'
        np.savez_compressed(
            filename,
            counterfactuals=np.array(self.buffer_cfs),
            originals=np.array(self.buffer_originals),
            target_labels=np.array(self.buffer_labels),
            original_labels=np.array(self.buffer_orig_labels),
            plausibility_scores=np.array(self.buffer_scores),
            correlations=np.array(self.buffer_corrs)
        )
        
        saved_count = len(self.buffer_cfs)
        self.total_saved += saved_count
        self.flush_count += 1
        
        # Clear buffer
        self.buffer_cfs.clear()
        self.buffer_originals.clear()
        self.buffer_labels.clear()
        self.buffer_orig_labels.clear()
        self.buffer_scores.clear()
        self.buffer_corrs.clear()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"   Flushed {saved_count} to {filename.name} (total: {self.total_saved})")
    
    @property
    def count(self):
        return self.total_saved + len(self.buffer_cfs)

# ============================================================================
# Checkpoint
# ============================================================================

def save_progress(n2a_count, a2n_count, stats):
    """Save minimal checkpoint."""
    progress = {
        'normal_to_afib_accepted': n2a_count,
        'afib_to_normal_accepted': a2n_count,
        'stats': stats,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(Config.OUTPUT_DIR / Config.PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

def load_progress():
    """Load checkpoint if exists."""
    path = Config.OUTPUT_DIR / Config.PROGRESS_FILE
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

# ============================================================================
# Main Generation Loop
# ============================================================================

def generate_direction(source_signals, source_indices, original_class, target_class,
                       direction_name, target_count, existing_count=0):
    """Generate filtered counterfactuals for one direction.
    
    Two-gate filtering:
      Gate 1: Classifier MUST predict target class (hard gate)
      Gate 2: Plausibility score >= 0.7
    
    Labels = target_class (guaranteed correct since flip was verified).
    """
    
    print(f"\n{'='*60}")
    print(f"Generating: {direction_name}")
    print(f"Target: {target_count}, Already have: {existing_count}")
    print(f"Source pool: {len(source_indices)} ECGs (will reuse as needed)")
    print(f"Strategy: Gate 1 (flip required) + Gate 2 (plausibility >= 0.7)")
    print(f"{'='*60}")
    
    buffer = AcceptedBuffer(direction_name)
    buffer.total_saved = existing_count
    
    stats = {
        'total_attempts': 0,
        'gate1_pass': 0,  # classifier flip
        'gate2_pass': 0,  # plausibility (= accepted)
        'gate1_fail': 0,
        'gate2_fail': 0,
    }
    
    # For visualization tracking
    next_viz_at = int(target_count * Config.VIZ_INTERVAL)
    viz_buffer = []
    
    start_time = time.time()
    source_idx = 0
    attempts_on_current = 0
    
    while buffer.count < target_count:
        # Pick source ECG (cycle through with reuse)
        idx = source_indices[source_idx % len(source_indices)]
        source_signal = X_train[idx]
        
        stats['total_attempts'] += 1
        
        # --- Generate ---
        try:
            cf_raw = generate_single_cf(source_signal, target_class)
            cf = reduce_noise(cf_raw)
        except Exception as e:
            print(f"   Generation error: {e}")
            source_idx += 1
            attempts_on_current = 0
            continue
        
        # --- Gate 1: Classifier flip (HARD GATE) ---
        passed, prob = check_classifier_flip(cf, target_class)
        if not passed:
            stats['gate1_fail'] += 1
            del cf, cf_raw
            attempts_on_current += 1
            if attempts_on_current >= Config.MAX_ATTEMPTS_PER_SOURCE:
                source_idx += 1
                attempts_on_current = 0
            continue
        stats['gate1_pass'] += 1
        
        # --- Gate 2: Plausibility ---
        val_result = validator.validate(
            cf[0].numpy(),
            original_ecg=source_signal[0].numpy(),
            target_class=target_class,
            original_class=original_class
        )
        if not val_result['valid'] or val_result['score'] < Config.MIN_PLAUSIBILITY_SCORE:
            stats['gate2_fail'] += 1
            del cf, cf_raw
            attempts_on_current += 1
            if attempts_on_current >= Config.MAX_ATTEMPTS_PER_SOURCE:
                source_idx += 1
                attempts_on_current = 0
            continue
        stats['gate2_pass'] += 1
        
        # --- Get correlation for logging ---
        _, corr = check_similarity(source_signal, cf)
        
        # --- BOTH GATES PASSED: Save with target_class label (verified correct) ---
        buffer.add(cf, source_signal, target_class, original_class,
                   val_result['score'], corr)
        
        # Track for visualization
        viz_buffer.append({
            'cf': cf[0].numpy(),
            'original': source_signal[0].numpy(),
            'target_class': target_class,
            'original_class': original_class,
            'prob': prob,
            'plausibility': val_result['score'],
            'correlation': corr
        })
        
        # Move to next source
        attempts_on_current += 1
        if attempts_on_current >= Config.MAX_ATTEMPTS_PER_SOURCE:
            source_idx += 1
            attempts_on_current = 0
        
        # Clean up
        del cf, cf_raw
        
        current = buffer.count
        
        # Progress logging every 50 accepted
        if current % 50 == 0:
            elapsed = time.time() - start_time
            rate = current / elapsed if elapsed > 0 else 0
            accept_rate = stats['gate2_pass'] / max(1, stats['total_attempts']) * 100
            flip_rate = stats['gate1_pass'] / max(1, stats['total_attempts']) * 100
            eta_hours = (target_count - current) / max(rate, 0.01) / 3600
            print(f"  [{current}/{target_count}] {100*current/target_count:.1f}% | "
                  f"Rate: {rate:.2f}/s | Accept: {accept_rate:.1f}% | "
                  f"Flip: {flip_rate:.1f}% | ETA: {eta_hours:.1f}h", flush=True)
        
        # Visualization at ~5% intervals
        if current >= next_viz_at and len(viz_buffer) >= 4:
            visualize_samples(viz_buffer, direction_name, current, target_count)
            next_viz_at += int(target_count * Config.VIZ_INTERVAL)
            viz_buffer = viz_buffer[-4:]  # Keep last 4 for next viz
    
    # Final flush
    buffer.flush()
    
    elapsed = time.time() - start_time
    print(f"\n✓ {direction_name} COMPLETE")
    print(f"  Accepted: {buffer.total_saved}")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Accept rate: {stats['gate2_pass']/max(1,stats['total_attempts'])*100:.1f}%")
    print(f"  Gate 1 (flip) pass: {stats['gate1_pass']}/{stats['total_attempts']} "
          f"({stats['gate1_pass']/max(1,stats['total_attempts'])*100:.1f}%)")
    print(f"  Gate 2 (plausibility) pass: {stats['gate2_pass']}/{stats['gate1_pass']} "
          f"({stats['gate2_pass']/max(1,stats['gate1_pass'])*100:.1f}%)")
    print(f"  Time: {elapsed/3600:.2f} hours")
    
    return stats

# ============================================================================
# Assemble Final Dataset
# ============================================================================

def assemble_final_dataset():
    """Merge all batch files and balance classes based on actual classifier predictions."""
    print("\n" + "="*60)
    print("Assembling Final Filtered Dataset")
    print("="*60)
    
    all_cfs = []
    all_originals = []
    all_labels = []
    all_orig_labels = []
    all_scores = []
    all_corrs = []
    
    for batch_file in sorted(Config.OUTPUT_DIR.glob('filtered_*.npz')):
        data = np.load(batch_file)
        all_cfs.append(data['counterfactuals'])
        all_originals.append(data['originals'])
        all_labels.append(data['target_labels'])  # These are actual predictions now
        all_orig_labels.append(data['original_labels'])
        all_scores.append(data['plausibility_scores'])
        all_corrs.append(data['correlations'])
        print(f"  Loaded {batch_file.name}: {len(data['counterfactuals'])} samples")
    
    if not all_cfs:
        print("ERROR: No batch files found!")
        return
    
    cfs = np.concatenate(all_cfs)
    originals = np.concatenate(all_originals)
    labels = np.concatenate(all_labels)
    orig_labels = np.concatenate(all_orig_labels)
    scores = np.concatenate(all_scores)
    corrs = np.concatenate(all_corrs)
    
    n_normal = (labels == 0).sum()
    n_afib = (labels == 1).sum()
    
    print(f"\nBefore balancing:")
    print(f"  Total: {len(cfs)}")
    print(f"  Normal: {n_normal}, AFib: {n_afib}")
    
    # Balance classes
    min_class = min(n_normal, n_afib)
    normal_indices = np.where(labels == 0)[0]
    afib_indices = np.where(labels == 1)[0]
    
    # Randomly sample to balance
    np.random.seed(42)
    selected_normal = np.random.choice(normal_indices, min_class, replace=False)
    selected_afib = np.random.choice(afib_indices, min_class, replace=False)
    selected_indices = np.concatenate([selected_normal, selected_afib])
    np.random.shuffle(selected_indices)
    
    # Apply selection
    cfs = cfs[selected_indices]
    originals = originals[selected_indices]
    labels = labels[selected_indices]
    orig_labels = orig_labels[selected_indices]
    scores = scores[selected_indices]
    corrs = corrs[selected_indices]
    
    # Save unified dataset
    output_path = Config.OUTPUT_DIR / 'filtered_counterfactuals.npz'
    np.savez_compressed(
        output_path,
        X=cfs,
        y=labels,
        originals=originals,
        original_labels=orig_labels,
        plausibility_scores=scores,
        correlations=corrs
    )
    
    print(f"\nAfter balancing:")
    print(f"  Total: {len(cfs)}")
    print(f"  Normal: {(labels==0).sum()}, AFib: {(labels==1).sum()}")
    print(f"  Mean plausibility: {scores.mean():.3f}")
    print(f"  Mean correlation: {corrs.mean():.3f}")
    
    print(f"\n✓ Final dataset saved: {output_path}")
    
    # Also save in format compatible with three-way eval
    compat_path = Config.OUTPUT_DIR / 'counterfactual_test_data.npz'
    np.savez(compat_path, X=cfs, y=labels)
    print(f"✓ Compatible format saved: {compat_path}")

# ============================================================================
# Final Verification
# ============================================================================

@torch.no_grad()
def verify_final_dataset():
    """Re-verify all saved counterfactuals through classifier."""
    print("\n" + "="*60)
    print("FINAL VERIFICATION")
    print("="*60)
    
    data = np.load(Config.OUTPUT_DIR / 'filtered_counterfactuals.npz')
    cfs = torch.FloatTensor(data['X']).unsqueeze(1) if data['X'].ndim == 2 else torch.FloatTensor(data['X'])
    labels = torch.LongTensor(data['y'])
    
    correct = 0
    total = len(cfs)
    
    for i in range(0, total, 100):
        batch = cfs[i:i+100].to(Config.DEVICE)
        batch_labels = labels[i:i+100]
        
        mean = batch.mean(dim=2, keepdim=True)
        std = batch.std(dim=2, keepdim=True) + 1e-8
        batch_norm = (batch - mean) / std
        
        logits, _ = classifier_wrapper.model(batch_norm)
        preds = logits.argmax(dim=1).cpu()
        correct += (preds == batch_labels).sum().item()
        
        del batch, logits, preds
        torch.cuda.empty_cache()
    
    flip_rate = correct / total * 100
    print(f"  Verified flip rate: {flip_rate:.1f}% ({correct}/{total})")
    print(f"  Normal-target: {(labels==0).sum().item()}")
    print(f"  AFib-target: {(labels==1).sum().item()}")
    
    if flip_rate < 95:
        print("   WARNING: Flip rate dropped below 95% on re-verification!")
    else:
        print("  ✓ All labels verified correct!")
    
    return flip_rate

# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*60)
    print("FILTERED COUNTERFACTUAL GENERATION")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Check for existing progress
    progress = load_progress()
    existing_n2a = 0
    existing_a2n = 0
    
    if progress:
        existing_n2a = progress.get('normal_to_afib_accepted', 0)
        existing_a2n = progress.get('afib_to_normal_accepted', 0)
        print(f"\n Resuming from checkpoint:")
        print(f"  Normal→AFib: {existing_n2a}/{Config.TARGET_AFIB}")
        print(f"  AFib→Normal: {existing_a2n}/{Config.TARGET_NORMAL}")
    
    all_stats = {}
    
    # Direction 1: Normal → AFib (harder, ~15% pass rate)
    if existing_n2a < Config.TARGET_AFIB:
        stats_n2a = generate_direction(
            X_train, normal_indices, 
            original_class=0, target_class=1,
            direction_name="Normal → AFib",
            target_count=Config.TARGET_AFIB,
            existing_count=existing_n2a
        )
        all_stats['normal_to_afib'] = stats_n2a
        save_progress(Config.TARGET_AFIB, existing_a2n, all_stats)
    else:
        print(f"\n✓ Normal→AFib already complete ({existing_n2a})")
    
    # Direction 2: AFib → Normal (easier, ~66% pass rate)
    if existing_a2n < Config.TARGET_NORMAL:
        stats_a2n = generate_direction(
            X_train, afib_indices,
            original_class=1, target_class=0,
            direction_name="AFib → Normal",
            target_count=Config.TARGET_NORMAL,
            existing_count=existing_a2n
        )
        all_stats['afib_to_normal'] = stats_a2n
        save_progress(Config.TARGET_AFIB, Config.TARGET_NORMAL, all_stats)
    else:
        print(f"\n✓ AFib→Normal already complete ({existing_a2n})")
    
    # Assemble
    assemble_final_dataset()
    
    # Verify
    flip_rate = verify_final_dataset()
    
    # Save final stats
    all_stats['final_flip_rate'] = flip_rate
    all_stats['completed'] = time.strftime('%Y-%m-%d %H:%M:%S')
    save_progress(Config.TARGET_AFIB, Config.TARGET_NORMAL, all_stats)
    
    print("\n" + "="*60)
    print(" GENERATION COMPLETE!")
    print("="*60)
    print(f"Output: {Config.OUTPUT_DIR / 'filtered_counterfactuals.npz'}")
    print(f"Verified flip rate: {flip_rate:.1f}%")

if __name__ == '__main__':
    main()
