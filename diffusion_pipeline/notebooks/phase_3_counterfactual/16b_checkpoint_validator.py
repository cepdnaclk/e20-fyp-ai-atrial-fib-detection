#!/usr/bin/env python3
"""
Checkpoint Plausibility Validator

This script validates the clinical plausibility of counterfactuals generated
from training checkpoints. Run this in parallel with training to monitor
plausibility progression without interrupting training.

Usage:
    python 16b_checkpoint_validator.py --checkpoint checkpoint_stage2_epoch_060.pth
    python 16b_checkpoint_validator.py --latest  # Use latest checkpoint
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from datetime import datetime
import sys

# Add project root to path
PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
sys.path.append(str(PROJECT_ROOT / 'notebooks/phase_3_counterfactual'))

from plausibility_validator import PlausibilityValidator
from shared_models import load_classifier, ClassifierWrapper
import math

# ============================================================================
# Configuration
# ============================================================================

class Config:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/enhanced_diffusion_cf'
    DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
    OUTPUT_DIR = MODEL_DIR / 'checkpoint_validation'
    
    # Model architecture
    IN_CHANNELS = 1
    SIGNAL_LENGTH = 2500
    CONTENT_DIM = 256
    STYLE_DIM = 128
    MODEL_CHANNELS = 64
    ENCODER_CHANNELS = 64
    NUM_CLASSES = 2
    DIFFUSION_TIMESTEPS = 1000
    
    # Generation parameters
    SDEDIT_STRENGTH = 0.6
    CFG_SCALE = 3.0
    SAVGOL_WINDOW = 11
    SAVGOL_POLY = 3
    NUM_SAMPLES_PER_CLASS = 50  # Samples to validate per direction

Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Import Model Architectures from Training Script
# ============================================================================

# We need to import the same architectures used in training
exec(open(PROJECT_ROOT / 'notebooks/phase_3_counterfactual/16_enhanced_diffusion_cf.py').read().split('# ============================================================================\n# Initialize Models')[0])

# ============================================================================
# Helper Functions
# ============================================================================

def get_latest_checkpoint(checkpoint_dir):
    """Find the latest checkpoint file."""
    checkpoints = list(checkpoint_dir.glob('checkpoint_*.pth'))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)

def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path.name}")
    
    # Initialize models
    content_encoder = ContentEncoder(
        in_channels=Config.IN_CHANNELS,
        hidden_dim=Config.ENCODER_CHANNELS,
        content_dim=Config.CONTENT_DIM
    ).to(device)
    
    style_encoder = StyleEncoder(
        in_channels=Config.IN_CHANNELS,
        hidden_dim=Config.ENCODER_CHANNELS,
        style_dim=Config.STYLE_DIM,
        num_classes=Config.NUM_CLASSES
    ).to(device)
    
    unet = ConditionalUNet(
        in_ch=Config.IN_CHANNELS,
        model_ch=Config.MODEL_CHANNELS,
        content_dim=Config.CONTENT_DIM,
        style_dim=Config.STYLE_DIM,
        num_classes=Config.NUM_CLASSES
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    content_encoder.load_state_dict(checkpoint['content_encoder'])
    style_encoder.load_state_dict(checkpoint['style_encoder'])
    unet.load_state_dict(checkpoint['unet'])
    
    content_encoder.eval()
    style_encoder.eval()
    unet.eval()
    
    epoch = checkpoint.get('epoch', 0)
    stage = checkpoint.get('stage', 1)
    
    return content_encoder, style_encoder, unet, epoch, stage

def load_test_data():
    """Load test data for validation."""
    test_data = np.load(Config.DATA_DIR / 'test_data.npz')
    signals = torch.tensor(test_data['X'], dtype=torch.float32)
    labels = torch.tensor(test_data['y'], dtype=torch.long)
    return signals, labels

@torch.no_grad()
def validate_checkpoint(checkpoint_path):
    """Validate clinical plausibility of a checkpoint."""
    
    device = Config.DEVICE
    
    # Load checkpoint
    content_encoder, style_encoder, unet, epoch, stage = load_checkpoint(checkpoint_path, device)
    
    # Initialize scheduler and validator
    scheduler = DDIMScheduler(Config.DIFFUSION_TIMESTEPS, device)
    validator = PlausibilityValidator()
    classifier = load_classifier(device)
    classifier_wrapper = ClassifierWrapper(classifier).to(device)
    
    # Load test data
    test_signals, test_labels = load_test_data()
    
    # Select samples for each direction
    normal_idx = (test_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (test_labels == 1).nonzero(as_tuple=True)[0]
    
    n_samples = Config.NUM_SAMPLES_PER_CLASS
    normal_samples = test_signals[normal_idx[:n_samples]].to(device)
    afib_samples = test_signals[afib_idx[:n_samples]].to(device)
    
    print(f"\nValidating {n_samples} samples per direction...")
    
    results = {
        'checkpoint': checkpoint_path.name,
        'epoch': epoch,
        'stage': stage,
        'timestamp': datetime.now().isoformat(),
        'normal_to_afib': {},
        'afib_to_normal': {}
    }
    
    # ========== Normal → AFib ==========
    print("\n1. Generating Normal → AFib counterfactuals...")
    normal_content, _, _ = content_encoder(normal_samples)
    normal_style, _ = style_encoder(normal_samples)
    target_afib = torch.ones(n_samples, dtype=torch.long, device=device)
    
    cf_n2a = scheduler.sdedit_sample(
        unet, normal_content, normal_style, target_afib,
        normal_samples, strength=Config.SDEDIT_STRENGTH,
        num_steps=50, cfg_scale=Config.CFG_SCALE
    )
    cf_n2a = reduce_noise(cf_n2a, Config.SAVGOL_WINDOW, Config.SAVGOL_POLY)
    
    # Validate
    val_results_n2a = []
    for i in range(len(cf_n2a)):
        result = validator.validate(
            cf_n2a[i].cpu().numpy(),
            original_ecg=normal_samples[i].cpu().numpy(),
            target_class=1,
            original_class=0
        )
        val_results_n2a.append(result)
    
    stats_n2a = validator.compute_plausibility_stats(val_results_n2a)
    
    # Check flip rate
    cf_logits_n2a, _ = classifier_wrapper(cf_n2a)
    cf_pred_n2a = torch.argmax(cf_logits_n2a, dim=1)
    flip_rate_n2a = (cf_pred_n2a == 1).float().mean().item()
    
    # Compute similarity
    corr_n2a = np.mean([np.corrcoef(normal_samples[i, 0].cpu(), cf_n2a[i, 0].cpu())[0, 1] 
                        for i in range(len(cf_n2a))])
    
    results['normal_to_afib'] = {
        'flip_rate': flip_rate_n2a,
        'correlation': float(corr_n2a),
        'plausibility': stats_n2a
    }
    
    print(f"  Flip Rate: {flip_rate_n2a:.2%}")
    print(f"  Correlation: {corr_n2a:.3f}")
    print(f"  Plausibility Score: {stats_n2a['mean_score']:.3f}")
    print(f"  Valid Rate: {stats_n2a['valid_rate']:.2%}")
    
    # ========== AFib → Normal ==========
    print("\n2. Generating AFib → Normal counterfactuals...")
    afib_content, _, _ = content_encoder(afib_samples)
    afib_style, _ = style_encoder(afib_samples)
    target_normal = torch.zeros(n_samples, dtype=torch.long, device=device)
    
    cf_a2n = scheduler.sdedit_sample(
        unet, afib_content, afib_style, target_normal,
        afib_samples, strength=Config.SDEDIT_STRENGTH,
        num_steps=50, cfg_scale=Config.CFG_SCALE
    )
    cf_a2n = reduce_noise(cf_a2n, Config.SAVGOL_WINDOW, Config.SAVGOL_POLY)
    
    # Validate
    val_results_a2n = []
    for i in range(len(cf_a2n)):
        result = validator.validate(
            cf_a2n[i].cpu().numpy(),
            original_ecg=afib_samples[i].cpu().numpy(),
            target_class=0,
            original_class=1
        )
        val_results_a2n.append(result)
    
    stats_a2n = validator.compute_plausibility_stats(val_results_a2n)
    
    # Check flip rate
    cf_logits_a2n, _ = classifier_wrapper(cf_a2n)
    cf_pred_a2n = torch.argmax(cf_logits_a2n, dim=1)
    flip_rate_a2n = (cf_pred_a2n == 0).float().mean().item()
    
    # Compute similarity
    corr_a2n = np.mean([np.corrcoef(afib_samples[i, 0].cpu(), cf_a2n[i, 0].cpu())[0, 1] 
                        for i in range(len(cf_a2n))])
    
    results['afib_to_normal'] = {
        'flip_rate': flip_rate_a2n,
        'correlation': float(corr_a2n),
        'plausibility': stats_a2n
    }
    
    print(f"  Flip Rate: {flip_rate_a2n:.2%}")
    print(f"  Correlation: {corr_a2n:.3f}")
    print(f"  Plausibility Score: {stats_a2n['mean_score']:.3f}")
    print(f"  Valid Rate: {stats_a2n['valid_rate']:.2%}")
    
    # ========== Visualize ==========
    print("\n3. Generating visualizations...")
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    
    for i in range(5):
        # Normal → AFib
        axes[0, i].plot(normal_samples[i, 0].cpu().numpy(), 'b-', alpha=0.7, linewidth=0.8)
        axes[0, i].set_title(f'Original Normal {i+1}', fontsize=10)
        axes[0, i].set_ylim([-2, 2])
        axes[0, i].grid(True, alpha=0.3)
        
        axes[1, i].plot(cf_n2a[i, 0].cpu().numpy(), 'r-', alpha=0.7, linewidth=0.8)
        plaus_score = val_results_n2a[i]['plausibility_score']
        axes[1, i].set_title(f'CF (N→A) - Plaus: {plaus_score:.2f}', fontsize=10)
        axes[1, i].set_ylim([-2, 2])
        axes[1, i].grid(True, alpha=0.3)
        
        # AFib → Normal
        axes[2, i].plot(afib_samples[i, 0].cpu().numpy(), 'b-', alpha=0.7, linewidth=0.8)
        axes[2, i].set_title(f'Original AFib {i+1}', fontsize=10)
        axes[2, i].set_ylim([-2, 2])
        axes[2, i].grid(True, alpha=0.3)
        
        axes[3, i].plot(cf_a2n[i, 0].cpu().numpy(), 'g-', alpha=0.7, linewidth=0.8)
        plaus_score = val_results_a2n[i]['plausibility_score']
        axes[3, i].set_title(f'CF (A→N) - Plaus: {plaus_score:.2f}', fontsize=10)
        axes[3, i].set_ylim([-2, 2])
        axes[3, i].grid(True, alpha=0.3)
    
    plt.suptitle(f'Epoch {epoch} - Stage {stage} Checkpoint Validation\n'
                 f'N→A: Flip={flip_rate_n2a:.1%}, Corr={corr_n2a:.2f}, Plaus={stats_n2a["mean_score"]:.2f} | '
                 f'A→N: Flip={flip_rate_a2n:.1%}, Corr={corr_a2n:.2f}, Plaus={stats_a2n["mean_score"]:.2f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_file = Config.OUTPUT_DIR / f'validation_epoch_{epoch:03d}_stage{stage}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved visualization: {output_file}")
    
    # ========== Save Results ==========
    results_file = Config.OUTPUT_DIR / f'validation_epoch_{epoch:03d}_stage{stage}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Saved results: {results_file}")
    
    return results

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Validate training checkpoint plausibility')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint file to validate')
    parser.add_argument('--latest', action='store_true', help='Validate latest checkpoint')
    parser.add_argument('--all', action='store_true', help='Validate all checkpoints')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Checkpoint Plausibility Validator")
    print("="*70)
    
    if args.all:
        # Validate all checkpoints
        checkpoints = sorted(Config.MODEL_DIR.glob('checkpoint_*.pth'))
        if not checkpoints:
            print("No checkpoints found!")
            return
        
        print(f"\nFound {len(checkpoints)} checkpoints")
        
        all_results = []
        for ckpt in checkpoints:
            print(f"\n{'='*70}")
            results = validate_checkpoint(ckpt)
            all_results.append(results)
        
        # Save summary
        summary_file = Config.OUTPUT_DIR / 'validation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n\nSaved summary: {summary_file}")
        
    elif args.latest:
        # Validate latest checkpoint
        checkpoint_path = get_latest_checkpoint(Config.MODEL_DIR)
        if checkpoint_path is None:
            print("No checkpoints found!")
            return
        
        validate_checkpoint(checkpoint_path)
        
    elif args.checkpoint:
        # Validate specific checkpoint
        checkpoint_path = Config.MODEL_DIR / args.checkpoint
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        validate_checkpoint(checkpoint_path)
        
    else:
        print("Please specify --checkpoint, --latest, or --all")
        parser.print_help()

if __name__ == '__main__':
    main()
