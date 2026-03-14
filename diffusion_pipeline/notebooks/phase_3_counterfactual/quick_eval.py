#!/usr/bin/env python3
"""
Quick Evaluation: Generate and Visualize Counterfactual ECGs

This script loads the latest checkpoint and generates sample counterfactuals
to verify training quality. Runs independently of training process.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
sys.path.append(str(PROJECT_ROOT / 'notebooks/phase_3_counterfactual'))

from shared_models import load_classifier, ClassifierWrapper
from plausibility_validator import PlausibilityValidator
import math
from scipy.signal import savgol_filter

# Load model architectures
exec(open(PROJECT_ROOT / 'notebooks/phase_3_counterfactual/16_enhanced_diffusion_cf.py').read().split('# ============================================================================\\n# Initialize Models')[0])

# ============================================================================
# Configuration
# ============================================================================

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/enhanced_diffusion_cf'
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
OUTPUT_DIR = MODEL_DIR / 'quick_eval'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Generation params
SDEDIT_STRENGTH = 0.6
CFG_SCALE = 3.0
NUM_SAMPLES = 10  # Samples per direction

print("="*70)
print("QUICK EVALUATION: Counterfactual ECG Generation")
print("="*70)

# ============================================================================
# Load Latest Checkpoint
# ============================================================================

checkpoints = sorted(MODEL_DIR.glob('checkpoint_*.pth'), key=lambda p: p.stat().st_mtime)
if not checkpoints:
    print("ERROR: No checkpoints found!")
    sys.exit(1)

checkpoint_path = checkpoints[-1]
print(f"\nLoading checkpoint: {checkpoint_path.name}")

checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
epoch = checkpoint.get('epoch', 0)
stage = checkpoint.get('stage', 1)

print(f"Epoch: {epoch}, Stage: {stage}")

# Initialize models
content_encoder = ContentEncoder(1, 64, 256).to(DEVICE)
style_encoder = StyleEncoder(1, 64, 128, 2).to(DEVICE)
unet = ConditionalUNet(1, 64, 256, 128, 2).to(DEVICE)

content_encoder.load_state_dict(checkpoint['content_encoder'])
style_encoder.load_state_dict(checkpoint['style_encoder'])
unet.load_state_dict(checkpoint['unet'])

content_encoder.eval()
style_encoder.eval()
unet.eval()

scheduler = DDIMScheduler(1000, DEVICE)
classifier = load_classifier(DEVICE)
classifier_wrapper = ClassifierWrapper(classifier).to(DEVICE)
validator = PlausibilityValidator()

print("✓ Models loaded successfully")

# ============================================================================
# Load Test Data
# ============================================================================

test_data = np.load(DATA_DIR / 'test_data.npz')
test_signals = torch.tensor(test_data['X'], dtype=torch.float32)
test_labels = torch.tensor(test_data['y'], dtype=torch.long)

normal_idx = (test_labels == 0).nonzero(as_tuple=True)[0]
afib_idx = (test_labels == 1).nonzero(as_tuple=True)[0]

# Select samples
normal_samples = test_signals[normal_idx[:NUM_SAMPLES]].to(DEVICE)
afib_samples = test_signals[afib_idx[:NUM_SAMPLES]].to(DEVICE)

print(f"\nSelected {NUM_SAMPLES} Normal and {NUM_SAMPLES} AFib samples")

# ============================================================================
# Generate Counterfactuals
# ============================================================================

print("\n" + "="*70)
print("GENERATING COUNTERFACTUALS")
print("="*70)

with torch.no_grad():
    # Normal → AFib
    print("\n1. Normal → AFib...")
    normal_content, _, _ = content_encoder(normal_samples)
    normal_style, _ = style_encoder(normal_samples)
    target_afib = torch.ones(NUM_SAMPLES, dtype=torch.long, device=DEVICE)
    
    cf_n2a = scheduler.sdedit_sample(
        unet, normal_content, normal_style, target_afib,
        normal_samples, strength=SDEDIT_STRENGTH,
        num_steps=50, cfg_scale=CFG_SCALE
    )
    cf_n2a = reduce_noise(cf_n2a, 11, 3)
    
    # Evaluate
    cf_logits_n2a = classifier_wrapper(cf_n2a)
    cf_pred_n2a = torch.argmax(cf_logits_n2a, dim=1)
    flip_rate_n2a = (cf_pred_n2a == 1).float().mean().item()
    corr_n2a = np.mean([np.corrcoef(normal_samples[i, 0].cpu(), cf_n2a[i, 0].cpu())[0, 1] 
                        for i in range(NUM_SAMPLES)])
    
    # Plausibility
    plaus_n2a = []
    for i in range(NUM_SAMPLES):
        result = validator.validate(cf_n2a[i].cpu().numpy(), 
                                    original_ecg=normal_samples[i].cpu().numpy(),
                                    target_class=1, original_class=0)
        plaus_n2a.append(result['plausibility_score'])
    
    print(f"  Flip Rate: {flip_rate_n2a:.1%}")
    print(f"  Correlation: {corr_n2a:.3f}")
    print(f"  Plausibility: {np.mean(plaus_n2a):.3f}")
    
    # AFib → Normal
    print("\n2. AFib → Normal...")
    afib_content, _, _ = content_encoder(afib_samples)
    afib_style, _ = style_encoder(afib_samples)
    target_normal = torch.zeros(NUM_SAMPLES, dtype=torch.long, device=DEVICE)
    
    cf_a2n = scheduler.sdedit_sample(
        unet, afib_content, afib_style, target_normal,
        afib_samples, strength=SDEDIT_STRENGTH,
        num_steps=50, cfg_scale=CFG_SCALE
    )
    cf_a2n = reduce_noise(cf_a2n, 11, 3)
    
    # Evaluate
    cf_logits_a2n = classifier_wrapper(cf_a2n)
    cf_pred_a2n = torch.argmax(cf_logits_a2n, dim=1)
    flip_rate_a2n = (cf_pred_a2n == 0).float().mean().item()
    corr_a2n = np.mean([np.corrcoef(afib_samples[i, 0].cpu(), cf_a2n[i, 0].cpu())[0, 1] 
                        for i in range(NUM_SAMPLES)])
    
    # Plausibility
    plaus_a2n = []
    for i in range(NUM_SAMPLES):
        result = validator.validate(cf_a2n[i].cpu().numpy(),
                                    original_ecg=afib_samples[i].cpu().numpy(),
                                    target_class=0, original_class=1)
        plaus_a2n.append(result['plausibility_score'])
    
    print(f"  Flip Rate: {flip_rate_a2n:.1%}")
    print(f"  Correlation: {corr_a2n:.3f}")
    print(f"  Plausibility: {np.mean(plaus_a2n):.3f}")

# ============================================================================
# Visualizations
# ============================================================================

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Figure 1: Sample Comparisons
fig, axes = plt.subplots(4, 5, figsize=(20, 12))

for i in range(5):
    # Normal → AFib
    axes[0, i].plot(normal_samples[i, 0].cpu().numpy(), 'b-', linewidth=0.8)
    axes[0, i].set_title(f'Original Normal {i+1}', fontsize=10)
    axes[0, i].set_ylim([-2.5, 2.5])
    axes[0, i].grid(True, alpha=0.3)
    axes[0, i].set_ylabel('Amplitude', fontsize=8)
    
    axes[1, i].plot(cf_n2a[i, 0].cpu().numpy(), 'r-', linewidth=0.8)
    pred_label = 'AFib' if cf_pred_n2a[i] == 1 else 'Normal'
    corr_val = np.corrcoef(normal_samples[i, 0].cpu(), cf_n2a[i, 0].cpu())[0, 1]
    axes[1, i].set_title(f'CF→{pred_label} (r={corr_val:.2f}, p={plaus_n2a[i]:.2f})', fontsize=9)
    axes[1, i].set_ylim([-2.5, 2.5])
    axes[1, i].grid(True, alpha=0.3)
    axes[1, i].set_ylabel('Amplitude', fontsize=8)
    
    # AFib → Normal
    axes[2, i].plot(afib_samples[i, 0].cpu().numpy(), 'b-', linewidth=0.8)
    axes[2, i].set_title(f'Original AFib {i+1}', fontsize=10)
    axes[2, i].set_ylim([-2.5, 2.5])
    axes[2, i].grid(True, alpha=0.3)
    axes[2, i].set_ylabel('Amplitude', fontsize=8)
    
    axes[3, i].plot(cf_a2n[i, 0].cpu().numpy(), 'g-', linewidth=0.8)
    pred_label = 'Normal' if cf_pred_a2n[i] == 0 else 'AFib'
    corr_val = np.corrcoef(afib_samples[i, 0].cpu(), cf_a2n[i, 0].cpu())[0, 1]
    axes[3, i].set_title(f'CF→{pred_label} (r={corr_val:.2f}, p={plaus_a2n[i]:.2f})', fontsize=9)
    axes[3, i].set_ylim([-2.5, 2.5])
    axes[3, i].grid(True, alpha=0.3)
    axes[3, i].set_ylabel('Amplitude', fontsize=8)
    axes[3, i].set_xlabel('Time (samples)', fontsize=8)

plt.suptitle(f'Checkpoint Evaluation - Epoch {epoch} (Stage {stage})\\n'
             f'N→A: Flip={flip_rate_n2a:.1%}, Corr={corr_n2a:.2f}, Plaus={np.mean(plaus_n2a):.2f} | '
             f'A→N: Flip={flip_rate_a2n:.1%}, Corr={corr_a2n:.2f}, Plaus={np.mean(plaus_a2n):.2f}',
             fontsize=14, fontweight='bold')
plt.tight_layout()

output_file = OUTPUT_DIR / f'eval_epoch{epoch:03d}_stage{stage}_samples.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_file}")
plt.close()

# Figure 2: Detailed Overlay
fig, axes = plt.subplots(2, 3, figsize=(18, 8))

for i in range(3):
    # Normal → AFib overlay
    axes[0, i].plot(normal_samples[i, 0].cpu().numpy(), 'b-', alpha=0.6, label='Original', linewidth=1.2)
    axes[0, i].plot(cf_n2a[i, 0].cpu().numpy(), 'r-', alpha=0.8, label='Counterfactual', linewidth=1.2)
    axes[0, i].set_title(f'Normal→AFib #{i+1}\\nFlip: {cf_pred_n2a[i]==1}, Plaus: {plaus_n2a[i]:.2f}', fontsize=11)
    axes[0, i].set_ylim([-2.5, 2.5])
    axes[0, i].grid(True, alpha=0.3)
    axes[0, i].legend(fontsize=9)
    axes[0, i].set_ylabel('Amplitude')
    
    # AFib → Normal overlay
    axes[1, i].plot(afib_samples[i, 0].cpu().numpy(), 'b-', alpha=0.6, label='Original', linewidth=1.2)
    axes[1, i].plot(cf_a2n[i, 0].cpu().numpy(), 'g-', alpha=0.8, label='Counterfactual', linewidth=1.2)
    axes[1, i].set_title(f'AFib→Normal #{i+1}\\nFlip: {cf_pred_a2n[i]==0}, Plaus: {plaus_a2n[i]:.2f}', fontsize=11)
    axes[1, i].set_ylim([-2.5, 2.5])
    axes[1, i].grid(True, alpha=0.3)
    axes[1, i].legend(fontsize=9)
    axes[1, i].set_ylabel('Amplitude')
    axes[1, i].set_xlabel('Time (samples)')

plt.suptitle(f'Detailed Counterfactual Comparison - Epoch {epoch} Stage {stage}', fontsize=14, fontweight='bold')
plt.tight_layout()

output_file = OUTPUT_DIR / f'eval_epoch{epoch:03d}_stage{stage}_overlay.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# ============================================================================
# Summary Report
# ============================================================================

print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

print(f"\nCheckpoint: Epoch {epoch}, Stage {stage}")
print(f"\nNormal → AFib:")
print(f"  Flip Rate:      {flip_rate_n2a:.1%} ({int(flip_rate_n2a*NUM_SAMPLES)}/{NUM_SAMPLES} flipped)")
print(f"  Correlation:    {corr_n2a:.3f}")
print(f"  Plausibility:   {np.mean(plaus_n2a):.3f} ± {np.std(plaus_n2a):.3f}")
print(f"\nAFib → Normal:")
print(f"  Flip Rate:      {flip_rate_a2n:.1%} ({int(flip_rate_a2n*NUM_SAMPLES)}/{NUM_SAMPLES} flipped)")
print(f"  Correlation:    {corr_a2n:.3f}")
print(f"  Plausibility:   {np.mean(plaus_a2n):.3f} ± {np.std(plaus_a2n):.3f}")

# Assessment
print(f"\n{'='*70}")
print("ASSESSMENT")
print("="*70)

avg_flip = (flip_rate_n2a + flip_rate_a2n) / 2
avg_corr = (corr_n2a + corr_a2n) / 2
avg_plaus = (np.mean(plaus_n2a) + np.mean(plaus_a2n)) / 2

if stage == 1:
    print("\n📊 Stage 1 (Reconstruction):")
    if avg_corr > 0.8:
        print(f"  ✓ Excellent correlation ({avg_corr:.2f}) - reconstruction quality is good")
    else:
        print(f"  ⚠ Low correlation ({avg_corr:.2f}) - needs more training")
    print(f"  ℹ Flip rates not critical in Stage 1 (current: {avg_flip:.1%})")
else:
    print("\n📊 Stage 2 (Counterfactual Fine-tuning):")
    if avg_flip > 0.8:
        print(f"  ✓ Excellent flip rate ({avg_flip:.1%})")
    elif avg_flip > 0.65:
        print(f"  ⚙ Good flip rate ({avg_flip:.1%}) - improving")
    else:
        print(f"  ⚠ Low flip rate ({avg_flip:.1%}) - needs more training")
    
    if avg_corr > 0.65:
        print(f"  ✓ Good similarity ({avg_corr:.2f})")
    elif avg_corr > 0.5:
        print(f"  ⚙ Moderate similarity ({avg_corr:.2f})")
    else:
        print(f"  ⚠ Low similarity ({avg_corr:.2f}) - counterfactuals too different")
    
    if avg_plaus > 0.7:
        print(f"  ✓ Clinically plausible ({avg_plaus:.2f})")
    elif avg_plaus > 0.5:
        print(f"  ⚙ Moderate plausibility ({avg_plaus:.2f})")
    else:
        print(f"  ⚠ Low plausibility ({avg_plaus:.2f}) - signals may not be clinically valid")

print(f"\n{'='*70}")
print(f"Visualizations saved to: {OUTPUT_DIR}")
print("="*70)
