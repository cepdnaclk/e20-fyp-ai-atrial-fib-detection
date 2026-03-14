"""
Phase 3: Gradient-Based Minimal Perturbation Counterfactual V6
==============================================================

Key insight: Diffusion-based methods either:
1. Preserve signal too well → can't flip classifier
2. Change too much → lose similarity

New approach: DIRECT GRADIENT OPTIMIZATION
- Start from original signal
- Use gradient descent to find MINIMAL perturbation that flips classifier
- Add strong regularization to preserve signal similarity
- Smooth perturbations to ensure realistic output (no high-frequency artifacts)

This is similar to adversarial examples but with:
1. Smoothness constraint (no high-freq noise)
2. Similarity constraint (perturbation must be small)
3. Validity constraint (output must be valid ECG)
"""

import os
import sys
import subprocess

def get_free_gpu():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        gpu_info = [(int(l.split(',')[0]), int(l.split(',')[1])) for l in lines if l.strip()]
        if gpu_info:
            return str(max(gpu_info, key=lambda x: x[1])[0])
    except:
        pass
    return '0'

os.environ['CUDA_VISIBLE_DEVICES'] = get_free_gpu()

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from scipy import signal as scipy_signal
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, '/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')
from model_architecture import AFibResLSTM, ModelConfig

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/gradient_perturbation_v6'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

REAL_MEAN, REAL_STD = -0.00396, 0.14716
FS = 500


# ============================================================================
# Classifier
# ============================================================================

def load_classifier():
    classifier_path = PROJECT_ROOT / 'models/afib_reslstm_final.pth'
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(DEVICE)
    checkpoint = torch.load(classifier_path, map_location=DEVICE)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    return classifier

class ClassifierWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
    
    def forward(self, x):
        self.model.train()  # Required for LSTM backward
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True) + 1e-8
        x_norm = (x - mean) / std
        logits, _ = self.model(x_norm)
        return logits


# ============================================================================
# Smooth Perturbation Generator
# ============================================================================

class SmoothPerturbationNet(nn.Module):
    """
    Generates smooth perturbations that are band-limited.
    This ensures the perturbation doesn't add high-frequency noise.
    """
    def __init__(self, signal_length=2500, low_rank=64):
        super().__init__()
        # Use low-rank factorization to generate smooth perturbations
        self.signal_length = signal_length
        
        # Generate smooth basis functions (sine/cosine at low frequencies)
        basis = []
        for k in range(1, low_rank + 1):
            freq = k  # Low frequency components
            t = torch.linspace(0, 2 * np.pi * freq, signal_length)
            basis.append(torch.sin(t))
            basis.append(torch.cos(t))
        
        self.register_buffer('basis', torch.stack(basis))  # (2*low_rank, L)
        
        # Learnable coefficients for each basis function
        self.coeffs = nn.Parameter(torch.zeros(2 * low_rank))
    
    def forward(self):
        # Weighted sum of smooth basis functions
        perturbation = torch.einsum('k,kl->l', self.coeffs, self.basis)
        return perturbation.unsqueeze(0).unsqueeze(0)  # (1, 1, L)


# ============================================================================
# Gradient-Based Counterfactual Generator
# ============================================================================

class GradientCounterfactualGenerator:
    def __init__(self, classifier, device='cuda'):
        self.classifier = classifier
        self.device = device
    
    def generate(self, original_signal, target_class, 
                 max_iterations=500, 
                 lr=0.01,
                 flip_weight=1.0,
                 similarity_weight=10.0,
                 smoothness_weight=0.1,
                 early_stop_prob=0.85,
                 min_similarity=0.7):
        """
        Generate counterfactual by gradient optimization.
        
        Loss = flip_loss + similarity_weight * similarity_loss + smoothness_weight * smoothness_loss
        
        - flip_loss: cross-entropy toward target class
        - similarity_loss: L2 distance to original
        - smoothness_loss: penalize high-frequency perturbations
        """
        original = original_signal.clone().to(self.device)
        
        # Initialize perturbation as zeros (start from original)
        perturbation = torch.zeros_like(original, requires_grad=True)
        
        optimizer = torch.optim.Adam([perturbation], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iterations)
        
        best_cf = None
        best_score = -float('inf')
        
        target = torch.tensor([target_class], device=self.device)
        
        for i in range(max_iterations):
            optimizer.zero_grad()
            
            # Counterfactual = original + perturbation
            cf = original + perturbation
            
            # Clamp to valid range
            cf = torch.clamp(cf, -3, 3)
            
            # Classification loss (toward target class)
            self.classifier.train()  # LSTM needs train mode
            logits = self.classifier(cf)
            flip_loss = F.cross_entropy(logits, target)
            
            # Similarity loss (stay close to original)
            similarity_loss = F.mse_loss(cf, original)
            
            # Smoothness loss (penalize high-frequency changes)
            # Use total variation as smoothness measure
            diff = perturbation[:, :, 1:] - perturbation[:, :, :-1]
            smoothness_loss = diff.abs().mean()
            
            # Total loss
            loss = (flip_weight * flip_loss + 
                    similarity_weight * similarity_loss + 
                    smoothness_weight * smoothness_loss)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Evaluate current solution
            with torch.no_grad():
                prob = F.softmax(self.classifier(cf), dim=1)[0, target_class].item()
                corr, _ = pearsonr(original[0, 0].cpu().numpy(), cf[0, 0].detach().cpu().numpy())
            
            # Score: prioritize flip while maintaining similarity
            if prob > 0.5:  # Flipped
                score = prob + corr
            else:
                score = prob * 0.3 + corr * 0.7  # Prioritize similarity until flipped
            
            if score > best_score:
                best_score = score
                best_cf = cf.detach().clone()
            
            # Early stopping
            if prob > early_stop_prob and corr > min_similarity:
                break
        
        return best_cf
    
    def generate_adaptive(self, original_signal, target_class, 
                          max_retries=5, target_similarity=0.8):
        """
        Adaptive generation: adjust weights based on results.
        """
        original = original_signal.clone().to(self.device)
        
        # Start with high similarity weight
        sim_weight = 50.0
        
        for retry in range(max_retries):
            cf = self.generate(
                original, target_class,
                max_iterations=300,
                lr=0.01,
                flip_weight=1.0,
                similarity_weight=sim_weight,
                smoothness_weight=0.1,
                early_stop_prob=0.8,
                min_similarity=0.7
            )
            
            with torch.no_grad():
                prob = F.softmax(self.classifier(cf), dim=1)[0, target_class].item()
            
            corr, _ = pearsonr(original[0, 0].cpu().numpy(), cf[0, 0].cpu().numpy())
            
            if prob > 0.5 and corr > target_similarity:
                return cf  # Success!
            
            if prob < 0.5:
                # Not flipped - reduce similarity weight
                sim_weight *= 0.5
            else:
                # Flipped but low similarity - increase similarity weight
                sim_weight *= 1.5
        
        return cf  # Return best effort


# ============================================================================
# Analysis functions
# ============================================================================

def detect_r_peaks(signal, fs=500):
    signal = signal.flatten()
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    peaks, _ = scipy_signal.find_peaks(signal_norm, height=0.3, distance=int(0.3*fs), prominence=0.2)
    return peaks

def compute_rr_features(signal, fs=500):
    peaks = detect_r_peaks(signal, fs)
    if len(peaks) < 2:
        return {'rr_cv': 0, 'hr': 0, 'num_beats': 0}
    rr = np.diff(peaks) / fs * 1000
    return {
        'rr_cv': np.std(rr) / (np.mean(rr) + 1e-8),
        'hr': 60000 / np.mean(rr) if np.mean(rr) > 0 else 0,
        'num_beats': len(peaks)
    }

def check_validity(signal, fs=500):
    signal = signal.flatten()
    amp = np.max(signal) - np.min(signal)
    peaks = detect_r_peaks(signal, fs)
    if len(peaks) >= 2:
        hr = 60 / (np.mean(np.diff(peaks)) / fs)
    else:
        hr = 0
    return (0.01 < amp < 10) and (len(peaks) >= 2) and (30 < hr < 200)


def create_visualization(orig, cf, perturbation, orig_class, target_class, 
                          orig_prob, cf_prob, idx, save_path):
    """Create detailed comparison with perturbation analysis."""
    orig_np = orig.cpu().numpy().flatten()
    cf_np = cf.cpu().numpy().flatten()
    pert_np = perturbation.cpu().numpy().flatten()
    
    # Convert to mV
    orig_mv = orig_np * REAL_STD + REAL_MEAN
    cf_mv = cf_np * REAL_STD + REAL_MEAN
    pert_mv = pert_np * REAL_STD  # Perturbation in mV
    
    time = np.arange(len(orig_np)) / FS
    
    corr, _ = pearsonr(orig_np, cf_np)
    orig_feat = compute_rr_features(orig_np, FS)
    cf_feat = compute_rr_features(cf_np, FS)
    
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(5, 4, figure=fig, height_ratios=[1.5, 1, 1, 1, 0.8])
    
    class_names = ['Normal', 'AFib']
    
    # Row 1: Full signal overlay
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.plot(time, orig_mv, 'b-', lw=1, alpha=0.8, label=f'Original ({class_names[orig_class]})')
    ax_full.plot(time, cf_mv, 'r-', lw=1, alpha=0.7, label=f'CF (→{class_names[target_class]})')
    ax_full.legend(loc='upper right', fontsize=11)
    ax_full.set_xlabel('Time (s)')
    ax_full.set_ylabel('Amplitude (mV)')
    ax_full.set_title(f'Sample {idx} | Correlation: {corr:.4f} | P({class_names[target_class]}): {orig_prob:.3f} → {cf_prob:.3f}', 
                      fontsize=14, fontweight='bold')
    ax_full.grid(True, alpha=0.3)
    
    # Row 2: Zoomed segments
    for i, (s, e) in enumerate([(0, 500), (1000, 1500), (2000, 2500)]):
        ax = fig.add_subplot(gs[1, i])
        ax.plot(time[s:e], orig_mv[s:e], 'b-', lw=1.5, label='Orig')
        ax.plot(time[s:e], cf_mv[s:e], 'r-', lw=1.5, alpha=0.8, label='CF')
        ax.set_title(f'{time[s]:.1f}-{time[e-1]:.1f}s')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Perturbation signal
    ax_pert = fig.add_subplot(gs[1, 3])
    ax_pert.plot(time, pert_mv, 'purple', lw=0.8)
    ax_pert.fill_between(time, 0, pert_mv, alpha=0.3, color='purple')
    ax_pert.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax_pert.set_title(f'Perturbation | Max: {np.max(np.abs(pert_mv)):.4f} mV')
    ax_pert.set_xlabel('Time (s)')
    ax_pert.grid(True, alpha=0.3)
    
    # Row 3: Perturbation spectrum
    ax_spec = fig.add_subplot(gs[2, 0:2])
    f, psd = scipy_signal.welch(pert_np, fs=FS, nperseg=256)
    ax_spec.semilogy(f[:100], psd[:100], 'purple', lw=1.5)
    ax_spec.set_xlabel('Frequency (Hz)')
    ax_spec.set_ylabel('PSD')
    ax_spec.set_title('Perturbation Spectrum (should be low-frequency)')
    ax_spec.grid(True, alpha=0.3)
    
    # Signal spectra comparison
    ax_orig_spec = fig.add_subplot(gs[2, 2:])
    f_o, psd_o = scipy_signal.welch(orig_np, fs=FS, nperseg=256)
    f_c, psd_c = scipy_signal.welch(cf_np, fs=FS, nperseg=256)
    ax_orig_spec.semilogy(f_o[:50], psd_o[:50], 'b-', lw=1.5, label='Orig')
    ax_orig_spec.semilogy(f_c[:50], psd_c[:50], 'r-', lw=1.5, label='CF')
    ax_orig_spec.set_xlabel('Frequency (Hz)')
    ax_orig_spec.set_ylabel('PSD')
    ax_orig_spec.set_title('Signal Spectrum Comparison')
    ax_orig_spec.legend()
    ax_orig_spec.grid(True, alpha=0.3)
    
    # Row 4: R-R analysis
    orig_peaks = detect_r_peaks(orig_np, FS)
    cf_peaks = detect_r_peaks(cf_np, FS)
    
    ax_rr = fig.add_subplot(gs[3, 0:2])
    if len(orig_peaks) > 1:
        orig_rr = np.diff(orig_peaks) / FS * 1000
        ax_rr.bar(np.arange(len(orig_rr)) - 0.2, orig_rr, 0.4, color='blue', alpha=0.7, label='Orig')
    if len(cf_peaks) > 1:
        cf_rr = np.diff(cf_peaks) / FS * 1000
        ax_rr.bar(np.arange(len(cf_rr)) + 0.2, cf_rr, 0.4, color='red', alpha=0.7, label='CF')
    ax_rr.set_xlabel('Beat #')
    ax_rr.set_ylabel('RR (ms)')
    ax_rr.set_title(f'RR Intervals | CV: {orig_feat["rr_cv"]:.3f} → {cf_feat["rr_cv"]:.3f}')
    ax_rr.legend()
    ax_rr.grid(True, alpha=0.3)
    
    # Poincare
    ax_poin = fig.add_subplot(gs[3, 2])
    if len(orig_peaks) > 2:
        orig_rr = np.diff(orig_peaks) / FS * 1000
        ax_poin.scatter(orig_rr[:-1], orig_rr[1:], c='blue', alpha=0.6, s=60, label='Orig')
    if len(cf_peaks) > 2:
        cf_rr = np.diff(cf_peaks) / FS * 1000
        ax_poin.scatter(cf_rr[:-1], cf_rr[1:], c='red', alpha=0.6, s=60, label='CF')
    ax_poin.plot([200, 1500], [200, 1500], 'k--', alpha=0.3)
    ax_poin.set_xlabel('RR_n (ms)')
    ax_poin.set_ylabel('RR_{n+1} (ms)')
    ax_poin.set_title('Poincaré')
    ax_poin.legend()
    ax_poin.grid(True, alpha=0.3)
    
    # Probabilities
    ax_prob = fig.add_subplot(gs[3, 3])
    if target_class == 1:
        probs = [1-orig_prob, orig_prob, 1-cf_prob, cf_prob]
    else:
        probs = [orig_prob, 1-orig_prob, cf_prob, 1-cf_prob]
    colors = ['lightblue', 'lightcoral', 'blue', 'red']
    bars = ax_prob.bar(['O:P(N)', 'O:P(AF)', 'C:P(N)', 'C:P(AF)'], probs, color=colors, edgecolor='black')
    ax_prob.axhline(0.5, color='gray', linestyle='--', lw=2)
    ax_prob.set_ylim([0, 1])
    for bar, p in zip(bars, probs):
        ax_prob.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{p:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax_prob.set_title('Classification')
    
    # Row 5: Summary
    ax_summary = fig.add_subplot(gs[4, :])
    ax_summary.axis('off')
    
    rr_cv_change = cf_feat['rr_cv'] - orig_feat['rr_cv']
    expected = 'INCREASE' if target_class == 1 else 'DECREASE'
    actual = 'INCREASED' if rr_cv_change > 0 else 'DECREASED'
    correct_rr = (target_class == 1 and rr_cv_change > 0) or (target_class == 0 and rr_cv_change < 0)
    flipped = cf_prob > 0.5
    valid = check_validity(cf_np, FS)
    
    summary = f"""
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║  CORRELATION: {corr:.4f} {'✓ HIGH (>0.7)' if corr > 0.7 else '✗ LOW (<0.7)':20}  |  CLASSIFIER: {'✓ FLIPPED' if flipped else '✗ NOT FLIPPED':15}  |  VALID: {'✓' if valid else '✗'}                     ║
    ║  RR CV: {orig_feat['rr_cv']:.3f} → {cf_feat['rr_cv']:.3f} ({'+' if rr_cv_change > 0 else ''}{rr_cv_change:.3f})  |  Expected: {expected}  Actual: {actual}  {'✓' if correct_rr else '✗'}                                           ║
    ║  MAX PERTURBATION: {np.max(np.abs(pert_mv)):.4f} mV  |  PERTURBATION L2: {np.sqrt(np.mean(pert_np**2)):.6f}                                                             ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    ax_summary.text(0.02, 0.5, summary, fontfamily='monospace', fontsize=10, va='center', transform=ax_summary.transAxes)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return {
        'corr': corr, 'flipped': flipped, 'valid': valid,
        'orig_rr_cv': orig_feat['rr_cv'], 'cf_rr_cv': cf_feat['rr_cv'],
        'correct_rr': correct_rr, 'orig_prob': orig_prob, 'cf_prob': cf_prob,
        'max_pert': np.max(np.abs(pert_np)), 'pert_l2': np.sqrt(np.mean(pert_np**2))
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*70)
    print("GRADIENT-BASED MINIMAL PERTURBATION COUNTERFACTUAL V6")
    print("="*70)
    print(f"Results: {RESULTS_DIR}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load classifier
    print("\nLoading classifier...")
    raw_classifier = load_classifier()
    classifier = ClassifierWrapper(raw_classifier)
    
    # Load data
    val_data = np.load(DATA_DIR / 'val_data.npz')
    val_signals = torch.tensor(val_data['X'], dtype=torch.float32)
    val_labels = torch.tensor(val_data['y'], dtype=torch.long)
    if val_signals.dim() == 2:
        val_signals = val_signals.unsqueeze(1)
    
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    
    print(f"Data: {len(normal_idx)} Normal, {len(afib_idx)} AFib")
    
    # Test classifier
    with torch.no_grad():
        test = val_signals[:200].to(DEVICE)
        preds = classifier(test).argmax(dim=1)
        acc = (preds == val_labels[:200].to(DEVICE)).float().mean().item() * 100
        print(f"Classifier accuracy: {acc:.1f}%")
    
    generator = GradientCounterfactualGenerator(classifier, DEVICE)
    
    # ========================================================================
    # Hyperparameter search
    # ========================================================================
    print("\n" + "="*70)
    print("HYPERPARAMETER SEARCH")
    print("="*70)
    
    configs = [
        # (lr, flip_weight, similarity_weight, smoothness_weight)
        (0.01, 1.0, 20.0, 0.1),
        (0.02, 1.0, 30.0, 0.1),
        (0.01, 2.0, 50.0, 0.2),
        (0.02, 1.0, 100.0, 0.1),
    ]
    
    best_config = None
    best_score = 0
    
    for lr, flip_w, sim_w, smooth_w in configs:
        flips = 0
        corrs = []
        
        for i in range(3):
            x = val_signals[normal_idx[i]:normal_idx[i]+1].to(DEVICE)
            cf = generator.generate(x, 1, max_iterations=200, lr=lr, 
                                     flip_weight=flip_w, similarity_weight=sim_w, 
                                     smoothness_weight=smooth_w)
            with torch.no_grad():
                p = F.softmax(classifier(cf), dim=1)[0, 1].item()
            if p > 0.5:
                flips += 1
            c, _ = pearsonr(x[0,0].cpu().numpy(), cf[0,0].cpu().numpy())
            corrs.append(c)
        
        for i in range(3):
            x = val_signals[afib_idx[i]:afib_idx[i]+1].to(DEVICE)
            cf = generator.generate(x, 0, max_iterations=200, lr=lr,
                                     flip_weight=flip_w, similarity_weight=sim_w,
                                     smoothness_weight=smooth_w)
            with torch.no_grad():
                p = F.softmax(classifier(cf), dim=1)[0, 0].item()
            if p > 0.5:
                flips += 1
            c, _ = pearsonr(x[0,0].cpu().numpy(), cf[0,0].cpu().numpy())
            corrs.append(c)
        
        flip_rate = flips / 6
        mean_corr = np.mean(corrs)
        score = flip_rate * 0.5 + mean_corr * 0.5
        
        print(f"  lr={lr}, flip={flip_w}, sim={sim_w}, smooth={smooth_w} -> "
              f"flip={flip_rate*100:.0f}%, corr={mean_corr:.3f}, score={score:.3f}")
        
        if score > best_score:
            best_score = score
            best_config = (lr, flip_w, sim_w, smooth_w)
    
    lr, flip_w, sim_w, smooth_w = best_config
    print(f"\nBest: lr={lr}, flip={flip_w}, sim={sim_w}, smooth={smooth_w}")
    
    # ========================================================================
    # Generate counterfactuals
    # ========================================================================
    print("\n" + "="*70)
    print("GENERATING COUNTERFACTUALS")
    print("="*70)
    
    num_samples = 10
    results = {'normal_to_afib': [], 'afib_to_normal': []}
    
    # Normal → AFib
    print("\n--- Normal → AFib ---")
    for i in tqdm(range(num_samples), desc="N→AF"):
        x = val_signals[normal_idx[i]:normal_idx[i]+1].to(DEVICE)
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 1].item()
        
        cf = generator.generate(
            x, 1, max_iterations=500, lr=lr,
            flip_weight=flip_w, similarity_weight=sim_w, smoothness_weight=smooth_w,
            early_stop_prob=0.8, min_similarity=0.75
        )
        
        with torch.no_grad():
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 1].item()
        
        perturbation = cf - x
        save_path = RESULTS_DIR / f'n2a_{i+1:02d}.png'
        result = create_visualization(x[0,0], cf[0,0], perturbation[0,0], 0, 1, 
                                       orig_prob, cf_prob, i+1, save_path)
        results['normal_to_afib'].append(result)
        print(f"  #{i+1}: P(AF) {orig_prob:.3f}→{cf_prob:.3f} | Corr: {result['corr']:.3f}")
    
    # AFib → Normal
    print("\n--- AFib → Normal ---")
    for i in tqdm(range(num_samples), desc="AF→N"):
        x = val_signals[afib_idx[i]:afib_idx[i]+1].to(DEVICE)
        with torch.no_grad():
            orig_prob = F.softmax(classifier(x), dim=1)[0, 0].item()
        
        cf = generator.generate(
            x, 0, max_iterations=500, lr=lr,
            flip_weight=flip_w, similarity_weight=sim_w, smoothness_weight=smooth_w,
            early_stop_prob=0.8, min_similarity=0.75
        )
        
        with torch.no_grad():
            cf_prob = F.softmax(classifier(cf), dim=1)[0, 0].item()
        
        perturbation = cf - x
        save_path = RESULTS_DIR / f'a2n_{i+1:02d}.png'
        result = create_visualization(x[0,0], cf[0,0], perturbation[0,0], 1, 0,
                                       orig_prob, cf_prob, i+1, save_path)
        results['afib_to_normal'].append(result)
        print(f"  #{i+1}: P(N) {orig_prob:.3f}→{cf_prob:.3f} | Corr: {result['corr']:.3f}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    n2a = results['normal_to_afib']
    a2n = results['afib_to_normal']
    
    def summarize(data, name):
        valid = sum(r['valid'] for r in data)
        flipped = sum(r['flipped'] for r in data)
        corr = np.mean([r['corr'] for r in data])
        correct_rr = sum(r['correct_rr'] for r in data)
        print(f"\n{name}:")
        print(f"  Valid: {valid}/{len(data)} ({100*valid/len(data):.0f}%)")
        print(f"  Flipped: {flipped}/{len(data)} ({100*flipped/len(data):.0f}%)")
        print(f"  Correlation: {corr:.4f}")
        print(f"  Correct RR: {correct_rr}/{len(data)} ({100*correct_rr/len(data):.0f}%)")
        return valid, flipped, corr, correct_rr
    
    n2a_valid, n2a_flip, n2a_corr, n2a_rr = summarize(n2a, "Normal → AFib")
    a2n_valid, a2n_flip, a2n_corr, a2n_rr = summarize(a2n, "AFib → Normal")
    
    # Save results
    summary = {
        'config': {'lr': lr, 'flip_weight': flip_w, 'similarity_weight': sim_w, 'smoothness_weight': smooth_w},
        'normal_to_afib': {'valid': n2a_valid/num_samples, 'flip': n2a_flip/num_samples, 'corr': float(n2a_corr), 'rr': n2a_rr/num_samples},
        'afib_to_normal': {'valid': a2n_valid/num_samples, 'flip': a2n_flip/num_samples, 'corr': float(a2n_corr), 'rr': a2n_rr/num_samples}
    }
    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST CASE RESULTS")
    print("="*70)
    
    total = 2 * num_samples
    all_valid = n2a_valid + a2n_valid
    all_flip = n2a_flip + a2n_flip
    mean_corr = (n2a_corr + a2n_corr) / 2
    
    print(f"\n1. REALISTIC (valid): {all_valid}/{total} ({100*all_valid/total:.0f}%) {'✓' if all_valid >= 0.9*total else '✗'}")
    print(f"2. SIMILARITY (corr): {mean_corr:.4f} {'✓' if mean_corr > 0.7 else '✗'}")
    print(f"3. FLIP RATE: {all_flip}/{total} ({100*all_flip/total:.0f}%) {'✓' if all_flip >= 0.8*total else '✗'}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == '__main__':
    main()
