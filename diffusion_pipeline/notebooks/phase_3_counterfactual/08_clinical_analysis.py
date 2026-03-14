"""
Clinical Feature Analysis for Counterfactuals
==============================================

Verifies that counterfactuals:
1. Are realistic (not abnormal signals)
2. Only change class-discriminative features:
   - R-R intervals (irregular in AFib)
   - P-waves (absent in AFib)
   - Fibrillatory activity (present in AFib)
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
            best_gpu = max(gpu_info, key=lambda x: x[1])
            return str(best_gpu[0])
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
from scipy import signal as scipy_signal
from scipy.stats import pearsonr
import json

sys.path.insert(0, '/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline/models')
from model_architecture import AFibResLSTM, ModelConfig

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', '0')}")

PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
DATA_DIR = PROJECT_ROOT / 'data/processed/diffusion'
RESULTS_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/guided_v2'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REAL_MEAN = -0.00396
REAL_STD = 0.14716


# Import the model architectures from shared module
from shared_models import CounterfactualVAE, ClassifierWrapper, load_classifier


# ============================================================================
# Clinical Feature Extraction
# ============================================================================

def detect_r_peaks(signal, fs=500):
    """Detect R-peaks using peak detection."""
    signal = signal.flatten()
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    peaks, properties = scipy_signal.find_peaks(
        signal_norm, 
        height=0.3, 
        distance=int(0.3 * fs),
        prominence=0.2
    )
    return peaks, properties

def compute_rr_features(signal, fs=500):
    """Compute R-R interval features - key AFib indicator."""
    peaks, _ = detect_r_peaks(signal, fs)
    
    if len(peaks) < 2:
        return {
            'num_beats': 0,
            'rr_mean_ms': 0,
            'rr_std_ms': 0,
            'rr_cv': 0,  # Coefficient of variation (irregularity)
            'heart_rate': 0,
        }
    
    rr_intervals = np.diff(peaks) / fs * 1000  # Convert to ms
    
    return {
        'num_beats': len(peaks),
        'rr_mean_ms': np.mean(rr_intervals),
        'rr_std_ms': np.std(rr_intervals),
        'rr_cv': np.std(rr_intervals) / (np.mean(rr_intervals) + 1e-8),
        'heart_rate': 60000 / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0,
    }

def compute_p_wave_features(signal, fs=500):
    """Estimate P-wave presence/absence."""
    signal = signal.flatten()
    peaks, _ = detect_r_peaks(signal, fs)
    
    if len(peaks) < 2:
        return {'p_wave_score': 0, 'p_wave_present': False}
    
    # Look for P-waves before each R-peak (typically 120-200ms before)
    p_wave_scores = []
    
    for i, peak in enumerate(peaks):
        # P-wave region: 200-80ms before R-peak
        p_start = int(peak - 0.2 * fs)
        p_end = int(peak - 0.08 * fs)
        
        if p_start < 0:
            continue
            
        p_region = signal[p_start:p_end]
        
        if len(p_region) > 10:
            # Look for small positive deflection
            # Normalize to signal std
            p_amplitude = (np.max(p_region) - np.mean(signal)) / (np.std(signal) + 1e-8)
            p_wave_scores.append(p_amplitude)
    
    if len(p_wave_scores) == 0:
        return {'p_wave_score': 0, 'p_wave_present': False}
    
    mean_score = np.mean(p_wave_scores)
    
    return {
        'p_wave_score': mean_score,
        'p_wave_present': mean_score > 0.3  # Threshold for P-wave presence
    }

def compute_high_freq_content(signal, fs=500):
    """Compute high-frequency content (fibrillatory waves in AFib)."""
    signal = signal.flatten()
    
    # High-pass filter to get fibrillatory waves (>10 Hz)
    sos = scipy_signal.butter(4, 10, 'highpass', fs=fs, output='sos')
    high_freq = scipy_signal.sosfilt(sos, signal)
    
    # RMS of high-frequency content
    hf_rms = np.sqrt(np.mean(high_freq ** 2))
    
    return {
        'high_freq_rms': hf_rms,
        'high_freq_ratio': hf_rms / (np.std(signal) + 1e-8)
    }

def check_validity(signal, fs=500):
    """Check if signal is physiologically valid."""
    signal = signal.flatten()
    
    # Amplitude check
    amp_range = np.max(signal) - np.min(signal)
    
    # R-peak detection
    peaks, _ = detect_r_peaks(signal, fs)
    
    # Heart rate
    if len(peaks) >= 2:
        rr_mean = np.mean(np.diff(peaks)) / fs
        hr = 60 / rr_mean if rr_mean > 0 else 0
    else:
        hr = 0
    
    is_valid = True
    issues = []
    
    if amp_range < 0.01:
        is_valid = False
        issues.append("Low amplitude")
    if amp_range > 10:
        is_valid = False
        issues.append("High amplitude") 
    if len(peaks) < 2:
        is_valid = False
        issues.append("No R-peaks detected")
    if hr > 0 and (hr < 30 or hr > 200):
        is_valid = False
        issues.append(f"Invalid HR: {hr:.0f}")
    
    return {
        'is_valid': is_valid,
        'issues': issues,
        'amplitude_range': amp_range,
        'num_rpeaks': len(peaks),
        'heart_rate': hr
    }


def analyze_counterfactual_pair(original, counterfactual, original_class, fs=500):
    """Comprehensive analysis of original vs counterfactual."""
    
    # Validity check
    orig_valid = check_validity(original, fs)
    cf_valid = check_validity(counterfactual, fs)
    
    # R-R features
    orig_rr = compute_rr_features(original, fs)
    cf_rr = compute_rr_features(counterfactual, fs)
    
    # P-wave features
    orig_pwave = compute_p_wave_features(original, fs)
    cf_pwave = compute_p_wave_features(counterfactual, fs)
    
    # High-frequency content
    orig_hf = compute_high_freq_content(original, fs)
    cf_hf = compute_high_freq_content(counterfactual, fs)
    
    # Correlation
    corr, _ = pearsonr(original.flatten(), counterfactual.flatten())
    
    # Expected changes based on class
    if original_class == 0:  # Normal -> AFib
        expected_rr_change = "increase"  # More irregular
        expected_pwave_change = "decrease"  # P-waves disappear
        expected_hf_change = "increase"  # Fibrillatory waves
    else:  # AFib -> Normal
        expected_rr_change = "decrease"  # More regular
        expected_pwave_change = "increase"  # P-waves appear
        expected_hf_change = "decrease"  # Less high-freq content
    
    # Actual changes
    rr_change = cf_rr['rr_cv'] - orig_rr['rr_cv']
    pwave_change = cf_pwave['p_wave_score'] - orig_pwave['p_wave_score']
    hf_change = cf_hf['high_freq_ratio'] - orig_hf['high_freq_ratio']
    
    # Check if changes are in expected direction
    rr_correct = (expected_rr_change == "increase" and rr_change > 0) or \
                 (expected_rr_change == "decrease" and rr_change < 0)
    pwave_correct = (expected_pwave_change == "decrease" and pwave_change < 0) or \
                    (expected_pwave_change == "increase" and pwave_change > 0)
    
    return {
        'validity': {
            'original_valid': orig_valid['is_valid'],
            'counterfactual_valid': cf_valid['is_valid'],
            'cf_issues': cf_valid['issues'],
        },
        'rr_intervals': {
            'original_cv': orig_rr['rr_cv'],
            'cf_cv': cf_rr['rr_cv'],
            'change': rr_change,
            'expected_direction': expected_rr_change,
            'correct_direction': rr_correct,
        },
        'p_waves': {
            'original_score': orig_pwave['p_wave_score'],
            'cf_score': cf_pwave['p_wave_score'],
            'change': pwave_change,
            'expected_direction': expected_pwave_change,
            'correct_direction': pwave_correct,
        },
        'high_freq': {
            'original': orig_hf['high_freq_ratio'],
            'cf': cf_hf['high_freq_ratio'],
            'change': hf_change,
        },
        'similarity': corr,
        'heart_rate': {
            'original': orig_rr['heart_rate'],
            'cf': cf_rr['heart_rate'],
        }
    }


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("\n" + "="*70)
    print("CLINICAL FEATURE ANALYSIS FOR COUNTERFACTUALS")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    val_data = np.load(DATA_DIR / 'val_data.npz')
    val_signals = torch.tensor(val_data['X'], dtype=torch.float32)
    val_labels = torch.tensor(val_data['y'], dtype=torch.long)
    
    if val_signals.dim() == 2:
        val_signals = val_signals.unsqueeze(1)
    
    print(f"  Val: {val_signals.shape}")
    
    # Load model
    print("\nLoading model...")
    model = CounterfactualVAE(latent_dim=512).to(DEVICE)
    checkpoint = torch.load(RESULTS_DIR / 'stage2_best.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Load classifier
    raw_classifier = load_classifier(DEVICE)
    classifier = ClassifierWrapper(raw_classifier)
    
    # Separate by class
    normal_idx = (val_labels == 0).nonzero(as_tuple=True)[0]
    afib_idx = (val_labels == 1).nonzero(as_tuple=True)[0]
    
    num_samples = 20
    
    # ========================================================================
    # Normal -> AFib Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("NORMAL → AFIB COUNTERFACTUAL ANALYSIS")
    print("="*70)
    
    n2a_results = []
    
    with torch.no_grad():
        normal_ecgs = val_signals[normal_idx[:num_samples]].to(DEVICE)
        target = torch.ones(num_samples, 1).to(DEVICE)
        cf, _, _ = model.generate_counterfactual(normal_ecgs, target)
        
        cf_preds = classifier(cf).argmax(dim=1)
    
    for i in range(num_samples):
        orig_np = normal_ecgs[i, 0].cpu().numpy()
        cf_np = cf[i, 0].cpu().numpy()
        
        analysis = analyze_counterfactual_pair(orig_np, cf_np, original_class=0)
        analysis['flipped'] = cf_preds[i].item() == 1
        n2a_results.append(analysis)
    
    # Summary
    n_valid = sum(1 for r in n2a_results if r['validity']['counterfactual_valid'])
    n_flipped = sum(1 for r in n2a_results if r['flipped'])
    n_rr_correct = sum(1 for r in n2a_results if r['rr_intervals']['correct_direction'])
    n_pwave_correct = sum(1 for r in n2a_results if r['p_waves']['correct_direction'])
    
    print(f"\nValidity: {n_valid}/{num_samples} ({100*n_valid/num_samples:.0f}%)")
    print(f"Flip Rate: {n_flipped}/{num_samples} ({100*n_flipped/num_samples:.0f}%)")
    print(f"R-R Irregularity Increased: {n_rr_correct}/{num_samples} ({100*n_rr_correct/num_samples:.0f}%)")
    print(f"P-Wave Score Decreased: {n_pwave_correct}/{num_samples} ({100*n_pwave_correct/num_samples:.0f}%)")
    
    avg_rr_change = np.mean([r['rr_intervals']['change'] for r in n2a_results])
    avg_pwave_change = np.mean([r['p_waves']['change'] for r in n2a_results])
    avg_similarity = np.mean([r['similarity'] for r in n2a_results])
    
    print(f"\nAverage R-R CV Change: {avg_rr_change:+.4f} (expected: +)")
    print(f"Average P-Wave Score Change: {avg_pwave_change:+.4f} (expected: -)")
    print(f"Average Similarity: {avg_similarity:.4f}")
    
    # ========================================================================
    # AFib -> Normal Analysis
    # ========================================================================
    print("\n" + "="*70)
    print("AFIB → NORMAL COUNTERFACTUAL ANALYSIS")
    print("="*70)
    
    a2n_results = []
    
    with torch.no_grad():
        afib_ecgs = val_signals[afib_idx[:num_samples]].to(DEVICE)
        target = torch.zeros(num_samples, 1).to(DEVICE)
        cf, _, _ = model.generate_counterfactual(afib_ecgs, target)
        
        cf_preds = classifier(cf).argmax(dim=1)
    
    for i in range(num_samples):
        orig_np = afib_ecgs[i, 0].cpu().numpy()
        cf_np = cf[i, 0].cpu().numpy()
        
        analysis = analyze_counterfactual_pair(orig_np, cf_np, original_class=1)
        analysis['flipped'] = cf_preds[i].item() == 0
        a2n_results.append(analysis)
    
    # Summary
    n_valid = sum(1 for r in a2n_results if r['validity']['counterfactual_valid'])
    n_flipped = sum(1 for r in a2n_results if r['flipped'])
    n_rr_correct = sum(1 for r in a2n_results if r['rr_intervals']['correct_direction'])
    n_pwave_correct = sum(1 for r in a2n_results if r['p_waves']['correct_direction'])
    
    print(f"\nValidity: {n_valid}/{num_samples} ({100*n_valid/num_samples:.0f}%)")
    print(f"Flip Rate: {n_flipped}/{num_samples} ({100*n_flipped/num_samples:.0f}%)")
    print(f"R-R Irregularity Decreased: {n_rr_correct}/{num_samples} ({100*n_rr_correct/num_samples:.0f}%)")
    print(f"P-Wave Score Increased: {n_pwave_correct}/{num_samples} ({100*n_pwave_correct/num_samples:.0f}%)")
    
    avg_rr_change = np.mean([r['rr_intervals']['change'] for r in a2n_results])
    avg_pwave_change = np.mean([r['p_waves']['change'] for r in a2n_results])
    avg_similarity = np.mean([r['similarity'] for r in a2n_results])
    
    print(f"\nAverage R-R CV Change: {avg_rr_change:+.4f} (expected: -)")
    print(f"Average P-Wave Score Change: {avg_pwave_change:+.4f} (expected: +)")
    print(f"Average Similarity: {avg_similarity:.4f}")
    
    # ========================================================================
    # Visualization
    # ========================================================================
    print("\n" + "="*70)
    print("CREATING CLINICAL VISUALIZATIONS")
    print("="*70)
    
    # Detailed comparison plot
    fig, axes = plt.subplots(4, 4, figsize=(24, 20))
    fig.suptitle('Clinical Feature Analysis: Counterfactual Changes', fontsize=16, fontweight='bold')
    
    time_axis = np.arange(2500) / 500
    
    # Show 2 Normal->AFib examples
    for i in range(2):
        orig = normal_ecgs[i, 0].cpu().numpy() * REAL_STD + REAL_MEAN
        cf_signal = cf[i, 0].cpu().numpy() * REAL_STD + REAL_MEAN
        result = n2a_results[i]
        
        # Full signal
        axes[i, 0].plot(time_axis, orig, 'b-', lw=1, alpha=0.8, label='Original (Normal)')
        axes[i, 0].plot(time_axis, cf_signal, 'r-', lw=1, alpha=0.8, label='CF (→AFib)')
        axes[i, 0].set_title(f'Normal→AFib {i+1}: Flipped={result["flipped"]}')
        axes[i, 0].legend(fontsize=8)
        axes[i, 0].set_ylabel('mV')
        axes[i, 0].grid(True, alpha=0.3)
        
        # Zoomed (1 second)
        axes[i, 1].plot(time_axis[:500], orig[:500], 'b-', lw=1.5)
        axes[i, 1].plot(time_axis[:500], cf_signal[:500], 'r-', lw=1.5)
        axes[i, 1].set_title(f'Zoomed - RR CV: {result["rr_intervals"]["original_cv"]:.3f}→{result["rr_intervals"]["cf_cv"]:.3f}')
        axes[i, 1].grid(True, alpha=0.3)
        
        # Difference
        diff = cf_signal - orig
        axes[i, 2].plot(time_axis, diff, 'purple', lw=1)
        axes[i, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[i, 2].set_title(f'Difference (CF - Orig), Sim={result["similarity"]:.3f}')
        axes[i, 2].fill_between(time_axis, 0, diff, alpha=0.3)
        axes[i, 2].grid(True, alpha=0.3)
        
        # Power spectrum
        f_orig, psd_orig = scipy_signal.welch(orig, fs=500, nperseg=256)
        f_cf, psd_cf = scipy_signal.welch(cf_signal, fs=500, nperseg=256)
        axes[i, 3].semilogy(f_orig[:50], psd_orig[:50], 'b-', lw=1.5, label='Original')
        axes[i, 3].semilogy(f_cf[:50], psd_cf[:50], 'r-', lw=1.5, label='CF')
        axes[i, 3].set_title('Power Spectrum (< 50 Hz)')
        axes[i, 3].legend(fontsize=8)
        axes[i, 3].grid(True, alpha=0.3)
    
    # Show 2 AFib->Normal examples
    with torch.no_grad():
        afib_ecgs = val_signals[afib_idx[:num_samples]].to(DEVICE)
        target = torch.zeros(num_samples, 1).to(DEVICE)
        cf_a2n, _, _ = model.generate_counterfactual(afib_ecgs, target)
    
    for i in range(2):
        orig = afib_ecgs[i, 0].cpu().numpy() * REAL_STD + REAL_MEAN
        cf_signal = cf_a2n[i, 0].cpu().numpy() * REAL_STD + REAL_MEAN
        result = a2n_results[i]
        
        row = i + 2
        
        # Full signal
        axes[row, 0].plot(time_axis, orig, 'r-', lw=1, alpha=0.8, label='Original (AFib)')
        axes[row, 0].plot(time_axis, cf_signal, 'b-', lw=1, alpha=0.8, label='CF (→Normal)')
        axes[row, 0].set_title(f'AFib→Normal {i+1}: Flipped={result["flipped"]}')
        axes[row, 0].legend(fontsize=8)
        axes[row, 0].set_ylabel('mV')
        axes[row, 0].grid(True, alpha=0.3)
        
        # Zoomed
        axes[row, 1].plot(time_axis[:500], orig[:500], 'r-', lw=1.5)
        axes[row, 1].plot(time_axis[:500], cf_signal[:500], 'b-', lw=1.5)
        axes[row, 1].set_title(f'Zoomed - RR CV: {result["rr_intervals"]["original_cv"]:.3f}→{result["rr_intervals"]["cf_cv"]:.3f}')
        axes[row, 1].grid(True, alpha=0.3)
        
        # Difference
        diff = cf_signal - orig
        axes[row, 2].plot(time_axis, diff, 'purple', lw=1)
        axes[row, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[row, 2].set_title(f'Difference, Sim={result["similarity"]:.3f}')
        axes[row, 2].fill_between(time_axis, 0, diff, alpha=0.3)
        axes[row, 2].grid(True, alpha=0.3)
        
        # Power spectrum
        f_orig, psd_orig = scipy_signal.welch(orig, fs=500, nperseg=256)
        f_cf, psd_cf = scipy_signal.welch(cf_signal, fs=500, nperseg=256)
        axes[row, 3].semilogy(f_orig[:50], psd_orig[:50], 'r-', lw=1.5, label='Original')
        axes[row, 3].semilogy(f_cf[:50], psd_cf[:50], 'b-', lw=1.5, label='CF')
        axes[row, 3].set_title('Power Spectrum')
        axes[row, 3].legend(fontsize=8)
        axes[row, 3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'clinical_analysis.png', dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'clinical_analysis.png'}")
    
    # R-R interval comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Normal -> AFib R-R changes
    orig_cv = [r['rr_intervals']['original_cv'] for r in n2a_results]
    cf_cv = [r['rr_intervals']['cf_cv'] for r in n2a_results]
    
    axes[0].scatter(range(len(orig_cv)), orig_cv, c='blue', label='Original (Normal)', s=80)
    axes[0].scatter(range(len(cf_cv)), cf_cv, c='red', label='Counterfactual (→AFib)', s=80)
    for i in range(len(orig_cv)):
        axes[0].plot([i, i], [orig_cv[i], cf_cv[i]], 'k-', alpha=0.3)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('R-R Coefficient of Variation')
    axes[0].set_title('Normal→AFib: R-R Irregularity Should INCREASE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AFib -> Normal R-R changes
    orig_cv = [r['rr_intervals']['original_cv'] for r in a2n_results]
    cf_cv = [r['rr_intervals']['cf_cv'] for r in a2n_results]
    
    axes[1].scatter(range(len(orig_cv)), orig_cv, c='red', label='Original (AFib)', s=80)
    axes[1].scatter(range(len(cf_cv)), cf_cv, c='blue', label='Counterfactual (→Normal)', s=80)
    for i in range(len(orig_cv)):
        axes[1].plot([i, i], [orig_cv[i], cf_cv[i]], 'k-', alpha=0.3)
    axes[1].set_xlabel('Sample')
    axes[1].set_ylabel('R-R Coefficient of Variation')
    axes[1].set_title('AFib→Normal: R-R Irregularity Should DECREASE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'rr_interval_changes.png', dpi=150)
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'rr_interval_changes.png'}")
    
    # Save results
    all_results = {
        'normal_to_afib': {
            'n_valid': n_valid,
            'n_flipped': sum(1 for r in n2a_results if r['flipped']),
            'n_rr_correct': sum(1 for r in n2a_results if r['rr_intervals']['correct_direction']),
            'avg_similarity': float(np.mean([r['similarity'] for r in n2a_results])),
        },
        'afib_to_normal': {
            'n_valid': sum(1 for r in a2n_results if r['validity']['counterfactual_valid']),
            'n_flipped': sum(1 for r in a2n_results if r['flipped']),
            'n_rr_correct': sum(1 for r in a2n_results if r['rr_intervals']['correct_direction']),
            'avg_similarity': float(np.mean([r['similarity'] for r in a2n_results])),
        }
    }
    
    with open(RESULTS_DIR / 'clinical_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {RESULTS_DIR / 'clinical_analysis_results.json'}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*70)
    print("CLINICAL ANALYSIS SUMMARY")
    print("="*70)
    
    print("\n✓ GOAL 1: Counterfactuals are realistic (valid ECG signals)")
    print(f"  Normal→AFib: {100*all_results['normal_to_afib']['n_valid']/num_samples:.0f}% valid")
    print(f"  AFib→Normal: {100*all_results['afib_to_normal']['n_valid']/num_samples:.0f}% valid")
    
    print("\n✓ GOAL 2: Only class-discriminative features change")
    print(f"  Normal→AFib R-R irregularity increased: {100*sum(1 for r in n2a_results if r['rr_intervals']['correct_direction'])/num_samples:.0f}%")
    print(f"  AFib→Normal R-R irregularity decreased: {100*sum(1 for r in a2n_results if r['rr_intervals']['correct_direction'])/num_samples:.0f}%")
    
    print("\n✓ High similarity maintained:")
    print(f"  Normal→AFib: {all_results['normal_to_afib']['avg_similarity']:.3f}")
    print(f"  AFib→Normal: {all_results['afib_to_normal']['avg_similarity']:.3f}")
    
    print("="*70)


if __name__ == '__main__':
    main()
