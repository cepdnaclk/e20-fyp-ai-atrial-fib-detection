"""
COUNTERFACTUAL QUALITY ANALYSIS
================================
This script analyzes whether counterfactuals preserve patient morphology
while making targeted AFib-specific changes.

What we WANT:
- High correlation between original and counterfactual (same patient morphology)
- Changed RR intervals (irregular for AFib, regular for Normal)
- P-wave presence/absence changes

What we DON'T want:
- Completely different ECG that just happens to flip the class
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import os
import json

# Setup
RESULTS_DIR = './counterfactual_results_enhanced/'
DATA_PATH = './ecg_afib_data/X_combined.npy'
LABELS_PATH = './ecg_afib_data/y_combined.npy'
NORM_PARAMS_PATH = './enhanced_counterfactual_training/norm_params.npy'

def load_data():
    """Load dataset and normalization params"""
    signals = np.load(DATA_PATH)
    labels = np.load(LABELS_PATH)
    norm_params = np.load(NORM_PARAMS_PATH, allow_pickle=True).item()
    
    # Convert labels
    if labels.dtype.kind in ['U', 'S', 'O']:
        labels = np.array([1 if l == 'A' else 0 for l in labels])
    
    return signals, labels, norm_params

def compute_morphology_metrics(original, counterfactual):
    """
    Compute metrics to assess morphology preservation
    
    Returns:
        dict with:
        - pearson_corr: Overall correlation (-1 to 1, higher = more similar)
        - mse: Mean squared error (lower = more similar)
        - psd_correlation: Power spectral density correlation
        - peak_count_diff: Difference in number of R-peaks
    """
    # 1. Pearson correlation (overall similarity)
    corr, _ = pearsonr(original.flatten(), counterfactual.flatten())
    
    # 2. MSE
    mse = np.mean((original - counterfactual) ** 2)
    
    # 3. PSD correlation (frequency content similarity)
    f_orig, psd_orig = signal.welch(original, fs=250, nperseg=256)
    f_cf, psd_cf = signal.welch(counterfactual, fs=250, nperseg=256)
    psd_corr, _ = pearsonr(psd_orig, psd_cf)
    
    # 4. R-peak detection (QRS complex count)
    # Simple peak detection
    orig_peaks, _ = signal.find_peaks(original, height=np.std(original)*0.5, distance=50)
    cf_peaks, _ = signal.find_peaks(counterfactual, height=np.std(counterfactual)*0.5, distance=50)
    peak_diff = abs(len(orig_peaks) - len(cf_peaks))
    
    return {
        'pearson_correlation': float(corr),
        'mse': float(mse),
        'psd_correlation': float(psd_corr),
        'peak_count_original': len(orig_peaks),
        'peak_count_counterfactual': len(cf_peaks),
        'peak_count_diff': peak_diff
    }

def compute_rr_irregularity(ecg, fs=250):
    """
    Compute RR interval irregularity (coefficient of variation)
    Higher = more irregular (AFib characteristic)
    """
    # Find R-peaks
    peaks, _ = signal.find_peaks(ecg, height=np.std(ecg)*0.5, distance=50)
    
    if len(peaks) < 3:
        return None
    
    # Compute RR intervals
    rr_intervals = np.diff(peaks) / fs  # in seconds
    
    # Coefficient of variation (std/mean)
    rr_cv = np.std(rr_intervals) / np.mean(rr_intervals) if np.mean(rr_intervals) > 0 else 0
    
    return {
        'mean_rr': float(np.mean(rr_intervals)),
        'std_rr': float(np.std(rr_intervals)),
        'cv_rr': float(rr_cv),  # Higher = more irregular
        'num_beats': len(peaks)
    }

def visualize_comparison(original, counterfactual, idx, label, save_path=None):
    """Create detailed comparison visualization"""
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    t = np.arange(len(original)) / 250  # Time in seconds
    
    # Plot 1: Full signals overlay
    ax1 = axes[0]
    ax1.plot(t, original, 'b-', alpha=0.7, label='Original', linewidth=1)
    ax1.plot(t, counterfactual, 'r-', alpha=0.7, label='Counterfactual', linewidth=1)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'Sample {idx} - Full ECG Overlay (Original: {"AFib" if label == 1 else "Normal"})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed first 3 seconds
    ax2 = axes[1]
    zoom_samples = int(3 * 250)
    ax2.plot(t[:zoom_samples], original[:zoom_samples], 'b-', alpha=0.7, label='Original', linewidth=1.5)
    ax2.plot(t[:zoom_samples], counterfactual[:zoom_samples], 'r-', alpha=0.7, label='Counterfactual', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('First 3 Seconds - Zoomed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Difference signal
    ax3 = axes[2]
    diff = counterfactual - original
    ax3.plot(t, diff, 'g-', linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Difference')
    ax3.set_title(f'Difference (Counterfactual - Original) | Mean: {diff.mean():.4f}, Std: {diff.std():.4f}')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Power spectral density
    ax4 = axes[3]
    f_orig, psd_orig = signal.welch(original, fs=250, nperseg=256)
    f_cf, psd_cf = signal.welch(counterfactual, fs=250, nperseg=256)
    ax4.semilogy(f_orig, psd_orig, 'b-', alpha=0.7, label='Original')
    ax4.semilogy(f_cf, psd_cf, 'r-', alpha=0.7, label='Counterfactual')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power Spectral Density')
    ax4.set_title('Power Spectral Density Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 50])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close()

def main():
    print("="*70)
    print("COUNTERFACTUAL QUALITY ANALYSIS")
    print("="*70)
    
    # Load data
    signals, labels, norm_params = load_data()
    means = norm_params['means']
    stds = norm_params['stds']
    
    # Sample indices from generation (from the log file)
    # These are the indices we generated counterfactuals for
    sample_indices = [35857, 21260, 37601, 83859, 77126, 39401, 6524, 83317, 55174, 45509]
    
    # For now, we need to regenerate counterfactuals to get the actual values
    # Let's use a modified generation script that saves the actual values
    
    print("\nNote: To properly analyze, we need the actual counterfactual numpy arrays.")
    print("The PNG files don't contain the raw data.")
    print("\nLet's regenerate a few samples and analyze them...\n")
    
    # Quick analysis using the visualization images
    # Count how many PNG files we have
    png_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.png')]
    print(f"Generated counterfactual visualizations: {len(png_files)}")
    
    # Load summary
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'r') as f:
        summary = json.load(f)
    
    print(f"\nSummary from generation:")
    print(f"  Total samples: {summary['total_samples']}")
    print(f"  Flip success rate: {summary['flip_success_rate']*100:.1f}%")
    print(f"  AFib → Normal: {summary['afib_to_normal']['rate']*100:.1f}%")
    print(f"  Normal → AFib: {summary['normal_to_afib']['rate']*100:.1f}%")
    
    print("\n" + "="*70)
    print("THE REAL PROBLEM")
    print("="*70)
    print("""
The current approach generates counterfactuals FROM RANDOM NOISE,
conditioned on content/style/class. This means:

1. The model is NOT making minimal changes to the original ECG
2. It's generating a NEW ECG that matches the target class
3. The "content encoder" is supposed to preserve morphology, but
   starting from noise means it's reconstructing, not editing

WHAT WE NEED:
=============
A MINIMAL INTERVENTION approach that:
1. Starts from the original ECG (not noise)
2. Makes targeted changes ONLY to AFib-specific features:
   - RR interval regularity
   - P-wave presence/absence
3. Preserves everything else (QRS shape, T-wave, amplitude patterns)

POSSIBLE SOLUTIONS:
==================
1. Use DDIM inversion: Encode the original ECG to noise, then decode
   with different class conditioning

2. Add STRONG reconstruction loss: Force counterfactual to be close
   to original except for class-specific features

3. Use a different architecture: Instead of diffusion, use a 
   targeted editing network that learns to modify only specific
   ECG features

4. Latent space interpolation: Find the direction in latent space
   that corresponds to AFib vs Normal, and move along it minimally

5. Cycle consistency: Require that Normal→AFib→Normal ≈ Original
    """)

if __name__ == "__main__":
    main()
