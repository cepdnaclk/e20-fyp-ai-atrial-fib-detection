"""
Diagnostic script to analyze what changes the counterfactual model is actually making.

Key questions:
1. How large are the edits (in terms of signal amplitude)?
2. Are the edits adversarial noise or meaningful changes?
3. What frequency components are being modified?
4. Are the changes visually perceptible?
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq
import os
from datetime import datetime

# Import the models
from counterfactual_training import AFibResLSTM, ModelConfig
from train_minimal_edit_model import ResidualEditModel

def load_models(device):
    """Load classifier and counterfactual model"""
    # Load classifier
    config = ModelConfig()
    classifier = AFibResLSTM(config).to(device)
    classifier_ckpt = torch.load('./best_model/best_model.pth', map_location=device)
    classifier.load_state_dict(classifier_ckpt['model_state_dict'])
    classifier.eval()
    
    # Load counterfactual model
    cf_model = ResidualEditModel(hidden_dim=256).to(device)
    cf_ckpt = torch.load('./minimal_edit_training/best_model.pth', map_location=device)
    cf_model.load_state_dict(cf_ckpt['model_state_dict'])
    cf_model.eval()
    
    return classifier, cf_model

def analyze_edit_magnitudes(original, counterfactual, edit):
    """Analyze the magnitude of edits relative to signal"""
    results = {}
    
    # Signal statistics
    signal_std = np.std(original)
    signal_range = np.max(original) - np.min(original)
    signal_mean_abs = np.mean(np.abs(original))
    
    # Edit statistics
    edit_std = np.std(edit)
    edit_range = np.max(edit) - np.min(edit)
    edit_mean_abs = np.mean(np.abs(edit))
    edit_max = np.max(np.abs(edit))
    
    # Relative magnitudes
    results['signal_std'] = signal_std
    results['signal_range'] = signal_range
    results['edit_std'] = edit_std
    results['edit_range'] = edit_range
    results['edit_max'] = edit_max
    
    # Key ratio: edit magnitude relative to signal
    results['edit_to_signal_ratio'] = edit_std / signal_std if signal_std > 0 else 0
    results['edit_max_to_signal_range'] = edit_max / signal_range if signal_range > 0 else 0
    
    # SNR: treat original as "signal" and edit as "noise"
    # If edits are meaningful, this should be moderate (10-30 dB)
    # If edits are adversarial, this will be very high (>40 dB)
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(edit ** 2)
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    results['snr_db'] = snr_db
    
    return results

def analyze_frequency_changes(original, counterfactual, fs=250):
    """Analyze frequency domain changes"""
    results = {}
    
    # Compute FFT
    n = len(original)
    freqs = fftfreq(n, 1/fs)[:n//2]
    
    fft_orig = np.abs(fft(original))[:n//2]
    fft_cf = np.abs(fft(counterfactual))[:n//2]
    fft_diff = fft_cf - fft_orig
    
    # Frequency bands for ECG
    # Delta (0-1 Hz): Baseline
    # Heart rate band (0.5-3 Hz): ~30-180 bpm
    # QRS complex (5-15 Hz): Sharp deflections
    # High frequency (15-40 Hz): Fine details, noise
    
    bands = {
        'baseline': (0, 1),
        'heart_rate': (0.5, 3),
        'qrs': (5, 15),
        'high_freq': (15, 40),
        'very_high': (40, 100)
    }
    
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if np.sum(mask) > 0:
            orig_power = np.mean(fft_orig[mask] ** 2)
            cf_power = np.mean(fft_cf[mask] ** 2)
            change_power = np.mean(np.abs(fft_diff[mask]) ** 2)
            
            results[f'{band_name}_orig_power'] = orig_power
            results[f'{band_name}_cf_power'] = cf_power
            results[f'{band_name}_change_ratio'] = change_power / orig_power if orig_power > 0 else 0
    
    return results, freqs, fft_orig, fft_cf

def check_adversarial_nature(classifier, original_tensor, cf_tensor, device):
    """Check if the flip is due to adversarial-like perturbation"""
    results = {}
    
    with torch.no_grad():
        # Get classifier outputs
        orig_logits, _ = classifier(original_tensor)
        cf_logits, _ = classifier(cf_tensor)
        
        orig_probs = torch.softmax(orig_logits, dim=1)
        cf_probs = torch.softmax(cf_logits, dim=1)
        
        orig_pred = orig_logits.argmax(dim=1).item()
        cf_pred = cf_logits.argmax(dim=1).item()
        
        results['orig_pred'] = orig_pred
        results['cf_pred'] = cf_pred
        results['flipped'] = orig_pred != cf_pred
        
        results['orig_confidence'] = orig_probs.max().item()
        results['cf_confidence'] = cf_probs.max().item()
        
        # Check confidence margin
        results['orig_margin'] = (orig_probs[0, orig_pred] - orig_probs[0, 1-orig_pred]).item()
        results['cf_margin'] = (cf_probs[0, cf_pred] - cf_probs[0, 1-cf_pred]).item()
        
    return results

def visualize_detailed_analysis(original, counterfactual, edit, analysis, freq_analysis, 
                                 freqs, fft_orig, fft_cf, sample_idx, save_dir):
    """Create detailed visualization"""
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Full signal comparison
    ax1 = fig.add_subplot(4, 2, 1)
    ax1.plot(original, 'b-', alpha=0.7, label='Original', linewidth=0.8)
    ax1.plot(counterfactual, 'r-', alpha=0.7, label='Counterfactual', linewidth=0.8)
    ax1.set_title(f'Sample {sample_idx}: Full Signal Comparison')
    ax1.legend()
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    
    # 2. Edit signal (what was added/subtracted)
    ax2 = fig.add_subplot(4, 2, 2)
    ax2.plot(edit, 'g-', linewidth=0.8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_title(f'Edit Signal (max={analysis["edit_max"]:.4f}, SNR={analysis["snr_db"]:.1f} dB)')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Edit Amplitude')
    
    # 3. Zoomed view (1 second)
    ax3 = fig.add_subplot(4, 2, 3)
    start = 500
    end = 750  # 1 second at 250 Hz
    ax3.plot(range(start, end), original[start:end], 'b-', linewidth=1.5, label='Original')
    ax3.plot(range(start, end), counterfactual[start:end], 'r--', linewidth=1.5, label='Counterfactual')
    ax3.set_title('Zoomed View (1 second)')
    ax3.legend()
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Amplitude')
    
    # 4. Edit zoomed
    ax4 = fig.add_subplot(4, 2, 4)
    ax4.plot(range(start, end), edit[start:end], 'g-', linewidth=1.5)
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.fill_between(range(start, end), edit[start:end], alpha=0.3, color='green')
    ax4.set_title('Edit Zoomed (1 second)')
    ax4.set_xlabel('Sample')
    ax4.set_ylabel('Edit Amplitude')
    
    # 5. Frequency spectrum
    ax5 = fig.add_subplot(4, 2, 5)
    ax5.semilogy(freqs[:500], fft_orig[:500], 'b-', alpha=0.7, label='Original')
    ax5.semilogy(freqs[:500], fft_cf[:500], 'r-', alpha=0.7, label='Counterfactual')
    ax5.set_title('Frequency Spectrum (0-50 Hz)')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Magnitude (log)')
    ax5.legend()
    ax5.set_xlim(0, 50)
    
    # 6. Frequency difference
    ax6 = fig.add_subplot(4, 2, 6)
    fft_diff = np.abs(fft_cf - fft_orig)
    ax6.bar(freqs[:500], fft_diff[:500], width=0.2, color='purple', alpha=0.7)
    ax6.set_title('Frequency Change Magnitude')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('|FFT(CF) - FFT(Orig)|')
    ax6.set_xlim(0, 50)
    
    # 7. Edit histogram
    ax7 = fig.add_subplot(4, 2, 7)
    ax7.hist(edit, bins=100, color='green', alpha=0.7, edgecolor='black')
    ax7.axvline(x=0, color='r', linestyle='--')
    ax7.set_title(f'Edit Distribution (std={analysis["edit_std"]:.4f})')
    ax7.set_xlabel('Edit Value')
    ax7.set_ylabel('Count')
    
    # 8. Summary statistics
    ax8 = fig.add_subplot(4, 2, 8)
    ax8.axis('off')
    
    summary_text = f"""
    ANALYSIS SUMMARY
    ================
    
    Signal Statistics:
      Signal STD: {analysis['signal_std']:.4f}
      Signal Range: {analysis['signal_range']:.4f}
    
    Edit Statistics:
      Edit STD: {analysis['edit_std']:.6f}
      Edit Max: {analysis['edit_max']:.6f}
      Edit Range: {analysis['edit_range']:.6f}
    
    Key Ratios:
      Edit/Signal Ratio: {analysis['edit_to_signal_ratio']:.4%}
      SNR (orig/edit): {analysis['snr_db']:.1f} dB
    
    Interpretation:
      {'⚠️ ADVERSARIAL: Edits are imperceptibly small!' if analysis['snr_db'] > 40 else ''}
      {'⚠️ MODERATE: Edits are small but may be visible' if 25 < analysis['snr_db'] <= 40 else ''}
      {'✓ MEANINGFUL: Edits are clinically visible' if analysis['snr_db'] <= 25 else ''}
    
    Frequency Band Changes:
      Heart Rate Band: {freq_analysis.get('heart_rate_change_ratio', 0):.4%}
      QRS Band: {freq_analysis.get('qrs_change_ratio', 0):.4%}
      High Freq Band: {freq_analysis.get('high_freq_change_ratio', 0):.4%}
    """
    
    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'analysis_sample_{sample_idx}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Force CPU since GPU may be occupied
    device = torch.device('cpu')
    print(f"Using device: {device} (forced CPU for analysis)")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f'./counterfactual_analysis_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X = np.load('./ecg_afib_data/X_combined.npy')
    y = np.load('./ecg_afib_data/y_combined.npy')
    
    # Load models
    print("Loading models...")
    classifier, cf_model = load_models(device)
    
    # Get edit strength
    with torch.no_grad():
        edit_strength = torch.sigmoid(cf_model.edit_strength).item()
    print(f"\nModel edit_strength (after sigmoid): {edit_strength:.4f}")
    
    # Analyze random samples
    n_samples = 50
    np.random.seed(42)
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    all_analyses = []
    adversarial_count = 0
    flip_count = 0
    
    print(f"\nAnalyzing {n_samples} samples...")
    
    for i, idx in enumerate(indices):
        # Get sample
        ecg = X[idx]
        label = 0 if y[idx] == 'A' else 1  # 0=AFib, 1=Normal
        target_class = 1 - label  # Flip target
        
        # Convert to tensor
        ecg_tensor = torch.FloatTensor(ecg).unsqueeze(0).unsqueeze(0).to(device)
        target_tensor = torch.LongTensor([target_class]).to(device)
        
        # Generate counterfactual
        with torch.no_grad():
            cf_output, edit = cf_model(ecg_tensor, target_tensor)
            strength = torch.sigmoid(cf_model.edit_strength).item()
            
        # Convert to numpy
        original = ecg
        counterfactual = cf_output.squeeze().cpu().numpy()
        edit_np = edit.squeeze().cpu().numpy() * strength
        
        # Analyze
        analysis = analyze_edit_magnitudes(original, counterfactual, edit_np)
        freq_analysis, freqs, fft_orig, fft_cf = analyze_frequency_changes(original, counterfactual)
        classifier_results = check_adversarial_nature(classifier, ecg_tensor, cf_output, device)
        
        analysis.update(classifier_results)
        analysis['sample_idx'] = idx
        analysis['original_label'] = 'AFib' if label == 0 else 'Normal'
        analysis['target_label'] = 'Normal' if label == 0 else 'AFib'
        
        all_analyses.append(analysis)
        
        if analysis['snr_db'] > 40:
            adversarial_count += 1
        if analysis['flipped']:
            flip_count += 1
        
        # Visualize first 10 samples
        if i < 10:
            visualize_detailed_analysis(original, counterfactual, edit_np, analysis, 
                                        freq_analysis, freqs, fft_orig, fft_cf, idx, save_dir)
            print(f"  Sample {idx}: SNR={analysis['snr_db']:.1f}dB, "
                  f"Edit/Signal={analysis['edit_to_signal_ratio']:.4%}, "
                  f"Flipped={analysis['flipped']}")
    
    # Summary statistics
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    snrs = [a['snr_db'] for a in all_analyses if not np.isinf(a['snr_db'])]
    ratios = [a['edit_to_signal_ratio'] for a in all_analyses]
    
    print(f"\nEdit Magnitude Statistics:")
    print(f"  Mean SNR: {np.mean(snrs):.1f} dB")
    print(f"  Min SNR: {np.min(snrs):.1f} dB")
    print(f"  Max SNR: {np.max(snrs):.1f} dB")
    print(f"  Mean Edit/Signal Ratio: {np.mean(ratios):.4%}")
    print(f"  Max Edit/Signal Ratio: {np.max(ratios):.4%}")
    
    print(f"\nAdversarial Analysis:")
    print(f"  Samples with SNR > 40dB (adversarial): {adversarial_count}/{n_samples} ({100*adversarial_count/n_samples:.1f}%)")
    print(f"  Samples with SNR 25-40dB (borderline): {sum(1 for a in all_analyses if 25 < a['snr_db'] <= 40)}/{n_samples}")
    print(f"  Samples with SNR < 25dB (visible): {sum(1 for a in all_analyses if a['snr_db'] <= 25)}/{n_samples}")
    
    print(f"\nClassifier Results:")
    print(f"  Successful flips: {flip_count}/{n_samples} ({100*flip_count/n_samples:.1f}%)")
    
    # CRITICAL INSIGHT
    print("\n" + "="*60)
    print("🔍 DIAGNOSIS")
    print("="*60)
    
    if np.mean(snrs) > 40:
        print("""
⚠️  THE MODEL IS GENERATING ADVERSARIAL PERTURBATIONS!

The edits are so small (SNR > 40dB) that they're essentially invisible.
The classifier is being fooled by imperceptible noise, not meaningful changes.

This happens because:
1. The similarity loss (MSE) pushes edits to be tiny
2. The classifier has sharp decision boundaries that can be crossed with tiny perturbations
3. There's no constraint forcing edits to be VISIBLE/INTERPRETABLE

For clinical teaching, we need edits that are:
- Large enough to be visible (SNR < 25dB or edit/signal ratio > 5%)
- Focused on clinically meaningful features (P-waves, RR intervals)
- NOT just adversarial noise

RECOMMENDATION: Need to add a MINIMUM EDIT constraint or use
gradient-guided approach that targets specific clinical features.
""")
    elif np.mean(snrs) > 25:
        print("""
⚠️  THE MODEL IS GENERATING BORDERLINE EDITS

The edits are small but may be barely visible.
They might be exploiting classifier weaknesses rather than meaningful features.

Consider increasing the minimum edit size or using clinical feature constraints.
""")
    else:
        print("""
✓  THE MODEL IS GENERATING VISIBLE EDITS

The edits appear to be large enough to be visible.
Check the visualizations to confirm they're clinically meaningful.
""")
    
    # Save summary
    summary_file = os.path.join(save_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Counterfactual Analysis Summary\n")
        f.write(f"================================\n\n")
        f.write(f"Model edit_strength: {edit_strength:.4f}\n")
        f.write(f"Samples analyzed: {n_samples}\n\n")
        f.write(f"Edit Magnitude:\n")
        f.write(f"  Mean SNR: {np.mean(snrs):.1f} dB\n")
        f.write(f"  Mean Edit/Signal Ratio: {np.mean(ratios):.4%}\n\n")
        f.write(f"Adversarial (SNR>40dB): {adversarial_count}/{n_samples}\n")
        f.write(f"Flip Rate: {flip_count}/{n_samples}\n")
    
    print(f"\nResults saved to: {save_dir}")

if __name__ == '__main__':
    main()
