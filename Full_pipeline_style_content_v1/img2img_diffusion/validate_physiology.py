"""
Physiological Validation of Generated ECG Counterfactuals
==========================================================

Validates that generated ECGs maintain physiological plausibility:
1. Heart rate within normal/tachycardia/bradycardia ranges
2. QRS detection success rate
3. RR interval variability (should differ between AFib and Normal)
4. Signal quality metrics (SNR, baseline)
5. Morphology preservation (P-wave, QRS amplitude)

Usage:
    python validate_physiology.py --output_dir ./outputs
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.stats import pearsonr
import json
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False
    print("Warning: neurokit2 not installed. Install with: pip install neurokit2")


class PhysiologicalValidator:
    """Validate ECG physiological plausibility"""
    
    def __init__(self, fs=250):
        self.fs = fs
        
    def detect_r_peaks(self, ecg):
        """Detect R-peaks using simple derivative approach"""
        # Bandpass filter (5-15 Hz for QRS)
        b, a = signal.butter(2, [5, 15], btype='band', fs=self.fs)
        filtered = signal.filtfilt(b, a, ecg)
        
        # Squared derivative
        diff = np.diff(filtered)
        squared = diff ** 2
        
        # Moving average
        window = int(0.1 * self.fs)  # 100ms
        ma = np.convolve(squared, np.ones(window)/window, mode='same')
        
        # Find peaks
        threshold = np.mean(ma) + 0.5 * np.std(ma)
        peaks, _ = signal.find_peaks(ma, height=threshold, distance=int(0.3 * self.fs))
        
        return peaks
    
    def compute_heart_rate(self, ecg):
        """Compute heart rate from R-peaks"""
        peaks = self.detect_r_peaks(ecg)
        
        if len(peaks) < 2:
            return None, None, []
        
        # RR intervals in ms
        rr_intervals = np.diff(peaks) / self.fs * 1000
        
        # Heart rate in bpm
        hr = 60000 / np.mean(rr_intervals) if len(rr_intervals) > 0 else None
        hr_std = 60000 / np.std(rr_intervals) if len(rr_intervals) > 1 and np.std(rr_intervals) > 0 else None
        
        return hr, hr_std, rr_intervals
    
    def compute_hrv_metrics(self, rr_intervals):
        """Compute HRV metrics (key for AFib vs Normal distinction)"""
        if len(rr_intervals) < 3:
            return {}
        
        # SDNN - Standard deviation of RR intervals
        sdnn = np.std(rr_intervals)
        
        # RMSSD - Root mean square of successive differences
        successive_diff = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(successive_diff ** 2))
        
        # pNN50 - Percentage of successive RR differences > 50ms
        pnn50 = np.sum(np.abs(successive_diff) > 50) / len(successive_diff) * 100
        
        # Coefficient of variation
        cv = np.std(rr_intervals) / np.mean(rr_intervals) * 100
        
        return {
            'sdnn_ms': sdnn,
            'rmssd_ms': rmssd,
            'pnn50_percent': pnn50,
            'cv_percent': cv,
            'mean_rr_ms': np.mean(rr_intervals),
            'num_beats': len(rr_intervals) + 1
        }
    
    def compute_signal_quality(self, ecg):
        """Compute signal quality metrics"""
        # SNR estimation
        f, psd = signal.welch(ecg, fs=self.fs, nperseg=512)
        
        # Signal power (1-40 Hz)
        signal_band = (f >= 1) & (f <= 40)
        signal_power = np.sum(psd[signal_band])
        
        # Noise power (>50 Hz)
        noise_band = f > 50
        noise_power = np.sum(psd[noise_band]) + 1e-10
        
        snr = 10 * np.log10(signal_power / noise_power)
        
        # Baseline wander (power < 1 Hz)
        baseline_band = f < 1
        baseline_power = np.sum(psd[baseline_band])
        baseline_ratio = baseline_power / signal_power
        
        return {
            'snr_db': snr,
            'baseline_ratio': baseline_ratio,
            'signal_power': signal_power
        }
    
    def compute_morphology_metrics(self, original, generated):
        """Compare morphological features"""
        # Overall correlation
        corr, _ = pearsonr(original.flatten(), generated.flatten())
        
        # QRS amplitude comparison
        peaks_orig = self.detect_r_peaks(original)
        peaks_gen = self.detect_r_peaks(generated)
        
        if len(peaks_orig) > 0:
            qrs_amp_orig = np.mean([np.ptp(original[max(0, p-10):min(len(original), p+10)]) for p in peaks_orig])
        else:
            qrs_amp_orig = 0
            
        if len(peaks_gen) > 0:
            qrs_amp_gen = np.mean([np.ptp(generated[max(0, p-10):min(len(generated), p+10)]) for p in peaks_gen])
        else:
            qrs_amp_gen = 0
        
        amp_ratio = qrs_amp_gen / (qrs_amp_orig + 1e-10)
        
        # Spectral similarity
        f_orig, psd_orig = signal.welch(original, fs=self.fs, nperseg=256)
        f_gen, psd_gen = signal.welch(generated, fs=self.fs, nperseg=256)
        spectral_corr, _ = pearsonr(psd_orig, psd_gen)
        
        return {
            'correlation': corr,
            'qrs_amplitude_ratio': amp_ratio,
            'spectral_correlation': spectral_corr
        }
    
    def validate_ecg(self, ecg):
        """Full validation of single ECG"""
        results = {
            'valid': True,
            'issues': []
        }
        
        # Heart rate check
        hr, hr_std, rr_intervals = self.compute_heart_rate(ecg)
        
        if hr is None:
            results['valid'] = False
            results['issues'].append('No R-peaks detected')
            results['heart_rate'] = None
        else:
            results['heart_rate'] = hr
            
            if hr < 30:
                results['issues'].append(f'Severe bradycardia ({hr:.0f} bpm)')
            elif hr < 50:
                results['issues'].append(f'Bradycardia ({hr:.0f} bpm)')
            elif hr > 150:
                results['issues'].append(f'Tachycardia ({hr:.0f} bpm)')
            elif hr > 200:
                results['valid'] = False
                results['issues'].append(f'Implausible HR ({hr:.0f} bpm)')
        
        # HRV metrics
        results['hrv'] = self.compute_hrv_metrics(rr_intervals)
        
        # Signal quality
        results['quality'] = self.compute_signal_quality(ecg)
        
        if results['quality']['snr_db'] < 5:
            results['issues'].append(f"Low SNR ({results['quality']['snr_db']:.1f} dB)")
        
        return results
    
    def validate_counterfactual(self, original, counterfactual, orig_class, cf_class):
        """Validate counterfactual specifically"""
        results = {
            'original': self.validate_ecg(original),
            'counterfactual': self.validate_ecg(counterfactual),
            'morphology': self.compute_morphology_metrics(original, counterfactual)
        }
        
        # Check if RR variability changed appropriately
        orig_hrv = results['original']['hrv']
        cf_hrv = results['counterfactual']['hrv']
        
        if orig_hrv and cf_hrv:
            # AFib should have higher RMSSD than Normal
            orig_rmssd = orig_hrv.get('rmssd_ms', 0)
            cf_rmssd = cf_hrv.get('rmssd_ms', 0)
            
            results['hrv_change'] = cf_rmssd - orig_rmssd
            
            # Expected changes based on direction
            if orig_class == 0 and cf_class == 1:  # Normal → AFib
                # Expect increased variability
                results['expected_change'] = 'increase'
                results['change_correct'] = cf_rmssd > orig_rmssd
            else:  # AFib → Normal
                # Expect decreased variability
                results['expected_change'] = 'decrease'
                results['change_correct'] = cf_rmssd < orig_rmssd
        
        return results


def run_validation(output_dir, data_path=None):
    """Run validation on generated outputs"""
    validator = PhysiologicalValidator()
    
    output_path = Path(output_dir)
    cf_dir = output_path / 'counterfactuals'
    
    if not cf_dir.exists():
        print(f"No counterfactuals found in {cf_dir}")
        return
    
    # Load generation results
    results_file = output_path / 'generation_results.json'
    if results_file.exists():
        with open(results_file) as f:
            gen_results = json.load(f)
    else:
        gen_results = {'samples': []}
    
    # Load original data if provided
    if data_path:
        X = np.load(Path(data_path) / 'X_combined.npy')
        y = np.load(Path(data_path) / 'y_combined.npy')
        y_int = np.array([1 if l == 'A' else 0 for l in y])
    else:
        X = None
    
    print("\n" + "="*70)
    print("Physiological Validation Results")
    print("="*70)
    
    validation_results = {
        'total': 0,
        'valid_original': 0,
        'valid_counterfactual': 0,
        'plausible_hr_cf': 0,
        'good_correlation': 0,
        'correct_hrv_change': 0,
        'samples': []
    }
    
    cf_files = list(cf_dir.glob('cf_*.npy'))
    
    for cf_file in tqdm(cf_files, desc='Validating'):
        idx = int(cf_file.stem.split('_')[1])
        
        # Load counterfactual
        cf = np.load(cf_file)
        
        # Get original if available
        if X is not None and idx < len(X):
            original = X[idx]
            original = (original - original.mean()) / (original.std() + 1e-6)  # Normalize
            orig_class = y_int[idx]
            
            # Get counterfactual class from generation results
            sample_info = next((s for s in gen_results.get('samples', []) if s['idx'] == idx), None)
            cf_class = sample_info['cf_class'] if sample_info else 1 - orig_class
            
            # Validate pair
            results = validator.validate_counterfactual(original, cf, orig_class, cf_class)
        else:
            results = {'counterfactual': validator.validate_ecg(cf)}
        
        validation_results['total'] += 1
        
        if results.get('original', {}).get('valid', True):
            validation_results['valid_original'] += 1
        
        if results['counterfactual']['valid']:
            validation_results['valid_counterfactual'] += 1
        
        hr = results['counterfactual'].get('heart_rate')
        if hr and 40 <= hr <= 180:
            validation_results['plausible_hr_cf'] += 1
        
        corr = results.get('morphology', {}).get('correlation', 0)
        if corr > 0.8:
            validation_results['good_correlation'] += 1
        
        if results.get('change_correct', False):
            validation_results['correct_hrv_change'] += 1
        
        validation_results['samples'].append({
            'idx': idx,
            'cf_valid': results['counterfactual']['valid'],
            'cf_hr': results['counterfactual'].get('heart_rate'),
            'correlation': corr,
            'issues': results['counterfactual'].get('issues', [])
        })
    
    # Print summary
    total = validation_results['total']
    print(f"\n{'Metric':<35} {'Value':>10} {'Percent':>10}")
    print("-"*60)
    print(f"{'Total Samples':<35} {total:>10}")
    print(f"{'Valid Counterfactuals':<35} {validation_results['valid_counterfactual']:>10} {100*validation_results['valid_counterfactual']/total:>9.1f}%")
    print(f"{'Plausible HR (40-180 bpm)':<35} {validation_results['plausible_hr_cf']:>10} {100*validation_results['plausible_hr_cf']/total:>9.1f}%")
    print(f"{'Good Correlation (>0.8)':<35} {validation_results['good_correlation']:>10} {100*validation_results['good_correlation']/total:>9.1f}%")
    if X is not None:
        print(f"{'Correct HRV Change Direction':<35} {validation_results['correct_hrv_change']:>10} {100*validation_results['correct_hrv_change']/total:>9.1f}%")
    
    # Save results
    with open(output_path / 'physiology_validation.json', 'w') as f:
        json.dump(validation_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nResults saved to: {output_path / 'physiology_validation.json'}")
    
    # Analyze issues
    all_issues = []
    for s in validation_results['samples']:
        all_issues.extend(s['issues'])
    
    if all_issues:
        print("\n--- Common Issues ---")
        from collections import Counter
        for issue, count in Counter(all_issues).most_common(5):
            print(f"  {issue}: {count} samples")
    
    return validation_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--data_path', type=str, default='../ecg_afib_data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_validation(args.output_dir, args.data_path)
