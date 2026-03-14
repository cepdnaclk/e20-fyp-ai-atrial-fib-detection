"""
Clinical Plausibility Validator for ECG Counterfactuals
========================================================

Multi-level validation system to ensure generated ECGs are clinically valid.

Validation Levels:
1. Morphological: R-peak detection, QRS verification, amplitude checks
2. Physiological: Heart rate, RR intervals, variability constraints
3. Clinical: Feature changes appropriate for target class


"""

import numpy as np
import torch
from scipy import signal
from scipy.stats import pearsonr
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class PlausibilityValidator:
    """
    Multi-level clinical plausibility validator for ECG signals.
    
    Validates:
    - Morphological correctness (R-peaks, QRS complexes)
    - Physiological constraints (HR, RR intervals)
    - Clinical appropriateness (feature changes for class)
    """
    
    def __init__(self, fs=250, signal_length=2500):
        """
        Args:
            fs: Sampling frequency (Hz)
            signal_length: Expected signal length in samples
        """
        self.fs = fs
        self.signal_length = signal_length
        self.beat_duration = 10.0  # seconds
        
        # Physiological constraints
        self.min_hr = 30   # bpm
        self.max_hr = 200  # bpm
        self.min_rr = 0.3  # seconds (200 bpm)
        self.max_rr = 2.0  # seconds (30 bpm)
        self.max_rr_cv = 0.6  # Maximum coefficient of variation
        
        # Morphological constraints
        self.min_amplitude = -3.0
        self.max_amplitude = 3.0
        self.min_qrs_width = int(0.06 * fs)  # 60ms
        self.max_qrs_width = int(0.12 * fs)  # 120ms
        self.min_qrs_amplitude = 0.2
        
    def validate(self, 
                 ecg: np.ndarray, 
                 original_ecg: Optional[np.ndarray] = None,
                 target_class: Optional[int] = None,
                 original_class: Optional[int] = None) -> Dict:
        """
        Comprehensive validation of ECG signal.
        
        Args:
            ecg: ECG signal (1, length) or (length,)
            original_ecg: Original signal for comparison (optional)
            target_class: Target class (0=Normal, 1=AFib) (optional)
            original_class: Original class (optional)
            
        Returns:
            Dictionary with:
                - valid: bool, overall validity
                - score: float, plausibility score [0-1]
                - morphology_score: float
                - physiology_score: float
                - clinical_score: float
                - details: dict with detailed checks
        """
        # Ensure 1D
        if isinstance(ecg, torch.Tensor):
            ecg = ecg.cpu().numpy()
        ecg = np.squeeze(ecg)
        
        if original_ecg is not None:
            if isinstance(original_ecg, torch.Tensor):
                original_ecg = original_ecg.cpu().numpy()
            original_ecg = np.squeeze(original_ecg)
        
        # Level 1: Morphological checks
        morph_result = self._check_morphology(ecg)
        
        # Level 2: Physiological checks
        physio_result = self._check_physiology(ecg, morph_result['r_peaks'])
        
        # Level 3: Clinical feature validation
        if original_ecg is not None and target_class is not None:
            clinical_result = self._check_clinical_features(
                ecg, original_ecg, target_class, original_class,
                morph_result['r_peaks']
            )
        else:
            clinical_result = {'score': 1.0, 'checks': {}}
        
        # Compute overall score
        morphology_score = morph_result['score']
        physiology_score = physio_result['score']
        clinical_score = clinical_result['score']
        
        overall_score = (
            0.3 * morphology_score +
            0.3 * physiology_score +
            0.4 * clinical_score
        )
        
        # Valid if score > 0.7
        valid = overall_score > 0.7
        
        return {
            'valid': valid,
            'score': overall_score,
            'morphology_score': morphology_score,
            'physiology_score': physiology_score,
            'clinical_score': clinical_score,
            'details': {
                'morphology': morph_result['checks'],
                'physiology': physio_result['checks'],
                'clinical': clinical_result['checks']
            }
        }
    
    def _check_morphology(self, ecg: np.ndarray) -> Dict:
        """Level 1: Morphological checks."""
        checks = {}
        score = 0.0
        max_score = 4.0
        
        # Check 1: Amplitude range
        in_range = np.all((ecg >= self.min_amplitude) & (ecg <= self.max_amplitude))
        checks['amplitude_valid'] = in_range
        if in_range:
            score += 1.0
        
        # Check 2: R-peak detection
        r_peaks = self._detect_r_peaks(ecg)
        checks['num_r_peaks'] = len(r_peaks)
        checks['r_peaks_detected'] = len(r_peaks) >= 3  # At least 3 beats
        if len(r_peaks) >= 3:
            score += 1.0
        
        # Check 3: QRS complexes
        valid_qrs = 0
        if len(r_peaks) > 0:
            for r_idx in r_peaks:
                # Check QRS amplitude
                qrs_amp = ecg[r_idx]
                if abs(qrs_amp) > self.min_qrs_amplitude:
                    valid_qrs += 1
            
            qrs_valid = valid_qrs >= len(r_peaks) * 0.7  # 70% valid
            checks['valid_qrs_complexes'] = qrs_valid
            if qrs_valid:
                score += 1.0
        
        # Check 4: No extreme spikes (outlier detection)
        median = np.median(np.abs(ecg))
        mad = np.median(np.abs(ecg - np.median(ecg)))
        outliers = np.abs(ecg) > (median + 10 * mad)
        no_extreme_spikes = np.sum(outliers) < len(ecg) * 0.01  # <1% outliers
        checks['no_extreme_spikes'] = no_extreme_spikes
        if no_extreme_spikes:
            score += 1.0
        
        return {
            'score': score / max_score,
            'checks': checks,
            'r_peaks': r_peaks
        }
    
    def _check_physiology(self, ecg: np.ndarray, r_peaks: np.ndarray) -> Dict:
        """Level 2: Physiological constraints."""
        checks = {}
        score = 0.0
        max_score = 3.0
        
        if len(r_peaks) < 2:
            return {'score': 0.0, 'checks': checks}
        
        # Compute RR intervals
        rr_intervals = np.diff(r_peaks) / self.fs  # in seconds
        checks['mean_rr'] = float(np.mean(rr_intervals))
        
        # Check 1: RR intervals in valid range
        rr_valid = np.all((rr_intervals >= self.min_rr) & (rr_intervals <= self.max_rr))
        checks['rr_intervals_valid'] = rr_valid
        if rr_valid:
            score += 1.0
        
        # Check 2: Heart rate in valid range
        hr = 60.0 / np.mean(rr_intervals)
        checks['heart_rate'] = float(hr)
        hr_valid = (hr >= self.min_hr) and (hr <= self.max_hr)
        checks['heart_rate_valid'] = hr_valid
        if hr_valid:
            score += 1.0
        
        # Check 3: RR variability not chaotic
        rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
        checks['rr_cv'] = float(rr_cv)
        cv_valid = rr_cv < self.max_rr_cv
        checks['rr_cv_valid'] = cv_valid
        if cv_valid:
            score += 1.0
        
        return {
            'score': score / max_score,
            'checks': checks
        }
    
    def _check_clinical_features(self, 
                                  ecg: np.ndarray,
                                  original_ecg: np.ndarray,
                                  target_class: int,
                                  original_class: Optional[int],
                                  r_peaks: np.ndarray) -> Dict:
        """Level 3: Clinical feature appropriateness."""
        checks = {}
        score = 0.0
        max_score = 3.0
        
        # Detect original R-peaks
        original_r_peaks = self._detect_r_peaks(original_ecg)
        
        if len(r_peaks) < 2 or len(original_r_peaks) < 2:
            return {'score': 0.0, 'checks': checks}
        
        # Compute RR variability
        rr_intervals = np.diff(r_peaks) / self.fs
        original_rr = np.diff(original_r_peaks) / self.fs
        
        rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
        original_rr_cv = np.std(original_rr) / np.mean(original_rr)
        
        checks['original_rr_cv'] = float(original_rr_cv)
        checks['new_rr_cv'] = float(rr_cv)
        
        # Check 1: RR variability changes in correct direction
        if target_class == 1:  # Normal → AFib: RR-CV should increase
            rr_increased = rr_cv > original_rr_cv * 1.3  # At least 30% increase
            checks['rr_direction_correct'] = rr_increased
            if rr_increased:
                score += 1.5  # Important feature
        elif target_class == 0:  # AFib → Normal: RR-CV should decrease
            rr_decreased = rr_cv < original_rr_cv * 0.8  # At least 20% decrease
            checks['rr_direction_correct'] = rr_decreased
            if rr_decreased:
                score += 1.5  # Important feature
        
        # Check 2: Overall morphology similarity
        # Signals should be similar but with rhythm changes
        if len(ecg) == len(original_ecg):
            corr, _ = pearsonr(ecg, original_ecg)
            checks['correlation'] = float(corr)
            good_similarity = corr > 0.6  # Not too different
            checks['good_similarity'] = good_similarity
            if good_similarity:
                score += 1.0
        
        # Check 3: Signal quality preserved
        # Standard deviation should be similar (not collapsed or exploded)
        std_ratio = np.std(ecg) / (np.std(original_ecg) + 1e-8)
        checks['std_ratio'] = float(std_ratio)
        std_preserved = 0.5 < std_ratio < 2.0
        checks['std_preserved'] = std_preserved
        if std_preserved:
            score += 0.5
        
        return {
            'score': score / max_score,
            'checks': checks
        }
    
    def _detect_r_peaks(self, ecg: np.ndarray) -> np.ndarray:
        """
        Detect R-peaks using adaptive threshold method.
        
        Args:
            ecg: ECG signal (1D array)
            
        Returns:
            Array of R-peak indices
        """
        # Preprocess: bandpass filter
        sos = signal.butter(4, [5, 40], btype='band', fs=self.fs, output='sos')
        ecg_filtered = signal.sosfilt(sos, ecg)
        
        # Differentiate to emphasize QRS
        diff_ecg = np.diff(ecg_filtered)
        diff_ecg = np.abs(diff_ecg)
        
        # Find peaks with adaptive threshold
        threshold = np.mean(diff_ecg) + 2 * np.std(diff_ecg)
        min_distance = int(0.3 * self.fs)  # Minimum 300ms between beats
        
        peaks, _ = signal.find_peaks(
            diff_ecg,
            height=threshold,
            distance=min_distance
        )
        
        # Map back to original indices (account for diff)
        return peaks
    
    def compute_plausibility_stats(self, results: list) -> Dict:
        """
        Compute aggregate statistics from multiple validation results.
        
        Args:
            results: List of validation result dictionaries
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {}
        
        scores = [r['score'] for r in results]
        valid_count = sum([r['valid'] for r in results])
        
        # Extract specific checks
        morph_scores = [r['morphology_score'] for r in results]
        physio_scores = [r['physiology_score'] for r in results]
        clinical_scores = [r['clinical_score'] for r in results]
        
        # RR direction correctness (if available)
        rr_correct = []
        for r in results:
            if 'rr_direction_correct' in r['details'].get('clinical', {}):
                rr_correct.append(r['details']['clinical']['rr_direction_correct'])
        
        stats = {
            'total_samples': len(results),
            'valid_samples': valid_count,
            'valid_rate': valid_count / len(results),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'mean_morphology_score': float(np.mean(morph_scores)),
            'mean_physiology_score': float(np.mean(physio_scores)),
            'mean_clinical_score': float(np.mean(clinical_scores)),
        }
        
        if rr_correct:
            stats['rr_direction_correctness'] = sum(rr_correct) / len(rr_correct)
        
        return stats


def test_validator():
    """Quick test of the validator."""
    validator = PlausibilityValidator()
    
    # Create synthetic ECG
    t = np.linspace(0, 10, 2500)
    # Simulate regular rhythm (Normal)
    ecg_normal = np.zeros_like(t)
    for beat_time in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        # Add QRS complex
        qrs_center = int(beat_time * 250)
        qrs = signal.gaussian(50, 5)
        start = max(0, qrs_center - 25)
        end = min(len(ecg_normal), qrs_center + 25)
        ecg_normal[start:end] += qrs[:end-start]
    
    # Simulate irregular rhythm (AFib)
    ecg_afib = np.zeros_like(t)
    beat_times = [1, 1.7, 2.6, 3.3, 4.5, 5.2, 6.3, 7.1, 8.4, 9.2]
    for beat_time in beat_times:
        qrs_center = int(beat_time * 250)
        qrs = signal.gaussian(50, 5)
        start = max(0, qrs_center - 25)
        end = min(len(ecg_afib), qrs_center + 25)
        ecg_afib[start:end] += qrs[:end-start]
    
    # Test Normal ECG
    result = validator.validate(ecg_normal)
    print("Normal ECG Validation:")
    print(f"  Valid: {result['valid']}")
    print(f"  Score: {result['score']:.3f}")
    print(f"  Morphology: {result['morphology_score']:.3f}")
    print(f"  Physiology: {result['physiology_score']:.3f}")
    print(f"  Clinical: {result['clinical_score']:.3f}")
    
    # Test AFib ECG with comparison
    result = validator.validate(
        ecg_afib, 
        original_ecg=ecg_normal,
        target_class=1,  # AFib
        original_class=0  # Normal
    )
    print("\nAFib Counterfactual Validation:")
    print(f"  Valid: {result['valid']}")
    print(f"  Score: {result['score']:.3f}")
    print(f"  RR Direction Correct: {result['details']['clinical'].get('rr_direction_correct', 'N/A')}")


if __name__ == '__main__':
    test_validator()
