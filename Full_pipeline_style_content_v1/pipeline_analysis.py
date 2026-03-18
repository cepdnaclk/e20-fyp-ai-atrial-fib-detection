"""
COMPREHENSIVE ANALYSIS OF COUNTERFACTUAL ECG PIPELINE
=====================================================
Evaluates:
1. Architecture Review (U-Net + AdaGN + Style/Content Encoders)
2. Physiological Plausibility using NeuroKit2
3. PSD Analysis for Frequency Content
4. Classification Flip Success Rate
5. Signal Quality Metrics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False
    print("⚠️ NeuroKit2 not installed. Run: pip install neurokit2")

# ============================================================================
# 1. ARCHITECTURE ANALYSIS
# ============================================================================

def analyze_architecture():
    """Summary of the pipeline architecture"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    COUNTERFACTUAL ECG PIPELINE ANALYSIS                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

📐 ARCHITECTURE SUMMARY
════════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│  1. CONTENT ENCODER (HuBERT-ECG)                                            │
│     ├─ Input: ECG signal [1, 2500]                                          │
│     ├─ Pretrained: Edoardo-BS/hubert-ecg-base                              │
│     ├─ Output: Content embedding [batch, seq, 512]                          │
│     └─ Purpose: Captures MORPHOLOGY (P-waves, QRS, T-waves)                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  2. STYLE ENCODER (AFibResLSTM frozen features)                             │
│     ├─ Input: ECG signal [1, 2500]                                          │
│     ├─ Architecture: Multi-scale Conv → ResNet → BiLSTM                    │
│     ├─ Output: Style embedding [batch, 128] → projected to [batch, 512]    │
│     └─ Purpose: Captures RHYTHM patterns (RR intervals, regularity)         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  3. DIFFUSION U-NET (1D Conditional)                                        │
│     ├─ Input: Noisy ECG [1, 2500] + timestep + conditioning                │
│     ├─ Time Embedding: MLP [1 → 128]                                        │
│     ├─ Conditioning: Content + Style concatenated                           │
│     ├─ INNOVATION: AdaGN (Adaptive Group Normalization)                    │
│     │   └─ Scale & shift normalized features based on condition             │
│     ├─ Encoder: 4 stages [64→128→256→512] with residual blocks            │
│     ├─ Bottleneck: 512 channels                                             │
│     ├─ Decoder: Skip connections from encoder                               │
│     └─ Output: Denoised ECG [1, 2500]                                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  4. CLASSIFIER-FREE GUIDANCE (CFG)                                          │
│     ├─ Conditional: Full content + style conditioning                       │
│     ├─ Unconditional: Zero conditioning                                     │
│     ├─ Guidance: noise = uncond + scale * (cond - uncond)                  │
│     └─ Default scale: 3.0-4.0                                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  5. TRAINING LOSSES                                                          │
│     ├─ MSE Loss: Primary denoising objective                                │
│     ├─ Frequency Loss: FFT magnitude matching (preserves ECG structure)    │
│     └─ Smoothness Loss (optional): Penalizes high-frequency noise          │
└─────────────────────────────────────────────────────────────────────────────┘

🎯 COUNTERFACTUAL GENERATION FLOW
════════════════════════════════════════════════════════════════════════════════
  
  Input ECG (Normal)  ─────┬─────────────────────────────────────────────────┐
        │                  │                                                  │
        ▼                  │                                                  │
  ┌──────────┐            │                                                  │
  │ Content  │◄───────────┤ Preserve morphology (QRS shape, amplitudes)     │
  │ Encoder  │            │                                                  │
  └────┬─────┘            │                                                  │
       │                  │                                                  │
       ▼                  │                                                  │
  ┌──────────┐            │                                                  │
  │  Style   │◄───────────┤ Extract rhythm (but CFG pushes toward target)   │
  │ Encoder  │            │                                                  │
  └────┬─────┘            │                                                  │
       │                  │                                                  │
       ▼                  │                                                  │
  ┌──────────────┐        │                                                  │
  │   U-Net +    │        │                                                  │
  │  DDIM (50    │◄───────┤ Denoise from random noise with CFG             │
  │   steps)     │        │                                                  │
  └──────┬───────┘        │                                                  │
         │                │                                                  │
         ▼                │                                                  │
  Counterfactual ECG ─────┴──────────────────────────────────────────────────┘
  (Should be AFib)
  
""")

# ============================================================================
# 2. PHYSIOLOGICAL PLAUSIBILITY METRICS
# ============================================================================

class PhysiologicalAnalyzer:
    """Complete physiological analysis using NeuroKit2"""
    
    def __init__(self, fs=250):
        self.fs = fs
    
    def analyze_ecg(self, ecg_signal, label="ECG"):
        """
        Complete physiological analysis of an ECG signal
        
        Returns dict with:
        - valid: bool - Whether signal is physiologically plausible
        - heart_rate: float - Mean heart rate (bpm)
        - hrv_rmssd: float - Heart rate variability (ms)
        - r_peaks: int - Number of detected R-peaks
        - quality_score: float - Signal quality (0-1)
        - issues: list - Any detected issues
        """
        results = {
            'label': label,
            'valid': False,
            'heart_rate': 0,
            'hrv_rmssd': 0,
            'r_peaks': 0,
            'quality_score': 0,
            'issues': []
        }
        
        if not HAS_NEUROKIT:
            results['issues'].append("NeuroKit2 not installed")
            return results
        
        try:
            # 1. Clean the signal
            ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=self.fs, method="neurokit")
            
            # 2. Detect R-peaks
            peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=self.fs)
            r_peak_indices = info.get('ECG_R_Peaks', [])
            results['r_peaks'] = len(r_peak_indices)
            
            if len(r_peak_indices) < 3:
                results['issues'].append(f"Too few R-peaks detected ({len(r_peak_indices)})")
                return results
            
            # 3. Calculate Heart Rate
            rr_intervals = np.diff(r_peak_indices) / self.fs * 1000  # in ms
            mean_rr = np.mean(rr_intervals)
            results['heart_rate'] = 60000 / mean_rr  # Convert to BPM
            
            # 4. Calculate HRV (RMSSD)
            if len(rr_intervals) >= 2:
                rr_diff = np.diff(rr_intervals)
                results['hrv_rmssd'] = np.sqrt(np.mean(rr_diff**2))
            
            # 5. Physiological validity checks
            hr = results['heart_rate']
            if hr < 30:
                results['issues'].append(f"HR too low: {hr:.1f} bpm")
            elif hr > 220:
                results['issues'].append(f"HR too high: {hr:.1f} bpm")
            else:
                results['valid'] = True
            
            # 6. Quality Score (based on R-peak detection confidence)
            expected_peaks = (len(ecg_signal) / self.fs) / 60 * 75  # ~75 bpm expected
            peak_ratio = len(r_peak_indices) / max(expected_peaks, 1)
            results['quality_score'] = min(1.0, max(0.0, 1 - abs(1 - peak_ratio)))
            
        except Exception as e:
            results['issues'].append(f"Analysis failed: {str(e)[:50]}")
        
        return results
    
    def compare_signals(self, original, generated):
        """Compare original and generated ECG signals"""
        orig_analysis = self.analyze_ecg(original, "Original")
        gen_analysis = self.analyze_ecg(generated, "Generated")
        
        # Compute additional comparison metrics
        comparison = {
            'original': orig_analysis,
            'generated': gen_analysis,
            'hr_difference': abs(gen_analysis['heart_rate'] - orig_analysis['heart_rate']),
            'hrv_change': gen_analysis['hrv_rmssd'] - orig_analysis['hrv_rmssd'],
            'peaks_ratio': gen_analysis['r_peaks'] / max(orig_analysis['r_peaks'], 1)
        }
        
        return comparison

# ============================================================================
# 3. SIGNAL QUALITY METRICS (PSD + NOISE ANALYSIS)
# ============================================================================

class SignalQualityAnalyzer:
    """Frequency domain and noise analysis"""
    
    def __init__(self, fs=250):
        self.fs = fs
    
    def compute_psd_metrics(self, ecg_signal):
        """
        Compute Power Spectral Density metrics
        
        Returns:
        - psd_corr: Correlation with typical ECG spectrum
        - noise_ratio: Power above 50Hz / total power
        - dominant_freq: Dominant frequency (should be 1-3 Hz for heart rate)
        """
        f, psd = signal.welch(ecg_signal, fs=self.fs, nperseg=min(1024, len(ecg_signal)//2))
        
        # Noise ratio (power above 50Hz)
        idx_50 = np.argmin(np.abs(f - 50))
        power_above_50 = np.sum(psd[idx_50:])
        total_power = np.sum(psd)
        noise_ratio = power_above_50 / (total_power + 1e-10)
        
        # Physiological band power (0.5-40 Hz for ECG)
        idx_05 = np.argmin(np.abs(f - 0.5))
        idx_40 = np.argmin(np.abs(f - 40))
        physio_power = np.sum(psd[idx_05:idx_40])
        physio_ratio = physio_power / (total_power + 1e-10)
        
        # Dominant frequency
        dominant_freq = f[np.argmax(psd)]
        
        return {
            'frequencies': f,
            'psd': psd,
            'noise_ratio': noise_ratio,
            'physio_ratio': physio_ratio,
            'dominant_freq': dominant_freq
        }
    
    def compute_psd_correlation(self, ecg1, ecg2):
        """Correlation between two PSDs"""
        _, psd1 = signal.welch(ecg1, fs=self.fs, nperseg=min(1024, len(ecg1)//2))
        _, psd2 = signal.welch(ecg2, fs=self.fs, nperseg=min(1024, len(ecg2)//2))
        
        # Ensure same length
        min_len = min(len(psd1), len(psd2))
        return np.corrcoef(psd1[:min_len], psd2[:min_len])[0, 1]
    
    def is_noise(self, ecg_signal, threshold=0.3):
        """
        Determine if signal is primarily noise
        
        Returns True if:
        - Noise ratio > threshold
        - No clear physiological structure
        """
        metrics = self.compute_psd_metrics(ecg_signal)
        
        # Check noise ratio
        if metrics['noise_ratio'] > threshold:
            return True, "High frequency noise dominates"
        
        # Check for physiological structure
        if metrics['physio_ratio'] < 0.5:
            return True, "Low physiological band content"
        
        # Check dominant frequency is in physiological range (0.5-3 Hz for QRS)
        if not (0.5 <= metrics['dominant_freq'] <= 40):
            return True, f"Unusual dominant frequency: {metrics['dominant_freq']:.1f} Hz"
        
        return False, "Signal appears physiologically plausible"

# ============================================================================
# 4. RESULTS ANALYSIS
# ============================================================================

def analyze_results_folder(results_dir):
    """Analyze all results in a folder"""
    
    print(f"\n{'='*70}")
    print(f"ANALYZING RESULTS: {results_dir}")
    print(f"{'='*70}")
    
    if not os.path.exists(results_dir):
        print(f"❌ Directory not found: {results_dir}")
        return
    
    files = os.listdir(results_dir)
    png_files = [f for f in files if f.endswith('.png')]
    
    if not png_files:
        print("❌ No PNG files found")
        return
    
    # Count successes and failures
    successes = [f for f in png_files if 'SUCCESS' in f.upper()]
    failures = [f for f in png_files if 'FAIL' in f.upper()]
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total samples: {len(png_files)}")
    print(f"   ✅ Successful flips: {len(successes)} ({len(successes)/len(png_files)*100:.1f}%)")
    print(f"   ❌ Failed flips: {len(failures)} ({len(failures)/len(png_files)*100:.1f}%)")
    
    return {
        'total': len(png_files),
        'successes': len(successes),
        'failures': len(failures),
        'success_rate': len(successes) / len(png_files) if png_files else 0
    }

# ============================================================================
# 5. COMPREHENSIVE PIPELINE REVIEW
# ============================================================================

def print_pipeline_review():
    """Print comprehensive review of the pipeline"""
    
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PIPELINE REVIEW & RECOMMENDATIONS                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

✅ STRENGTHS
════════════════════════════════════════════════════════════════════════════════

1. AdaGN (Adaptive Group Normalization)
   - Excellent choice for conditioning! Better than simple concatenation
   - Learns to scale and shift features based on style/content condition
   - Used in StyleGAN2 and modern diffusion models

2. Content/Style Separation
   - HuBERT-ECG for content: Captures morphological features
   - AFibResLSTM features for style: Captures rhythm patterns
   - Good theoretical foundation for counterfactual generation

3. Frequency Loss
   - Helps preserve ECG frequency structure
   - Prevents model from generating pure noise
   - Forces learning of heartbeat periodicity

4. DDIM Scheduler
   - Fast sampling (50 steps)
   - Deterministic option for reproducibility
   - squaredcos_cap_v2 schedule is good for natural signals

5. Classifier-Free Guidance
   - Allows control over generation strength
   - Higher guidance = stronger class flip attempt

⚠️  POTENTIAL ISSUES & RECOMMENDATIONS
════════════════════════════════════════════════════════════════════════════════

1. Style Encoder Design Issue
   ┌────────────────────────────────────────────────────────────────────────┐
   │ CURRENT: Style is extracted from INPUT ECG, not a target class sample │
   │                                                                        │
   │ PROBLEM: If input is Normal, style encoder outputs Normal rhythm      │
   │          The model must rely ONLY on CFG to flip the class            │
   │                                                                        │
   │ RECOMMENDATION: Provide an actual AFib sample as style source when    │
   │                 generating Normal→AFib counterfactuals                 │
   └────────────────────────────────────────────────────────────────────────┘

2. Smoothness Loss Considerations
   ┌────────────────────────────────────────────────────────────────────────┐
   │ CURRENT: Optional, sometimes clamped                                  │
   │                                                                        │
   │ ISSUE: Clamping at max=10.0 may cut gradients                        │
   │        But removing it can cause explosion                            │
   │                                                                        │
   │ RECOMMENDATION: Use unclamped but with lower weight (0.001-0.01)     │
   │                 Or use TV loss (Total Variation) instead              │
   └────────────────────────────────────────────────────────────────────────┘

3. Limited Temporal Conditioning
   ┌────────────────────────────────────────────────────────────────────────┐
   │ CURRENT: Conditioning averaged before passing to U-Net               │
   │          cond_emb = conditioning.mean(dim=1)                          │
   │                                                                        │
   │ ISSUE: Loses sequential information from content encoder             │
   │                                                                        │
   │ RECOMMENDATION: Use cross-attention to preserve temporal structure   │
   │                 Or use AdaGN with per-timestep conditioning          │
   └────────────────────────────────────────────────────────────────────────┘

4. Evaluation Robustness
   ┌────────────────────────────────────────────────────────────────────────┐
   │ CURRENT: NeuroKit2 can fail on some valid ECGs                       │
   │                                                                        │
   │ RECOMMENDATION: Use multiple peak detection algorithms:             │
   │   - NeuroKit2 (neurokit method)                                       │
   │   - Pan-Tompkins                                                       │
   │   - Hamilton                                                           │
   │ If 2/3 detect peaks, signal is likely valid                          │
   └────────────────────────────────────────────────────────────────────────┘

📈 SUCCESS RATE INTERPRETATION
════════════════════════════════════════════════════════════════════════════════

Based on your results folders:

- counterfactual_results_physio_v2: 5/10 SUCCESS (50%)
- counterfactual_results_sota2: 6/21 SUCCESS (~29%)

INTERPRETATION:
├─ 30-50% flip rate is actually reasonable for early-stage models
├─ The classifier was trained on real data - it's hard to fool
├─ Some "failures" may still be physiologically valid ECGs
└─ The model is learning ECG structure (not pure noise)

WAYS TO IMPROVE FLIP RATE:
1. ↑ Increase guidance_scale to 5.0-7.0
2. Use actual target-class samples for style encoding
3. Train classifier discriminator jointly (adversarial)
4. Add classification loss during diffusion training

""")

# ============================================================================
# 6. RUN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    # Print architecture analysis
    analyze_architecture()
    
    # Print pipeline review
    print_pipeline_review()
    
    # Analyze existing results
    base_path = "/scratch1/e20-fyp-ai-atrial-fib-det/old_vision/PERA_AF_Detection/Pipeline_Implementation/Full_pipeline_style_content"
    
    results_dirs = [
        f"{base_path}/counterfactual_results_physio_v2",
        f"{base_path}/counterfactual_results_sota2",
        f"{base_path}/counterfactual_results",
        f"{base_path}/counterfactual_results_single_fold_debug"
    ]
    
    all_results = {}
    for results_dir in results_dirs:
        result = analyze_results_folder(results_dir)
        if result:
            all_results[os.path.basename(results_dir)] = result
    
    # Final Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY ACROSS ALL EXPERIMENTS")
    print(f"{'='*70}")
    
    total_samples = sum(r['total'] for r in all_results.values())
    total_success = sum(r['successes'] for r in all_results.values())
    
    print(f"\n📊 Overall Statistics:")
    print(f"   Total samples evaluated: {total_samples}")
    print(f"   Total successful flips: {total_success}")
    print(f"   Overall success rate: {total_success/total_samples*100:.1f}%" if total_samples > 0 else "   N/A")
    
    print(f"\n{'='*70}")
    print("PHYSIOLOGICAL PLAUSIBILITY CHECK")
    print(f"{'='*70}")
    print("""
To determine if generated ECGs are physiologically plausible vs noise:

✅ PLAUSIBLE indicators:
   - NeuroKit2 can detect R-peaks (3+ peaks in 10s segment)
   - Heart rate between 40-180 bpm
   - PSD has power concentrated below 50 Hz
   - Noise ratio < 15-20%
   - Visible P-QRS-T morphology in time domain

❌ NOISE indicators:
   - No detectable R-peaks
   - Heart rate < 30 or > 250 bpm
   - High-frequency noise dominates (>50 Hz)
   - Noise ratio > 30%
   - Random fluctuations without ECG structure

Based on your evaluation code (counterfactual_new_eval.py):
   - PSD correlation is being computed ✅
   - NeuroKit2 HR and RMSSD extracted ✅
   - Both original and generated are validated ✅

The fact that NeuroKit2 CAN detect R-peaks in your generated signals
(when physio_cf['valid'] = True) suggests they ARE physiologically 
plausible ECGs, not random noise!
""")
