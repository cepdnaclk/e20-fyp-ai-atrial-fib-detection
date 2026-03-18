"""
DETAILED PHYSIOLOGICAL PLAUSIBILITY EVALUATION
===============================================
Actually generates counterfactuals and evaluates them with NeuroKit2
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

import neurokit2 as nk

# Import models
from counterfactual_training import (
    UNet1DConditional,
    StyleEncoderWrapper,
    PretrainedContentEncoder,
    AFibResLSTM,
    ModelConfig
)
from diffusers import DDIMScheduler

# ============================================================================
# COMPREHENSIVE PHYSIOLOGICAL VALIDATOR
# ============================================================================

class ComprehensiveValidator:
    """Multi-method ECG validation"""
    
    def __init__(self, fs=250):
        self.fs = fs
    
    def validate_signal(self, ecg_signal, label="ECG"):
        """
        Comprehensive validation using multiple methods
        """
        results = {
            'label': label,
            'is_noise': False,
            'is_physiological': False,
            'metrics': {},
            'checks': {}
        }
        
        # ============================================
        # CHECK 1: Basic Statistics
        # ============================================
        results['metrics']['mean'] = float(np.mean(ecg_signal))
        results['metrics']['std'] = float(np.std(ecg_signal))
        results['metrics']['min'] = float(np.min(ecg_signal))
        results['metrics']['max'] = float(np.max(ecg_signal))
        
        # Check for flat signal or extreme values
        if results['metrics']['std'] < 0.01:
            results['checks']['variance'] = "❌ FLAT SIGNAL"
            results['is_noise'] = True
        elif results['metrics']['std'] > 100:
            results['checks']['variance'] = "⚠️ HIGH VARIANCE"
        else:
            results['checks']['variance'] = "✅ NORMAL"
        
        # ============================================
        # CHECK 2: Power Spectral Density
        # ============================================
        f, psd = signal.welch(ecg_signal, fs=self.fs, nperseg=min(1024, len(ecg_signal)//2))
        
        # Noise ratio (power above 50Hz)
        idx_50 = np.argmin(np.abs(f - 50))
        power_above_50 = np.sum(psd[idx_50:])
        total_power = np.sum(psd)
        noise_ratio = power_above_50 / (total_power + 1e-10)
        
        results['metrics']['noise_ratio'] = float(noise_ratio)
        results['metrics']['dominant_freq'] = float(f[np.argmax(psd)])
        
        if noise_ratio > 0.3:
            results['checks']['frequency'] = f"❌ HIGH NOISE ({noise_ratio*100:.1f}%)"
            results['is_noise'] = True
        elif noise_ratio > 0.15:
            results['checks']['frequency'] = f"⚠️ MODERATE NOISE ({noise_ratio*100:.1f}%)"
        else:
            results['checks']['frequency'] = f"✅ LOW NOISE ({noise_ratio*100:.1f}%)"
        
        # ============================================
        # CHECK 3: NeuroKit2 R-Peak Detection
        # ============================================
        try:
            ecg_clean = nk.ecg_clean(ecg_signal, sampling_rate=self.fs, method="neurokit")
            peaks, info = nk.ecg_peaks(ecg_clean, sampling_rate=self.fs)
            r_peaks = info.get('ECG_R_Peaks', [])
            
            results['metrics']['r_peaks'] = len(r_peaks)
            
            if len(r_peaks) >= 3:
                # Calculate HR
                rr_intervals = np.diff(r_peaks) / self.fs * 1000
                mean_rr = np.mean(rr_intervals)
                hr = 60000 / mean_rr
                results['metrics']['heart_rate'] = float(hr)
                
                # Calculate HRV (RMSSD)
                if len(rr_intervals) >= 2:
                    rr_diff = np.diff(rr_intervals)
                    rmssd = np.sqrt(np.mean(rr_diff**2))
                    results['metrics']['hrv_rmssd'] = float(rmssd)
                
                # Validate HR range
                if 40 <= hr <= 200:
                    results['checks']['heart_rate'] = f"✅ NORMAL HR ({hr:.1f} bpm)"
                    results['is_physiological'] = True
                else:
                    results['checks']['heart_rate'] = f"⚠️ UNUSUAL HR ({hr:.1f} bpm)"
            else:
                results['checks']['heart_rate'] = f"❌ TOO FEW R-PEAKS ({len(r_peaks)})"
                results['is_noise'] = True
                
        except Exception as e:
            results['checks']['heart_rate'] = f"❌ ANALYSIS FAILED"
            results['is_noise'] = True
        
        # ============================================
        # CHECK 4: Try alternative peak detection
        # ============================================
        try:
            # Pan-Tompkins method
            _, info_pt = nk.ecg_peaks(ecg_signal, sampling_rate=self.fs, method="pantompkins1985")
            pt_peaks = len(info_pt.get('ECG_R_Peaks', []))
            results['metrics']['pantompkins_peaks'] = pt_peaks
            
            if pt_peaks >= 3:
                results['checks']['alt_detection'] = f"✅ Pan-Tompkins found {pt_peaks} peaks"
            else:
                results['checks']['alt_detection'] = f"⚠️ Pan-Tompkins found only {pt_peaks} peaks"
        except:
            results['checks']['alt_detection'] = "⚠️ Pan-Tompkins failed"
        
        # ============================================
        # FINAL VERDICT
        # ============================================
        if results['is_physiological'] and not results['is_noise']:
            results['verdict'] = "✅ PHYSIOLOGICALLY PLAUSIBLE ECG"
        elif results['is_noise']:
            results['verdict'] = "❌ LIKELY NOISE / NOT ECG"
        else:
            results['verdict'] = "⚠️ UNCERTAIN - NEEDS MANUAL REVIEW"
        
        return results

def run_evaluation(num_samples=10):
    """Run evaluation on actual data"""
    
    print("\n" + "="*70)
    print("PHYSIOLOGICAL PLAUSIBILITY EVALUATION WITH NEUROKIT2")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Paths
    BASE = "/scratch1/e20-fyp-ai-atrial-fib-det/old_vision/PERA_AF_Detection/Pipeline_Implementation/Full_pipeline_style_content"
    UNET_PATH = f"{BASE}/single_fold_debug/best_model.pth"
    CLF_PATH = f"{BASE}/best_model/best_model.pth"
    DATA_PATH = f"{BASE}/ecg_afib_data/X_combined.npy"
    
    # Check paths
    for path, name in [(UNET_PATH, "U-Net"), (CLF_PATH, "Classifier"), (DATA_PATH, "Data")]:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name} NOT FOUND: {path}")
            return
    
    # Load models
    print("\n📦 Loading models...")
    
    # U-Net
    unet = UNet1DConditional().to(device)
    checkpoint = torch.load(UNET_PATH, map_location=device)
    unet.load_state_dict(checkpoint['unet_state_dict'])
    unet.eval()
    print("   ✅ U-Net loaded")
    
    # Style components
    style_net = StyleEncoderWrapper(CLF_PATH, device)
    style_proj = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 512)).to(device)
    style_proj.load_state_dict(checkpoint['style_proj_state_dict'])
    style_proj.eval()
    print("   ✅ Style encoder loaded")
    
    # Content encoder
    content_net = PretrainedContentEncoder().to(device)
    content_net.eval()
    print("   ✅ Content encoder loaded")
    
    # Classifier
    classifier = AFibResLSTM(ModelConfig()).to(device)
    clf_checkpoint = torch.load(CLF_PATH, map_location=device)
    classifier.load_state_dict(clf_checkpoint['model_state_dict'])
    classifier.eval()
    print("   ✅ Classifier loaded")
    
    # Scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=False
    )
    
    # Load data
    print("\n📂 Loading data...")
    signals = np.load(DATA_PATH)
    
    # Try to load normalization params
    try:
        stats = np.load(f"{BASE}/single_fold_debug/norm_params.npy", allow_pickle=True).item()
        means = stats['means']
        stds = stats['stds']
    except:
        print("   ⚠️ Using per-sample normalization")
        means = np.mean(signals, axis=1)
        stds = np.std(signals, axis=1)
    
    print(f"   Dataset size: {len(signals)}")
    
    # Initialize validator
    validator = ComprehensiveValidator(fs=250)
    
    # Select random samples
    np.random.seed(42)  # For reproducibility
    sample_indices = np.random.choice(len(signals), num_samples, replace=False)
    
    # Results storage
    all_results = []
    
    print("\n" + "="*70)
    print("GENERATING AND EVALUATING COUNTERFACTUALS")
    print("="*70)
    
    for i, idx in enumerate(sample_indices):
        print(f"\n{'─'*70}")
        print(f"SAMPLE {i+1}/{num_samples} (Index: {idx})")
        print(f"{'─'*70}")
        
        # Get raw signal
        raw_signal = signals[idx]
        mean, std = means[idx], stds[idx]
        
        # Normalize
        normalized = (raw_signal - mean) / (std + 1e-6)
        normalized = normalized.reshape(1, -1)
        
        # Validate original
        print("\n📊 ORIGINAL ECG:")
        orig_validation = validator.validate_signal(raw_signal, "Original")
        for check, status in orig_validation['checks'].items():
            print(f"   {check}: {status}")
        print(f"   VERDICT: {orig_validation['verdict']}")
        
        # Classify original
        with torch.no_grad():
            x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = classifier(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            orig_class = np.argmax(probs)
            orig_conf = probs[orig_class]
        
        print(f"   Classification: {'AFib' if orig_class == 1 else 'Normal'} ({orig_conf*100:.1f}%)")
        
        # Generate counterfactual
        target_class = 1 - orig_class
        print(f"\n🔮 GENERATING COUNTERFACTUAL (Target: {'AFib' if target_class == 1 else 'Normal'})...")
        
        with torch.no_grad():
            input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Extract features
            content = content_net(input_tensor)
            style = style_net(input_tensor)
            style_emb = style_proj(style).unsqueeze(1)
            
            # Conditioning
            cond = torch.cat([content, style_emb], dim=1)
            uncond = torch.zeros_like(cond)
            
            # DDIM sampling
            scheduler.set_timesteps(50)
            latents = torch.randn_like(input_tensor)
            
            guidance_scale = 4.0
            
            for t in scheduler.timesteps:
                latent_input = torch.cat([latents] * 2)
                t_input = torch.cat([t.unsqueeze(0).to(device)] * 2)
                cond_input = torch.cat([cond, uncond])
                
                noise_pred = unet(latent_input, t_input, cond_input)
                noise_cond, noise_uncond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            cf_normalized = latents.squeeze().cpu().numpy()
        
        # Denormalize
        cf_raw = cf_normalized * std + mean
        
        # Validate counterfactual
        print("\n📊 COUNTERFACTUAL ECG:")
        cf_validation = validator.validate_signal(cf_raw, "Counterfactual")
        for check, status in cf_validation['checks'].items():
            print(f"   {check}: {status}")
        print(f"   VERDICT: {cf_validation['verdict']}")
        
        # Classify counterfactual
        with torch.no_grad():
            cf_tensor = torch.tensor(cf_normalized.reshape(1, -1), dtype=torch.float32).unsqueeze(0).to(device)
            logits, _ = classifier(cf_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            cf_class = np.argmax(probs)
            cf_conf = probs[cf_class]
        
        flip_success = (cf_class == target_class)
        print(f"   Classification: {'AFib' if cf_class == 1 else 'Normal'} ({cf_conf*100:.1f}%)")
        print(f"   FLIP: {'✅ SUCCESS' if flip_success else '❌ FAILED'}")
        
        # PSD Correlation
        _, psd_orig = signal.welch(raw_signal, fs=250, nperseg=512)
        _, psd_cf = signal.welch(cf_raw, fs=250, nperseg=512)
        psd_corr = np.corrcoef(psd_orig, psd_cf)[0, 1]
        print(f"\n   PSD Correlation: {psd_corr:.4f}")
        
        # Store results
        all_results.append({
            'idx': idx,
            'orig_class': orig_class,
            'cf_class': cf_class,
            'target_class': target_class,
            'flip_success': flip_success,
            'orig_physiological': orig_validation['is_physiological'],
            'cf_physiological': cf_validation['is_physiological'],
            'orig_noise': orig_validation['is_noise'],
            'cf_noise': cf_validation['is_noise'],
            'psd_corr': psd_corr,
            'orig_hr': orig_validation['metrics'].get('heart_rate', 0),
            'cf_hr': cf_validation['metrics'].get('heart_rate', 0),
            'orig_rmssd': orig_validation['metrics'].get('hrv_rmssd', 0),
            'cf_rmssd': cf_validation['metrics'].get('hrv_rmssd', 0),
        })
    
    # ============================================
    # SUMMARY STATISTICS
    # ============================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    total = len(all_results)
    flips = sum(1 for r in all_results if r['flip_success'])
    physio_cf = sum(1 for r in all_results if r['cf_physiological'])
    noise_cf = sum(1 for r in all_results if r['cf_noise'])
    
    print(f"""
📊 OVERALL RESULTS ({total} samples)
═══════════════════════════════════════════════════════════════════════════

CLASSIFICATION FLIP SUCCESS:
   ✅ Successful flips: {flips}/{total} ({flips/total*100:.1f}%)
   ❌ Failed flips: {total-flips}/{total} ({(total-flips)/total*100:.1f}%)

PHYSIOLOGICAL PLAUSIBILITY:
   ✅ Physiologically valid counterfactuals: {physio_cf}/{total} ({physio_cf/total*100:.1f}%)
   ❌ Likely noise/invalid: {noise_cf}/{total} ({noise_cf/total*100:.1f}%)
   ⚠️  Uncertain: {total - physio_cf - noise_cf}/{total}

PSD CORRELATION (Signal Fidelity):
   Mean: {np.mean([r['psd_corr'] for r in all_results]):.4f}
   Min:  {np.min([r['psd_corr'] for r in all_results]):.4f}
   Max:  {np.max([r['psd_corr'] for r in all_results]):.4f}

HEART RATE ANALYSIS:
   Original mean HR: {np.mean([r['orig_hr'] for r in all_results if r['orig_hr'] > 0]):.1f} bpm
   Generated mean HR: {np.mean([r['cf_hr'] for r in all_results if r['cf_hr'] > 0]):.1f} bpm

═══════════════════════════════════════════════════════════════════════════

INTERPRETATION:
""")
    
    if physio_cf / total > 0.7:
        print("   ✅ EXCELLENT: Most generated ECGs are physiologically plausible!")
    elif physio_cf / total > 0.4:
        print("   ⚠️  MODERATE: Some generated ECGs are valid, but quality is inconsistent")
    else:
        print("   ❌ POOR: Many generated ECGs are not physiologically valid")
    
    if noise_cf / total > 0.5:
        print("   ⚠️  HIGH NOISE: Model may need more training or architecture changes")
    else:
        print("   ✅ LOW NOISE: Generated signals have ECG-like structure")
    
    avg_psd = np.mean([r['psd_corr'] for r in all_results])
    if avg_psd > 0.8:
        print("   ✅ HIGH FIDELITY: Generated ECGs preserve frequency content well")
    elif avg_psd > 0.5:
        print("   ⚠️  MODERATE FIDELITY: Some frequency content preserved")
    else:
        print("   ❌ LOW FIDELITY: Significant frequency distortion")
    
    return all_results

if __name__ == "__main__":
    results = run_evaluation(num_samples=15)
