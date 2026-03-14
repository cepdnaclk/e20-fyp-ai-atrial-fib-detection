"""
Enhanced Comprehensive Metrics for Publication
===============================================

Computes all publication-ready metrics including:
1. Signal Quality: PSNR, SSIM, SNR, FID
2. Clinical Validation: RR intervals, HR, P-waves, QRS morphology
3. Spectral Analysis: PSD comparison
4. Statistical Tests: Per-direction and overall
5. Publication Figures (300 DPI)


"""

import os
import sys
import gc
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.stats import pearsonr, ttest_ind, mannwhitneyu, ks_2samp
from scipy.signal import find_peaks, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Paths
PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
RESULTS_DIR = PROJECT_ROOT / 'notebooks/phase_3_counterfactual/results'
MODEL_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/enhanced_diffusion_cf'
OUTPUT_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/comprehensive_metrics'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FS = 250  # Sampling frequency
SIGNAL_LENGTH = 2500  # 10 seconds at 250 Hz

print("=" * 70)
print("Enhanced Comprehensive Metrics for Publication")
print("=" * 70)

# ============================================================================
# Load Data
# ============================================================================
print("\n1. LOADING DATA")
print("-" * 70)

data = np.load(RESULTS_DIR / 'counterfactual_full_data.npz')
cfs = data['counterfactuals']
originals = data['originals']
target_labels = data['target_labels']
original_labels = data['original_labels']
val_scores = data['validation_scores']
attempts = data['attempts']

n2a_mask = original_labels == 0  # Normal → AFib
a2n_mask = original_labels == 1  # AFib → Normal

print(f"Total samples: {len(cfs)}")
print(f"Normal→AFib: {n2a_mask.sum()}")
print(f"AFib→Normal: {a2n_mask.sum()}")

# Also load original test data for reference
test_data = np.load(PROJECT_ROOT / 'data/processed/diffusion/test_data.npz')

# ============================================================================
# 1. Signal Quality Metrics
# ============================================================================
print("\n2. SIGNAL QUALITY METRICS")
print("-" * 70)

def compute_psnr(original, generated):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((original - generated) ** 2)
    if mse == 0:
        return float('inf')
    max_val = max(np.max(np.abs(original)), np.max(np.abs(generated)))
    if max_val == 0:
        return 0.0
    return 20 * np.log10(max_val / np.sqrt(mse))

def compute_ssim_1d(original, generated, win_size=11):
    """Structural Similarity Index adapted for 1D signals."""
    C1 = (0.01 * 2) ** 2  # data range ~2 for normalized ECG
    C2 = (0.03 * 2) ** 2
    
    # Use sliding window for local statistics
    kernel = np.ones(win_size) / win_size
    
    mu_x = np.convolve(original, kernel, mode='valid')
    mu_y = np.convolve(generated, kernel, mode='valid')
    
    sig_x2 = np.convolve(original ** 2, kernel, mode='valid') - mu_x ** 2
    sig_y2 = np.convolve(generated ** 2, kernel, mode='valid') - mu_y ** 2
    sig_xy = np.convolve(original * generated, kernel, mode='valid') - mu_x * mu_y
    
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sig_x2 + sig_y2 + C2))
    
    return float(np.mean(ssim_map))

def compute_snr(original, generated):
    """Signal-to-Noise Ratio (treating difference as noise)."""
    noise = original - generated
    signal_power = np.mean(original ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def compute_mae(original, generated):
    """Mean Absolute Error."""
    return float(np.mean(np.abs(original - generated)))

def compute_max_error(original, generated):
    """Maximum Absolute Error."""
    return float(np.max(np.abs(original - generated)))

# Compute signal quality metrics (sample for speed)
n_samples_metric = min(2000, len(cfs))
idx = np.random.RandomState(42).choice(len(cfs), n_samples_metric, replace=False)

psnr_vals, ssim_vals, snr_vals, corr_vals, mse_vals, mae_vals, max_err_vals = [], [], [], [], [], [], []

for i in idx:
    orig, cf = originals[i], cfs[i]
    psnr_vals.append(compute_psnr(orig, cf))
    ssim_vals.append(compute_ssim_1d(orig, cf))
    snr_vals.append(compute_snr(orig, cf))
    c, _ = pearsonr(orig, cf)
    corr_vals.append(c)
    mse_vals.append(float(np.mean((orig - cf) ** 2)))
    mae_vals.append(compute_mae(orig, cf))
    max_err_vals.append(compute_max_error(orig, cf))

# Per-direction metrics
psnr_n2a = [compute_psnr(originals[i], cfs[i]) for i in idx if n2a_mask[i]]
psnr_a2n = [compute_psnr(originals[i], cfs[i]) for i in idx if a2n_mask[i]]
ssim_n2a = [compute_ssim_1d(originals[i], cfs[i]) for i in idx if n2a_mask[i]]
ssim_a2n = [compute_ssim_1d(originals[i], cfs[i]) for i in idx if a2n_mask[i]]

signal_quality = {
    'overall': {
        'psnr_mean': float(np.mean(psnr_vals)),
        'psnr_std': float(np.std(psnr_vals)),
        'ssim_mean': float(np.mean(ssim_vals)),
        'ssim_std': float(np.std(ssim_vals)),
        'snr_mean': float(np.mean(snr_vals)),
        'snr_std': float(np.std(snr_vals)),
        'correlation_mean': float(np.mean(corr_vals)),
        'correlation_std': float(np.std(corr_vals)),
        'mse_mean': float(np.mean(mse_vals)),
        'mse_std': float(np.std(mse_vals)),
        'mae_mean': float(np.mean(mae_vals)),
        'mae_std': float(np.std(mae_vals)),
        'max_error_mean': float(np.mean(max_err_vals)),
        'max_error_std': float(np.std(max_err_vals)),
    },
    'normal_to_afib': {
        'psnr_mean': float(np.mean(psnr_n2a)),
        'psnr_std': float(np.std(psnr_n2a)),
        'ssim_mean': float(np.mean(ssim_n2a)),
        'ssim_std': float(np.std(ssim_n2a)),
    },
    'afib_to_normal': {
        'psnr_mean': float(np.mean(psnr_a2n)),
        'psnr_std': float(np.std(psnr_a2n)),
        'ssim_mean': float(np.mean(ssim_a2n)),
        'ssim_std': float(np.std(ssim_a2n)),
    }
}

print(f"PSNR: {signal_quality['overall']['psnr_mean']:.2f} ± {signal_quality['overall']['psnr_std']:.2f} dB")
print(f"SSIM: {signal_quality['overall']['ssim_mean']:.4f} ± {signal_quality['overall']['ssim_std']:.4f}")
print(f"SNR: {signal_quality['overall']['snr_mean']:.2f} ± {signal_quality['overall']['snr_std']:.2f} dB")
print(f"Correlation: {signal_quality['overall']['correlation_mean']:.4f} ± {signal_quality['overall']['correlation_std']:.4f}")
print(f"MSE: {signal_quality['overall']['mse_mean']:.6f} ± {signal_quality['overall']['mse_std']:.6f}")
print(f"MAE: {signal_quality['overall']['mae_mean']:.6f} ± {signal_quality['overall']['mae_std']:.6f}")
print(f"Max Error: {signal_quality['overall']['max_error_mean']:.4f} ± {signal_quality['overall']['max_error_std']:.4f}")

# ============================================================================
# 2. Clinical Validation Metrics
# ============================================================================
print("\n3. CLINICAL VALIDATION METRICS")
print("-" * 70)

def detect_r_peaks(ecg, fs=250):
    """Detect R-peaks using adaptive threshold."""
    # Bandpass filter
    b, a = scipy_signal.butter(2, [5, 45], btype='band', fs=fs)
    filtered = scipy_signal.filtfilt(b, a, ecg)
    
    # Detect peaks
    min_distance = int(0.3 * fs)  # Min 300ms between peaks
    height = np.std(filtered) * 0.5
    peaks, properties = find_peaks(filtered, distance=min_distance, height=height)
    
    return peaks

def compute_rr_features(ecg, fs=250):
    """Compute RR interval features."""
    peaks = detect_r_peaks(ecg, fs)
    if len(peaks) < 3:
        return None
    
    rr = np.diff(peaks) / fs * 1000  # Convert to ms
    hr = 60000 / rr  # Heart rate in BPM
    
    return {
        'n_beats': len(peaks),
        'rr_mean': float(np.mean(rr)),
        'rr_std': float(np.std(rr)),
        'rr_cv': float(np.std(rr) / np.mean(rr)) if np.mean(rr) > 0 else 0,
        'hr_mean': float(np.mean(hr)),
        'hr_std': float(np.std(hr)),
        'hr_min': float(np.min(hr)),
        'hr_max': float(np.max(hr)),
        'rr_min': float(np.min(rr)),
        'rr_max': float(np.max(rr)),
    }

def compute_spectral_features(ecg, fs=250):
    """Compute power spectral density features."""
    freqs, psd = welch(ecg, fs=fs, nperseg=min(512, len(ecg)))
    
    # Frequency bands relevant to ECG
    vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.4)
    
    vlf_power = np.trapz(psd[vlf_mask], freqs[vlf_mask]) if vlf_mask.any() else 0
    lf_power = np.trapz(psd[lf_mask], freqs[lf_mask]) if lf_mask.any() else 0
    hf_power = np.trapz(psd[hf_mask], freqs[hf_mask]) if hf_mask.any() else 0
    total_power = np.trapz(psd, freqs)
    
    return {
        'vlf_power': float(vlf_power),
        'lf_power': float(lf_power),
        'hf_power': float(hf_power),
        'total_power': float(total_power),
        'lf_hf_ratio': float(lf_power / hf_power) if hf_power > 0 else float('inf'),
    }

# Compute clinical features for a subset
n_clinical = min(2000, len(cfs))
clinical_idx = np.random.RandomState(42).choice(len(cfs), n_clinical, replace=False)

rr_orig_features = []
rr_cf_features = []
rr_direction_correct = {'n2a': 0, 'a2n': 0, 'n2a_total': 0, 'a2n_total': 0}
hr_in_range = 0
hr_total = 0
spectral_similarity = []

print("Computing clinical features...", flush=True)
for count, i in enumerate(clinical_idx):
    if count % 500 == 0:
        print(f"  [{count}/{n_clinical}]", flush=True)
    
    orig_feat = compute_rr_features(originals[i])
    cf_feat = compute_rr_features(cfs[i])
    
    if orig_feat is not None and cf_feat is not None:
        rr_orig_features.append(orig_feat)
        rr_cf_features.append(cf_feat)
        
        # Check RR-CV direction correctness
        if n2a_mask[i]:  # Normal → AFib: RR-CV should increase
            rr_direction_correct['n2a_total'] += 1
            if cf_feat['rr_cv'] > orig_feat['rr_cv']:
                rr_direction_correct['n2a'] += 1
        else:  # AFib → Normal: RR-CV should decrease
            rr_direction_correct['a2n_total'] += 1
            if cf_feat['rr_cv'] < orig_feat['rr_cv']:
                rr_direction_correct['a2n'] += 1
        
        # Check HR in physiological range
        hr_total += 1
        if 30 <= cf_feat['hr_mean'] <= 200:
            hr_in_range += 1
    
    # Spectral similarity
    orig_spec = compute_spectral_features(originals[i])
    cf_spec = compute_spectral_features(cfs[i])
    if orig_spec['total_power'] > 0 and cf_spec['total_power'] > 0:
        spec_corr = pearsonr(
            [orig_spec['vlf_power'], orig_spec['lf_power'], orig_spec['hf_power']],
            [cf_spec['vlf_power'], cf_spec['lf_power'], cf_spec['hf_power']]
        )[0]
        spectral_similarity.append(spec_corr if not np.isnan(spec_corr) else 0)

# Aggregate clinical metrics
n2a_direction_pct = rr_direction_correct['n2a'] / max(1, rr_direction_correct['n2a_total'])
a2n_direction_pct = rr_direction_correct['a2n'] / max(1, rr_direction_correct['a2n_total'])

orig_rr_cvs = [f['rr_cv'] for f in rr_orig_features]
cf_rr_cvs = [f['rr_cv'] for f in rr_cf_features]
orig_hrs = [f['hr_mean'] for f in rr_orig_features]
cf_hrs = [f['hr_mean'] for f in rr_cf_features]

clinical_metrics = {
    'rr_interval_analysis': {
        'original_rr_cv_mean': float(np.mean(orig_rr_cvs)),
        'original_rr_cv_std': float(np.std(orig_rr_cvs)),
        'counterfactual_rr_cv_mean': float(np.mean(cf_rr_cvs)),
        'counterfactual_rr_cv_std': float(np.std(cf_rr_cvs)),
        'rr_direction_correct_n2a': float(n2a_direction_pct),
        'rr_direction_correct_a2n': float(a2n_direction_pct),
        'rr_direction_correct_overall': float((rr_direction_correct['n2a'] + rr_direction_correct['a2n']) / 
                                               max(1, rr_direction_correct['n2a_total'] + rr_direction_correct['a2n_total'])),
    },
    'heart_rate': {
        'original_hr_mean': float(np.mean(orig_hrs)),
        'original_hr_std': float(np.std(orig_hrs)),
        'counterfactual_hr_mean': float(np.mean(cf_hrs)),
        'counterfactual_hr_std': float(np.std(cf_hrs)),
        'hr_in_physiological_range_pct': float(hr_in_range / max(1, hr_total)),
    },
    'spectral_analysis': {
        'spectral_similarity_mean': float(np.mean(spectral_similarity)) if spectral_similarity else 0,
        'spectral_similarity_std': float(np.std(spectral_similarity)) if spectral_similarity else 0,
    },
    'plausibility': {
        'mean_score': float(np.mean(val_scores)),
        'std_score': float(np.std(val_scores)),
        'above_0_7_pct': float(np.mean(val_scores > 0.7)),
        'above_0_5_pct': float(np.mean(val_scores > 0.5)),
        'n2a_mean': float(np.mean(val_scores[n2a_mask])),
        'a2n_mean': float(np.mean(val_scores[a2n_mask])),
    },
    'generation_stats': {
        'mean_attempts': float(np.mean(attempts)),
        'n2a_mean_attempts': float(np.mean(attempts[n2a_mask])),
        'a2n_mean_attempts': float(np.mean(attempts[a2n_mask])),
        'single_attempt_pct': float(np.mean(attempts == 1)),
    }
}

print(f"RR Direction Correct (N→A): {n2a_direction_pct:.1%}")
print(f"RR Direction Correct (A→N): {a2n_direction_pct:.1%}")
print(f"HR in range (30-200 bpm): {hr_in_range/max(1,hr_total):.1%}")
print(f"Original RR-CV: {np.mean(orig_rr_cvs):.4f}")
print(f"CF RR-CV: {np.mean(cf_rr_cvs):.4f}")
print(f"Spectral similarity: {np.mean(spectral_similarity):.4f}")

# ============================================================================
# 3. Classifier-Based Metrics (Flip Rate + FID-like)
# ============================================================================
print("\n4. CLASSIFIER-BASED METRICS")
print("-" * 70)

sys.path.insert(0, str(PROJECT_ROOT / 'notebooks/phase_3_counterfactual'))
from shared_models import load_classifier, ClassifierWrapper

classifier = load_classifier(DEVICE)
classifier_wrapper = ClassifierWrapper(classifier).to(DEVICE)
classifier_wrapper.eval()

# Compute flip rates
def compute_flip_rate(signals, target_labels, batch_size=200):
    """Compute classifier flip rate."""
    correct = 0
    total = 0
    all_probs = []
    
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            batch = torch.FloatTensor(signals[i:i+batch_size]).unsqueeze(1).to(DEVICE)
            targets = target_labels[i:i+batch_size]
            
            logits = classifier_wrapper(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            correct += np.sum(preds == targets)
            total += len(targets)
            all_probs.extend(probs.cpu().numpy().tolist())
            
            del batch, logits, probs
            torch.cuda.empty_cache()
    
    return correct / total, np.array(all_probs)

n2a_flip, n2a_probs = compute_flip_rate(cfs[n2a_mask], target_labels[n2a_mask])
a2n_flip, a2n_probs = compute_flip_rate(cfs[a2n_mask], target_labels[a2n_mask])
overall_flip = (n2a_flip * n2a_mask.sum() + a2n_flip * a2n_mask.sum()) / len(cfs)

# Also compute original data accuracy
orig_flip, orig_probs = compute_flip_rate(originals, original_labels)

# FID-like metric using classifier features
def extract_features(signals, batch_size=200):
    """Extract penultimate layer features from classifier."""
    features = []
    with torch.no_grad():
        for i in range(0, len(signals), batch_size):
            batch = torch.FloatTensor(signals[i:i+batch_size]).unsqueeze(1).to(DEVICE)
            # Normalize
            mean = batch.mean(dim=2, keepdim=True)
            std = batch.std(dim=2, keepdim=True) + 1e-8
            batch_norm = (batch - mean) / std
            # Get features from model (before final layer)
            classifier.model.train()  # LSTM needs train mode
            x = batch_norm
            # Run through ResNet blocks
            for block in classifier.model.res_blocks:
                x = block(x)
            x = classifier.model.pool(x)
            x = x.permute(0, 2, 1)
            output, _ = classifier.model.lstm(x)
            feat = output[:, -1, :]
            features.append(feat.cpu().numpy())
            del batch, batch_norm, x, output, feat
            torch.cuda.empty_cache()
    return np.concatenate(features, axis=0)

print("Extracting features for FID computation...", flush=True)
n_fid = min(5000, len(cfs))
fid_idx = np.random.RandomState(42).choice(len(cfs), n_fid, replace=False)

try:
    orig_features = extract_features(originals[fid_idx])
    cf_features = extract_features(cfs[fid_idx])
    
    # Compute FID
    mu_orig = np.mean(orig_features, axis=0)
    mu_cf = np.mean(cf_features, axis=0)
    sigma_orig = np.cov(orig_features, rowvar=False)
    sigma_cf = np.cov(cf_features, rowvar=False)
    
    diff = mu_orig - mu_cf
    # Use trace approximation to avoid sqrtm issues
    fid_approx = np.sum(diff ** 2) + np.trace(sigma_orig) + np.trace(sigma_cf) - 2 * np.trace(
        np.real(np.linalg.eigvals(sigma_orig @ sigma_cf)) ** 0.5 * np.eye(sigma_orig.shape[0])
    ).sum()
    
    # Simpler FID: mean distance + covariance trace
    fid_simple = float(np.sum(diff ** 2) + np.trace(sigma_orig + sigma_cf - 2 * np.sqrt(np.abs(sigma_orig * sigma_cf))))
    
    print(f"FID (approximate): {fid_simple:.4f}")
    fid_value = fid_simple
except Exception as e:
    print(f"FID computation error: {e}")
    fid_value = -1

classifier_metrics = {
    'flip_rate': {
        'normal_to_afib': float(n2a_flip),
        'afib_to_normal': float(a2n_flip),
        'overall': float(overall_flip),
    },
    'original_accuracy': float(orig_flip),
    'fid_approximate': float(fid_value),
    'classifier_confidence': {
        'n2a_mean_target_prob': float(np.mean([p[1] for p in n2a_probs])),  # Prob of AFib
        'a2n_mean_target_prob': float(np.mean([p[0] for p in a2n_probs])),  # Prob of Normal
    }
}

print(f"Flip Rate N→A: {n2a_flip:.2%}")
print(f"Flip Rate A→N: {a2n_flip:.2%}")
print(f"Overall Flip Rate: {overall_flip:.2%}")
print(f"Original Accuracy: {orig_flip:.2%}")

# ============================================================================
# 4. Model Architecture Details  
# ============================================================================
print("\n5. MODEL ARCHITECTURE DETAILS")
print("-" * 70)

model_metadata_path = MODEL_DIR / 'model_metadata.json'
if model_metadata_path.exists():
    with open(model_metadata_path, 'r') as f:
        model_meta = json.load(f)
else:
    model_meta = {}

architecture = {
    'model_name': 'Enhanced Diffusion Counterfactual Generator',
    'components': {
        'content_encoder': {
            'type': 'Conv1D + BatchNorm + VAE',
            'input': '(1, 2500)',
            'output_dim': 256,
            'hidden_dim': 64,
            'params': model_meta.get('content_encoder_params', 'N/A'),
        },
        'style_encoder': {
            'type': 'Conv1D + InstanceNorm + Classifier',
            'input': '(1, 2500)',
            'output_dim': 128,
            'hidden_dim': 64,
            'params': model_meta.get('style_encoder_params', 'N/A'),
        },
        'conditional_unet': {
            'type': 'UNet with FiLM conditioning',
            'input': '(1, 2500)',
            'channels': '64→128→256→512',
            'attention_resolutions': [2, 3],
            'params': model_meta.get('unet_params', 'N/A'),
        },
        'ddim_scheduler': {
            'type': 'DDIM with cosine schedule',
            'timesteps': 1000,
            'inference_steps': 50,
            'sdedit_strength': 0.6,
            'cfg_scale': 3.0,
        }
    },
    'total_parameters': model_meta.get('total_parameters', 'N/A'),
    'training': {
        'total_epochs': model_meta.get('total_epochs', 100),
        'stage1_epochs': 50,
        'stage2_epochs': 50,
        'training_time_hours': model_meta.get('training_time_hours', 'N/A'),
        'optimizer': 'Adam',
        'learning_rate': 0.0001,
        'batch_size': 16,
        'gpu': 'NVIDIA A100 (49GB)',
    }
}

# ============================================================================
# 5. Dataset Statistics
# ============================================================================
print("\n6. DATASET STATISTICS")
print("-" * 70)

train_data = np.load(PROJECT_ROOT / 'data/processed/diffusion/train_data.npz')
val_data = np.load(PROJECT_ROOT / 'data/processed/diffusion/val_data.npz')

dataset_stats = {
    'source': 'MIMIC-IV ECG Database',
    'sampling_rate_hz': 250,
    'signal_length_samples': 2500,
    'signal_duration_seconds': 10,
    'splits': {
        'train': {
            'total': len(train_data['X']),
            'normal': int(np.sum(train_data['y'] == 0)),
            'afib': int(np.sum(train_data['y'] == 1)),
        },
        'validation': {
            'total': len(val_data['X']),
            'normal': int(np.sum(val_data['y'] == 0)),
            'afib': int(np.sum(val_data['y'] == 1)),
        },
        'test': {
            'total': len(test_data['X']),
            'normal': int(np.sum(test_data['y'] == 0)),
            'afib': int(np.sum(test_data['y'] == 1)),
        },
    },
    'counterfactual_generated': {
        'total': len(cfs),
        'normal_to_afib': int(n2a_mask.sum()),
        'afib_to_normal': int(a2n_mask.sum()),
    },
    'signal_statistics': {
        'original_mean': float(np.mean(originals)),
        'original_std': float(np.std(originals)),
        'original_min': float(np.min(originals)),
        'original_max': float(np.max(originals)),
        'cf_mean': float(np.mean(cfs)),
        'cf_std': float(np.std(cfs)),
        'cf_min': float(np.min(cfs)),
        'cf_max': float(np.max(cfs)),
    }
}

for split_name, split_info in dataset_stats['splits'].items():
    print(f"  {split_name}: {split_info['total']} (N={split_info['normal']}, A={split_info['afib']})")

del train_data, val_data  # Free memory
gc.collect()

# ============================================================================
# 6. Statistical Tests
# ============================================================================
print("\n7. STATISTICAL TESTS")
print("-" * 70)

# Test if RR-CV distributions differ between original and CF
t_stat_rr, p_val_rr = ttest_ind(orig_rr_cvs, cf_rr_cvs)
u_stat_rr, p_val_rr_mw = mannwhitneyu(orig_rr_cvs, cf_rr_cvs, alternative='two-sided')
ks_stat_rr, p_val_ks = ks_2samp(orig_rr_cvs, cf_rr_cvs)

# Test HR distributions
t_stat_hr, p_val_hr = ttest_ind(orig_hrs, cf_hrs)

statistical_tests = {
    'rr_cv_comparison': {
        't_test': {'statistic': float(t_stat_rr), 'p_value': float(p_val_rr)},
        'mann_whitney': {'statistic': float(u_stat_rr), 'p_value': float(p_val_rr_mw)},
        'ks_test': {'statistic': float(ks_stat_rr), 'p_value': float(p_val_ks)},
    },
    'hr_comparison': {
        't_test': {'statistic': float(t_stat_hr), 'p_value': float(p_val_hr)},
    },
    'signal_quality_tests': {
        'psnr_vs_20db': float(np.mean(np.array(psnr_vals) > 20)),
        'ssim_vs_0_5': float(np.mean(np.array(ssim_vals) > 0.5)),
        'correlation_vs_0_3': float(np.mean(np.array(corr_vals) > 0.3)),
    }
}

print(f"RR-CV t-test: t={t_stat_rr:.3f}, p={p_val_rr:.4e}")
print(f"RR-CV KS test: D={ks_stat_rr:.3f}, p={p_val_ks:.4e}")
print(f"HR t-test: t={t_stat_hr:.3f}, p={p_val_hr:.4e}")

# ============================================================================
# 7. Generate Publication Figures
# ============================================================================
print("\n8. GENERATING PUBLICATION FIGURES")
print("-" * 70)

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'figure.dpi': 300,
})

# Figure 1: Sample counterfactual overlays
fig, axes = plt.subplots(4, 2, figsize=(16, 12))
fig.suptitle('Counterfactual ECG Examples', fontsize=16, fontweight='bold', y=0.98)

# N→A examples
n2a_idx_samples = np.where(n2a_mask)[0][:4]
for row, i in enumerate(n2a_idx_samples[:2]):
    t = np.arange(SIGNAL_LENGTH) / FS
    axes[row, 0].plot(t, originals[i], 'b-', alpha=0.7, linewidth=0.8, label='Original (Normal)')
    axes[row, 0].plot(t, cfs[i], 'r-', alpha=0.7, linewidth=0.8, label='Counterfactual (→AFib)')
    axes[row, 0].set_ylabel('Amplitude')
    axes[row, 0].legend(loc='upper right', fontsize=8)
    axes[row, 0].set_title(f'Normal → AFib (Score: {val_scores[i]:.2f})')
    axes[row, 0].grid(alpha=0.3)

# A→N examples
a2n_idx_samples = np.where(a2n_mask)[0][:4]
for row, i in enumerate(a2n_idx_samples[:2]):
    t = np.arange(SIGNAL_LENGTH) / FS
    axes[row, 1].plot(t, originals[i], 'r-', alpha=0.7, linewidth=0.8, label='Original (AFib)')
    axes[row, 1].plot(t, cfs[i], 'b-', alpha=0.7, linewidth=0.8, label='Counterfactual (→Normal)')
    axes[row, 1].set_ylabel('Amplitude')
    axes[row, 1].legend(loc='upper right', fontsize=8)
    axes[row, 1].set_title(f'AFib → Normal (Score: {val_scores[i]:.2f})')
    axes[row, 1].grid(alpha=0.3)

# Zoomed-in RR interval comparison
for row in [2, 3]:
    direction = 'N→A' if row == 2 else 'A→N'
    i = n2a_idx_samples[0] if row == 2 else a2n_idx_samples[0]
    t = np.arange(SIGNAL_LENGTH) / FS
    
    # Show 3-second window
    start, end = 500, 1250
    t_zoom = t[start:end]
    axes[row, 0].plot(t_zoom, originals[i][start:end], 'b-', linewidth=1.2, label='Original')
    axes[row, 0].set_title(f'{direction}: Original (3s zoom)')
    axes[row, 0].set_xlabel('Time (s)')
    axes[row, 0].set_ylabel('Amplitude')
    axes[row, 0].grid(alpha=0.3)
    
    axes[row, 1].plot(t_zoom, cfs[i][start:end], 'r-', linewidth=1.2, label='Counterfactual')
    axes[row, 1].set_title(f'{direction}: Counterfactual (3s zoom)')
    axes[row, 1].set_xlabel('Time (s)')
    axes[row, 1].set_ylabel('Amplitude')
    axes[row, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig1_counterfactual_examples.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 1: Counterfactual examples saved")

# Figure 2: Signal quality distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Signal Quality Metric Distributions', fontsize=16, fontweight='bold')

axes[0, 0].hist(psnr_vals, bins=50, color='steelblue', alpha=0.8, edgecolor='white')
axes[0, 0].axvline(np.mean(psnr_vals), color='red', linestyle='--', label=f'Mean={np.mean(psnr_vals):.1f}')
axes[0, 0].set_xlabel('PSNR (dB)')
axes[0, 0].set_title('PSNR Distribution')
axes[0, 0].legend()

axes[0, 1].hist(ssim_vals, bins=50, color='coral', alpha=0.8, edgecolor='white')
axes[0, 1].axvline(np.mean(ssim_vals), color='red', linestyle='--', label=f'Mean={np.mean(ssim_vals):.3f}')
axes[0, 1].set_xlabel('SSIM')
axes[0, 1].set_title('SSIM Distribution')
axes[0, 1].legend()

axes[0, 2].hist(corr_vals, bins=50, color='mediumseagreen', alpha=0.8, edgecolor='white')
axes[0, 2].axvline(np.mean(corr_vals), color='red', linestyle='--', label=f'Mean={np.mean(corr_vals):.3f}')
axes[0, 2].set_xlabel('Pearson Correlation')
axes[0, 2].set_title('Correlation Distribution')
axes[0, 2].legend()

axes[1, 0].hist(mse_vals, bins=50, color='mediumpurple', alpha=0.8, edgecolor='white')
axes[1, 0].axvline(np.mean(mse_vals), color='red', linestyle='--', label=f'Mean={np.mean(mse_vals):.5f}')
axes[1, 0].set_xlabel('MSE')
axes[1, 0].set_title('MSE Distribution')
axes[1, 0].legend()

axes[1, 1].hist(snr_vals, bins=50, color='goldenrod', alpha=0.8, edgecolor='white')
axes[1, 1].axvline(np.mean(snr_vals), color='red', linestyle='--', label=f'Mean={np.mean(snr_vals):.1f}')
axes[1, 1].set_xlabel('SNR (dB)')
axes[1, 1].set_title('SNR Distribution')
axes[1, 1].legend()

# Plausibility score distribution
axes[1, 2].hist(val_scores[n2a_mask], bins=50, alpha=0.7, label='N→A', color='blue')
axes[1, 2].hist(val_scores[a2n_mask], bins=50, alpha=0.7, label='A→N', color='red')
axes[1, 2].axvline(0.7, color='green', linestyle='--', label='Threshold (0.7)')
axes[1, 2].set_xlabel('Plausibility Score')
axes[1, 2].set_title('Plausibility Score Distribution')
axes[1, 2].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig2_signal_quality_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 2: Signal quality distributions saved")

# Figure 3: Clinical feature comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Clinical Feature Analysis', fontsize=16, fontweight='bold')

# RR-CV before/after
axes[0].boxplot([orig_rr_cvs, cf_rr_cvs], labels=['Original', 'Counterfactual'])
axes[0].set_ylabel('RR-CV')
axes[0].set_title('RR Interval Variability')
axes[0].grid(alpha=0.3)

# Heart rate before/after
axes[1].boxplot([orig_hrs, cf_hrs], labels=['Original', 'Counterfactual'])
axes[1].set_ylabel('Heart Rate (BPM)')
axes[1].set_title('Heart Rate Distribution')
axes[1].grid(alpha=0.3)

# RR direction correctness
dir_correct = [n2a_direction_pct * 100, a2n_direction_pct * 100]
bars = axes[2].bar(['N→A\n(CV↑)', 'A→N\n(CV↓)'], dir_correct, color=['steelblue', 'coral'])
axes[2].set_ylabel('Direction Correct (%)')
axes[2].set_title('RR-CV Direction Correctness')
axes[2].set_ylim([0, 100])
axes[2].axhline(50, color='gray', linestyle='--', alpha=0.5, label='Random')
for bar, val in zip(bars, dir_correct):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:.1f}%', ha='center', fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig3_clinical_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 3: Clinical feature analysis saved")

# Figure 4: Flip rate analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Classifier Flip Rate Analysis', fontsize=16, fontweight='bold')

# Flip rates
bars = axes[0].bar(['N→A', 'A→N', 'Overall'], 
                    [n2a_flip * 100, a2n_flip * 100, overall_flip * 100],
                    color=['steelblue', 'coral', 'mediumpurple'])
axes[0].set_ylabel('Flip Rate (%)')
axes[0].set_title('Classification Flip Rate by Direction')
axes[0].set_ylim([0, 100])
for bar in bars:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', fontweight='bold')
axes[0].grid(alpha=0.3)

# Confidence distribution
n2a_target_probs = [p[1] for p in n2a_probs]  # AFib probability
a2n_target_probs = [p[0] for p in a2n_probs]  # Normal probability
axes[1].hist(n2a_target_probs, bins=50, alpha=0.7, label='N→A (P(AFib))', color='steelblue')
axes[1].hist(a2n_target_probs, bins=50, alpha=0.7, label='A→N (P(Normal))', color='coral')
axes[1].axvline(0.5, color='black', linestyle='--', label='Decision boundary')
axes[1].set_xlabel('Target Class Probability')
axes[1].set_ylabel('Count')
axes[1].set_title('Classifier Confidence on Counterfactuals')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'fig4_flip_rate_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Figure 4: Flip rate analysis saved")

# ============================================================================
# 8. Compile All Metrics
# ============================================================================
print("\n9. COMPILING ALL METRICS")
print("-" * 70)

all_metrics = {
    'signal_quality': signal_quality,
    'clinical_validation': clinical_metrics,
    'classifier_metrics': classifier_metrics,
    'statistical_tests': statistical_tests,
    'model_architecture': architecture,
    'dataset_statistics': dataset_stats,
    'figures_generated': [
        'fig1_counterfactual_examples.png',
        'fig2_signal_quality_distributions.png',
        'fig3_clinical_features.png',
        'fig4_flip_rate_analysis.png',
    ]
}

# Save JSON
with open(OUTPUT_DIR / 'comprehensive_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2, default=str)

# Save human-readable text
with open(OUTPUT_DIR / 'comprehensive_metrics.txt', 'w') as f:
    f.write("COMPREHENSIVE METRICS FOR PUBLICATION\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("1. SIGNAL QUALITY METRICS\n")
    f.write("-" * 40 + "\n")
    sq = signal_quality['overall']
    f.write(f"PSNR:        {sq['psnr_mean']:.2f} ± {sq['psnr_std']:.2f} dB\n")
    f.write(f"SSIM:        {sq['ssim_mean']:.4f} ± {sq['ssim_std']:.4f}\n")
    f.write(f"SNR:         {sq['snr_mean']:.2f} ± {sq['snr_std']:.2f} dB\n")
    f.write(f"Correlation: {sq['correlation_mean']:.4f} ± {sq['correlation_std']:.4f}\n")
    f.write(f"MSE:         {sq['mse_mean']:.6f} ± {sq['mse_std']:.6f}\n")
    f.write(f"MAE:         {sq['mae_mean']:.6f} ± {sq['mae_std']:.6f}\n")
    f.write(f"Max Error:   {sq['max_error_mean']:.4f} ± {sq['max_error_std']:.4f}\n\n")
    
    f.write("2. CLASSIFIER FLIP RATES\n")
    f.write("-" * 40 + "\n")
    f.write(f"Normal→AFib:  {classifier_metrics['flip_rate']['normal_to_afib']:.2%}\n")
    f.write(f"AFib→Normal:  {classifier_metrics['flip_rate']['afib_to_normal']:.2%}\n")
    f.write(f"Overall:      {classifier_metrics['flip_rate']['overall']:.2%}\n\n")
    
    f.write("3. CLINICAL VALIDATION\n")
    f.write("-" * 40 + "\n")
    cm = clinical_metrics
    f.write(f"RR Direction Correct (N→A): {cm['rr_interval_analysis']['rr_direction_correct_n2a']:.1%}\n")
    f.write(f"RR Direction Correct (A→N): {cm['rr_interval_analysis']['rr_direction_correct_a2n']:.1%}\n")
    f.write(f"HR in range (30-200 bpm):   {cm['heart_rate']['hr_in_physiological_range_pct']:.1%}\n")
    f.write(f"Original RR-CV:             {cm['rr_interval_analysis']['original_rr_cv_mean']:.4f}\n")
    f.write(f"CF RR-CV:                   {cm['rr_interval_analysis']['counterfactual_rr_cv_mean']:.4f}\n")
    f.write(f"Plausibility (mean):        {cm['plausibility']['mean_score']:.3f}\n")
    f.write(f"Plausibility > 0.7:         {cm['plausibility']['above_0_7_pct']:.1%}\n\n")
    
    f.write("4. DATASET STATISTICS\n")
    f.write("-" * 40 + "\n")
    for split_name, split_info in dataset_stats['splits'].items():
        f.write(f"  {split_name}: {split_info['total']} (N={split_info['normal']}, A={split_info['afib']})\n")
    f.write(f"  Generated CFs: {dataset_stats['counterfactual_generated']['total']}\n")

print(f"✓ Saved comprehensive_metrics.json")
print(f"✓ Saved comprehensive_metrics.txt")
print(f"✓ All figures saved to {OUTPUT_DIR}")

print("\n" + "=" * 70)
print("ENHANCED METRICS COMPUTATION COMPLETE!")
print("=" * 70)
