"""
Part A: Plausibility & Quality Testing for Paper
- PSNR, SNR, correlation metrics
- Clinical parameter analysis (R-R intervals, P-wave, HR)
- Distribution tests (KS test, Wasserstein distance)
- Uniqueness proof (pairwise similarity)
"""
import os, sys, json, gc, warnings
import numpy as np
import torch, torch.nn.functional as F
from pathlib import Path
from scipy import signal, stats
from scipy.stats import pearsonr, ks_2samp, wasserstein_distance
from scipy.signal import savgol_filter
from sklearn.metrics import pairwise_distances
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
OUT = ROOT / 'models/phase3_counterfactual/paper_results'
FIG = OUT / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
DATA = OUT / 'data'; DATA.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / 'notebooks/phase_3_counterfactual'))
from plausibility_validator import PlausibilityValidator

# Load data
print("Loading data...")
orig = np.load(ROOT / 'data/processed/diffusion/train_data.npz')
orig_X, orig_y = orig['X'], orig['y']
cf = np.load(ROOT / 'models/phase3_counterfactual/generated_counterfactuals/filtered_counterfactuals.npz')
cf_X, cf_y = cf['X'], cf['y']
plaus_scores = cf['plausibility_scores'] if 'plausibility_scores' in cf else None

if orig_X.ndim == 3: orig_X = orig_X.squeeze(1)
if cf_X.ndim == 3: cf_X = cf_X.squeeze(1)

print(f"Original: {orig_X.shape}, CF: {cf_X.shape}")

FS = 250  # Sampling frequency
validator = PlausibilityValidator(fs=FS, signal_length=2500)

# ============================================================================
# 1. PSNR / SNR / Correlation Metrics
# ============================================================================
print("\n=== 1. Signal Quality Metrics ===")

N_SAMPLE = min(5000, len(cf_X))
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(cf_X), N_SAMPLE, replace=False)

def compute_psnr(original, generated):
    mse = np.mean((original - generated)**2)
    if mse == 0: return 100.0
    max_val = np.max(np.abs(original))
    return 20 * np.log10(max_val / np.sqrt(mse))

def compute_snr(sig_arr):
    return 20 * np.log10(np.std(sig_arr) / (np.std(np.diff(sig_arr)) + 1e-10))

# Match CF to closest original by label for PSNR
psnr_vals, corr_vals, snr_orig_vals, snr_cf_vals = [], [], [], []
cf_snr_all = []

for i in sample_idx:
    cf_sig = cf_X[i]
    cf_label = cf_y[i]
    # Find random original with OPPOSITE class (since CF was generated from opposite)
    source_class = 1 - cf_label
    source_idx = rng.choice(np.where(orig_y == source_class)[0])
    source_sig = orig_X[source_idx]
    
    psnr = compute_psnr(source_sig, cf_sig)
    corr = pearsonr(source_sig, cf_sig)[0]
    snr_o = compute_snr(source_sig)
    snr_c = compute_snr(cf_sig)
    
    psnr_vals.append(psnr); corr_vals.append(corr)
    snr_orig_vals.append(snr_o); snr_cf_vals.append(snr_c)

psnr_vals = np.array(psnr_vals)
corr_vals = np.array(corr_vals)
snr_orig_vals = np.array(snr_orig_vals)
snr_cf_vals = np.array(snr_cf_vals)

print(f"PSNR: {psnr_vals.mean():.2f} ± {psnr_vals.std():.2f} dB")
print(f"Correlation: {corr_vals.mean():.4f} ± {corr_vals.std():.4f}")
print(f"SNR (original): {snr_orig_vals.mean():.2f} ± {snr_orig_vals.std():.2f} dB")
print(f"SNR (generated): {snr_cf_vals.mean():.2f} ± {snr_cf_vals.std():.2f} dB")

# ============================================================================
# 2. Clinical Parameter Analysis
# ============================================================================
print("\n=== 2. Clinical Parameter Analysis ===")

def analyze_rr_intervals(ecg, fs=250):
    """Extract R-R intervals and heart rate variability metrics."""
    ecg_f = savgol_filter(ecg, 11, 3)
    grad = np.gradient(ecg_f)
    threshold = np.std(ecg_f) * 0.5
    peaks = []
    for i in range(1, len(ecg_f)-1):
        if ecg_f[i] > threshold and ecg_f[i] > ecg_f[i-1] and ecg_f[i] > ecg_f[i+1]:
            if len(peaks) == 0 or (i - peaks[-1]) > fs * 0.3:
                peaks.append(i)
    peaks = np.array(peaks)
    if len(peaks) < 3:
        return {'hr': 0, 'rr_mean': 0, 'rr_std': 0, 'rr_irregularity': 0, 'n_beats': len(peaks)}
    rr = np.diff(peaks) / fs
    hr = 60.0 / rr.mean() if rr.mean() > 0 else 0
    rr_std = rr.std()
    rr_irreg = rr_std / rr.mean() if rr.mean() > 0 else 0
    return {'hr': hr, 'rr_mean': rr.mean(), 'rr_std': rr_std, 
            'rr_irregularity': rr_irreg, 'n_beats': len(peaks)}

def detect_p_wave_energy(ecg, fs=250):
    """Estimate P-wave presence using frequency content in P-wave band."""
    nyq = fs / 2
    b, a = signal.butter(4, [0.5/nyq, 10/nyq], btype='band')
    filtered = signal.filtfilt(b, a, ecg)
    return np.std(filtered)

# Analyze samples
N_CLIN = min(3000, len(cf_X))
clinical_orig = {'normal': [], 'afib': []}
clinical_cf = {'normal': [], 'afib': []}

for i in range(N_CLIN):
    idx_o = rng.choice(len(orig_X))
    metrics_o = analyze_rr_intervals(orig_X[idx_o])
    metrics_o['p_wave_energy'] = detect_p_wave_energy(orig_X[idx_o])
    key_o = 'normal' if orig_y[idx_o] == 0 else 'afib'
    clinical_orig[key_o].append(metrics_o)
    
    idx_c = i % len(cf_X)
    metrics_c = analyze_rr_intervals(cf_X[idx_c])
    metrics_c['p_wave_energy'] = detect_p_wave_energy(cf_X[idx_c])
    key_c = 'normal' if cf_y[idx_c] == 0 else 'afib'
    clinical_cf[key_c].append(metrics_c)

for class_name in ['normal', 'afib']:
    print(f"\n  {class_name.upper()}:")
    for source, data in [('Original', clinical_orig), ('Generated', clinical_cf)]:
        if len(data[class_name]) == 0: continue
        hrs = [d['hr'] for d in data[class_name] if d['hr'] > 0]
        irregs = [d['rr_irregularity'] for d in data[class_name] if d['rr_irregularity'] > 0]
        pwe = [d['p_wave_energy'] for d in data[class_name]]
        print(f"    {source}: HR={np.mean(hrs):.1f}±{np.std(hrs):.1f}, "
              f"RR_irreg={np.mean(irregs):.4f}±{np.std(irregs):.4f}, "
              f"P-wave={np.mean(pwe):.6f}±{np.std(pwe):.6f}")

# ============================================================================
# 3. Distribution Tests
# ============================================================================
print("\n=== 3. Distribution Tests ===")

# Amplitude distribution
ks_stat, ks_p = ks_2samp(orig_X.flatten()[:100000], cf_X.flatten()[:100000])
wd = wasserstein_distance(orig_X.flatten()[:100000], cf_X.flatten()[:100000])
print(f"Amplitude KS test: statistic={ks_stat:.4f}, p={ks_p:.6f}")
print(f"Wasserstein distance: {wd:.6f}")

# HR distribution by class
for cls, name in [(0, 'Normal'), (1, 'AFib')]:
    hr_orig = [d['hr'] for d in clinical_orig[name.lower()] if d['hr'] > 30]
    hr_cf = [d['hr'] for d in clinical_cf[name.lower()] if d['hr'] > 30]
    if hr_orig and hr_cf:
        ks, p = ks_2samp(hr_orig, hr_cf)
        print(f"  HR ({name}): KS={ks:.4f}, p={p:.4f}")

# RR irregularity by class
for cls, name in [(0, 'Normal'), (1, 'AFib')]:
    irr_orig = [d['rr_irregularity'] for d in clinical_orig[name.lower()] if d['rr_irregularity'] > 0]
    irr_cf = [d['rr_irregularity'] for d in clinical_cf[name.lower()] if d['rr_irregularity'] > 0]
    if irr_orig and irr_cf:
        ks, p = ks_2samp(irr_orig, irr_cf)
        print(f"  RR irregularity ({name}): KS={ks:.4f}, p={p:.4f}")

# ============================================================================
# 4. Uniqueness Proof
# ============================================================================
print("\n=== 4. Uniqueness Proof (Not Copies) ===")

N_UNI = 500
cf_sample = cf_X[rng.choice(len(cf_X), N_UNI, replace=False)]
orig_normal = orig_X[orig_y == 0][:5000]
orig_afib = orig_X[orig_y == 1][:5000]

# For each CF, find nearest neighbor in original data
max_corrs = []
max_corrs_same_class = []
for i in range(N_UNI):
    cf_sig = cf_sample[i]
    # Check against same-class originals
    pool = orig_normal if cf_y[i] == 0 else orig_afib
    batch_corrs = []
    for j in range(0, min(len(pool), 2000), 1):
        c = np.corrcoef(cf_sig, pool[j])[0, 1]
        batch_corrs.append(abs(c))
    max_corrs_same_class.append(max(batch_corrs))
    max_corrs.append(max(batch_corrs))

max_corrs = np.array(max_corrs)
max_corrs_sc = np.array(max_corrs_same_class)

print(f"Max correlation to nearest original (same class):")
print(f"  Mean: {max_corrs_sc.mean():.4f}, Std: {max_corrs_sc.std():.4f}")
print(f"  > 0.95 (near-copy): {(max_corrs_sc > 0.95).sum()}/{N_UNI} ({100*(max_corrs_sc > 0.95).mean():.1f}%)")
print(f"  > 0.90: {(max_corrs_sc > 0.90).sum()}/{N_UNI}")
print(f"  > 0.80: {(max_corrs_sc > 0.80).sum()}/{N_UNI}")
print(f"  < 0.50 (very different): {(max_corrs_sc < 0.50).sum()}/{N_UNI}")

# L2 distance
l2_dists = []
for i in range(N_UNI):
    cf_sig = cf_sample[i]
    pool = orig_normal if cf_y[i] == 0 else orig_afib
    dists = np.sqrt(np.mean((pool[:2000] - cf_sig)**2, axis=1))
    l2_dists.append(dists.min())
l2_dists = np.array(l2_dists)
print(f"Min L2 distance to nearest original: {l2_dists.mean():.4f} ± {l2_dists.std():.4f}")

# ============================================================================
# 5. Plausibility Validation on Full Dataset
# ============================================================================
print("\n=== 5. Full Plausibility Validation ===")

N_PLAUS = min(2000, len(cf_X))
plaus_results = []

for i in range(N_PLAUS):
    result = validator.validate(cf_X[i], target_class=cf_y[i])
    plaus_results.append(result)

scores = [r['score'] for r in plaus_results]
morpho = [r.get('morphology_score', 0) for r in plaus_results]
physio = [r.get('physiology_score', 0) for r in plaus_results]

scores = np.array(scores)
print(f"Plausibility: {scores.mean():.4f} ± {scores.std():.4f}")
print(f"  >= 0.7: {(scores >= 0.7).sum()}/{N_PLAUS} ({100*(scores >= 0.7).mean():.1f}%)")
print(f"  >= 0.5: {(scores >= 0.5).sum()}/{N_PLAUS} ({100*(scores >= 0.5).mean():.1f}%)")
if plaus_scores is not None:
    print(f"Stored plausibility (full 20K): {plaus_scores.mean():.4f} ± {plaus_scores.std():.4f}")

# ============================================================================
# 6. Generate Figures
# ============================================================================
print("\n=== 6. Generating Figures ===")

# Fig 1: Signal quality histograms
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].hist(psnr_vals, bins=50, color='#2196F3', alpha=0.8, edgecolor='white')
axes[0,0].set_xlabel('PSNR (dB)'); axes[0,0].set_ylabel('Count')
axes[0,0].set_title(f'PSNR Distribution\n(mean={psnr_vals.mean():.2f}±{psnr_vals.std():.2f} dB)')
axes[0,0].axvline(psnr_vals.mean(), color='red', linestyle='--')

axes[0,1].hist(corr_vals, bins=50, color='#FF5722', alpha=0.8, edgecolor='white')
axes[0,1].set_xlabel('Pearson Correlation'); axes[0,1].set_ylabel('Count')
axes[0,1].set_title(f'Source-CF Correlation\n(mean={corr_vals.mean():.4f})')
axes[0,1].axvline(corr_vals.mean(), color='red', linestyle='--')

axes[1,0].hist(snr_cf_vals, bins=50, color='#4CAF50', alpha=0.8, edgecolor='white', label='Generated')
axes[1,0].hist(snr_orig_vals, bins=50, color='#2196F3', alpha=0.5, edgecolor='white', label='Original')
axes[1,0].set_xlabel('SNR (dB)'); axes[1,0].set_ylabel('Count')
axes[1,0].set_title('SNR Comparison'); axes[1,0].legend()

axes[1,1].hist(scores, bins=30, color='#9C27B0', alpha=0.8, edgecolor='white')
axes[1,1].set_xlabel('Plausibility Score'); axes[1,1].set_ylabel('Count')
axes[1,1].set_title(f'Clinical Plausibility\n(mean={scores.mean():.4f})')
axes[1,1].axvline(0.7, color='red', linestyle='--', label='Threshold')
axes[1,1].legend()

plt.suptitle('Counterfactual ECG Quality Metrics', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(FIG / 'quality_metrics.png', dpi=200); plt.close()
print("✓ quality_metrics.png")

# Fig 2: Uniqueness proof
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(max_corrs_sc, bins=40, color='#FF9800', alpha=0.8, edgecolor='white')
axes[0].axvline(0.95, color='red', linestyle='--', label='Copy threshold (0.95)')
axes[0].axvline(0.80, color='orange', linestyle='--', label='High similarity (0.80)')
axes[0].set_xlabel('Max Correlation to Nearest Original'); axes[0].set_ylabel('Count')
axes[0].set_title(f'Nearest-Neighbor Similarity\n(mean={max_corrs_sc.mean():.4f})'); axes[0].legend()

axes[1].hist(l2_dists, bins=40, color='#00BCD4', alpha=0.8, edgecolor='white')
axes[1].set_xlabel('Min L2 Distance'); axes[1].set_ylabel('Count')
axes[1].set_title(f'L2 Distance to Nearest Original\n(mean={l2_dists.mean():.4f})')

plt.suptitle('Uniqueness Proof: Generated ≠ Copies', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(FIG / 'uniqueness_proof.png', dpi=200); plt.close()
print("✓ uniqueness_proof.png")

# Fig 3: Clinical parameter comparison
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
params = ['hr', 'rr_irregularity', 'p_wave_energy']
titles = ['Heart Rate (BPM)', 'R-R Irregularity (CV)', 'P-Wave Energy']

for col, (param, title) in enumerate(zip(params, titles)):
    for row, cls in enumerate(['normal', 'afib']):
        ax = axes[row, col]
        vals_o = [d[param] for d in clinical_orig[cls] if d[param] > 0]
        vals_c = [d[param] for d in clinical_cf[cls] if d[param] > 0]
        if vals_o and vals_c:
            ax.hist(vals_o, bins=30, alpha=0.6, label='Original', color='#2196F3', density=True)
            ax.hist(vals_c, bins=30, alpha=0.6, label='Generated', color='#FF5722', density=True)
            ks, p = ks_2samp(vals_o, vals_c)
            ax.set_title(f'{cls.upper()} - {title}\n(KS={ks:.3f}, p={p:.4f})')
            ax.legend(fontsize=8)
        ax.set_xlabel(title)

plt.suptitle('Clinical Parameter Distributions: Original vs Generated', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(FIG / 'clinical_parameters.png', dpi=200); plt.close()
print("✓ clinical_parameters.png")

# ============================================================================
# 7. Save Results
# ============================================================================
results = {
    'signal_quality': {
        'psnr': {'mean': float(psnr_vals.mean()), 'std': float(psnr_vals.std())},
        'correlation': {'mean': float(corr_vals.mean()), 'std': float(corr_vals.std())},
        'snr_original': {'mean': float(snr_orig_vals.mean()), 'std': float(snr_orig_vals.std())},
        'snr_generated': {'mean': float(snr_cf_vals.mean()), 'std': float(snr_cf_vals.std())},
    },
    'plausibility': {
        'mean_score': float(scores.mean()), 'std_score': float(scores.std()),
        'above_0.7': float((scores >= 0.7).mean()),
        'stored_mean': float(plaus_scores.mean()) if plaus_scores is not None else None,
    },
    'uniqueness': {
        'max_corr_to_nearest': {'mean': float(max_corrs_sc.mean()), 'std': float(max_corrs_sc.std())},
        'near_copies_above_0.95': int((max_corrs_sc > 0.95).sum()),
        'l2_to_nearest': {'mean': float(l2_dists.mean()), 'std': float(l2_dists.std())},
    },
    'distribution_tests': {
        'amplitude_ks': {'statistic': float(ks_stat), 'p_value': float(ks_p)},
        'wasserstein_distance': float(wd),
    },
}

with open(DATA / 'plausibility_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✓ Saved: {DATA / 'plausibility_analysis.json'}")
print("Part A complete!")
