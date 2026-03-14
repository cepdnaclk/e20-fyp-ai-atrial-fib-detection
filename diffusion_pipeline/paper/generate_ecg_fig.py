"""
Publication ECG Figure — 4 rows × 3 columns
With classifier predictions on each panel showing class + confidence
"""


import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8.5,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.linewidth': 0.4,
    'lines.linewidth': 0.55,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.04,
    'mathtext.fontset': 'stix',
})

DATA_DIR = PROJECT_ROOT / 'models/phase3_counterfactual/generated_counterfactuals'
OUTPUT_DIR = PROJECT_ROOT / 'paper/figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FS = 250
SHOW_SEC = 4.0
SHOW_N = int(SHOW_SEC * FS)

# Colors
COLOR_ORIG = '#1A5276'   # dark teal-blue
COLOR_CF   = '#A04000'   # warm brown

# ============================================================================
# Classifier — AFibResLSTM (same as used in counterfactual generation)
# ============================================================================
import sys
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'models'))
from model_architecture import AFibResLSTM, ModelConfig

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

CLASSIFIER_PATH = PROJECT_ROOT / 'models/afib_reslstm_final.pth'
classifier = AFibResLSTM(ModelConfig()).to(DEVICE)
cls_ckpt = torch.load(CLASSIFIER_PATH, map_location=DEVICE, weights_only=False)
classifier.load_state_dict(cls_ckpt['model_state_dict'])
classifier.eval()
print(f"✅ Classifier loaded ({sum(p.numel() for p in classifier.parameters()):,} params)")

CLASS_NAMES = {0: 'Normal', 1: 'AFib'}

@torch.no_grad()
def classify(signal_1d):
    """Return (pred_class_name, confidence%).
    Applies per-signal normalization (same as ClassifierWrapper in pipeline)."""
    ecg = torch.tensor(signal_1d, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    # Normalize — identical to ClassifierWrapper in shared_models.py
    mean = ecg.mean(dim=2, keepdim=True)
    std = ecg.std(dim=2, keepdim=True) + 1e-8
    ecg_norm = (ecg - mean) / std
    logits, _ = classifier(ecg_norm)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(np.argmax(probs))
    return CLASS_NAMES[pred], float(probs[pred]) * 100

# ============================================================================
# Helper functions
# ============================================================================
def rr_cv(signal, fs=250):
    peaks, _ = find_peaks(signal, height=np.mean(signal)+0.4*np.std(signal), distance=int(0.4*fs))
    if len(peaks) < 3: return peaks, 99.0
    rr = np.diff(peaks) / fs
    return peaks, np.std(rr) / np.mean(rr)

def ecg_quality(sig):
    """Strict quality check: signal must look like a proper ECG."""
    zoomed = sig[:SHOW_N]
    amp_range = np.max(zoomed) - np.min(zoomed)
    if amp_range < 0.08 or amp_range > 0.6:
        return False
    diff = np.abs(np.diff(zoomed))
    if np.percentile(diff, 99.5) > 0.035:
        return False
    peaks, _ = find_peaks(zoomed, height=np.mean(zoomed)+0.4*np.std(zoomed), distance=int(0.4*FS))
    if len(peaks) < 3:
        return False
    peak_heights = zoomed[peaks]
    if np.std(peak_heights) / (np.mean(peak_heights) + 1e-10) > 1.5:
        return False
    if abs(np.median(zoomed)) > 0.1:
        return False
    baseline_std = np.std(zoomed[zoomed < np.percentile(zoomed, 60)])
    peak_mean = np.mean(peak_heights)
    if peak_mean < 3 * baseline_std:
        return False
    return True

def score_pair(orig, cf, direction):
    p_o, cv_o = rr_cv(orig)
    p_c, cv_c = rr_cv(cf)
    if len(p_o) < 4 or len(p_c) < 4: return -999
    if not ecg_quality(orig) or not ecg_quality(cf): return -999
    zp_o = [p for p in p_o if p < SHOW_N]
    zp_c = [p for p in p_c if p < SHOW_N]
    if len(zp_o) < 3 or len(zp_c) < 3: return -999
    if direction == 'n2a':
        if cv_o > 0.08 or cv_c < 0.25: return -999
        return (cv_c - cv_o) * 3.0 + min(len(zp_o), 6) * 0.15
    else:
        if cv_c > 0.10 or cv_o < 0.20: return -999
        return (cv_o - cv_c) * 3.0 + min(len(zp_c), 6) * 0.15

# ============================================================================
# Load data
# ============================================================================
print("Loading data...")
n2a_orig, n2a_cf, n2a_plaus = [], [], []
for i in range(50):
    f = DATA_DIR / f'filtered_Normal_to_AFib_batch_{i:04d}.npz'
    if not f.exists(): break
    d = np.load(f)
    n2a_orig.append(d['originals']); n2a_cf.append(d['counterfactuals']); n2a_plaus.append(d['plausibility_scores'])
n2a_orig = np.concatenate(n2a_orig); n2a_cf = np.concatenate(n2a_cf); n2a_plaus = np.concatenate(n2a_plaus)

a2n_orig, a2n_cf, a2n_plaus = [], [], []
for i in range(50):
    f = DATA_DIR / f'filtered_AFib_to_Normal_batch_{i:04d}.npz'
    if not f.exists(): break
    d = np.load(f)
    a2n_orig.append(d['originals']); a2n_cf.append(d['counterfactuals']); a2n_plaus.append(d['plausibility_scores'])
a2n_orig = np.concatenate(a2n_orig); a2n_cf = np.concatenate(a2n_cf); a2n_plaus = np.concatenate(a2n_plaus)
print(f"  N→A: {len(n2a_orig)},  A→N: {len(a2n_orig)}")

# ============================================================================
# Select best examples
# ============================================================================
print("Selecting best examples (strict quality)...")

def select_best_2(orig_arr, cf_arr, plaus_arr, direction):
    scores = []
    for i in range(len(orig_arr)):
        if plaus_arr[i] < 0.75: continue
        s = score_pair(orig_arr[i], cf_arr[i], direction)
        if s > 0: scores.append((i, s))
    scores.sort(key=lambda x: x[1], reverse=True)
    print(f"    Eligible ({direction}): {len(scores)}")
    sel = [scores[0]]
    for idx, sc in scores[1:]:
        _, cv_prev = rr_cv(cf_arr[sel[-1][0]])
        _, cv_this = rr_cv(cf_arr[idx])
        if abs(cv_this - cv_prev) > 0.06:
            sel.append((idx, sc)); break
    if len(sel) < 2: sel.append(scores[1])
    return sel

sel_n2a = select_best_2(n2a_orig, n2a_cf, n2a_plaus, 'n2a')
sel_a2n = select_best_2(a2n_orig, a2n_cf, a2n_plaus, 'a2n')

panels = []
for idx, _ in sel_n2a:
    panels.append((n2a_orig[idx], n2a_cf[idx], 'Normal $\\rightarrow$ AFib', 'Normal', 'AFib'))
for idx, _ in sel_a2n:
    panels.append((a2n_orig[idx], a2n_cf[idx], 'AFib $\\rightarrow$ Normal', 'AFib', 'Normal'))

# ============================================================================
# Plot
# ============================================================================
print("Classifying and plotting...")

fig, axes = plt.subplots(4, 3, figsize=(7.16, 8.5),
                          gridspec_kw={'hspace': 0.50, 'wspace': 0.28})

time_ax = np.arange(SHOW_N) / FS
row_labels = ['(a)', '(b)', '(c)', '(d)']

for row, (orig, cf, direction, orig_cls, cf_cls) in enumerate(panels):
    sig_o = orig[:SHOW_N]
    sig_c = cf[:SHOW_N]

    # Classifier predictions
    pred_o, conf_o = classify(orig)
    pred_c, conf_c = classify(cf)

    ymin = min(sig_o.min(), sig_c.min())
    ymax = max(sig_o.max(), sig_c.max())
    ypad = (ymax - ymin) * 0.15
    ylims = [ymin - ypad, ymax + ypad]

    peaks_o, _ = find_peaks(sig_o, height=np.mean(sig_o)+0.4*np.std(sig_o), distance=int(0.4*FS))
    peaks_c, _ = find_peaks(sig_c, height=np.mean(sig_c)+0.4*np.std(sig_c), distance=int(0.4*FS))

    # Annotation style — green for correct prediction, red for mis-match
    def pred_badge(ax, pred_name, confidence, expected_cls):
        """Place a classifier-prediction badge in the top-right corner."""
        color = '#1a7a2e' if pred_name == expected_cls else '#b03030'
        ax.text(0.97, 0.95,
                f'{pred_name}  {confidence:.0f}%',
                transform=ax.transAxes, ha='right', va='top', fontsize=6.8,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.25', fc=color, ec='none', alpha=0.92))

    # --- Col 0: Original ---
    ax = axes[row, 0]
    ax.plot(time_ax, sig_o, color=COLOR_ORIG, linewidth=0.6)
    ax.plot(peaks_o/FS, sig_o[peaks_o], 'v', color=COLOR_ORIG, markersize=3.5,
            markeredgewidth=0.3, markeredgecolor='white', zorder=4)
    ax.set_ylim(ylims); ax.set_xlim([0, SHOW_SEC])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.10, linewidth=0.3); ax.tick_params(axis='both', length=2)
    if row == 0: ax.set_title('Original', fontsize=9.5, fontweight='bold')
    ax.set_ylabel(f'{row_labels[row]} {direction}\n\nAmplitude', fontsize=7.5)
    pred_badge(ax, pred_o, conf_o, orig_cls)

    # --- Col 1: Counterfactual ---
    ax = axes[row, 1]
    ax.plot(time_ax, sig_c, color=COLOR_CF, linewidth=0.6)
    ax.plot(peaks_c/FS, sig_c[peaks_c], '^', color=COLOR_CF, markersize=3.5,
            markeredgewidth=0.3, markeredgecolor='white', zorder=4)
    ax.set_ylim(ylims); ax.set_xlim([0, SHOW_SEC])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.10, linewidth=0.3); ax.tick_params(axis='both', length=2)
    if row == 0: ax.set_title('Counterfactual', fontsize=9.5, fontweight='bold')
    pred_badge(ax, pred_c, conf_c, cf_cls)

    # --- Col 2: Overlay ---
    ax = axes[row, 2]
    ax.plot(time_ax, sig_o, color=COLOR_ORIG, linewidth=0.55, alpha=0.80, label=f'Original ({orig_cls})')
    ax.plot(time_ax, sig_c, color=COLOR_CF,   linewidth=0.55, alpha=0.80, label=f'CF ({cf_cls})')
    ax.set_ylim(ylims); ax.set_xlim([0, SHOW_SEC])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.10, linewidth=0.3); ax.tick_params(axis='both', length=2)
    if row == 0:
        ax.set_title('Overlay', fontsize=9.5, fontweight='bold')
        ax.legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='gray',
                  fontsize=6.5, handlelength=1.2)
    corr = np.corrcoef(sig_o, sig_c)[0, 1]
    ax.text(0.97, 0.95, f'$r$ = {corr:.2f}', transform=ax.transAxes, ha='right', va='top',
            fontsize=7, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.85, lw=0.3))

for col in range(3):
    axes[3, col].set_xlabel('Time (s)', fontsize=8.5)

pdf_path = OUTPUT_DIR / 'ecg_comparison.pdf'
png_path = OUTPUT_DIR / 'ecg_comparison.png'
fig.savefig(pdf_path, format='pdf')
fig.savefig(png_path, format='png', dpi=300)
plt.close()

print(f"\n✅ {pdf_path}")
print(f"✅ {png_path}")

for i, (orig, cf, direction, orig_cls, cf_cls) in enumerate(panels):
    pred_o, conf_o = classify(orig)
    pred_c, conf_c = classify(cf)
    corr = np.corrcoef(orig[:SHOW_N], cf[:SHOW_N])[0, 1]
    print(f"  {row_labels[i]}: {pred_o} ({conf_o:.0f}%) → {pred_c} ({conf_c:.0f}%), r={corr:.2f}")
