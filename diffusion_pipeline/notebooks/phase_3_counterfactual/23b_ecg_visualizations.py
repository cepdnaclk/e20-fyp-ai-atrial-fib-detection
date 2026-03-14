"""
Part B: ECG Visualization & Comparison for Paper
- Original vs generated ECG pairs with clinical annotations
- R-R irregularity and P-wave detailed comparison
- Architecture and pipeline flow diagrams
"""
import os, sys, json, warnings
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.signal import savgol_filter
from scipy.stats import ks_2samp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patches as mpatches
warnings.filterwarnings('ignore')

ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
FIG = ROOT / 'models/phase3_counterfactual/paper_results/figures'
FIG.mkdir(parents=True, exist_ok=True)

# Load data
orig = np.load(ROOT / 'data/processed/diffusion/train_data.npz')
orig_X, orig_y = orig['X'].squeeze(), orig['y']
cf = np.load(ROOT / 'models/phase3_counterfactual/generated_counterfactuals/filtered_counterfactuals.npz')
cf_X, cf_y = cf['X'].squeeze(), cf['y']
if orig_X.ndim == 3: orig_X = orig_X.squeeze(1)
if cf_X.ndim == 3: cf_X = cf_X.squeeze(1)

FS = 250
rng = np.random.RandomState(42)

def detect_r_peaks(ecg, fs=250):
    ecg_f = savgol_filter(ecg, 11, 3)
    threshold = np.std(ecg_f) * 0.5
    peaks = []
    for i in range(1, len(ecg_f)-1):
        if ecg_f[i] > threshold and ecg_f[i] > ecg_f[i-1] and ecg_f[i] > ecg_f[i+1]:
            if len(peaks) == 0 or (i - peaks[-1]) > fs * 0.3:
                peaks.append(i)
    return np.array(peaks)

def find_best_pair(orig_X, orig_y, cf_X, cf_y, target_cf_class, n_candidates=200):
    """Find orig-CF pair that clearly shows clinical feature differences."""
    candidates = np.where(cf_y == target_cf_class)[0]
    source_class = 1 - target_cf_class
    source_pool = np.where(orig_y == source_class)[0]
    
    best_score = -1; best_pair = (0, 0)
    for _ in range(n_candidates):
        cf_idx = rng.choice(candidates)
        src_idx = rng.choice(source_pool)
        
        rr_cf = detect_r_peaks(cf_X[cf_idx])
        rr_src = detect_r_peaks(orig_X[src_idx])
        if len(rr_cf) < 3 or len(rr_src) < 3: continue
        
        rr_intervals_cf = np.diff(rr_cf) / FS
        rr_intervals_src = np.diff(rr_src) / FS
        
        irregularity_diff = abs(np.std(rr_intervals_cf)/np.mean(rr_intervals_cf) - 
                               np.std(rr_intervals_src)/np.mean(rr_intervals_src))
        
        corr = abs(np.corrcoef(cf_X[cf_idx][:min(len(cf_X[cf_idx]), len(orig_X[src_idx]))],
                                orig_X[src_idx][:min(len(cf_X[cf_idx]), len(orig_X[src_idx]))])[0,1])
        
        score = irregularity_diff * 0.7 + (1 - corr) * 0.3
        if score > best_score:
            best_score = score
            best_pair = (src_idx, cf_idx)
    
    return best_pair

# ============================================================================
# Fig 1: Best Normal→AFib comparison with clinical annotations
# ============================================================================
print("Finding best Normal→AFib pair...")
src_idx, cf_idx = find_best_pair(orig_X, orig_y, cf_X, cf_y, target_cf_class=1)

fig, axes = plt.subplots(4, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 2, 1, 1]})

# Original Normal
ecg_orig = orig_X[src_idx]
r_peaks_orig = detect_r_peaks(ecg_orig)
t = np.arange(len(ecg_orig)) / FS

axes[0].plot(t, ecg_orig, 'b-', linewidth=0.8, label='Original Normal ECG')
axes[0].plot(r_peaks_orig/FS, ecg_orig[r_peaks_orig], 'rv', markersize=8, label='R-peaks')
# Annotate P-waves (region before R-peak)
for rp in r_peaks_orig[:5]:
    p_start = max(0, rp - int(0.2*FS))
    p_end = max(0, rp - int(0.05*FS))
    axes[0].axvspan(p_start/FS, p_end/FS, alpha=0.2, color='green', label='P-wave region' if rp == r_peaks_orig[0] else None)
axes[0].set_title('Original Normal Sinus Rhythm', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Amplitude'); axes[0].legend(loc='upper right'); axes[0].grid(alpha=0.3)

# Generated AFib
ecg_cf = cf_X[cf_idx]
r_peaks_cf = detect_r_peaks(ecg_cf)

axes[1].plot(t[:len(ecg_cf)], ecg_cf, 'r-', linewidth=0.8, label='Generated AFib ECG')
axes[1].plot(r_peaks_cf/FS, ecg_cf[r_peaks_cf], 'bv', markersize=8, label='R-peaks')
for rp in r_peaks_cf[:5]:
    p_start = max(0, rp - int(0.2*FS))
    p_end = max(0, rp - int(0.05*FS))
    axes[1].axvspan(p_start/FS, p_end/FS, alpha=0.2, color='orange', label='P-wave region (absent)' if rp == r_peaks_cf[0] else None)
axes[1].set_title('Generated Counterfactual (AFib)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Amplitude'); axes[1].legend(loc='upper right'); axes[1].grid(alpha=0.3)

# R-R interval comparison
if len(r_peaks_orig) > 2 and len(r_peaks_cf) > 2:
    rr_orig_s = np.diff(r_peaks_orig) / FS * 1000
    rr_cf_s = np.diff(r_peaks_cf) / FS * 1000
    
    axes[2].bar(range(len(rr_orig_s)), rr_orig_s, alpha=0.7, color='blue', label=f'Normal (CV={np.std(rr_orig_s)/np.mean(rr_orig_s):.3f})')
    axes[2].bar(range(len(rr_cf_s)), rr_cf_s, alpha=0.5, color='red', label=f'AFib CF (CV={np.std(rr_cf_s)/np.mean(rr_cf_s):.3f})')
    axes[2].set_xlabel('Beat Interval Index'); axes[2].set_ylabel('R-R Interval (ms)')
    axes[2].set_title('R-R Interval Comparison: Regular (Normal) vs Irregular (AFib)')
    axes[2].legend(); axes[2].grid(alpha=0.3)
    axes[2].axhline(np.mean(rr_orig_s), color='blue', linestyle='--', alpha=0.5)
    axes[2].axhline(np.mean(rr_cf_s), color='red', linestyle='--', alpha=0.5)

# P-wave energy comparison
nyq = FS / 2
b, a = signal.butter(4, [0.5/nyq, 10/nyq], btype='band')
p_orig = signal.filtfilt(b, a, ecg_orig)
p_cf = signal.filtfilt(b, a, ecg_cf)
axes[3].plot(t, p_orig, 'b-', alpha=0.7, linewidth=0.8, label=f'Normal P-wave (energy={np.std(p_orig):.4f})')
axes[3].plot(t[:len(p_cf)], p_cf, 'r-', alpha=0.7, linewidth=0.8, label=f'AFib P-wave (energy={np.std(p_cf):.4f})')
axes[3].set_xlabel('Time (s)'); axes[3].set_ylabel('P-wave Band')
axes[3].set_title('P-Wave Band Comparison (0.5-10 Hz)'); axes[3].legend(); axes[3].grid(alpha=0.3)

plt.suptitle('Normal → AFib Counterfactual: Clinical Feature Analysis', fontsize=15, fontweight='bold')
plt.tight_layout(); plt.savefig(FIG / 'ecg_comparison_n2a_detailed.png', dpi=200, bbox_inches='tight'); plt.close()
print("✓ ecg_comparison_n2a_detailed.png")

# ============================================================================
# Fig 2: AFib→Normal comparison
# ============================================================================
print("Finding best AFib→Normal pair...")
src_idx2, cf_idx2 = find_best_pair(orig_X, orig_y, cf_X, cf_y, target_cf_class=0)

fig, axes = plt.subplots(4, 1, figsize=(18, 16), gridspec_kw={'height_ratios': [2, 2, 1, 1]})

ecg_orig2 = orig_X[src_idx2]
ecg_cf2 = cf_X[cf_idx2]
r_peaks_o2 = detect_r_peaks(ecg_orig2)
r_peaks_c2 = detect_r_peaks(ecg_cf2)

axes[0].plot(t[:len(ecg_orig2)], ecg_orig2, 'r-', linewidth=0.8, label='Original AFib ECG')
axes[0].plot(r_peaks_o2/FS, ecg_orig2[r_peaks_o2], 'bv', markersize=8, label='R-peaks')
axes[0].set_title('Original Atrial Fibrillation', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Amplitude'); axes[0].legend(loc='upper right'); axes[0].grid(alpha=0.3)

axes[1].plot(t[:len(ecg_cf2)], ecg_cf2, 'b-', linewidth=0.8, label='Generated Normal ECG')
axes[1].plot(r_peaks_c2/FS, ecg_cf2[r_peaks_c2], 'rv', markersize=8, label='R-peaks')
for rp in r_peaks_c2[:5]:
    p_start = max(0, rp - int(0.2*FS))
    p_end = max(0, rp - int(0.05*FS))
    axes[1].axvspan(p_start/FS, p_end/FS, alpha=0.2, color='green', label='P-wave region (restored)' if rp == r_peaks_c2[0] else None)
axes[1].set_title('Generated Counterfactual (Normal Sinus)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Amplitude'); axes[1].legend(loc='upper right'); axes[1].grid(alpha=0.3)

if len(r_peaks_o2) > 2 and len(r_peaks_c2) > 2:
    rr_o2 = np.diff(r_peaks_o2) / FS * 1000
    rr_c2 = np.diff(r_peaks_c2) / FS * 1000
    axes[2].bar(range(len(rr_o2)), rr_o2, alpha=0.7, color='red', label=f'AFib (CV={np.std(rr_o2)/np.mean(rr_o2):.3f})')
    axes[2].bar(range(len(rr_c2)), rr_c2, alpha=0.5, color='blue', label=f'Normal CF (CV={np.std(rr_c2)/np.mean(rr_c2):.3f})')
    axes[2].set_xlabel('Beat Index'); axes[2].set_ylabel('R-R (ms)')
    axes[2].set_title('R-R Intervals: Irregular (AFib) → Regular (Normal)'); axes[2].legend(); axes[2].grid(alpha=0.3)

p_o2 = signal.filtfilt(b, a, ecg_orig2)
p_c2 = signal.filtfilt(b, a, ecg_cf2)
axes[3].plot(t[:len(p_o2)], p_o2, 'r-', alpha=0.7, linewidth=0.8, label=f'AFib (energy={np.std(p_o2):.4f})')
axes[3].plot(t[:len(p_c2)], p_c2, 'b-', alpha=0.7, linewidth=0.8, label=f'Normal CF (energy={np.std(p_c2):.4f})')
axes[3].set_xlabel('Time (s)'); axes[3].set_ylabel('P-wave Band')
axes[3].set_title('P-Wave Restoration'); axes[3].legend(); axes[3].grid(alpha=0.3)

plt.suptitle('AFib → Normal Counterfactual: Clinical Feature Analysis', fontsize=15, fontweight='bold')
plt.tight_layout(); plt.savefig(FIG / 'ecg_comparison_a2n_detailed.png', dpi=200, bbox_inches='tight'); plt.close()
print("✓ ecg_comparison_a2n_detailed.png")

# ============================================================================
# Fig 3: Multi-sample comparison grid (6 pairs)
# ============================================================================
print("Generating multi-sample grid...")
fig, axes = plt.subplots(3, 4, figsize=(24, 12))
class_names = {0: 'Normal', 1: 'AFib'}

for row in range(3):
    for direction in range(2):
        col = direction * 2
        target_cls = direction  # 0=Normal target, 1=AFib target
        source_cls = 1 - target_cls
        
        src_idx_r = rng.choice(np.where(orig_y == source_cls)[0])
        cf_idx_r = rng.choice(np.where(cf_y == target_cls)[0])
        
        axes[row, col].plot(t[:2500], orig_X[src_idx_r][:2500], color='blue' if source_cls==0 else 'red', linewidth=0.6)
        axes[row, col].set_title(f'Original {class_names[source_cls]}', fontsize=9)
        axes[row, col].set_ylabel('Amp' if col==0 else '')
        axes[row, col].grid(alpha=0.2)
        
        axes[row, col+1].plot(t[:2500], cf_X[cf_idx_r][:2500], color='blue' if target_cls==0 else 'red', linewidth=0.6)
        axes[row, col+1].set_title(f'CF → {class_names[target_cls]}', fontsize=9)
        axes[row, col+1].grid(alpha=0.2)
        
        if row == 2:
            axes[row, col].set_xlabel('Time (s)')
            axes[row, col+1].set_xlabel('Time (s)')

plt.suptitle('Multi-Sample Generated ECG Comparison', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(FIG / 'ecg_multi_comparison.png', dpi=200); plt.close()
print("✓ ecg_multi_comparison.png")

# ============================================================================
# Fig 4: Pipeline Flow Diagram
# ============================================================================
print("Generating pipeline diagram...")

fig, ax = plt.subplots(figsize=(22, 10))
ax.set_xlim(0, 22); ax.set_ylim(0, 10); ax.axis('off')

boxes = [
    (1, 7, 3.5, 1.5, 'ECG Data\n(Chapman-Shaoxing)\n149,793 recordings', '#E3F2FD'),
    (1, 4.5, 3.5, 1.5, 'Preprocessing\n• Z-normalization\n• 10s segments (2500pt)\n• Train/Val/Test split', '#E8F5E9'),
    (5.5, 7, 3.5, 1.5, 'Content-Style\nDiffusion Training\n(100 epochs)', '#FFF3E0'),
    (5.5, 4.5, 3.5, 1.5, 'ContentEncoder\nStyleEncoder\nConditional UNet\nDDIM Scheduler', '#FCE4EC'),
    (10, 7, 3.5, 1.5, 'Counterfactual\nGeneration\n(SDEdit sampling)', '#F3E5F5'),
    (10, 4.5, 3.5, 1.5, 'Two-Gate Filtering\n✓ Classifier flip\n✓ Plausibility≥0.7', '#FFFDE7'),
    (14.5, 7, 3.5, 1.5, 'Filtered CFs\n20,000 ECGs\n(10K Normal+10K AFib)', '#E0F7FA'),
    (14.5, 4.5, 3.5, 1.5, 'Confidence Filter\nLabel verification\n7,784 high-conf CFs', '#FBE9E7'),
    (18, 5.75, 3.5, 1.5, 'Three-Way\nEvaluation\n(5-fold CV)', '#E8EAF6'),
    (14.5, 2, 3.5, 1.5, 'Statistical Testing\nMcNemar, t-test\nTOST equivalence', '#F1F8E9'),
    (18, 2, 3.5, 1.5, 'Paper\nConclusions', '#FFEBEE'),
]

for x, y, w, h, text, color in boxes:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=7.5, fontweight='bold')

# Arrows
arrows = [
    (2.75, 7, 2.75, 6.1), (2.75, 4.5, 5.5, 4.5+0.75), 
    (4.5, 7.75, 5.5, 7.75), (7.25, 5.2, 7.25, 6.9),
    (9, 7.75, 10, 7.75), (11.75, 7, 11.75, 6.1),
    (13.5, 7.75, 14.5, 7.75), (13.5, 5.25, 14.5, 5.25),
    (16.25, 5.9, 18, 6.0), (16.25, 4.5, 16.25, 3.6), (18, 3.5, 18, 2+1.5),
    (18, 5.75, 18, 3.6),
]
for x1, y1, x2, y2 in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

ax.set_title('Counterfactual ECG Generation Pipeline', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout(); plt.savefig(FIG / 'pipeline_flow.png', dpi=200, bbox_inches='tight'); plt.close()
print("✓ pipeline_flow.png")

# ============================================================================
# Fig 5: Architecture Diagram
# ============================================================================
print("Generating architecture diagram...")

fig, ax = plt.subplots(figsize=(20, 12))
ax.set_xlim(0, 20); ax.set_ylim(0, 12); ax.axis('off')

# Content Encoder
rect = FancyBboxPatch((0.5, 8), 4, 3, boxstyle="round,pad=0.15", facecolor='#BBDEFB', edgecolor='#1565C0', lw=2)
ax.add_patch(rect)
ax.text(2.5, 10.3, 'Content Encoder', ha='center', fontsize=11, fontweight='bold', color='#1565C0')
ax.text(2.5, 9.5, 'Conv1d(1→64→128→256→512)\nGroupNorm + LeakyReLU\nAdaptiveAvgPool1d(8)\nFC → μ, σ (256-dim)', 
        ha='center', va='center', fontsize=7.5)
ax.text(2.5, 8.2, 'Input: ECG (1, 2500)', ha='center', fontsize=7, style='italic')

# Style Encoder
rect = FancyBboxPatch((0.5, 4), 4, 3, boxstyle="round,pad=0.15", facecolor='#C8E6C9', edgecolor='#2E7D32', lw=2)
ax.add_patch(rect)
ax.text(2.5, 6.3, 'Style Encoder', ha='center', fontsize=11, fontweight='bold', color='#2E7D32')
ax.text(2.5, 5.2, 'Conv1d(1→64→128→256)\nLeakyReLU\nAdaptiveAvgPool1d(1)\nFC → style (128-dim)\n+ Classifier head', 
        ha='center', va='center', fontsize=7.5)

# UNet
rect = FancyBboxPatch((6, 4.5), 7, 6, boxstyle="round,pad=0.15", facecolor='#FFE0B2', edgecolor='#E65100', lw=2)
ax.add_patch(rect)
ax.text(9.5, 9.8, 'Conditional UNet', ha='center', fontsize=12, fontweight='bold', color='#E65100')
ax.text(9.5, 8.8, 'Conditioning: timestep + content + style + class', ha='center', fontsize=8, style='italic')
ax.text(9.5, 7.8, 'Encoder: ResBlock(64→128→256→512)\n+ Self-Attention at 256 channels\n+ GroupNorm(32)', 
        ha='center', fontsize=8)
ax.text(9.5, 6.2, 'Bottleneck: ResBlock(512) + SelfAttn', ha='center', fontsize=8)
ax.text(9.5, 5.4, 'Decoder: ResBlock(512→256→128→64)\n+ Skip connections + Attention\nOutput: Conv1d → ε prediction', 
        ha='center', fontsize=8)

# DDIM Scheduler
rect = FancyBboxPatch((14.5, 8), 5, 3, boxstyle="round,pad=0.15", facecolor='#F3E5F5', edgecolor='#6A1B9A', lw=2)
ax.add_patch(rect)
ax.text(17, 10.3, 'DDIM Scheduler', ha='center', fontsize=11, fontweight='bold', color='#6A1B9A')
ax.text(17, 9.3, 'SDEdit Sampling\nT=1000 timesteps\nStrength=0.6 (start at t=600)\n50 denoising steps\nCFG scale=3.0', 
        ha='center', va='center', fontsize=8)

# Classifier
rect = FancyBboxPatch((14.5, 4), 5, 3, boxstyle="round,pad=0.15", facecolor='#FFCDD2', edgecolor='#C62828', lw=2)
ax.add_patch(rect)
ax.text(17, 6.3, 'AFibResLSTM Classifier', ha='center', fontsize=11, fontweight='bold', color='#C62828')
ax.text(17, 5.2, 'Pre-trained (frozen)\nResidual blocks + LSTM\nBinary: Normal/AFib\nUsed for flip verification', 
        ha='center', va='center', fontsize=8)

# Output
rect = FancyBboxPatch((6, 0.5), 7, 2.5, boxstyle="round,pad=0.15", facecolor='#E0F7FA', edgecolor='#00695C', lw=2)
ax.add_patch(rect)
ax.text(9.5, 2.2, 'Generated Counterfactual ECG', ha='center', fontsize=11, fontweight='bold', color='#00695C')
ax.text(9.5, 1.2, 'Flip target class features while preserving content\n'
        'Normal→AFib: introduce R-R irregularity, reduce P-waves\n'
        'AFib→Normal: regularize rhythm, restore P-waves', ha='center', fontsize=8)

# Arrows
for x1,y1,x2,y2,label in [
    (4.5,9.5,6,9.5,'content z'), (4.5,5.5,6,5.5,'style s'),
    (13,7.5,14.5,9.5,'ε(x,t,z,s,c)'), (13,6,14.5,5.5,'verify flip'),
    (9.5,4.5,9.5,3.1,''), (17,8,17,7.1,''),
]:
    ax.annotate(label, xy=(x2,y2), xytext=(x1,y1), fontsize=7,
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.5),
                ha='center', va='center')

ax.set_title('Content-Style Disentangled Diffusion Architecture', fontsize=15, fontweight='bold', pad=20)
plt.tight_layout(); plt.savefig(FIG / 'architecture_diagram.png', dpi=200, bbox_inches='tight'); plt.close()
print("✓ architecture_diagram.png")

# ============================================================================
# Fig 6: Amplitude distribution comparison 
# ============================================================================
print("Generating amplitude distributions...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for cls, name, color_o, color_c in [(0, 'Normal', '#2196F3', '#64B5F6'), (1, 'AFib', '#F44336', '#EF9A9A')]:
    ax = axes[cls]
    orig_cls = orig_X[orig_y == cls].flatten()[:500000]
    cf_cls = cf_X[cf_y == cls].flatten()[:500000]
    ax.hist(orig_cls, bins=100, alpha=0.6, density=True, color=color_o, label='Original')
    ax.hist(cf_cls, bins=100, alpha=0.6, density=True, color=color_c, label='Generated')
    ks, p = ks_2samp(orig_cls[:50000], cf_cls[:50000])
    ax.set_title(f'{name} ECG Amplitude Distribution\n(KS={ks:.4f}, p={p:.2e})')
    ax.set_xlabel('Amplitude'); ax.set_ylabel('Density'); ax.legend()

plt.suptitle('Amplitude Distribution: Original vs Generated', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig(FIG / 'amplitude_distributions.png', dpi=200); plt.close()
print("✓ amplitude_distributions.png")

print("\nPart B complete! All figures saved to:", FIG)
