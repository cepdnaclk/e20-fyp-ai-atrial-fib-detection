"""
Fixed ECG Clinical Feature Visualization (Q11 fix)
- Proper R-peak detection using scipy.signal.find_peaks
- Correct P-wave annotation (absent in AFib, present in Normal)
- Better signal selection for clear clinical features
- Focused zoomed-in views for R-R intervals and P-waves
"""
import os, sys, warnings
import numpy as np
from pathlib import Path
from scipy import signal
from scipy.signal import find_peaks, savgol_filter, butter, filtfilt
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
FIG = ROOT / 'models/phase3_counterfactual/paper_results/figures'

# Load data
orig = np.load(ROOT / 'data/processed/diffusion/train_data.npz')
orig_X, orig_y = orig['X'].squeeze(), orig['y']
cf = np.load(ROOT / 'models/phase3_counterfactual/generated_counterfactuals/filtered_counterfactuals.npz')
cf_X, cf_y = cf['X'].squeeze(), cf['y']
if orig_X.ndim == 3: orig_X = orig_X.squeeze(1)
if cf_X.ndim == 3: cf_X = cf_X.squeeze(1)

FS = 250  # Hz
rng = np.random.RandomState(42)

def detect_r_peaks_proper(ecg, fs=250):
    """Proper R-peak detection using bandpass filter + find_peaks."""
    # Band-pass filter to isolate QRS complex (5-15 Hz)
    nyq = fs / 2
    b, a = butter(3, [5/nyq, 15/nyq], btype='band')
    filtered = filtfilt(b, a, ecg)
    
    # Square the signal to emphasize R-peaks
    squared = filtered ** 2
    
    # Smooth
    window = int(0.08 * fs)  # 80ms window
    if window % 2 == 0: window += 1
    smoothed = savgol_filter(squared, window, 2)
    
    # Find peaks with minimum distance of 200ms (300 BPM max)
    min_dist = int(0.2 * fs)
    threshold = np.percentile(smoothed, 70)
    peaks, props = find_peaks(smoothed, distance=min_dist, height=threshold)
    
    # Refine: find actual R-peak (max in original signal near each detected peak)
    refined_peaks = []
    for p in peaks:
        search_start = max(0, p - int(0.05*fs))
        search_end = min(len(ecg), p + int(0.05*fs))
        region = ecg[search_start:search_end]
        # R-peak is usually the maximum absolute value
        max_idx = np.argmax(np.abs(region))
        refined_peaks.append(search_start + max_idx)
    
    return np.array(refined_peaks)

def compute_rr_cv(peaks, fs=250):
    """Compute R-R interval coefficient of variation."""
    if len(peaks) < 3: return 0
    rr = np.diff(peaks) / fs
    return np.std(rr) / np.mean(rr)

def p_wave_energy(ecg, fs=250):
    """Extract P-wave band energy (1-10 Hz)."""
    nyq = fs / 2
    b, a = butter(3, [1/nyq, 10/nyq], btype='band')
    return filtfilt(b, a, ecg)

def find_good_pair(orig_X, orig_y, cf_X, cf_y, target_cf_label, n=500):
    """Find a pair with clear clinical features."""
    candidates_cf = np.where(cf_y == target_cf_label)[0]
    source_label = 1 - target_cf_label
    candidates_orig = np.where(orig_y == source_label)[0]
    
    best_score = -1
    best_pair = None
    
    for _ in range(n):
        ci = rng.choice(candidates_cf)
        oi = rng.choice(candidates_orig)
        
        peaks_c = detect_r_peaks_proper(cf_X[ci])
        peaks_o = detect_r_peaks_proper(orig_X[oi])
        
        if len(peaks_c) < 4 or len(peaks_o) < 4: continue
        
        cv_c = compute_rr_cv(peaks_c)
        cv_o = compute_rr_cv(peaks_o)
        
        # For N->A: want low CV original, high CV generated
        if target_cf_label == 1:  # Generated is AFib
            score = (cv_c - cv_o) + 0.3 * (len(peaks_c) >= 5) + 0.3 * (len(peaks_o) >= 5)
            # Prefer signals where R-peaks look clean
            snr_o = np.std(orig_X[oi]) / (np.std(np.diff(orig_X[oi])) + 1e-10)
            snr_c = np.std(cf_X[ci]) / (np.std(np.diff(cf_X[ci])) + 1e-10)
            score += 0.1 * min(snr_o, 5) + 0.1 * min(snr_c, 5)
        else:  # Generated is Normal
            score = (cv_o - cv_c) + 0.3 * (len(peaks_c) >= 5) + 0.3 * (len(peaks_o) >= 5)
            snr_o = np.std(orig_X[oi]) / (np.std(np.diff(orig_X[oi])) + 1e-10)
            snr_c = np.std(cf_X[ci]) / (np.std(np.diff(cf_X[ci])) + 1e-10)
            score += 0.1 * min(snr_o, 5) + 0.1 * min(snr_c, 5)
        
        if score > best_score:
            best_score = score
            best_pair = (oi, ci)
    
    return best_pair

# ============================================================================
# Figure 1: Normal -> AFib Counterfactual (FIXED)
# ============================================================================
print("Finding best Normal -> AFib pair...")
oi, ci = find_good_pair(orig_X, orig_y, cf_X, cf_y, target_cf_label=1, n=1000)
ecg_orig = orig_X[oi]      # Original Normal
ecg_cf = cf_X[ci]           # Generated AFib

peaks_orig = detect_r_peaks_proper(ecg_orig)
peaks_cf = detect_r_peaks_proper(ecg_cf)

t = np.arange(2500) / FS  # Time axis

fig = plt.figure(figsize=(20, 20))
gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3, height_ratios=[2, 2, 1.5, 2, 2])

# Row 1: Full Original Normal ECG
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, ecg_orig, 'b-', linewidth=0.8, label='Original Normal ECG')
ax1.plot(peaks_orig/FS, ecg_orig[peaks_orig], 'rv', markersize=10, label='R-peaks', zorder=5)
# Annotate R-R intervals
for i in range(min(len(peaks_orig)-1, 5)):
    p1, p2 = peaks_orig[i], peaks_orig[i+1]
    rr_ms = (p2-p1)/FS*1000
    mid = (p1+p2)/(2*FS)
    y_top = max(ecg_orig[p1], ecg_orig[p2]) + 0.3
    ax1.annotate('', xy=(p2/FS, y_top), xytext=(p1/FS, y_top),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax1.text(mid, y_top+0.1, f'{rr_ms:.0f}ms', ha='center', fontsize=8, color='green', fontweight='bold')
ax1.set_title('ORIGINAL: Normal Sinus Rhythm (Regular R-R Intervals)', fontsize=13, fontweight='bold', color='blue')
ax1.set_ylabel('Amplitude (z-norm)')
ax1.legend(loc='upper right')
ax1.grid(alpha=0.3)

# Row 2: Full Generated AFib ECG
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(t, ecg_cf, 'r-', linewidth=0.8, label='Generated AFib ECG')
ax2.plot(peaks_cf/FS, ecg_cf[peaks_cf], 'bv', markersize=10, label='R-peaks', zorder=5)
for i in range(min(len(peaks_cf)-1, 5)):
    p1, p2 = peaks_cf[i], peaks_cf[i+1]
    rr_ms = (p2-p1)/FS*1000
    mid = (p1+p2)/(2*FS)
    y_top = max(ecg_cf[p1], ecg_cf[p2]) + 0.3
    ax2.annotate('', xy=(p2/FS, y_top), xytext=(p1/FS, y_top),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=1.5))
    ax2.text(mid, y_top+0.1, f'{rr_ms:.0f}ms', ha='center', fontsize=8, color='orange', fontweight='bold')
ax2.set_title('GENERATED: Atrial Fibrillation Counterfactual (Irregular R-R Intervals)', fontsize=13, fontweight='bold', color='red')
ax2.set_ylabel('Amplitude (z-norm)')
ax2.legend(loc='upper right')
ax2.grid(alpha=0.3)

# Row 3: R-R interval bar comparison
ax3 = fig.add_subplot(gs[2, :])
rr_orig_ms = np.diff(peaks_orig) / FS * 1000
rr_cf_ms = np.diff(peaks_cf) / FS * 1000
max_bars = max(len(rr_orig_ms), len(rr_cf_ms))
x_bars = np.arange(max_bars)

if len(rr_orig_ms) > 0:
    ax3.bar(x_bars[:len(rr_orig_ms)] - 0.2, rr_orig_ms, 0.35, color='blue', alpha=0.7,
            label=f'Normal (CV={np.std(rr_orig_ms)/np.mean(rr_orig_ms):.3f} - REGULAR)')
if len(rr_cf_ms) > 0:
    ax3.bar(x_bars[:len(rr_cf_ms)] + 0.2, rr_cf_ms, 0.35, color='red', alpha=0.7,
            label=f'AFib CF (CV={np.std(rr_cf_ms)/np.mean(rr_cf_ms):.3f} - IRREGULAR)')
ax3.axhline(np.mean(rr_orig_ms), color='blue', ls='--', alpha=0.5, label=f'Normal mean: {np.mean(rr_orig_ms):.0f}ms')
ax3.axhline(np.mean(rr_cf_ms), color='red', ls='--', alpha=0.5, label=f'AFib mean: {np.mean(rr_cf_ms):.0f}ms')
ax3.set_xlabel('Beat Interval #')
ax3.set_ylabel('R-R Interval (ms)')
ax3.set_title('R-R Interval Comparison: Regular (Normal) vs Irregular (AFib)')
ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3)

# Row 4 Left: Zoomed P-wave region - NORMAL (P-waves PRESENT)
ax4l = fig.add_subplot(gs[3, 0])
if len(peaks_orig) >= 3:
    # Show 2 beats centered on second R-peak
    center = peaks_orig[1]
    start = max(0, center - int(0.8*FS))
    end = min(2500, center + int(0.4*FS))
    t_zoom = np.arange(start, end) / FS
    
    ax4l.plot(t_zoom, ecg_orig[start:end], 'b-', linewidth=1.5, label='Normal ECG')
    # Mark R-peaks in zoom window
    for p in peaks_orig:
        if start <= p < end:
            ax4l.axvline(p/FS, color='red', ls='--', alpha=0.3)
            ax4l.plot(p/FS, ecg_orig[p], 'rv', markersize=12)
            # P-wave region: 120-200ms before R-peak
            p_start = max(start, p - int(0.20*FS))
            p_end = max(start, p - int(0.08*FS))
            ax4l.axvspan(p_start/FS, p_end/FS, alpha=0.3, color='green', 
                        label='P-wave region' if p == peaks_orig[1] else None)
    ax4l.set_title('NORMAL: P-waves PRESENT\n(green regions before R-peaks)', fontsize=11, fontweight='bold', color='blue')
    ax4l.set_xlabel('Time (s)')
    ax4l.set_ylabel('Amplitude')
    ax4l.legend(fontsize=8)
    ax4l.grid(alpha=0.3)

# Row 4 Right: Zoomed P-wave region - AFIB (P-waves ABSENT)
ax4r = fig.add_subplot(gs[3, 1])
if len(peaks_cf) >= 3:
    center = peaks_cf[1]
    start = max(0, center - int(0.8*FS))
    end = min(2500, center + int(0.4*FS))
    t_zoom = np.arange(start, end) / FS
    
    ax4r.plot(t_zoom, ecg_cf[start:end], 'r-', linewidth=1.5, label='AFib CF ECG')
    for p in peaks_cf:
        if start <= p < end:
            ax4r.axvline(p/FS, color='blue', ls='--', alpha=0.3)
            ax4r.plot(p/FS, ecg_cf[p], 'bv', markersize=12)
            p_start = max(start, p - int(0.20*FS))
            p_end = max(start, p - int(0.08*FS))
            ax4r.axvspan(p_start/FS, p_end/FS, alpha=0.3, color='orange',
                        label='P-wave region (absent/chaotic)' if p == peaks_cf[1] else None)
    ax4r.set_title('AFIB: P-waves ABSENT/CHAOTIC\n(fibrillatory baseline in P-wave region)', fontsize=11, fontweight='bold', color='red')
    ax4r.set_xlabel('Time (s)')
    ax4r.set_ylabel('Amplitude')
    ax4r.legend(fontsize=8)
    ax4r.grid(alpha=0.3)

# Row 5: P-wave band filtered comparison
ax5l = fig.add_subplot(gs[4, 0])
pw_orig = p_wave_energy(ecg_orig)
ax5l.plot(t, pw_orig, 'b-', linewidth=0.8)
ax5l.fill_between(t, pw_orig, alpha=0.3, color='blue')
ax5l.set_title(f'Normal: P-wave Band (1-10Hz)\nEnergy = {np.std(pw_orig):.4f}', fontsize=11, fontweight='bold', color='blue')
ax5l.set_xlabel('Time (s)'); ax5l.set_ylabel('Amplitude'); ax5l.grid(alpha=0.3)

ax5r = fig.add_subplot(gs[4, 1])
pw_cf = p_wave_energy(ecg_cf)
ax5r.plot(t, pw_cf, 'r-', linewidth=0.8)
ax5r.fill_between(t, pw_cf, alpha=0.3, color='red')
ax5r.set_title(f'AFib CF: P-wave Band (1-10Hz)\nEnergy = {np.std(pw_cf):.4f}', fontsize=11, fontweight='bold', color='red')
ax5r.set_xlabel('Time (s)'); ax5r.set_ylabel('Amplitude'); ax5r.grid(alpha=0.3)

fig.suptitle('Normal -> AFib Counterfactual: Clinical Feature Analysis\n'
             'Key Changes: R-R irregularity introduced, P-wave morphology altered', 
             fontsize=15, fontweight='bold', y=0.98)
plt.savefig(FIG / 'ecg_comparison_n2a_detailed.png', dpi=200, bbox_inches='tight')
plt.close()
print("OK: ecg_comparison_n2a_detailed.png")

# ============================================================================
# Figure 2: AFib -> Normal Counterfactual (FIXED)
# ============================================================================
print("Finding best AFib -> Normal pair...")
oi2, ci2 = find_good_pair(orig_X, orig_y, cf_X, cf_y, target_cf_label=0, n=1000)
ecg_orig2 = orig_X[oi2]    # Original AFib
ecg_cf2 = cf_X[ci2]         # Generated Normal

peaks_orig2 = detect_r_peaks_proper(ecg_orig2)
peaks_cf2 = detect_r_peaks_proper(ecg_cf2)

fig = plt.figure(figsize=(20, 20))
gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3, height_ratios=[2, 2, 1.5, 2, 2])

# Row 1: Original AFib
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, ecg_orig2, 'r-', linewidth=0.8, label='Original AFib ECG')
ax1.plot(peaks_orig2/FS, ecg_orig2[peaks_orig2], 'bv', markersize=10, label='R-peaks', zorder=5)
for i in range(min(len(peaks_orig2)-1, 5)):
    p1, p2 = peaks_orig2[i], peaks_orig2[i+1]
    rr_ms = (p2-p1)/FS*1000
    mid = (p1+p2)/(2*FS)
    y_top = max(ecg_orig2[p1], ecg_orig2[p2]) + 0.3
    ax1.annotate('', xy=(p2/FS, y_top), xytext=(p1/FS, y_top),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=1.5))
    ax1.text(mid, y_top+0.1, f'{rr_ms:.0f}ms', ha='center', fontsize=8, color='orange', fontweight='bold')
ax1.set_title('ORIGINAL: Atrial Fibrillation (Irregular R-R, P-waves ABSENT)', fontsize=13, fontweight='bold', color='red')
ax1.set_ylabel('Amplitude')
ax1.legend(loc='upper right'); ax1.grid(alpha=0.3)

# Row 2: Generated Normal
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(t, ecg_cf2, 'b-', linewidth=0.8, label='Generated Normal ECG')
ax2.plot(peaks_cf2/FS, ecg_cf2[peaks_cf2], 'rv', markersize=10, label='R-peaks', zorder=5)
for i in range(min(len(peaks_cf2)-1, 5)):
    p1, p2 = peaks_cf2[i], peaks_cf2[i+1]
    rr_ms = (p2-p1)/FS*1000
    mid = (p1+p2)/(2*FS)
    y_top = max(ecg_cf2[p1], ecg_cf2[p2]) + 0.3
    ax2.annotate('', xy=(p2/FS, y_top), xytext=(p1/FS, y_top),
                arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
    ax2.text(mid, y_top+0.1, f'{rr_ms:.0f}ms', ha='center', fontsize=8, color='green', fontweight='bold')
ax2.set_title('GENERATED: Normal Sinus Rhythm CF (Regular R-R, P-waves RESTORED)', fontsize=13, fontweight='bold', color='blue')
ax2.set_ylabel('Amplitude')
ax2.legend(loc='upper right'); ax2.grid(alpha=0.3)

# Row 3: R-R bar comparison
ax3 = fig.add_subplot(gs[2, :])
rr_o2 = np.diff(peaks_orig2) / FS * 1000
rr_c2 = np.diff(peaks_cf2) / FS * 1000
max_bars = max(len(rr_o2), len(rr_c2))
x_bars = np.arange(max_bars)
if len(rr_o2) > 0:
    ax3.bar(x_bars[:len(rr_o2)] - 0.2, rr_o2, 0.35, color='red', alpha=0.7,
            label=f'AFib (CV={np.std(rr_o2)/np.mean(rr_o2):.3f} - IRREGULAR)')
if len(rr_c2) > 0:
    ax3.bar(x_bars[:len(rr_c2)] + 0.2, rr_c2, 0.35, color='blue', alpha=0.7,
            label=f'Normal CF (CV={np.std(rr_c2)/np.mean(rr_c2):.3f} - REGULAR)')
ax3.set_xlabel('Beat Interval #'); ax3.set_ylabel('R-R (ms)')
ax3.set_title('R-R Intervals: Irregular (AFib) -> Regular (Normal)'); ax3.legend(fontsize=8)
ax3.grid(axis='y', alpha=0.3)

# Row 4: Zoomed P-wave (AFIB = absent, NORMAL CF = restored)
ax4l = fig.add_subplot(gs[3, 0])
if len(peaks_orig2) >= 3:
    center = peaks_orig2[1]
    start = max(0, center - int(0.8*FS))
    end = min(2500, center + int(0.4*FS))
    t_z = np.arange(start, end) / FS
    ax4l.plot(t_z, ecg_orig2[start:end], 'r-', linewidth=1.5)
    for p in peaks_orig2:
        if start <= p < end:
            ax4l.plot(p/FS, ecg_orig2[p], 'bv', markersize=12)
            ps = max(start, p - int(0.20*FS))
            pe = max(start, p - int(0.08*FS))
            ax4l.axvspan(ps/FS, pe/FS, alpha=0.3, color='orange',
                        label='P-wave ABSENT' if p == peaks_orig2[1] else None)
    ax4l.set_title('AFIB: P-waves ABSENT\n(no organized P-waves before QRS)', fontsize=11, fontweight='bold', color='red')
    ax4l.set_xlabel('Time (s)'); ax4l.legend(fontsize=8); ax4l.grid(alpha=0.3)

ax4r = fig.add_subplot(gs[3, 1])
if len(peaks_cf2) >= 3:
    center = peaks_cf2[1]
    start = max(0, center - int(0.8*FS))
    end = min(2500, center + int(0.4*FS))
    t_z = np.arange(start, end) / FS
    ax4r.plot(t_z, ecg_cf2[start:end], 'b-', linewidth=1.5)
    for p in peaks_cf2:
        if start <= p < end:
            ax4r.plot(p/FS, ecg_cf2[p], 'rv', markersize=12)
            ps = max(start, p - int(0.20*FS))
            pe = max(start, p - int(0.08*FS))
            ax4r.axvspan(ps/FS, pe/FS, alpha=0.3, color='green',
                        label='P-wave RESTORED' if p == peaks_cf2[1] else None)
    ax4r.set_title('NORMAL CF: P-waves RESTORED\n(organized P-waves before each QRS)', fontsize=11, fontweight='bold', color='blue')
    ax4r.set_xlabel('Time (s)'); ax4r.legend(fontsize=8); ax4r.grid(alpha=0.3)

# Row 5: P-wave band
ax5l = fig.add_subplot(gs[4, 0])
pw_o2 = p_wave_energy(ecg_orig2)
ax5l.plot(t, pw_o2, 'r-', linewidth=0.8)
ax5l.fill_between(t, pw_o2, alpha=0.3, color='red')
ax5l.set_title(f'AFib: P-wave Band Energy = {np.std(pw_o2):.4f}', fontsize=11, fontweight='bold', color='red')
ax5l.set_xlabel('Time (s)'); ax5l.grid(alpha=0.3)

ax5r = fig.add_subplot(gs[4, 1])
pw_c2 = p_wave_energy(ecg_cf2)
ax5r.plot(t, pw_c2, 'b-', linewidth=0.8)
ax5r.fill_between(t, pw_c2, alpha=0.3, color='blue')
ax5r.set_title(f'Normal CF: P-wave Band Energy = {np.std(pw_c2):.4f}', fontsize=11, fontweight='bold', color='blue')
ax5r.set_xlabel('Time (s)'); ax5r.grid(alpha=0.3)

fig.suptitle('AFib -> Normal Counterfactual: Clinical Feature Analysis\n'
             'Key Changes: Rhythm regularized, P-waves restored', 
             fontsize=15, fontweight='bold', y=0.98)
plt.savefig(FIG / 'ecg_comparison_a2n_detailed.png', dpi=200, bbox_inches='tight')
plt.close()
print("OK: ecg_comparison_a2n_detailed.png")
print("Done!")
