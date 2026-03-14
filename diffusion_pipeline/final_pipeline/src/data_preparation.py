
"""
Phase 2 Diffusion - Data Preparation (FIXED - GLOBAL NORMALIZATION)

Load MIMIC-IV raw data and preprocess for diffusion training

KEY STRATEGY (FIXED):
- Extract and clean ECG signals
- Apply GLOBAL normalization across ALL signals (preserves relative amplitudes)
- This prevents artificial clipping at boundaries
"""

import sys
sys.path.append('D:/research/codes')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings('ignore')

# Signal processing
from scipy import signal
from scipy.signal import butter, filtfilt, resample
import neurokit2 as nk

# PyTorch
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# WFDB for MIMIC-IV
import wfdb

print("="*80)
print(" PHASE 2 DIFFUSION - DATA PREPARATION (FIXED VERSION)")
print("="*80)
print(f"PyTorch version: {torch.__version__}")
print(f"WFDB version: {wfdb.__version__}")
print(f"NeuroKit2 version: {nk.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
print("="*80)


"""
Configuration for MIMIC-IV data preparation
"""

class DataConfig:
    # ========================================================================
    # PATHS
    # ========================================================================
    
    PROJECT_ROOT = Path('/scratch1/e20-fyp-ai-atrial-fib-det/diffusion_pipeline')
    DATA_RAW = PROJECT_ROOT / 'data/raw'
    DATA_PROCESSED = PROJECT_ROOT / 'data/processed'
    
    # MIMIC-IV ECG Database path
    MIMIC_IV_PATH = DATA_RAW / 'mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0'
    
    # Output directory for diffusion
    DIFFUSION_DATA_DIR = DATA_PROCESSED / 'diffusion'
    DIFFUSION_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # ECG PARAMETERS
    # ========================================================================
    SAMPLING_RATE = 250
    SIGNAL_LENGTH = 2500
    DURATION = 10
    LEAD_TO_USE = 'II'
    
    # ========================================================================
    # PREPROCESSING STRATEGY (FIXED!)
    # ========================================================================
    USE_NEUROKIT = True
    
    # Filtering parameters
    LOWCUT = 0.5
    HIGHCUT = 40
    FILTER_ORDER = 4
    
    #  FIXED: Global normalization (not per-signal!)
    NORMALIZE_METHOD = 'global_clinical'  # Changed from 'clinical'
    CLINICAL_MIN = -1.5
    CLINICAL_MAX = 1.5
    
    # ========================================================================
    # DATA SAMPLING
    # ========================================================================
    TARGET_SAMPLES = 150000
    TARGET_AFIB_SAMPLES = 75000
    TARGET_NORMAL_SAMPLES = 75000
    
    # Train/Val/Test split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
   
    MIN_SIGNAL_QUALITY = 0.7
    MAX_FLATLINE_RATIO = 0.1
    REMOVE_ARTIFACTS = True
  
    BATCH_SIZE_PROCESSING = 1000
    NUM_WORKERS = 4
    
 
    SEED = 42
   
    PHASE1_CLASSIFIER_PATH = PROJECT_ROOT / 'models/afib_reslstm_final.pth'
    PHASE1_ARCHITECTURE_PATH = PROJECT_ROOT / 'models/model_architecture.py'

# Set random seeds
np.random.seed(DataConfig.SEED)
torch.manual_seed(DataConfig.SEED)

print("="*80)
print(" CONFIGURATION LOADED (FIXED VERSION)")
print("="*80)
print("\n KEY FIX:")
print("   • Changed to GLOBAL normalization (preserves relative amplitudes)")
print("   • No more per-signal min-max scaling")
print("   • Prevents artificial clipping at boundaries")
print("="*80)


"""
Signal preprocessing functions (FIXED VERSION)
"""

def clean_ecg_signal(ecg_signal, sampling_rate=250, use_neurokit=True):
    """
    Clean ECG signal using NeuroKit2
    """
    if use_neurokit:
        try:
            cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
            return cleaned
        except Exception as e:
            return butter_bandpass_filter(ecg_signal, 0.5, 40, sampling_rate, order=4)
    else:
        return butter_bandpass_filter(ecg_signal, 0.5, 40, sampling_rate, order=4)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Butterworth bandpass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def assess_signal_quality(ecg_signal, sampling_rate=250):
    """
    Assess ECG signal quality using simple heuristics
    """
    try:
        quality_score = 1.0
        
        # 1. Check for flatlines
        diff = np.diff(ecg_signal)
        flatline_ratio = np.sum(np.abs(diff) < 1e-8) / len(diff)
        if flatline_ratio > 0.5:
            quality_score *= 0.3
        elif flatline_ratio > 0.3:
            quality_score *= 0.6
        
        # 2. Check signal variance
        std_val = np.std(ecg_signal)
        if std_val < 0.01:
            quality_score *= 0.2
        elif std_val < 0.03:
            quality_score *= 0.7
        
        # 3. Check for extreme values
        extreme_ratio = np.sum(np.abs(ecg_signal) > 5) / len(ecg_signal)
        if extreme_ratio > 0.1:
            quality_score *= 0.3
        elif extreme_ratio > 0.01:
            quality_score *= 0.8
        
        # 4. Check for NaN/Inf
        if np.any(np.isnan(ecg_signal)) or np.any(np.isinf(ecg_signal)):
            quality_score = 0.0
        
        # 5. Check signal range
        signal_range = np.max(ecg_signal) - np.min(ecg_signal)
        if signal_range < 0.05:
            quality_score *= 0.3
        
    except Exception as e:
        quality_score = 0.5 if np.std(ecg_signal) > 0.01 else 0.0
    
    is_good = quality_score >= 0.5
    
    return quality_score, is_good


def preprocess_single_ecg_no_norm(ecg_signal,
                                   target_length=2500,
                                   target_fs=250):
    """
    Preprocessing pipeline WITHOUT normalization
    Normalization will be applied globally to ALL signals later
    
    Args:
        ecg_signal: Raw ECG signal
        target_length: Target length (2500)
        target_fs: Target sampling rate (250 Hz)
    
    Returns:
        Cleaned ECG (NOT normalized yet) or None
    """
    # 1. Resample if needed
    if len(ecg_signal) != target_length:
        ecg_signal = resample(ecg_signal, target_length)
    
    # 2. Clean with NeuroKit2
    cleaned = clean_ecg_signal(ecg_signal, target_fs, use_neurokit=True)
    
    # 3. Quality check
    quality_score, is_good = assess_signal_quality(cleaned, target_fs)
    
    if not is_good:
        return None
    
    # 4. Ensure correct length
    if len(cleaned) > target_length:
        cleaned = cleaned[:target_length]
    elif len(cleaned) < target_length:
        pad_length = target_length - len(cleaned)
        cleaned = np.pad(cleaned, (0, pad_length), mode='edge')
    
    # 5. NO NORMALIZATION YET!
    return cleaned.astype(np.float32)


print(" Preprocessing functions defined (FIXED)")
print("   • NeuroKit2 cleaning")
print("   • Quality assessment")
print("   • NO per-signal normalization (will be global)")


"""
Load MIMIC-IV ECG metadata
(This part stays the same)
"""

def load_mimic_iv_metadata(target_afib=75000, target_normal=75000):
    """
    Create dataset from raw MIMIC-IV
    """
    print("="*80)
    print(" LOADING MIMIC-IV METADATA FROM RAW DATA")
    print("="*80)
    
    mimic_base = DataConfig.MIMIC_IV_PATH
    
    if not mimic_base.exists():
        raise FileNotFoundError(f"MIMIC-IV not found at: {mimic_base}")
    
    # Load measurements
    measurements_path = mimic_base / 'machine_measurements.csv'
    measurements_df = pd.read_csv(measurements_path, low_memory=False)
    print(f" Loaded {len(measurements_df):,} measurement records")
    
    # Load record list
    records_path = mimic_base / 'record_list.csv'
    records_df = pd.read_csv(records_path)
    print(f" Loaded {len(records_df):,} file records")
    
    # Merge
    merged_df = measurements_df.merge(
        records_df[['study_id', 'subject_id', 'path']], 
        on='study_id',
        how='inner'
    )
    print(f" Merged: {len(merged_df):,} records")
    
    # Filter AFib
    afib_regex = '|'.join([
        'Atrial fibrillation',
        'atrial fibrillation',
        'ATRIAL FIBRILLATION'
    ])
    
    afib_mask = merged_df['report_0'].str.contains(
        afib_regex, case=False, na=False, regex=True
    )
    
    df_afib = merged_df[afib_mask].copy()
    df_afib['label'] = 1
    
    print(f" Found {len(df_afib):,} AFib records")
    
    if len(df_afib) < target_afib:
        target_afib = len(df_afib)
    
    df_afib_sampled = df_afib.sample(n=target_afib, random_state=42)
    
    # Filter Normal
    normal_patterns = ['Sinus rhythm', 'sinus rhythm', 'SINUS RHYTHM']
    normal_exact_mask = merged_df['report_0'].isin(normal_patterns)
    df_normal = merged_df[normal_exact_mask].copy()
    df_normal['label'] = 0
    
    print(f" Found {len(df_normal):,} Normal records")
    
    if len(df_normal) < target_normal:
        target_normal = len(df_normal)
    
    df_normal_sampled = df_normal.sample(n=target_normal, random_state=42)
    
    # Combine
    df_combined = pd.concat([df_afib_sampled, df_normal_sampled], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    df_combined = df_combined[['study_id', 'subject_id', 'path', 'label', 'report_0']]
    
    print(f" Combined dataset: {len(df_combined):,} records")
    
    # Save metadata
    output_path = DataConfig.DIFFUSION_DATA_DIR / 'mimic_iv_metadata_diffusion_150k.csv'
    df_combined.to_csv(output_path, index=False)
    print(f" Saved metadata: {output_path}")
    
    return df_combined


print("Loading metadata...")
metadata_df = load_mimic_iv_metadata(
    target_afib=DataConfig.TARGET_AFIB_SAMPLES,
    target_normal=DataConfig.TARGET_NORMAL_SAMPLES
)

print(f"\n Metadata ready: {len(metadata_df):,} records")


"""
Load ECG from MIMIC-IV
(This part stays the same)
"""

def load_ecg_from_mimic(record_path, lead='II', target_fs=250):
    """
    Load ECG signal from MIMIC-IV WFDB file
    """
    try:
        full_path = DataConfig.MIMIC_IV_PATH / record_path
        record = wfdb.rdrecord(str(full_path))
        
        # Extract lead
        if lead in record.sig_name:
            lead_idx = record.sig_name.index(lead)
        else:
            if 'II' in record.sig_name:
                lead_idx = record.sig_name.index('II')
            else:
                lead_idx = 0
        
        signal = record.p_signal[:, lead_idx]
        
        # Resample
        if record.fs != target_fs:
            target_length = int(len(signal) * (target_fs / record.fs))
            signal_resampled = resample(signal, target_length)
        else:
            signal_resampled = signal
        
        # Ensure 2500 samples
        if len(signal_resampled) > DataConfig.SIGNAL_LENGTH:
            signal_resampled = signal_resampled[:DataConfig.SIGNAL_LENGTH]
        elif len(signal_resampled) < DataConfig.SIGNAL_LENGTH:
            pad_length = DataConfig.SIGNAL_LENGTH - len(signal_resampled)
            signal_resampled = np.pad(signal_resampled, (0, pad_length), mode='edge')
        
        return signal_resampled
        
    except FileNotFoundError:
        return None
    except Exception as e:
        if np.random.random() < 0.01:
            print(f"   ️  Error: {e}")
        return None

print(" ECG loading function defined")


"""
Extract and preprocess ECG windows (FIXED - GLOBAL NORMALIZATION!)
"""

def extract_and_preprocess_ecg_windows_FIXED(metadata_df):
    """
    Extract and preprocess ECG windows with GLOBAL normalization
    
    KEY FIX: Normalization is applied to ALL signals at once,
    preserving relative amplitude differences between signals
    """
    print("="*80)
    print(" EXTRACTING ECG WINDOWS (FIXED - GLOBAL NORMALIZATION)")
    print("="*80)
    
    df_afib = metadata_df[metadata_df['label'] == 1].reset_index(drop=True)
    df_normal = metadata_df[metadata_df['label'] == 0].reset_index(drop=True)
    
    print(f"\n Records to Process:")
    print(f"   AFib: {len(df_afib):,}")
    print(f"   Normal: {len(df_normal):,}")
    
    # Storage
    X_windows = []
    y_labels = []
    groups_subjects = []
    
    # ========================================================================
    # PROCESS AFib RECORDS
    # ========================================================================
    print("\n PROCESSING AFib RECORDS...")
    
    afib_success = 0
    pbar = tqdm(df_afib.iterrows(), total=len(df_afib), desc='AFib')
    
    for idx, row in pbar:
        ecg = load_ecg_from_mimic(row['path'], lead=DataConfig.LEAD_TO_USE,
                                   target_fs=DataConfig.SAMPLING_RATE)
        
        if ecg is None:
            continue
        
        # Preprocess WITHOUT normalization
        processed = preprocess_single_ecg_no_norm(
            ecg,
            target_length=DataConfig.SIGNAL_LENGTH,
            target_fs=DataConfig.SAMPLING_RATE
        )
        
        if processed is None:
            continue
        
        X_windows.append(processed)
        y_labels.append(1)
        groups_subjects.append(row['subject_id'])
        afib_success += 1
        
        pbar.set_postfix({'Success': afib_success})
    
    print(f" AFib: {afib_success:,} signals extracted")
    
    # ========================================================================
    # PROCESS NORMAL RECORDS
    # ========================================================================
    print("\n PROCESSING NORMAL RECORDS...")
    
    normal_success = 0
    pbar = tqdm(df_normal.iterrows(), total=len(df_normal), desc='Normal')
    
    for idx, row in pbar:
        ecg = load_ecg_from_mimic(row['path'], lead=DataConfig.LEAD_TO_USE,
                                   target_fs=DataConfig.SAMPLING_RATE)
        
        if ecg is None:
            continue
        
        processed = preprocess_single_ecg_no_norm(
            ecg,
            target_length=DataConfig.SIGNAL_LENGTH,
            target_fs=DataConfig.SAMPLING_RATE
        )
        
        if processed is None:
            continue
        
        X_windows.append(processed)
        y_labels.append(0)
        groups_subjects.append(row['subject_id'])
        normal_success += 1
        
        pbar.set_postfix({'Success': normal_success})
    
    print(f" Normal: {normal_success:,} signals extracted")
    
    # ========================================================================
    # CONVERT TO ARRAYS
    # ========================================================================
    print("\n Converting to arrays...")
    X_data = np.array(X_windows, dtype=np.float32)
    y_data = np.array(y_labels, dtype=np.int64)
    groups_data = np.array(groups_subjects)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_data))
    X_data = X_data[shuffle_idx]
    y_data = y_data[shuffle_idx]
    groups_data = groups_data[shuffle_idx]
    
    print(f"\n BEFORE Normalization:")
    print(f"   Shape: {X_data.shape}")
    print(f"   Range: [{X_data.min():.4f}, {X_data.max():.4f}] mV")
    print(f"   Mean: {X_data.mean():.4f} mV")
    print(f"   Std: {X_data.std():.4f} mV")
    
    # ========================================================================
    #  CRITICAL FIX: GLOBAL NORMALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print(" APPLYING GLOBAL NORMALIZATION")
    print("="*80)
    
    global_min = float(X_data.min())
    global_max = float(X_data.max())
    
    print(f"\n Global Statistics:")
    print(f"   Global Min: {global_min:.4f} mV")
    print(f"   Global Max: {global_max:.4f} mV")
    print(f"   Global Range: {global_max - global_min:.4f} mV")
    
    # Apply global normalization
    if global_max - global_min > 1e-6:
        # Scale to [0, 1]
        X_normalized = (X_data - global_min) / (global_max - global_min)
        # Scale to [-1.5, 1.5]
        X_normalized = X_normalized * (DataConfig.CLINICAL_MAX - DataConfig.CLINICAL_MIN) + DataConfig.CLINICAL_MIN
    else:
        X_normalized = np.zeros_like(X_data)
    
    print(f"\n AFTER Global Normalization:")
    print(f"   Range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}] mV")
    print(f"   Mean: {X_normalized.mean():.4f} mV")
    print(f"   Std: {X_normalized.std():.4f} mV")
    
    # ========================================================================
    # CHECK FOR ARTIFICIAL CLIPPING
    # ========================================================================
    print(f"\n Clipping Check:")
    at_min = np.sum(np.abs(X_normalized - DataConfig.CLINICAL_MIN) < 1e-6)
    at_max = np.sum(np.abs(X_normalized - DataConfig.CLINICAL_MAX) < 1e-6)
    total_values = X_normalized.size
    
    print(f"   Values at {DataConfig.CLINICAL_MIN}: {at_min} ({100*at_min/total_values:.3f}%)")
    print(f"   Values at {DataConfig.CLINICAL_MAX}: {at_max} ({100*at_max/total_values:.3f}%)")
    
    if at_min/total_values < 0.001 and at_max/total_values < 0.001:
        print(f"    NO ARTIFICIAL CLIPPING - Normalization is correct!")
    else:
        print(f"   ️  WARNING: High concentration at boundaries")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print(" EXTRACTION COMPLETE")
    print("="*80)
    
    print(f"\n Final Dataset:")
    print(f"   Total samples: {len(X_normalized):,}")
    print(f"   AFib: {(y_data==1).sum():,} ({(y_data==1).sum()/len(y_data)*100:.1f}%)")
    print(f"   Normal: {(y_data==0).sum():,} ({(y_data==0).sum()/len(y_data)*100:.1f}%)")
    
    print(f"\n Signal Properties:")
    print(f"   Shape: {X_normalized.shape}")
    print(f"   Dtype: {X_normalized.dtype}")
    print(f"   Memory: {X_normalized.nbytes / 1e9:.2f} GB")
    print(f"   Range: [{X_normalized.min():.4f}, {X_normalized.max():.4f}] mV")
    
    # Integrity check
    n_nan = np.isnan(X_normalized).sum()
    n_inf = np.isinf(X_normalized).sum()
    print(f"\n Data Integrity:")
    print(f"   NaN values: {n_nan}")
    print(f"   Inf values: {n_inf}")
    
    if n_nan == 0 and n_inf == 0:
        print(f"    Data is clean and ready!")
    else:
        print(f"    Data contains invalid values!")
    
    print("="*80)
    
    return X_normalized, y_data, groups_data, global_min, global_max


# ============================================================================
# EXECUTE: Extract ECG windows with FIXED normalization
# ============================================================================

print("\n" + "="*80)
print(" STARTING ECG EXTRACTION (FIXED VERSION)")
print("="*80)

import time
print("\nStarting in 5 seconds... (Press Ctrl+C to cancel)")
for i in range(5, 0, -1):
    print(f"   {i}...")
    time.sleep(1)
print("   GO!\n")

# Extract with FIXED global normalization
X_raw, y_raw, subject_ids, global_min, global_max = extract_and_preprocess_ecg_windows_FIXED(metadata_df)

# Save normalization parameters for later use
norm_params = {
    'global_min': float(global_min),
    'global_max': float(global_max),
    'target_min': float(DataConfig.CLINICAL_MIN),
    'target_max': float(DataConfig.CLINICAL_MAX)
}

print("\n" + "="*80)
print(" EXTRACTION COMPLETE!")
print("="*80)
print(f"Ready for visualization and train/val/test split")
print("="*80)


"""
The rest of the script (visualization, splitting, saving) stays the same
Just continue with your existing cells for:
- Visualization
- Classifier compatibility test
- Train/Val/Test split
- Saving

The key difference: Now your data will have proper global normalization
with no artificial clipping!
"""

print("\n" + "="*80)
print(" NEXT STEPS:")
print("="*80)
print("1.  Data extracted with FIXED global normalization")
print("2. → Continue with your existing visualization cells")
print("3. → Continue with train/val/test split")
print("4. → Save the data")
print("\n The histogram should now show a natural distribution")
print("   with NO spike at -1.5!")
print("="*80)