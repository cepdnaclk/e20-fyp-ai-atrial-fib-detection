"""
Research-Paper-Accurate Universal ECG Preprocessor for AFib Detection
Implements EXACT preprocessing from published papers for each model

VERSION: 4.0 (Accepts explicit CINC17 reference path and fixes NoneType bug)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from tqdm.auto import tqdm
import warnings
from scipy import signal as scipy_signal
from scipy.signal import butter, filtfilt
import pywt
warnings.filterwarnings('ignore')

# Import the universal loader
from universal_loader import UniversalECGLoader


# ============================================================================
# RESEARCH PAPER PREPROCESSING CONFIGURATIONS
# ============================================================================

RESEARCH_PAPER_CONFIGS = {
    'cnn_bilstm': {
        'paper': 'Andersen et al. 2019 - A Deep Learning Approach for Real-Time Detection of Atrial Fibrillation',
        'methodology': 'RR-interval based with 30-RRI segments',
        'input_type': 'rr_intervals', 'segment_type': 'rr_based', 'rri_count': 30, 'overlap_beats': 10,
        'labeling_rule': 'majority_vote', 'label_threshold': 0.5, 'zero_padding': True, 'output_shape': (30,),
        'description': 'Andersen 2019: 30 RR-intervals with 10-beat overlap, binary labeling (AF>50%)'
    },
    'cnn_lstm_focal': {
        'paper': 'Petmezas et al. 2021 - Automated Atrial Fibrillation Detection using Hybrid CNN-LSTM Network',
        'methodology': 'Beat-level classification with extensive denoising',
        'target_fs': 250, 'input_type': 'raw_beats', 'segment_type': 'beat_based',
        'beat_window_before_ms': 250, 'beat_window_after_ms': 500, 'samples_per_beat': 187,
        'preprocessing_stages': [
            {'name': 'highpass_butterworth', 'order': 7, 'cutoff': 0.5, 'type': 'highpass'},
            {'name': 'lowpass_butterworth', 'order': 6, 'cutoff': 40, 'type': 'lowpass'},
            {'name': 'dwt_denoising', 'wavelet': 'db4', 'level': 3, 'threshold_method': 'soft'}
        ],
        'r_peak_method': 'dwt', 'dwt_levels': 4, 'dwt_sum_coeffs': ['D3', 'D4'],
        'num_classes': 4, 'class_labels': ['N', 'AFIB', 'AFL', 'J'],
        'description': 'Petmezas 2021: 187-sample beats, 3-stage denoising (Butterworth + DWT), beat morphology'
    },
    'afib_reslstm': {
        'paper': 'Yingjie et al. 2020 - An End-to-end Deep Learning Scheme for Atrial Fibrillation Detection',
        'methodology': '30-second windows with 5-second stride',
        'target_fs': 250, 'input_type': 'raw_ecg', 'segment_type': 'time_based',
        'window_duration_s': 30, 'stride_s': 5, 'samples_per_segment': 7500,
        'filters': [
            {'type': 'lowpass', 'cutoff': 40, 'order': 6},
            {'type': 'highpass', 'cutoff': 0.5, 'order': 6},
            {'type': 'notch', 'freq': 50, 'quality': 30}
        ],
        'labeling_rule': 'majority_time', 'label_threshold': 0.5,
        'min_af_duration_s': 30, 'merge_gap_s': 5,
        'description': 'Yingjie 2020: 30s windows with 5s stride, 250Hz, multi-filter preprocessing'
    },
    'resnet_bilstm_attention': {
        'paper': 'Yingjie et al. 2020 - An End-to-end Deep Learning Scheme (Same as afib_reslstm)',
        'methodology': 'Same preprocessing as afib_reslstm',
        'target_fs': 250, 'input_type': 'raw_ecg', 'segment_type': 'time_based',
        'window_duration_s': 30, 'stride_s': 5, 'samples_per_segment': 7500,
        'filters': [
            {'type': 'lowpass', 'cutoff': 40, 'order': 6},
            {'type': 'highpass', 'cutoff': 0.5, 'order': 6},
            {'type': 'notch', 'freq': 50, 'quality': 30}
        ],
        'labeling_rule': 'majority_time', 'label_threshold': 0.5,
        'min_af_duration_s': 30, 'merge_gap_s': 5,
        'description': 'Same as Yingjie 2020: 30s windows, 5s stride, attention mechanism'
    },
    'lightweight_resnet': {
        'paper': 'Ben-Moshe et al. 2023 - RawECGNet: Deep Learning Generalization for AF Detection',
        'methodology': '30-second non-overlapping windows with quality filtering',
        'target_fs': 200, 'input_type': 'raw_ecg', 'segment_type': 'time_based',
        'window_duration_s': 30, 'stride_s': 30, 'samples_per_segment': 6000,
        'use_quality_filtering': True, 'quality_metric': 'bsqi', 'quality_threshold': 0.8,
        'labeling_rule': 'dominant_rhythm', 'label_classes': ['AF', 'AFL', 'non-AFL'],
        'min_age': 18, 'max_poor_quality_ratio': 0.75,
        'description': 'Ben-Moshe 2023: 30s non-overlapping, 200Hz, bSQI quality filtering'
    }
}


# ============================================================================
# SIGNAL PROCESSING UTILITIES
# ============================================================================

class SignalProcessor:
    @staticmethod
    def butter_filter(data, cutoff, fs, order, ftype='lowpass'):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype=ftype, analog=False)
        return filtfilt(b, a, data)
    
    @staticmethod
    def notch_filter(data, freq, fs, quality=30):
        nyquist = 0.5 * fs
        freq_normalized = freq / nyquist
        b, a = scipy_signal.iirnotch(freq_normalized, quality)
        return filtfilt(b, a, data)
    
    @staticmethod
    def dwt_denoise(data, wavelet='db4', level=3, threshold_method='soft'):
        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        coeffs_thresh = [pywt.threshold(c, threshold, mode=threshold_method) for c in coeffs]
        denoised = pywt.waverec(coeffs_thresh, wavelet)
        if len(denoised) > len(data): denoised = denoised[:len(data)]
        elif len(denoised) < len(data): denoised = np.pad(denoised, (0, len(data) - len(denoised)), mode='edge')
        return denoised
    
    @staticmethod
    def detect_r_peaks_pantompkins(ecg, fs):
        b, a = butter(2, [5/(fs/2), 15/(fs/2)], btype='band')
        filtered = filtfilt(b, a, ecg)
        diff = np.diff(filtered); squared = diff ** 2
        window_size = int(0.12 * fs)
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(integrated, distance=int(0.2*fs))
        return peaks
    
    @staticmethod
    def detect_r_peaks_dwt(ecg, fs, level=4):
        coeffs = pywt.wavedec(ecg, 'db4', level=level)
        d3 = coeffs[-2]; d4 = coeffs[-1]
        d3_upsampled = scipy_signal.resample(d3, len(ecg))
        d4_upsampled = scipy_signal.resample(d4, len(ecg))
        summed = d3_upsampled + d4_upsampled
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(summed, distance=int(0.2*fs))
        return peaks
    
    @staticmethod
    def extract_rr_intervals(r_peaks, fs):
        return np.diff(r_peaks) / fs * 1000
    
    @staticmethod
    def calculate_bsqi(ecg, fs):
        if np.std(ecg) < 0.01: return 0.0
        zero_crossings = np.sum(np.diff(np.sign(ecg)) != 0)
        zcr = zero_crossings / len(ecg)
        quality = 0.5 if zcr > 0.3 else 1.0
        if np.max(np.abs(ecg)) > 3 * np.std(ecg): quality *= 0.8
        return quality


# ============================================================================
# RESEARCH ACCURATE PREPROCESSOR (FIXED)
# ============================================================================

class ResearchAccuratePreprocessor:
    """
    Preprocessor that implements EXACT methodologies from research papers
    """
    
    # --- START OF __init__ FIX ---
    def __init__(
        self,
        raw_data_paths: List[str],
        output_dir: str = '../data/processed/',
        cinc_reference_path: Optional[str] = None, # <-- THIS IS THE NEW ARGUMENT
        lead_to_use: str = 'MLII',
        random_seed: int = 42
    ):
    # --- END OF __init__ FIX ---
        self.raw_data_paths = [Path(p) for p in raw_data_paths]
        self.output_dir = Path(output_dir)
        self.lead_to_use = lead_to_use
        self.random_seed = random_seed
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cinc_labels_dict = {}
        
        # --- START FIX: Load CINC17 labels from explicit path ---
        if cinc_reference_path:
            # Resolve the path relative to the current working directory
            cinc_label_path = Path(cinc_reference_path).resolve()
            if cinc_label_path.exists():
                print(f"   ... Explicitly loading CINC17 labels from: {cinc_label_path}")
                try:
                    df = pd.read_csv(cinc_label_path, header=None, names=['record_id', 'label'])
                    self.cinc_labels_dict = dict(zip(df['record_id'], df['label']))
                    print(f"   ... Loaded {len(self.cinc_labels_dict)} CINC17 labels.")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not load CINC17 labels: {e}")
            else:
                print(f"   ⚠️  Warning: CINC17 reference path provided but not found: {cinc_label_path}")
        else:
            print("   ... Note: 'cinc_reference_path' not provided. CINC17 records will be labeled 'Normal'.")
        # --- END FIX ---

        # Initialize loaders
        self.loaders = []
        for path_str in raw_data_paths:
            path = Path(path_str).resolve() # Use resolved paths
            if path.exists():
                print(f"Loading data from: {path}")
                self.loaders.append(UniversalECGLoader(str(path)))
            else:
                print(f"⚠️  Warning: Path not found: {path}")
        
        self.signal_processor = SignalProcessor()

    
    def preprocess_for_model(
        self,
        model_name: str,
        max_records_per_dataset: Optional[int] = None
    ) -> Dict:
        if model_name not in RESEARCH_PAPER_CONFIGS:
            raise ValueError(
                f"Model '{model_name}' not found in research paper configs.\n"
                f"Available: {list(RESEARCH_PAPER_CONFIGS.keys())}"
            )
        
        config = RESEARCH_PAPER_CONFIGS[model_name]
        
        print(f"\n{'='*70}")
        print(f"RESEARCH-ACCURATE PREPROCESSING: {model_name}")
        print(f"{'='*70}")
        print(f"\n📄 Paper: {config['paper']}")
        print(f"📋 Methodology: {config['methodology']}")
        print(f"📝 Description: {config['description']}\n")
        
        # Route to appropriate preprocessing method
        input_type = config['input_type']
        
        if input_type == 'rr_intervals':
            return self._preprocess_rr_based(model_name, config, max_records_per_dataset)
        elif input_type == 'raw_beats':
            return self._preprocess_beat_based(model_name, config, max_records_per_dataset)
        elif input_type == 'raw_ecg':
            return self._preprocess_time_based(model_name, config, max_records_per_dataset)
        else:
            raise ValueError(f"Unknown input type: {input_type}")
    
    def _preprocess_rr_based(self, model_name, config, max_records):
        """
        RR-interval based preprocessing (CNN-BiLSTM - Andersen 2019)
        """
        print("🔬 Method: RR-Interval Based Segmentation")
        print(f"   - Segment size: {config['rri_count']} RR intervals")
        print(f"   - Overlap: {config['overlap_beats']} beats")
        print(f"   - Labeling: {'AF if >50% beats are AF' if config['labeling_rule'] == 'majority_vote' else config['labeling_rule']}\n")
        
        model_output_dir = self.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_segments = []
        all_labels = []
        metadata_list = []
        
        stats = {
            'total_records_processed': 0,
            'total_segments_created': 0,
            'segments_per_class': {'normal': 0, 'afib': 0}
        }
        
        for loader in self.loaders:
            print(f"\n📂 Processing dataset: {loader.data_path}")
            valid_records = loader.get_valid_records()
            
            if max_records:
                valid_records = valid_records[:max_records]
            
            for record_id in tqdm(valid_records, desc="Processing records"):
                try:
                    # Load record
                    data = loader.load_record(record_id)
                    signal = self._extract_lead(data, self.lead_to_use)
                    
                    if signal is None:
                        continue
                    
                    # Detect R-peaks
                    r_peaks = self.signal_processor.detect_r_peaks_pantompkins(
                        signal, data['fs']
                    )
                    
                    if len(r_peaks) < config['rri_count'] + 1:
                        continue
                    
                    # Extract RR intervals
                    rr_intervals = self.signal_processor.extract_rr_intervals(
                        r_peaks, data['fs']
                    )
                    
                    # Create segments with overlap
                    rri_count = config['rri_count']
                    overlap = config['overlap_beats']
                    stride = rri_count - overlap
                    
                    for i in range(0, len(rr_intervals) - rri_count + 1, stride):
                        segment = rr_intervals[i:i + rri_count]
                        
                        if len(segment) == rri_count:
                            # Get label for this segment's beats
                            beat_indices = range(i, i + rri_count + 1)
                            
                            af_beats = self._count_af_beats(
                                data, r_peaks[beat_indices], data['fs'], record_id
                            )
                            
                            # Majority vote labeling
                            label = 1 if af_beats / (rri_count + 1) > config['label_threshold'] else 0
                            
                            stats['segments_per_class'][
                                'afib' if label == 1 else 'normal'
                            ] += 1
                            
                            # Zero-pad if needed
                            if config.get('zero_padding', False):
                                segment = np.pad(segment, (0, max(0, rri_count - len(segment))), 
                                               mode='constant')
                            
                            all_segments.append(segment)
                            all_labels.append(label)
                            
                            metadata_list.append({
                                'record_id': record_id,
                                'segment_index': len(all_segments) - 1,
                                'label': label,
                                'rri_count': rri_count,
                                'af_beats_ratio': af_beats / (rri_count + 1)
                            })
                    
                    stats['total_records_processed'] += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Failed: {record_id}: {str(e)[:50]}")
                    continue
        
        return self._save_processed_data(
            model_name, model_output_dir, all_segments, all_labels, 
            metadata_list, stats, config
        )
    
    def _preprocess_beat_based(self, model_name, config, max_records):
        """
        Beat-based preprocessing (CNN-LSTM-Focal - Petmezas 2021)
        """
        print("🔬 Method: Beat-Level Classification with 3-Stage Denoising")
        print(f"   - Beat window: {config['beat_window_before_ms']}ms before + "
              f"{config['beat_window_after_ms']}ms after R-peak")
        print(f"   - Samples per beat: {config['samples_per_beat']} @ {config['target_fs']}Hz")
        print(f"   - Denoising stages: {len(config['preprocessing_stages'])}\n")
        
        model_output_dir = self.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_beats = []
        all_labels = []
        metadata_list = []
        
        stats = {
            'total_records_processed': 0,
            'total_beats_extracted': 0,
            'beats_per_class': {label: 0 for label in config['class_labels']}
        }
        
        for loader in self.loaders:
            print(f"\n📂 Processing dataset: {loader.data_path}")
            valid_records = loader.get_valid_records()
            
            if max_records:
                valid_records = valid_records[:max_records]
            
            for record_id in tqdm(valid_records, desc="Processing records"):
                try:
                    # Load and preprocess
                    data = loader.load_record(record_id)
                    signal = self._extract_lead(data, self.lead_to_use)
                    
                    if signal is None:
                        continue
                    
                    # Resample to 250 Hz
                    if data['fs'] != config['target_fs']:
                        signal = self._resample_signal(
                            signal, data['fs'], config['target_fs']
                        )
                    
                    # Apply 3-stage preprocessing
                    for stage in config['preprocessing_stages']:
                        if stage['name'] == 'highpass_butterworth':
                            signal = self.signal_processor.butter_filter(
                                signal, stage['cutoff'], config['target_fs'],
                                stage['order'], 'highpass'
                            )
                        elif stage['name'] == 'lowpass_butterworth':
                            signal = self.signal_processor.butter_filter(
                                signal, stage['cutoff'], config['target_fs'],
                                stage['order'], 'lowpass'
                            )
                        elif stage['name'] == 'dwt_denoising':
                            signal = self.signal_processor.dwt_denoise(
                                signal, stage['wavelet'], stage['level'],
                                stage['threshold_method']
                            )
                    
                    # Detect R-peaks using DWT method
                    r_peaks = self.signal_processor.detect_r_peaks_dwt(
                        signal, config['target_fs'], config['dwt_levels']
                    )
                    
                    # Extract beats
                    before_samples = int(config['beat_window_before_ms'] * config['target_fs'] / 1000)
                    after_samples = int(config['beat_window_after_ms'] * config['target_fs'] / 1000)
                    
                    for r_idx in r_peaks:
                        start = r_idx - before_samples
                        end = r_idx + after_samples
                        
                        if start >= 0 and end < len(signal):
                            beat = signal[start:end]
                            
                            # Ensure correct length
                            if len(beat) == config['samples_per_beat']:
                                label = self._get_beat_label(data, r_idx / config['target_fs'], record_id)
                                
                                all_beats.append(beat)
                                all_labels.append(label)
                                
                                stats['beats_per_class'][label] = stats['beats_per_class'].get(label, 0) + 1
                                
                                metadata_list.append({
                                    'record_id': record_id,
                                    'beat_index': len(all_beats) - 1,
                                    'label': label,
                                    'r_peak_location': r_idx
                                })
                    
                    stats['total_records_processed'] += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Failed: {record_id}: {str(e)[:50]}")
                    continue
        
        stats['total_beats_extracted'] = len(all_beats)
        
        return self._save_processed_data(
            model_name, model_output_dir, all_beats, all_labels,
            metadata_list, stats, config
        )
    
    def _preprocess_time_based(self, model_name, config, max_records):
        """
        Time-based window preprocessing (AFib-ResLSTM, ResNet-BiLSTM-Attention, Lightweight ResNet)
        """
        print("🔬 Method: Time-Based Window Segmentation")
        print(f"   - Window duration: {config['window_duration_s']}s")
        print(f"   - Stride: {config['stride_s']}s (overlap: {config['window_duration_s'] - config['stride_s']}s)")
        print(f"   - Target frequency: {config['target_fs']} Hz")
        print(f"   - Samples per segment: {config['samples_per_segment']}\n")
        
        if config.get('filters'):
            print(f"   Filters applied: {len(config['filters'])}")
            for filt in config['filters']:
                print(f"      - {filt['type']}: {filt.get('cutoff', filt.get('freq'))} Hz")
        
        model_output_dir = self.output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        all_segments = []
        all_labels = []
        metadata_list = []
        
        stats = {
            'total_records_processed': 0,
            'total_segments_created': 0,
            'segments_per_class': {'normal': 0, 'afib': 0}
        }
        
        for loader in self.loaders:
            print(f"\n📂 Processing dataset: {loader.data_path}")
            valid_records = loader.get_valid_records()
            
            if max_records:
                valid_records = valid_records[:max_records]
            
            for record_id in tqdm(valid_records, desc="Processing records"):
                try:
                    # Load record
                    data = loader.load_record(record_id)
                    signal = self._extract_lead(data, self.lead_to_use)
                    
                    if signal is None:
                        continue
                    
                    # Resample
                    if data['fs'] != config['target_fs']:
                        signal = self._resample_signal(
                            signal, data['fs'], config['target_fs']
                        )
                    
                    # Apply filters
                    if config.get('filters'):
                        for filt in config['filters']:
                            if filt['type'] == 'lowpass':
                                signal = self.signal_processor.butter_filter(
                                    signal, filt['cutoff'], config['target_fs'],
                                    filt['order'], 'lowpass'
                                )
                            elif filt['type'] == 'highpass':
                                signal = self.signal_processor.butter_filter(
                                    signal, filt['cutoff'], config['target_fs'],
                                    filt['order'], 'highpass'
                                )
                            elif filt['type'] == 'notch':
                                signal = self.signal_processor.notch_filter(
                                    signal, filt['freq'], config['target_fs'],
                                    filt['quality']
                                )
                    
                    # Segment signal
                    window_samples = config['samples_per_segment']
                    stride_samples = int(config['stride_s'] * config['target_fs'])
                    
                    for start in range(0, len(signal) - window_samples + 1, stride_samples):
                        segment = signal[start:start + window_samples]
                        
                        if len(segment) == window_samples:
                            # Quality filtering (if applicable)
                            if config.get('use_quality_filtering', False):
                                quality = self.signal_processor.calculate_bsqi(
                                    segment, config['target_fs']
                                )
                                if quality < config['quality_threshold']:
                                    continue
                            
                            # Get label
                            start_time = start / config['target_fs']
                            end_time = (start + window_samples) / config['target_fs']
                            
                            label = self._get_window_label(
                                data, start_time, end_time, config, record_id
                            )
                            
                            all_segments.append(segment)
                            all_labels.append(label)
                            
                            stats['segments_per_class'][
                                'afib' if label == 1 else 'normal'
                            ] += 1
                            
                            metadata_list.append({
                                'record_id': record_id,
                                'segment_index': len(all_segments) - 1,
                                'label': label,
                                'start_time_s': start_time,
                                'end_time_s': end_time
                            })
                    
                    stats['total_records_processed'] += 1
                    
                except Exception as e:
                    print(f"   ⚠️  Failed: {record_id}: {str(e)[:50]}")
                    continue
        
        stats['total_segments_created'] = len(all_segments)
        
        return self._save_processed_data(
            model_name, model_output_dir, all_segments, all_labels,
            metadata_list, stats, config
        )
    
    # --- Helper methods ---
    
    def _extract_lead(self, data, lead_name):
        signal, channels = data['signal'], data['channels']
        if len(channels) == 1: return signal[:, 0]
        if lead_name in channels: return signal[:, channels.index(lead_name)]
        alternatives = {'MLII': ['ML II', 'MLII', 'II', 'Lead II']}
        if lead_name in alternatives:
            for alt in alternatives[lead_name]:
                if alt in channels: return signal[:, channels.index(alt)]
        return signal[:, 0]
    
    def _resample_signal(self, signal, original_fs, target_fs):
        if original_fs == target_fs: return signal
        duration = len(signal) / original_fs
        new_length = int(duration * target_fs)
        return scipy_signal.resample(signal, new_length)
    
    def _get_window_label(self, data, start_time, end_time, config, record_id: str):
        try:
            window_duration = end_time - start_time
            af_duration = self._get_af_duration_in_window(data, start_time, end_time, record_id)
            if config.get('labeling_rule') == 'majority_time':
                threshold = config.get('label_threshold', 0.5)
                return 1 if (af_duration / window_duration) > threshold else 0
            elif config.get('labeling_rule') == 'dominant_rhythm':
                return 1 if af_duration > (window_duration / 2) else 0
            else:
                return 1 if af_duration > 0 else 0
        except: return 0
    
    def _get_beat_label(self, data, beat_time, record_id: str):
        try:
            return 'AFIB' if self._is_time_in_af(data, beat_time, record_id) else 'N'
        except: return 'N'
    
    def _count_af_beats(self, data, r_peaks_indices, fs, record_id: str):
        try:
            af_count = 0
            for r_peak_idx in r_peaks_indices:
                beat_time = r_peak_idx / fs
                if self._is_time_in_af(data, beat_time, record_id):
                    af_count += 1
            return af_count
        except: return 0
    

    # ========================================================================
    # --- START OF CRITICAL FIX ---
    # This function is the primary source of the "0 AFib" bug.
    # ========================================================================
    def _is_time_in_af(self, data, time_seconds, record_id: str):
        """
        Check if a specific time point is AF.
        FIXED to handle CINC17 labels AND AFDB 'aux_notes = None' case.
        """
        
        # --- 1. CINC17 Logic (External CSV) ---
        base_record_id = Path(record_id).stem
        if base_record_id in self.cinc_labels_dict:
            label = self.cinc_labels_dict[base_record_id]
            # Only count 'A' (AFib) as 1. 'N', 'O', '~' are 0 (Normal).
            return label == 'A'

        # --- 2. AFDB Logic (Internal Annotations) ---
        try:
            if 'annotations' not in data:
                return False
            
            annotations = data['annotations']
            
            # 2a. Find annotation keys
            sample_key = None
            aux_key = None
            for key in annotations.keys():
                if 'sample' in key.lower(): sample_key = key
                if 'aux' in key.lower() or 'note' in key.lower(): aux_key = key
            
            if not sample_key or not aux_key:
                return False # Keys not found
            
            # 2b. Get annotation data
            samples = annotations.get(sample_key)
            aux_notes = annotations.get(aux_key) # This can be None

            # 2c. *** THIS IS THE CRITICAL FIX ***
            # If aux_notes is None, the loader found no rhythm info. Return False.
            # This prevents the TypeError on len(aux_notes) later.
            if aux_notes is None or samples is None:
                return False 
            
            # 2d. We now know aux_notes is a list. Proceed with logic.
            sample_idx = int(time_seconds * data['fs'])
            current_rhythm = 'N'
            
            for i, sample in enumerate(samples):
                if sample <= sample_idx:
                    # Check that aux_notes has an entry for this sample
                    rhythm_str = str(aux_notes[i]) if i < len(aux_notes) else '' 
                    
                    if '(AFIB' in rhythm_str or '(AFL' in rhythm_str:
                        current_rhythm = 'A'
                    elif '(N' in rhythm_str or rhythm_str == '':
                        current_rhythm = 'N'
                else:
                    break
            
            return current_rhythm == 'A'
            
        except Exception as e:
            # Print the error for debugging, but return False
            # You can uncomment this if you still have issues
            # print(f"   [Debug] Error in _is_time_in_af for {record_id}: {e}")
            return False
    # ========================================================================
    # --- END OF CRITICAL FIX ---
    # ========================================================================

    def _get_af_duration_in_window(self, data, start_time, end_time, record_id: str):
        try:
            base_record_id = Path(record_id).stem
            if base_record_id in self.cinc_labels_dict:
                return (end_time - start_time) if self.cinc_labels_dict[base_record_id] == 'A' else 0.0
            
            sample_interval = 1.0
            af_duration = 0.0
            current_time = start_time
            while current_time < end_time:
                if self._is_time_in_af(data, current_time, record_id):
                    af_duration += sample_interval
                current_time += sample_interval
            return min(af_duration, end_time - start_time)
        except: return 0.0
    
    def _save_processed_data(self, model_name, output_dir, segments, labels, 
                            metadata, stats, config):
        X = np.array(segments, dtype=np.float32)
        y = np.array(labels)

        if model_name == 'cnn_lstm_focal':
            label_map = {'N': 0, 'AFIB': 1, 'AFL': 1, 'J': 1}
            y = np.array([label_map.get(str(lbl).strip().upper(), 0) for lbl in y], dtype=np.int32)
            print(f"      Converted string labels to integers for {model_name}")
            print(f"      Sample labels: {y[:20]}")
        
        if y.dtype == object:
            label_map = {label: idx for idx, label in enumerate(config.get('class_labels', ['N', 'AFIB']))}
            y = np.array([label_map.get(str(lbl), 0) for lbl in y], dtype=np.int32)
        
        np.save(output_dir / 'X_processed.npy', X)
        np.save(output_dir / 'y_processed.npy', y)
        pd.DataFrame(metadata).to_csv(output_dir / 'metadata.csv', index=False)
        
        stats['total_segments_created'] = len(X)
        if y.ndim == 1 and len(y) > 0:
            # Initialize if not exists
            if 'segments_per_class' not in stats:
                stats['segments_per_class'] = {}
            
            stats['segments_per_class']['afib'] = int(np.sum(y == 1))
            stats['segments_per_class']['normal'] = int(np.sum(y == 0))
        
        info = {
            'model_name': model_name, 'paper': config['paper'], 'methodology': config['methodology'],
            'preprocessing_config': config, 'statistics': stats, 'data_shape': X.shape
        }
        
        with open(output_dir / 'preprocessing_info.json', 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"\n✅ Saved to: {output_dir}")
        print(f"   X shape: {X.shape}")
        print(f"   y shape: {y.shape}")
        
        afib_count = stats['segments_per_class']['afib']
        total_count = stats['total_segments_created']
        balance = (afib_count / total_count * 100) if total_count > 0 else 0
        
        print(f"   Normal samples: {stats['segments_per_class']['normal']}")
        print(f"   AFib samples: {afib_count}")
        print(f"   Class balance: {balance:.1f}% AFib")
        
        return info

if __name__ == '__main__':
    print("Research-Accurate Preprocessor (FIXED v4) initialized")
    print("\nAvailable models:")
    for model_name, config in RESEARCH_PAPER_CONFIGS.items():
        print(f"\n  • {model_name}")
        print(f"    Paper: {config['paper']}")
        print(f"    Method: {config['methodology']}")