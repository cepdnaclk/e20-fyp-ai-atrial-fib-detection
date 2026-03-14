"""
Universal ECG Dataset Loader for AFib Research
Save this file as: src/data_loader.py
"""

import os
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATASET DETECTOR
# ============================================================================
class DatasetDetector:
    """
    Intelligently detects the format and structure of any ECG dataset.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.structure = {}

    def analyze_directory(self) -> Dict:
        """
        Analyzes directory structure and file types to understand the dataset.
        """
        print(f"\n{'='*60}")
        print(f"ANALYZING DATASET AT: {self.data_path}")
        print(f"{'='*60}\n")

        analysis = {
            'path': str(self.data_path),
            'file_extensions': {},
            'subdirectories': [],
            'metadata_files': [],
            'record_files': [],
            'format_type': None
        }

        # Scan all files
        for file_path in self.data_path.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()

                # Count file types
                if ext:
                    analysis['file_extensions'][ext] = analysis['file_extensions'].get(ext, 0) + 1

                # Identify metadata files
                if ext in ['.csv', '.json', '.txt', '.xlsx'] or 'metadata' in file_path.name.lower():
                    analysis['metadata_files'].append(str(file_path.relative_to(self.data_path)))

                # Identify record files (signal data)
                if ext in ['.dat', '.mat', '.edf', '.npy', '.npz']:
                    analysis['record_files'].append(str(file_path.relative_to(self.data_path)))

        # Identify subdirectories
        for item in self.data_path.iterdir():
            if item.is_dir():
                analysis['subdirectories'].append(item.name)

        # Determine format type
        analysis['format_type'] = self._determine_format(analysis)

        # Print analysis
        self._print_analysis(analysis)

        return analysis

    def _determine_format(self, analysis: Dict) -> str:
        """Determines the dataset format based on file extensions."""
        exts = analysis['file_extensions']

        if '.hea' in exts and '.dat' in exts:
            return 'WFDB'
        elif '.edf' in exts:
            return 'EDF'
        elif '.mat' in exts:
            return 'MATLAB'
        elif '.csv' in exts:
            return 'CSV'
        elif '.npy' in exts or '.npz' in exts:
            return 'NUMPY'
        else:
            return 'UNKNOWN'

    def _print_analysis(self, analysis: Dict):
        """Pretty print the analysis results."""
        print(f"📁 Directory Structure:")
        print(f"   Root: {analysis['path']}")
        if analysis['subdirectories']:
            print(f"   Subdirectories: {', '.join(analysis['subdirectories'][:5])}")
            if len(analysis['subdirectories']) > 5:
                print(f"   ... and {len(analysis['subdirectories']) - 5} more")

        print(f"\n📄 File Types Found:")
        for ext, count in sorted(analysis['file_extensions'].items()):
            print(f"   {ext}: {count} files")

        print(f"\n📊 Metadata Files:")
        if analysis['metadata_files']:
            for mf in analysis['metadata_files'][:5]:
                print(f"   - {mf}")
            if len(analysis['metadata_files']) > 5:
                print(f"   ... and {len(analysis['metadata_files']) - 5} more")
        else:
            print(f"   No metadata files found")

        print(f"\n🔍 Detected Format: {analysis['format_type']}")
        print(f"   Total record files: {len(analysis['record_files'])}")
        print(f"\n{'='*60}\n")


# ============================================================================
# FORMAT READERS
# ============================================================================
class FormatReader:
    """
    Contains readers for different ECG data formats.
    """

    @staticmethod
    def read_wfdb(file_path: Path, record_name: str) -> Dict:
        """Read WFDB format (.hea + .dat files)."""
        try:
            record = wfdb.rdrecord(str(file_path / record_name))

            # Try to load annotations
            annotation = None
            for ext in ['atr', 'qrs', 'ann']:
                try:
                    annotation = wfdb.rdann(str(file_path / record_name), ext)
                    break
                except:
                    continue

            data = {
                'signal': record.p_signal,
                'fs': record.fs,
                'channels': record.sig_name,
                'units': record.units,
                'duration': len(record.p_signal) / record.fs,
                'format': 'WFDB'
            }

            if annotation:
                data['annotations'] = {
                    'samples': annotation.sample,
                    'symbols': annotation.symbol,
                    'aux_notes': getattr(annotation, 'aux_note', None)
                }

            return data
        except Exception as e:
            raise Exception(f"Error reading WFDB file: {str(e)}")

    @staticmethod
    def read_edf(file_path: Path) -> Dict:
        """
        Read EDF format files.
        Note: Requires pyedflib (install with: pip install pyedflib)
        """
        try:
            import pyedflib
        except ImportError:
            raise ImportError(
                "pyedflib not installed. Install with: pip install pyedflib\n"
                "Or if you don't need EDF support, this can be ignored."
            )
        
        try:
            f = pyedflib.EdfReader(str(file_path))
            n_signals = f.signals_in_file

            signals = []
            channels = []
            fs_list = []

            for i in range(n_signals):
                signals.append(f.readSignal(i))
                channels.append(f.getLabel(i))
                fs_list.append(f.getSampleFrequency(i))

            signal = np.column_stack(signals)

            data = {
                'signal': signal,
                'fs': fs_list[0],
                'channels': channels,
                'duration': len(signal) / fs_list[0],
                'format': 'EDF'
            }

            f.close()
            return data

        except Exception as e:
            raise Exception(f"Error reading EDF file: {str(e)}")

    @staticmethod
    def read_matlab(file_path: Path) -> Dict:
        """Read MATLAB format ECG data with proper sampling frequency detection."""
        from scipy.io import loadmat
        
        mat_data = loadmat(str(file_path))
        
        # Find signal data
        signal = None
        for key in ['val', 'signal', 'data', 'ecg']:
            if key in mat_data:
                signal = mat_data[key]
                break
        
        if signal is None:
            # Take largest non-metadata array
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray):
                    if value.ndim >= 1:
                        signal = value
                        break
        
        # Fix signal shape (samples, channels)
        if signal.ndim == 1:
            signal = signal.reshape(-1, 1)
        elif signal.ndim == 2:
            if signal.shape[0] < signal.shape[1]:
                signal = signal.T
        
        # Try to extract sampling frequency from MATLAB file first
        fs = None
        for key in ['fs', 'Fs', 'freq', 'frequency', 'sampling_rate', 'sample_rate']:
            if key in mat_data:
                fs_value = mat_data[key]
                if isinstance(fs_value, np.ndarray):
                    fs = float(fs_value.flatten()[0])
                else:
                    fs = float(fs_value)
                break
        
        # If not found in .mat, try reading companion .hea file
        if fs is None:
            hea_path = file_path.with_suffix('.hea')
            if hea_path.exists():
                try:
                    fs = FormatReader._read_sampling_freq_from_hea(hea_path)
                except:
                    pass
        
        # Final fallback
        if fs is None:
            fs = 250  # Default
        
        # Determine number of channels
        n_channels = signal.shape[1] if signal.ndim > 1 else 1
        
        data = {
            'signal': signal,
            'fs': fs,
            'channels': [f'ECG_Lead_{i+1}' for i in range(n_channels)],
            'duration': len(signal) / fs,
            'format': 'MATLAB',
            'raw_keys': [k for k in mat_data.keys() if not k.startswith('__')]
        }
        
        return data
    
    @staticmethod
    def _read_sampling_freq_from_hea(hea_path: Path) -> float:
        """
        Parse a WFDB .hea (header) file to extract sampling frequency.
        Format: record_name num_signals sampling_freq num_samples ...
        """
        with open(hea_path, 'r') as f:
            first_line = f.readline().strip()
        
        parts = first_line.split()
        
        if len(parts) >= 3:
            try:
                fs = float(parts[2])
                return fs
            except ValueError:
                raise ValueError(f"Could not parse fs from: {first_line}")
        else:
            raise ValueError(f"Invalid .hea format: {first_line}")

    @staticmethod
    def read_csv(file_path: Path) -> Dict:
        """Read CSV format ECG data."""
        try:
            df = pd.read_csv(file_path)

            # Remove non-numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            signal = df[numeric_cols].values

            # Try to infer sampling frequency from time column
            fs = 250  # Default
            time_cols = [c for c in df.columns if c.lower() in ['time', 't', 'timestamp']]
            if time_cols:
                time_col = time_cols[0]
                time_diff = df[time_col].diff().median()
                if time_diff > 0:
                    fs = 1 / time_diff

            data = {
                'signal': signal,
                'fs': fs,
                'channels': numeric_cols,
                'duration': len(signal) / fs,
                'format': 'CSV',
                'all_columns': df.columns.tolist()
            }

            return data

        except Exception as e:
            raise Exception(f"Error reading CSV file: {str(e)}")

    @staticmethod
    def read_numpy(file_path: Path) -> Dict:
        """Read NumPy .npy or .npz files."""
        try:
            if file_path.suffix == '.npz':
                data_file = np.load(str(file_path))
                signal = data_file[data_file.files[0]]
                available_keys = data_file.files
            else:
                signal = np.load(str(file_path))
                available_keys = None

            data = {
                'signal': signal,
                'fs': 250,  # Must be specified externally
                'channels': [f'Channel_{i}' for i in range(signal.shape[1])] if signal.ndim > 1 else ['Channel_0'],
                'duration': len(signal) / 250,
                'format': 'NUMPY'
            }

            if available_keys:
                data['available_keys'] = available_keys

            return data

        except Exception as e:
            raise Exception(f"Error reading NumPy file: {str(e)}")


# ============================================================================
# UNIVERSAL ECG LOADER
# ============================================================================
class UniversalECGLoader:
    """
    Robust ECG loader that handles problematic records gracefully.
    """

    def __init__(self, data_path: str, auto_detect: bool = True):
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Path does not exist: {data_path}")

        self.detector = DatasetDetector(data_path)
        self.reader = FormatReader()

        if auto_detect:
            self.analysis = self.detector.analyze_directory()
            self.format_type = self.analysis['format_type']
        else:
            self.format_type = None

        self.failed_records = []
        self.valid_records_cache = None

    def get_records(self) -> List[str]:
        """Get list of all records in the dataset."""
        records = []

        if self.format_type == 'WFDB':
            for hea_file in self.data_path.rglob('*.hea'):
                if not hea_file.name.endswith('-'):
                    records.append(hea_file.stem)

        elif self.format_type in ['EDF', 'MATLAB', 'CSV', 'NUMPY']:
            ext_map = {
                'EDF': '.edf',
                'MATLAB': '.mat',
                'CSV': '.csv',
                'NUMPY': '.npy'
            }
            ext = ext_map.get(self.format_type, '')

            for data_file in self.data_path.rglob(f'*{ext}'):
                if 'metadata' not in data_file.name.lower():
                    records.append(str(data_file.relative_to(self.data_path)))

        print(f"✅ Found {len(records)} records in {self.format_type} format")
        return sorted(records)

    def get_valid_records(self, force_check: bool = False) -> List[str]:
        """Get list of only valid (loadable) records."""
        if self.valid_records_cache is not None and not force_check:
            return self.valid_records_cache

        all_records = self.get_records()
        valid_records = []
        failed_records = []

        print(f"\n🔍 Validating {len(all_records)} records...")

        for i, rec in enumerate(all_records, 1):
            try:
                _ = self.load_record(rec)
                valid_records.append(rec)

                if i % 10 == 0:
                    print(f"   Progress: {i}/{len(all_records)} ({len(valid_records)} valid)")

            except Exception as e:
                failed_records.append((rec, str(e)))
                if i % 10 == 0:
                    print(f"   Progress: {i}/{len(all_records)} ({len(valid_records)} valid)")

        self.valid_records_cache = valid_records
        self.failed_records = failed_records

        print(f"\n✅ Found {len(valid_records)} valid records")

        if failed_records:
            print(f"⚠️  Skipped {len(failed_records)} problematic records:")
            for rec, error in failed_records[:5]:
                print(f"   • {rec}: {error[:60]}...")
            if len(failed_records) > 5:
                print(f"   ... and {len(failed_records) - 5} more")

        return valid_records

    def load_record(self, record_id: str, **kwargs) -> Dict:
        """Load a single ECG record with error handling."""
        try:
            if self.format_type == 'WFDB':
                return self.reader.read_wfdb(self.data_path, record_id)

            elif self.format_type == 'EDF':
                file_path = self.data_path / record_id
                return self.reader.read_edf(file_path)

            elif self.format_type == 'MATLAB':
                file_path = self.data_path / record_id
                return self.reader.read_matlab(file_path)

            elif self.format_type == 'CSV':
                file_path = self.data_path / record_id
                return self.reader.read_csv(file_path)

            elif self.format_type == 'NUMPY':
                file_path = self.data_path / record_id
                return self.reader.read_numpy(file_path)

            else:
                raise ValueError(f"Unknown format: {self.format_type}")

        except Exception as e:
            raise Exception(f"Failed to load record '{record_id}': {str(e)}")

    def load_all(self, max_records: Optional[int] = None, use_valid_only: bool = True) -> List[Dict]:
        """Load all records from the dataset."""
        if use_valid_only:
            records = self.get_valid_records()
        else:
            records = self.get_records()

        if max_records:
            records = records[:max_records]

        print(f"\n🔄 Loading {len(records)} records...")
        all_data = []

        for i, rec in enumerate(records, 1):
            try:
                data = self.load_record(rec)
                data['record_id'] = rec
                all_data.append(data)

                if i % 10 == 0:
                    print(f"   Progress: {i}/{len(records)}")

            except Exception as e:
                print(f"   ⚠️  Failed to load {rec}: {str(e)[:60]}...")
                continue

        print(f"\n✅ Successfully loaded {len(all_data)}/{len(records)} records\n")
        return all_data

    def get_dataset_info(self, sample_size: int = 20) -> Dict:
        """Get comprehensive information about the dataset."""
        records = self.get_valid_records()

        if not records:
            return {"error": "No valid records found"}

        print(f"\n📊 Analyzing dataset characteristics...")

        info = {
            'total_records': len(self.get_records()),
            'valid_records': len(records),
            'failed_records': len(self.failed_records),
            'sampling_frequencies': {},
            'lead_configurations': {},
            'duration_stats': {
                'min': float('inf'),
                'max': 0,
                'durations': []
            },
            'formats': {},
            'sample_records': []
        }

        sample_size = min(sample_size, len(records))
        step = max(1, len(records) // sample_size)
        sample_records = records[::step][:sample_size]

        for rec in sample_records:
            try:
                data = self.load_record(rec)

                fs = data['fs']
                info['sampling_frequencies'][fs] = info['sampling_frequencies'].get(fs, 0) + 1

                num_leads = len(data['channels'])
                lead_config = f"{num_leads}-lead"
                info['lead_configurations'][lead_config] = info['lead_configurations'].get(lead_config, 0) + 1

                duration = data['duration']
                info['duration_stats']['durations'].append(duration)
                info['duration_stats']['min'] = min(info['duration_stats']['min'], duration)
                info['duration_stats']['max'] = max(info['duration_stats']['max'], duration)

                fmt = data['format']
                info['formats'][fmt] = info['formats'].get(fmt, 0) + 1

                info['sample_records'].append({
                    'record_id': rec,
                    'fs': fs,
                    'leads': data['channels'],
                    'duration': duration,
                    'signal_shape': data['signal'].shape
                })

            except Exception as e:
                print(f"   ⚠️  Could not analyze {rec}: {str(e)[:40]}...")
                continue

        if info['duration_stats']['durations']:
            info['duration_stats']['mean'] = np.mean(info['duration_stats']['durations'])
            info['duration_stats']['median'] = np.median(info['duration_stats']['durations'])

        self._print_dataset_info(info)

        return info

    def _print_dataset_info(self, info: Dict):
        """Pretty print dataset information."""
        print(f"\n{'='*60}")
        print(f"DATASET SUMMARY")
        print(f"{'='*60}\n")

        print(f"📈 Total Records: {info['total_records']}")
        print(f"✅ Valid Records: {info['valid_records']}")
        if info['failed_records'] > 0:
            print(f"⚠️  Failed Records: {info['failed_records']}")

        print(f"\n🔊 Sampling Frequencies Found:")
        for fs, count in sorted(info['sampling_frequencies'].items()):
            print(f"   • {fs} Hz: {count} record(s)")

        print(f"\n📡 Lead Configurations:")
        for config, count in sorted(info['lead_configurations'].items()):
            print(f"   • {config}: {count} record(s)")

        print(f"\n⏱️  Duration Statistics:")
        if info['duration_stats']['durations']:
            print(f"   • Min: {info['duration_stats']['min']:.2f} seconds")
            print(f"   • Max: {info['duration_stats']['max']:.2f} seconds")
            print(f"   • Mean: {info['duration_stats']['mean']:.2f} seconds")
            print(f"   • Median: {info['duration_stats']['median']:.2f} seconds")

        print(f"\n📝 Sample Records:")
        for sample in info['sample_records'][:3]:
            print(f"   • {sample['record_id']}:")
            print(f"     - Frequency: {sample['fs']} Hz")
            print(f"     - Leads: {', '.join(sample['leads'][:3])}{'...' if len(sample['leads']) > 3 else ''}")
            print(f"     - Duration: {sample['duration']:.2f}s")
            print(f"     - Shape: {sample['signal_shape']}")

        print(f"\n{'='*60}\n")