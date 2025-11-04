# %%

"""
Universal ECG Dataset Loader for AFib Research
A truly generalized approach that can adapt to any ECG dataset format
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

# %%
# ============================================================================
# STEP 1: DATASET DETECTOR - Automatically detects dataset format
# ============================================================================
class DatasetDetector:
    """
    Intelligently detects the format and structure of any ECG dataset.
    This is the KEY to making the loader work with unknown datasets.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.structure = {}

    def analyze_directory(self) -> Dict:
        """
        Analyzes directory structure and file types to understand the dataset.

        Returns a dictionary describing:
        - File formats found (.dat, .hea, .csv, .mat, .edf, etc.)
        - Directory structure
        - Potential metadata files
        - Number of records
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
        """
        Determines the dataset format based on file extensions.

        Common formats:
        - WFDB: .hea + .dat files (MIT-BIH, PhysioNet databases)
        - EDF: .edf files (European Data Format)
        - MATLAB: .mat files
        - CSV: .csv files with time series
        - NumPy: .npy or .npz files
        """
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


# %%
"""
Fixed FormatReader class with proper sampling frequency detection
for PhysioNet Challenge 2017 and other MATLAB-based datasets
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from scipy.io import loadmat


class FormatReader:
    """
    Contains readers for different ECG data formats.
    """

    @staticmethod
    def read_matlab(file_path: Path) -> Dict:
        """
        Read MATLAB format ECG data with proper sampling frequency detection.
        
        For PhysioNet Challenge 2017, reads the .hea file to get true fs.
        """
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
            signal = signal.reshape(-1, 1)  # (9000,) → (9000, 1)
        elif signal.ndim == 2:
            if signal.shape[0] < signal.shape[1]:
                signal = signal.T  # (1, 9000) → (9000, 1)
        
        # 🔥 FIX: Try to extract sampling frequency from MATLAB file first
        fs = None
        for key in ['fs', 'Fs', 'freq', 'frequency', 'sampling_rate', 'sample_rate']:
            if key in mat_data:
                fs_value = mat_data[key]
                if isinstance(fs_value, np.ndarray):
                    fs = float(fs_value.flatten()[0])
                else:
                    fs = float(fs_value)
                break
        
        # 🔥 FIX: If not found in .mat, try reading companion .hea file
        if fs is None:
            hea_path = file_path.with_suffix('.hea')
            if hea_path.exists():
                try:
                    fs = FormatReader._read_sampling_freq_from_hea(hea_path)
                    # Removed verbose logging - fs is silently read from .hea
                except Exception as e:
                    # Only print errors, not routine operations
                    pass
        
        # Final fallback (only warn if using default)
        if fs is None:
            fs = 250  # Default only if everything else fails
            # Optional: uncomment next line if you want to see when defaults are used
            # print(f"   ⚠️  Using default fs=250 Hz for {file_path.name}")
        
        # Determine number of channels AFTER reshaping
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
        Example: A00001 1 300 9000
        """
        with open(hea_path, 'r') as f:
            first_line = f.readline().strip()
        
        parts = first_line.split()
        
        if len(parts) >= 3:
            try:
                fs = float(parts[2])  # Third field is sampling frequency
                return fs
            except ValueError:
                raise ValueError(f"Could not parse fs from: {first_line}")
        else:
            raise ValueError(f"Invalid .hea format: {first_line}")

    @staticmethod
    def read_wfdb(file_path: Path, record_name: str) -> Dict:
        """Read WFDB format (.hea + .dat files)."""
        import wfdb
        
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

    @staticmethod
    def read_edf(file_path: Path) -> Dict:
        """Read EDF format files."""
        try:
            import pyedflib
            
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
            
        except ImportError:
            raise ImportError("pyedflib not installed. Install with: pip install pyedflib")
        except Exception as e:
            raise Exception(f"Error reading EDF file: {str(e)}")

# %%
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

class UniversalECGLoader:
    """
    Robust ECG loader that handles problematic records gracefully.

    Usage:
        loader = UniversalECGLoader('path/to/dataset')
        records = loader.get_valid_records()  # Only returns loadable records
        data = loader.load_record(records[0])
    """

    def __init__(self, data_path: str, auto_detect: bool = True):
        """
        Initialize the universal loader.

        Args:
            data_path: Path to the dataset directory
            auto_detect: If True, automatically detect dataset format
        """
        self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Path does not exist: {data_path}")

        # Assuming you have DatasetDetector and FormatReader classes
        # from your previous code
        self.detector = DatasetDetector(data_path)
        self.reader = FormatReader()

        if auto_detect:
            self.analysis = self.detector.analyze_directory()
            self.format_type = self.analysis['format_type']
        else:
            self.format_type = None

        # Track problematic records
        self.failed_records = []
        self.valid_records_cache = None

    def get_records(self) -> List[str]:
        """
        Get list of all records in the dataset.
        Returns record identifiers (file names without extensions).
        """
        records = []

        if self.format_type == 'WFDB':
            # Find all .hea files
            for hea_file in self.data_path.rglob('*.hea'):
                # Skip backup files
                if not hea_file.name.endswith('-'):
                    records.append(hea_file.stem)

        elif self.format_type in ['EDF', 'MATLAB', 'CSV', 'NUMPY']:
            # Find all data files
            ext_map = {
                'EDF': '.edf',
                'MATLAB': '.mat',
                'CSV': '.csv',
                'NUMPY': '.npy'
            }
            ext = ext_map.get(self.format_type, '')

            for data_file in self.data_path.rglob(f'*{ext}'):
                # Skip metadata files
                if 'metadata' not in data_file.name.lower():
                    records.append(str(data_file.relative_to(self.data_path)))

        print(f"✅ Found {len(records)} records in {self.format_type} format")
        return sorted(records)

    def get_valid_records(self, force_check: bool = False) -> List[str]:
        """
        Get list of only valid (loadable) records by checking each one.
        This is slower but ensures all returned records can be loaded.

        Args:
            force_check: If True, re-check all records even if cached

        Returns:
            List of valid record identifiers
        """
        if self.valid_records_cache is not None and not force_check:
            return self.valid_records_cache

        all_records = self.get_records()
        valid_records = []
        failed_records = []

        print(f"\n🔍 Validating {len(all_records)} records...")

        for i, rec in enumerate(all_records, 1):
            try:
                # Try loading just the header or minimal data
                _ = self.load_record(rec)
                valid_records.append(rec)

                if i % 10 == 0:
                    print(f"   Progress: {i}/{len(all_records)} ({len(valid_records)} valid)")

            except Exception as e:
                failed_records.append((rec, str(e)))
                if i % 10 == 0:
                    print(f"   Progress: {i}/{len(all_records)} ({len(valid_records)} valid)")

        # Cache results
        self.valid_records_cache = valid_records
        self.failed_records = failed_records

        print(f"\n✅ Found {len(valid_records)} valid records")

        if failed_records:
            print(f"⚠️  Skipped {len(failed_records)} problematic records:")
            for rec, error in failed_records[:5]:  # Show first 5
                print(f"   • {rec}: {error[:60]}...")
            if len(failed_records) > 5:
                print(f"   ... and {len(failed_records) - 5} more")

        return valid_records

    def load_record(self, record_id: str, **kwargs) -> Dict:
        """
        Load a single ECG record with error handling.

        Args:
            record_id: Record identifier (from get_records())
            **kwargs: Additional arguments passed to format-specific reader

        Returns:
            Dictionary with standardized format
        """
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
            # Re-raise with more context
            raise Exception(f"Failed to load record '{record_id}': {str(e)}")

    def load_all(self, max_records: Optional[int] = None, use_valid_only: bool = True) -> List[Dict]:
        """
        Load all records from the dataset.

        Args:
            max_records: Maximum number of records to load (None = all)
            use_valid_only: If True, only load pre-validated records

        Returns:
            List of loaded records
        """
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
        """
        Get comprehensive information about the loaded dataset.

        Args:
            sample_size: Number of records to sample for analysis
        """
        # Use valid records only
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

        # Sample records for analysis
        sample_size = min(sample_size, len(records))
        # Take evenly distributed samples
        step = max(1, len(records) // sample_size)
        sample_records = records[::step][:sample_size]

        for rec in sample_records:
            try:
                data = self.load_record(rec)

                # Track sampling frequency
                fs = data['fs']
                info['sampling_frequencies'][fs] = info['sampling_frequencies'].get(fs, 0) + 1

                # Track lead configuration
                num_leads = len(data['channels'])
                lead_config = f"{num_leads}-lead"
                info['lead_configurations'][lead_config] = info['lead_configurations'].get(lead_config, 0) + 1

                # Track duration
                duration = data['duration']
                info['duration_stats']['durations'].append(duration)
                info['duration_stats']['min'] = min(info['duration_stats']['min'], duration)
                info['duration_stats']['max'] = max(info['duration_stats']['max'], duration)

                # Track format
                fmt = data['format']
                info['formats'][fmt] = info['formats'].get(fmt, 0) + 1

                # Store sample record info
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

        # Calculate statistics
        if info['duration_stats']['durations']:
            info['duration_stats']['mean'] = np.mean(info['duration_stats']['durations'])
            info['duration_stats']['median'] = np.median(info['duration_stats']['durations'])

        # Print summary
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


# ============================================================================
# USAGE EXAMPLE for MIT-BIH AF Database
# ============================================================================
"""
# Initialize loader
loader = UniversalECGLoader('/content/drive/MyDrive/AF/files')

# Get only valid records (automatically filters out problematic ones)
valid_records = loader.get_valid_records()
print(f"Valid records: {valid_records}")

# Load a single record
if valid_records:
    print(f"Signal shape: {data['signal'].shape}")
    print(f"Sampling rate: {data['fs']} Hz")
    print(f"Channels: {data['channels']}")

# Get dataset information
info = loader.get_dataset_info()

# Load all valid records
all_data = loader.load_all(max_records=10)  # Load first 10
"""

# %%
# ============================================================================
# DATA EXPLORATION SCRIPT
# ============================================================================

import sys
# This line assumes your script is in the '/notebooks' folder and the loader
# code is in '/src'. It adds the 'src' folder to the path.
sys.path.append('../src')



# --- IMPORTANT: UPDATE THESE PATHS ---
afdb_path = '../data/raw/mit-bih-afdb/'
cinc_path = '../data/raw/physionet-challenge-2017/'
# -----------------------------------

print(f"\n{'='*80}")
print(" EXPLORING: MIT-BIH Atrial Fibrillation Database (AFDB)")
print(f"{'='*80}")

# --- Initialize loader for the first dataset ---
afdb_loader = UniversalECGLoader(afdb_path)

# --- Get only valid records (automatically filters out problematic ones) ---
afdb_valid_records = afdb_loader.get_valid_records()
# print(f"Found {len(afdb_valid_records)} valid records in AFDB.")

# --- Load a single record to inspect its structure ---
if afdb_valid_records:
    print("\n--- Loading a single sample record from AFDB ---")
    data = afdb_loader.load_record(afdb_valid_records[0])
    print(f"Signal shape: {data['signal'].shape}")
    print(f"Sampling rate: {data['fs']} Hz")
    print(f"Channels: {data['channels']}")

# --- Get a full statistical summary of the dataset ---
afdb_info = afdb_loader.get_dataset_info()


print(f"\n\n{'='*80}")
print(" EXPLORING: PhysioNet/Computing in Cardiology Challenge 2017")
print(f"{'='*80}")

# --- Initialize loader for the second dataset ---
cinc_loader = UniversalECGLoader(cinc_path)

# --- Get only valid records ---
cinc_valid_records = cinc_loader.get_valid_records()
# print(f"Found {len(cinc_valid_records)} valid records in CINC-2017.")

# --- Load a single record ---
if cinc_valid_records:
    print("\n--- Loading a single sample record from CINC-2017 ---")
    data = cinc_loader.load_record(cinc_valid_records[0])
    print(f"Signal shape: {data['signal'].shape}")
    print(f"Sampling rate: {data['fs']} Hz")
    print(f"Channels: {data['channels']}")

# --- Get a full statistical summary of the dataset ---
cinc_info = cinc_loader.get_dataset_info()

# --- Example of loading a small batch (from CINC-2017) ---
print("\n--- Example of loading a batch of 10 records from CINC-2017 ---")
all_data_batch = cinc_loader.load_all(max_records=10)
print(f"Successfully loaded a batch of {len(all_data_batch)} records.")
if all_data_batch:
    print(f"First record in batch has ID: {all_data_batch[0]['record_id']}")


