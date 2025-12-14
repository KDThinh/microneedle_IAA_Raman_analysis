"""
Utilities for compiling FFT (Fourier Transform) CSV files from multiple profiles.

This module provides functions to:
- Find FFT CSV files (fft_full_gband and fft_full_raman) for each profile
- Filter frequency data (0 to 0.5 cycles/hour)
- Compile all profiles' FFT data into a single CSV/Parquet file
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

try:
    import yaml
except ImportError:
    yaml = None

from .config_loader import load_profile_config, get_profiles_by_experiment
from .ingestion import load_raman_dataset
from .batch_summary_compiler import _optimize_dataframe_dtypes

FFT_LABEL_ALIASES = {
    'gband': 'gband_ratio',
    'gband_ratio': 'gband_ratio',
    'fluorescence_to_gband': 'gband_ratio',
    'raman': 'raman_850_ratio',
    'raman_850': 'raman_850_ratio',
    'raman_850_ratio': 'raman_850_ratio',
    'fluorescence_to_raman': 'raman_850_ratio',
}

FFT_LABEL_DISPLAY = {
    'gband_ratio': 'Fluorescence/G-band Ratio',
    'raman_850_ratio': 'Fluorescence/Raman Peak 850 Ratio',
}


def find_latest_fft_csv(profile_name: str, config_path: str, fft_label: str = 'gband_ratio', version: str = 'v4') -> Optional[Path]:
    """
    Find the latest FFT CSV file for a given profile and FFT type.
    
    Searches for directories ending with _YYYYMMDD_HHMMSS pattern (any folder name)
    and selects the one with the latest timestamp. Then looks for FFT CSV files
    matching the pattern fft_full_{fft_label}_*.csv.
    
    Parameters:
    -----------
    profile_name : str
        Name of the profile to search for
    config_path : str
        Path to pipeline.yml config file
    fft_label : str
        Internal FFT label (e.g., 'gband_ratio', 'raman_850_ratio')
    version : str
        Version string (currently not used for pattern matching, kept for compatibility)
    
    Returns:
    --------
    Path or None
        Path to the latest FFT CSV file, or None if not found
    """
    # Load profile config to get data source path
    try:
        config = load_profile_config(config_path, profile_name)
    except Exception as e:
        print(f"Warning: Could not load config for profile '{profile_name}': {e}")
        return None
    
    # Handle working directory change for requires_google_drive: false
    # (same logic as main_test_v4.py)
    data_source = config.get("data_source", {})
    requires_drive = data_source.get("requires_google_drive", True)
    original_cwd = os.getcwd()
    
    try:
        if not requires_drive:
            # Change to parent directory (same logic as main_test_v4.py)
            # Calculate project root from config path: config/pipeline.yml -> swnt_iaa_analysis_v2
            config_path_obj = Path(config_path).resolve()
            project_root = config_path_obj.parent.parent  # config -> swnt_iaa_analysis_v2
            # Go up two levels from project root: swnt_iaa_analysis_v2 -> Script -> IAA-MN longitudinal
            parent_dir = project_root.parent.parent
            if parent_dir.exists():
                os.chdir(parent_dir)
        
        # Load dataset to get source path
        try:
            dataset = load_raman_dataset(config)
            raw_data_dir = Path(dataset.source_path).parent
        except Exception as e:
            print(f"Warning: Could not load dataset for profile '{profile_name}': {e}")
            return None
    finally:
        # Restore original working directory
        if not requires_drive:
            os.chdir(original_cwd)
    
    # Pattern for batch directories ending with _YYYYMMDD_HHMMSS
    batch_pattern = re.compile(r'.*_(\d{8}_\d{6})$')
    
    # Pattern for FFT CSV files (matches fft_full_{label}_YYYYMMDD[_HHMMSS].csv)
    fft_pattern = re.compile(rf'fft_full_{re.escape(fft_label)}_(\d{{8}})(?:_(\d{{6}}))?\.csv')
    
    if not raw_data_dir.exists():
        print(f"Warning: Raw data directory does not exist: {raw_data_dir}")
        return None
    
    matching_files: List[Tuple[datetime, Path, str]] = []
    
    for item in raw_data_dir.iterdir():
        if not item.is_dir():
            continue
        match = batch_pattern.match(item.name)
        if not match:
            continue
        
        timestamp_dir = match.group(1)
        try:
            dir_timestamp = datetime.strptime(timestamp_dir, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        
        for file_path in item.glob(f"fft_full_{fft_label}_*.csv"):
            file_match = fft_pattern.match(file_path.name)
            if file_match:
                date_part = file_match.group(1)
                time_part = file_match.group(2) or "000000"
                timestamp_str = f"{date_part}_{time_part}"
                try:
                    file_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                except ValueError:
                    file_timestamp = dir_timestamp
            else:
                file_timestamp = dir_timestamp
            matching_files.append((file_timestamp, file_path, item.name))
    
    if not matching_files:
        print(f"Warning: No FFT CSV found for profile '{profile_name}' (type: {fft_label}) in {raw_data_dir}")
        return None
    
    matching_files.sort(key=lambda x: x[0], reverse=True)
    selected_folder = matching_files[0][2]
    selected_fft = matching_files[0][1]
    
    # Show which folder was selected
    print(f"  Selected folder: {selected_folder}")
    
    return selected_fft


def compile_fft_data(
    config_path: str,
    output_path: Optional[str] = None,
    profiles: Optional[List[str]] = None,
    experiment_type: Optional[str] = 'in planta',
    version: str = 'v4',
    fft_types: Optional[List[str]] = None,
    frequency_min: float = 0.0,
    frequency_max: float = 0.5,
    add_profile_column: bool = True,
    add_metadata_columns: bool = True,
    optimize_dtypes: bool = True,
    float_precision: int = 6,
    compression: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compile FFT CSV files from multiple profiles into a single DataFrame.
    
    For each profile, finds the latest FFT CSV files (fft_full_gband and fft_full_raman)
    and combines them all into one DataFrame, filtering by frequency range.
    
    Parameters:
    -----------
    config_path : str
        Path to pipeline.yml config file
    output_path : str, optional
        Path to save the compiled CSV/Parquet. If None, returns DataFrame without saving.
    profiles : List[str], optional
        List of specific profiles to compile. If None, uses all profiles matching experiment_type.
    experiment_type : str, optional
        Filter profiles by experiment type (default: 'in planta'). Ignored if profiles is provided.
    version : str
        Version to look for ('v3' or 'v4', default: 'v4')
    fft_types : List[str], optional
        List of FFT types to compile: ['gband', 'raman'] (default: both)
    frequency_min : float
        Minimum frequency to include (cycles/hour, default: 0.0)
    frequency_max : float
        Maximum frequency to include (cycles/hour, default: 0.5)
    add_profile_column : bool
        If True, adds a 'Profile' column to identify which profile each row came from (default: True)
    add_metadata_columns : bool
        If True, adds metadata columns (plant_type, treatment, light_cycle, etc.) from profile config (default: True)
    optimize_dtypes : bool
        If True, optimizes data types to reduce file size (default: True)
    float_precision : int
        Number of decimal places to keep for float columns when optimizing (default: 6)
    compression : str, optional
        Compression format: 'gzip', 'bz2', 'xz', 'zip', or 'parquet' (default: None).
        'parquet' uses Parquet format which is highly compressed and efficient.
    
    Returns:
    --------
    pd.DataFrame
        Compiled DataFrame with all FFT data combined
    """
    # Default to both FFT types if not specified
    if fft_types is None:
        fft_types = ['gband', 'raman']
    
    resolved_fft_labels: List[str] = []
    display_names: dict[str, str] = {}
    for fft_type in fft_types:
        alias = FFT_LABEL_ALIASES.get(fft_type.lower(), fft_type)
        if alias not in resolved_fft_labels:
            resolved_fft_labels.append(alias)
            display_names[alias] = FFT_LABEL_DISPLAY.get(alias, alias.replace('_', ' ').title())
    
    # Get list of profiles to process
    if profiles is None:
        if experiment_type:
            profiles = get_profiles_by_experiment(config_path, experiment_type)
        else:
            from .config_loader import list_profiles
            profiles = list_profiles(config_path, exclude_base_profiles=True)
    
    if not profiles:
        raise ValueError("No profiles found to compile")
    
    print(f"Compiling FFT data for {len(profiles)} profiles...")
    print(f"Version: {version}, Experiment type: {experiment_type or 'all'}")
    print(f"FFT types: {[display_names[label] for label in resolved_fft_labels]}")
    print(f"Frequency range: {frequency_min} to {frequency_max} cycles/hour")
    
    compiled_data = []
    successful_profiles = []
    failed_profiles = []
    rows_per_profile = {}  # Track rows per profile for summary
    
    for i, profile_name in enumerate(profiles, 1):
        print(f"\n[{i}/{len(profiles)}] Processing profile: {profile_name}")
        
        profile_data = []
        
        for fft_label in resolved_fft_labels:
            label_name = display_names.get(fft_label, fft_label)
            print(f"  Processing {label_name} FFT data...")
            
            # Find latest FFT CSV
            fft_path = find_latest_fft_csv(profile_name, config_path, fft_label=fft_label, version=version)
            
            if fft_path is None:
                print(f"    ✗ No {label_name} FFT data found")
                continue
            
            print(f"    ✓ Found: {fft_path}")
            
            try:
                # Load the CSV
                df = pd.read_csv(fft_path)
                
                # Check if required columns exist
                freq_col = None
                for col in df.columns:
                    if 'frequency' in col.lower() and 'cycle' in col.lower():
                        freq_col = col
                        break
                
                if freq_col is None:
                    print(f"    ⚠ Warning: No frequency column found in {fft_path}")
                    continue
                
                # Filter by frequency range
                df_filtered = df[(df[freq_col] >= frequency_min) & (df[freq_col] <= frequency_max)].copy()
                
                if df_filtered.empty:
                    print(f"    ⚠ Warning: No data in frequency range {frequency_min}-{frequency_max} for {label_name}")
                    continue
                
                # Add FFT type column
                df_filtered['FFT_Type'] = fft_label
                
                # Add profile column if requested
                if add_profile_column:
                    df_filtered['Profile'] = profile_name
                
                # Add metadata columns if requested
                if add_metadata_columns:
                    try:
                        # Load profile config to get metadata
                        profile_config = load_profile_config(config_path, profile_name)
                        metadata = profile_config.get('metadata', {})
                        
                        # Add each metadata field as a column
                        for key, value in metadata.items():
                            df_filtered[key] = value
                    except Exception as e:
                        print(f"    ⚠ Warning: Could not load metadata for profile '{profile_name}': {e}")
                        # Continue without metadata rather than failing
                
                profile_data.append(df_filtered)
                print(f"    ✓ Loaded {len(df_filtered)} rows (filtered from {len(df)} total)")
                
            except Exception as e:
                print(f"    ✗ Error loading {fft_type} FFT CSV: {e}")
                continue
        
        if profile_data:
            # Combine all FFT types for this profile
            profile_df = pd.concat(profile_data, ignore_index=True)
            compiled_data.append(profile_df)
            successful_profiles.append(profile_name)
            rows_per_profile[profile_name] = len(profile_df)
        else:
            print(f"  ✗ No FFT data found for profile")
            failed_profiles.append(profile_name)
    
    if not compiled_data:
        raise ValueError("No FFT data could be loaded. Check that profiles have been processed.")
    
    # Combine all DataFrames
    print(f"\n{'='*80}")
    print("Compiling DataFrames...")
    compiled_df = pd.concat(compiled_data, ignore_index=True)
    
    # Optimize data types if requested
    original_size_mb = compiled_df.memory_usage(deep=True).sum() / 1024 / 1024
    if optimize_dtypes:
        print("Optimizing data types to reduce file size...")
        compiled_df = _optimize_dataframe_dtypes(compiled_df, float_precision=float_precision)
        optimized_size_mb = compiled_df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (1 - optimized_size_mb / original_size_mb) * 100
        print(f"  Memory reduction: {original_size_mb:.2f} MB → {optimized_size_mb:.2f} MB ({reduction:.1f}% reduction)")
    
    print(f"\n{'='*80}")
    print("COMPILATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total profiles processed: {len(profiles)}")
    print(f"FFT files compiled: {len(successful_profiles)}")
    print(f"Successful: {len(successful_profiles)}")
    print(f"Failed: {len(failed_profiles)}")
    print(f"Total rows in compiled file: {len(compiled_df)}")
    print(f"Columns: {list(compiled_df.columns)}")
    
    if rows_per_profile:
        print(f"\nRows per profile:")
        for profile_name, row_count in sorted(rows_per_profile.items()):
            print(f"  {profile_name}: {row_count:,} rows")
    
    if failed_profiles:
        print(f"\nFailed profiles:")
        for profile in failed_profiles:
            print(f"  - {profile}")
    
    # Save if output path provided
    if output_path:
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine file format and compression based on extension and compression parameter
        output_str = str(output_path_obj)
        
        if compression == 'parquet' or output_str.endswith('.parquet'):
            # Save as Parquet (highly compressed, efficient)
            try:
                if not output_str.endswith('.parquet'):
                    output_path_obj = output_path_obj.with_suffix('.parquet')
                compiled_df.to_parquet(output_path_obj, index=False, compression='snappy')
                file_size_mb = output_path_obj.stat().st_size / 1024 / 1024
                print(f"\n✓ Compiled Parquet saved to: {output_path_obj}")
                print(f"  File size: {file_size_mb:.2f} MB")
            except ImportError:
                print("\n⚠ Warning: PyArrow is required for Parquet format. Install with: pip install pyarrow")
                print("  Falling back to CSV format...")
                output_path_obj = output_path_obj.with_suffix('.csv')
                compiled_df.to_csv(output_path_obj, index=False)
                file_size_mb = output_path_obj.stat().st_size / 1024 / 1024
                print(f"✓ Compiled CSV saved to: {output_path_obj}")
                print(f"  File size: {file_size_mb:.2f} MB")
        elif compression in ['gzip', 'bz2', 'xz', 'zip']:
            # Save as compressed CSV
            compiled_df.to_csv(output_path_obj, index=False, compression=compression)
            file_size_mb = output_path_obj.stat().st_size / 1024 / 1024
            print(f"\n✓ Compiled CSV ({compression}) saved to: {output_path_obj}")
            print(f"  File size: {file_size_mb:.2f} MB")
        else:
            # Save as regular CSV
            compiled_df.to_csv(output_path_obj, index=False)
            file_size_mb = output_path_obj.stat().st_size / 1024 / 1024
            print(f"\n✓ Compiled CSV saved to: {output_path_obj}")
            print(f"  File size: {file_size_mb:.2f} MB")
    
    return compiled_df

