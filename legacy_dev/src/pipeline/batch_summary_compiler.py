"""
Utilities for compiling batch summary CSV files from multiple profiles.

This module provides functions to:
- Find the latest batch summary CSV for each profile
- Compile all profiles' batch summaries into a single CSV file
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
from .utils import find_google_drive


def _get_default_output_path(config_path: str, version: str = 'v4') -> Path:
    """
    Get the default output path for compiled batch summaries.
    
    Returns path to: IAA Nanosensor Experiment/Result Summary/compiled_batch_summaries_{version}_{timestamp}.csv
    
    Parameters:
    -----------
    config_path : str
        Path to pipeline.yml config file (used to determine work root)
    version : str
        Version string (v3 or v4) to include in filename
    
    Returns:
    --------
    Path
        Default output path for compiled summaries
    """
    # Infer Work directory by loading a profile and extracting it from the resolved dataset path
    try:
        if yaml is None:
            raise ImportError("PyYAML is required")
        
        # Try to load any profile to get config structure
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}
        
        profiles = config_data.get('profiles', {})
        if not profiles:
            raise ValueError("No profiles found in config")
        
        # Get first non-base profile to resolve its path
        # Handle working directory change for requires_google_drive: false
        config_path_obj = Path(config_path).resolve()
        # Calculate project root from config path: config/pipeline.yml -> swnt_iaa_analysis_v2
        project_root = config_path_obj.parent.parent  # config -> swnt_iaa_analysis_v2
        # Go up two levels from project root: swnt_iaa_analysis_v2 -> Script -> IAA-MN longitudinal
        parent_dir = project_root.parent.parent
        original_cwd = os.getcwd()
        
        base_path = None
        for profile_name in profiles.keys():
            if profile_name == 'processing_default':
                continue
            try:
                test_config = load_profile_config(config_path, profile_name)
                data_source = test_config.get("data_source", {})
                requires_drive = data_source.get("requires_google_drive", True)
                
                # Change working directory if needed (same as main_test_v4.py)
                if not requires_drive and parent_dir.exists():
                    os.chdir(parent_dir)
                
                try:
                    test_dataset = load_raman_dataset(test_config)
                    # The source_path is already resolved by load_raman_dataset
                    source_path = Path(test_dataset.source_path)
                    # Path format: {Work}/DiSTAP/.../IAA Nanosensor Experiment/...
                    # Find "DiSTAP" in the path and get everything before it
                    parts = source_path.parts
                    if "DiSTAP" in parts:
                        distap_idx = parts.index("DiSTAP")
                        base_path = Path(*parts[:distap_idx])
                        break
                finally:
                    # Restore working directory
                    if not requires_drive:
                        os.chdir(original_cwd)
            except Exception:
                continue
        
        # Restore working directory if still changed
        if os.getcwd() != original_cwd:
            os.chdir(original_cwd)
        
        if base_path is None:
            # If no profile worked, try Google Drive
            try:
                drive_root = find_google_drive()
                base_path = Path(drive_root) / "My Drive" / "Work"
            except Exception:
                # Final fallback: infer from config file location
                # Config is at: {Work}/DiSTAP/.../Script/swnt_iaa_analysis_v2/config/pipeline.yml
                if "DiSTAP" in config_path_obj.parts:
                    distap_idx = config_path_obj.parts.index("DiSTAP")
                    base_path = Path(*config_path_obj.parts[:distap_idx])
                else:
                    raise ValueError("Could not determine Work directory")
        
        # Construct path to Result Summary directory
        result_summary_dir = base_path / "DiSTAP" / "Research" / "Auxin IAA" / "IAA-MN longitudinal" / "IAA Nanosensor Experiment" / "Result Summary"
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = result_summary_dir / f"compiled_batch_summaries_{version}_{timestamp}.csv"
        
        return output_file
        
    except Exception as e:
        # Fallback: use current directory
        print(f"Warning: Could not determine default output path: {e}")
        print("Using current directory as fallback")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path.cwd() / f"compiled_batch_summaries_{version}_{timestamp}.csv"


def _optimize_dataframe_dtypes(df: pd.DataFrame, float_precision: int = 6) -> pd.DataFrame:
    """
    Optimize DataFrame data types to reduce memory usage and file size.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to optimize
    float_precision : int
        Number of decimal places to keep for float columns (default: 6)
    
    Returns:
    --------
    pd.DataFrame
        Optimized DataFrame with reduced memory footprint
    """
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        # Optimize integer columns
        if col_type in ['int64', 'int32']:
            c_min = df_optimized[col].min()
            c_max = df_optimized[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df_optimized[col] = df_optimized[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df_optimized[col] = df_optimized[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df_optimized[col] = df_optimized[col].astype(np.int32)
        
        # Optimize float columns (reduce precision and use float32 if possible)
        elif col_type in ['float64', 'float32']:
            # Round to specified precision
            df_optimized[col] = df_optimized[col].round(float_precision)
            # Try to use float32 (half the size of float64)
            # Only if values are within float32 range
            if df_optimized[col].notna().any():
                try:
                    # Check if values fit in float32 range
                    if (df_optimized[col].abs() < np.finfo(np.float32).max).all():
                        df_optimized[col] = df_optimized[col].astype(np.float32)
                except (OverflowError, ValueError):
                    # Keep as float64 if conversion fails
                    pass
        
        # Convert object columns to category if they have few unique values
        elif col_type == 'object':
            unique_ratio = df_optimized[col].nunique() / len(df_optimized)
            if unique_ratio < 0.5:  # If less than 50% unique values
                try:
                    df_optimized[col] = df_optimized[col].astype('category')
                except (ValueError, TypeError):
                    # Keep as object if conversion fails
                    pass
    
    return df_optimized


def find_latest_batch_summary_csv(profile_name: str, config_path: str, version: str = 'v4') -> Optional[Path]:
    """
    Find the latest batch summary CSV file for a given profile.
    
    Searches for directories ending with _YYYYMMDD_HHMMSS pattern (any folder name)
    and selects the one with the latest timestamp. Then looks for any CSV file
    containing 'batch_summary' in its name within that directory.
    
    Parameters:
    -----------
    profile_name : str
        Name of the profile to search for
    config_path : str
        Path to pipeline.yml config file
    version : str
        Version string (currently not used for pattern matching, kept for compatibility)
    
    Returns:
    --------
    Path or None
        Path to the latest batch summary CSV file, or None if not found
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
    
    # Pattern for any directory ending with _YYYYMMDD_HHMMSS (flexible naming)
    # Matches any folder name that ends with underscore followed by timestamp pattern
    pattern = re.compile(r'.*_(\d{8}_\d{6})$')
    
    # Find all matching directories
    matching_dirs = []
    if not raw_data_dir.exists():
        print(f"Warning: Raw data directory does not exist: {raw_data_dir}")
        return None
    
    for item in raw_data_dir.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                timestamp_str = match.group(1)
                
                # Search for any CSV file containing "batch_summary" in its name
                csv_files = [f for f in item.iterdir() if f.is_file() and f.suffix.lower() == '.csv' and 'batch_summary' in f.name.lower()]
                
                if csv_files:
                    # Prefer files with "final" in the name, otherwise take the first one
                    final_csv = next((f for f in csv_files if 'final' in f.name.lower()), None)
                    csv_path = final_csv if final_csv else csv_files[0]
                    
                    try:
                        # Parse timestamp to get datetime for sorting (only criterion for selection)
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        matching_dirs.append((timestamp, csv_path, item.name))
                    except ValueError:
                        # Skip if timestamp can't be parsed
                        continue
    
    if not matching_dirs:
        print(f"Warning: No batch summary CSV found for profile '{profile_name}' in {raw_data_dir}")
        return None
    
    # Sort by timestamp only (latest first) and return the most recent
    matching_dirs.sort(key=lambda x: x[0], reverse=True)
    selected_folder = matching_dirs[0][2]
    selected_csv = matching_dirs[0][1]
    
    # Show which folder was selected
    print(f"  Selected folder: {selected_folder}")
    
    return selected_csv


def compile_batch_summaries(
    config_path: str,
    output_path: Optional[str] = None,
    profiles: Optional[List[str]] = None,
    experiment_type: Optional[str] = 'in planta',
    version: str = 'v4',
    add_profile_column: bool = True,
    add_metadata_columns: bool = True,
    optimize_dtypes: bool = True,
    float_precision: int = 6,
    compression: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compile batch summary CSV files from multiple profiles into a single DataFrame.
    
    For each profile, finds the latest batch summary CSV (based on timestamp in folder name)
    and combines them all into one DataFrame.
    
    Parameters:
    -----------
    config_path : str
        Path to pipeline.yml config file
    output_path : str, optional
        Path to save the compiled CSV. If None, returns DataFrame without saving.
    profiles : List[str], optional
        List of specific profiles to compile. If None, uses all profiles matching experiment_type.
    experiment_type : str, optional
        Filter profiles by experiment type (default: 'in planta'). Ignored if profiles is provided.
    version : str
        Version to look for ('v3' or 'v4', default: 'v4')
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
        Compiled DataFrame with all batch summaries combined
    """
    # Get list of profiles to process
    if profiles is None:
        if experiment_type:
            profiles = get_profiles_by_experiment(config_path, experiment_type)
        else:
            from .config_loader import list_profiles
            profiles = list_profiles(config_path, exclude_base_profiles=True)
    
    if not profiles:
        raise ValueError("No profiles found to compile")
    
    print(f"Compiling batch summaries for {len(profiles)} profiles...")
    print(f"Version: {version}, Experiment type: {experiment_type or 'all'}")
    
    compiled_data = []
    successful_profiles = []
    failed_profiles = []
    rows_per_profile = {}  # Track rows per profile for summary
    
    for i, profile_name in enumerate(profiles, 1):
        print(f"\n[{i}/{len(profiles)}] Processing profile: {profile_name}")
        
        # Find latest batch summary CSV
        csv_path = find_latest_batch_summary_csv(profile_name, config_path, version=version)
        
        if csv_path is None:
            print(f"  ✗ No batch summary found")
            failed_profiles.append(profile_name)
            continue
        
        print(f"  ✓ Found: {csv_path}")
        
        try:
            # Load the CSV
            df = pd.read_csv(csv_path)
            
            # Add profile column if requested
            if add_profile_column:
                df['Profile'] = profile_name
            
            # Add metadata columns if requested
            if add_metadata_columns:
                try:
                    # Load profile config to get metadata
                    profile_config = load_profile_config(config_path, profile_name)
                    metadata = profile_config.get('metadata', {})
                    
                    # Add each metadata field as a column
                    for key, value in metadata.items():
                        df[key] = value
                    
                    print(f"  ✓ Added metadata columns: {list(metadata.keys())}")
                except Exception as e:
                    print(f"  ⚠ Warning: Could not load metadata for profile '{profile_name}': {e}")
                    # Continue without metadata rather than failing
            
            compiled_data.append(df)
            successful_profiles.append(profile_name)
            rows_per_profile[profile_name] = len(df)
            print(f"  ✓ Loaded {len(df)} rows")
            
        except Exception as e:
            print(f"  ✗ Error loading CSV: {e}")
            failed_profiles.append(profile_name)
    
    if not compiled_data:
        raise ValueError("No batch summaries could be loaded. Check that profiles have been processed.")
    
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
    print(f"CSV files compiled: {len(successful_profiles)}")
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

