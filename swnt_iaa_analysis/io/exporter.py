"""Data export functions."""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd


def export_results(df: pd.DataFrame, output_dir: Path, filename: str = None):
    """
    Export results DataFrame to CSV.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Results DataFrame to export
    output_dir : Path
        Output directory
    filename : str, optional
        Output filename (default: results_YYYYMMDD.csv)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = "processed_data.csv"
    
    output_path = output_dir / filename
    df.to_csv(output_path, index=True)


def export_fft_results(fft_results: dict, output_dir: Path):
    """
    Export FFT results to CSV files.
    
    Parameters:
    -----------
    fft_results : dict
        Dictionary of FFT results
    output_dir : Path
        Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for key, result in fft_results.items():
        if 'peaks' in result and result['peaks'] is not None:
            peaks_path = output_dir / f"fft_peaks_{key}.csv"
            result['peaks'].to_csv(peaks_path, index=False)
        
        if 'complete' in result and result['complete'] is not None:
            complete_path = output_dir / f"fft_complete_{key}.csv"
            result['complete'].to_csv(complete_path, index=False)


def load_processed_data(csv_path: Path) -> pd.DataFrame:
    """
    Load processed_data.csv and restore proper index (Datetime or Scan Number).
    Validates that required columns exist.
    
    Parameters:
    -----------
    csv_path : Path
        Path to processed_data.csv file
    
    Returns:
    --------
    pd.DataFrame
        Loaded DataFrame with proper index restored
    
    Raises:
    -------
    FileNotFoundError
        If CSV file doesn't exist
    ValueError
        If required columns are missing
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {csv_path}")
    
    # Load CSV
    df = pd.read_csv(csv_path, index_col=0)
    
    # Try to parse index as datetime first, then fall back to numeric (Scan Number)
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError):
        # If datetime parsing fails, try numeric
        try:
            df.index = pd.to_numeric(df.index)
        except (ValueError, TypeError):
            # Keep as string if both fail
            pass
    
    # Validate required columns exist
    required_cols = [
        'Normalized_Fluorescence_Intensity',
        'Normalized_Gband_Area',
        'Normalized_Raman_Peak_850_Area'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing from processed_data.csv: {missing_cols}")
    
    return df

