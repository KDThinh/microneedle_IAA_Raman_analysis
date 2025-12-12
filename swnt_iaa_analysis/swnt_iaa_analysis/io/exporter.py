"""Data export functions."""

import os
from pathlib import Path
from datetime import datetime
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
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"results_{timestamp}.csv"
    
    output_path = output_dir / filename
    df.to_csv(output_path, index=True)
    print(f"Results exported to: {output_path}")


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
    timestamp = datetime.now().strftime("%Y%m%d")
    
    for key, result in fft_results.items():
        if 'peaks' in result and result['peaks'] is not None:
            peaks_path = output_dir / f"fft_peaks_{key}_{timestamp}.csv"
            result['peaks'].to_csv(peaks_path, index=False)
            print(f"FFT peaks exported to: {peaks_path}")
        
        if 'complete' in result and result['complete'] is not None:
            complete_path = output_dir / f"fft_complete_{key}_{timestamp}.csv"
            result['complete'].to_csv(complete_path, index=False)
            print(f"FFT complete data exported to: {complete_path}")

