"""Fourier transform analysis functions."""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


def compute_fourier_transform(signal, time_hours, top_peaks=10):
    """
    Compute Fourier transform of a signal.
    
    Parameters:
    -----------
    signal : array-like
        Input signal
    time_hours : array-like
        Time values in hours
    top_peaks : int
        Number of top peaks to extract
    
    Returns:
    --------
    dict
        Dictionary with 'frequencies', 'magnitude', 'peaks' (DataFrame), 'complete' (DataFrame)
    """
    # Detrend signal
    signal_detrended = signal - np.mean(signal)
    
    # Calculate sampling interval
    if len(time_hours) > 1:
        sampling_interval = np.mean(np.diff(time_hours))
    else:
        sampling_interval = 1.0
    
    # Compute FFT
    fft_result = np.fft.fft(signal_detrended)
    frequencies = np.fft.fftfreq(len(time_hours), d=sampling_interval)
    
    # Create complete FFT DataFrame
    complete_ft_df = pd.DataFrame({
        'Frequency (cycles/hour)': frequencies,
        'Real': np.real(fft_result),
        'Imag': np.imag(fft_result),
        'Magnitude': np.abs(fft_result),
        'Phase': np.angle(fft_result)
    })
    
    # Extract positive frequencies
    positive_mask = frequencies > 0
    positive_freqs = frequencies[positive_mask]
    magnitude = np.abs(fft_result)[positive_mask]
    
    # Find top peaks
    if len(magnitude) > 0:
        peak_indices = np.argsort(magnitude)[-top_peaks:][::-1]
        peak_freqs = positive_freqs[peak_indices]
        peak_magnitudes = magnitude[peak_indices]
        peak_periods = 1 / peak_freqs
        
        peaks_df = pd.DataFrame({
            'Frequency (cycles/hour)': peak_freqs,
            'Period (hours)': peak_periods,
            'Magnitude': peak_magnitudes
        })
    else:
        peaks_df = pd.DataFrame()
    
    return {
        'frequencies': frequencies,
        'magnitude': np.abs(fft_result),
        'peaks': peaks_df,
        'complete': complete_ft_df
    }


def compute_diurnal_average(results_df, bin_factor=2):
    """
    Compute diurnal average from time series data.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with datetime index and signal column
    bin_factor : int
        Binning factor for hour of day
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with diurnal average statistics
    """
    if not pd.api.types.is_datetime64_any_dtype(results_df.index):
        raise ValueError("DataFrame index must be datetime")
    
    df = results_df.copy()
    datetimes = df.index
    df['Hour of Day'] = datetimes.hour + datetimes.minute / 60.0
    df['Binned Hour'] = np.round(df['Hour of Day'] * bin_factor) / bin_factor
    
    return df

