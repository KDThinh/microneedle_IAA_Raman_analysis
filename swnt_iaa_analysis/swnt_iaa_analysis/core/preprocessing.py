"""Preprocessing functions for spectral data."""

import numpy as np
from scipy.signal import savgol_filter


def apply_savgol_filter(intensities, window_size, poly_order):
    """
    Apply Savitzky-Golay smoothing filter.
    
    Parameters:
    -----------
    intensities : array-like
        Input intensity array
    window_size : int
        Window size for smoothing (must be odd)
    poly_order : int
        Polynomial order for Savitzky-Golay filter
    
    Returns:
    --------
    smoothed : array
        Smoothed intensities
    """
    # Ensure window_size is valid
    if window_size >= len(intensities):
        window_size = len(intensities) - 1 if len(intensities) % 2 == 0 else len(intensities) - 2
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        window_size = 3
    
    return savgol_filter(intensities, window_size, poly_order)

