"""Peak fitting functions for Raman spectra."""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def lorentzian(x, amplitude, center, width, offset):
    """
    Lorentzian function for peak fitting.
    
    Parameters:
    -----------
    x : array-like
        Wavenumber values
    amplitude : float
        Peak amplitude
    center : float
        Peak center position
    width : float
        Half-width at half-maximum (HWHM) of the Lorentzian
    offset : float
        Baseline offset
    
    Returns:
    --------
    y : array
        Lorentzian function values
    """
    return amplitude / (1 + ((x - center) / width)**2) + offset


def calculate_peak_area_analytical(amplitude, width, peak_type='lorentzian'):
    """
    Calculate peak area using analytical formula.
    
    Parameters:
    -----------
    amplitude : float
        Peak amplitude
    width : float
        Peak width (HWHM for Lorentzian, sigma for Gaussian)
    peak_type : str
        'lorentzian' or 'gaussian'
    
    Returns:
    --------
    area : float
        Peak area
    """
    if peak_type == 'lorentzian':
        # Lorentzian: Area = π × amplitude × HWHM
        return np.pi * amplitude * width
    elif peak_type == 'gaussian':
        # Gaussian: Area = amplitude * sigma * sqrt(2π)
        return amplitude * width * np.sqrt(2 * np.pi)
    else:
        raise ValueError(f"Unknown peak_type: {peak_type}")


def find_peak_lorentzian(wavenumbers, intensities, wavenumber_min, wavenumber_max, 
                         prominence_factor=0.1, width_min=5, width_max=50):
    """
    Find peak using Lorentzian fitting.
    
    Parameters:
    -----------
    wavenumbers : array
        Wavenumber array
    intensities : array
        Intensity array
    wavenumber_min : float
        Minimum wavenumber for search range
    wavenumber_max : float
        Maximum wavenumber for search range
    prominence_factor : float
        Factor for peak prominence detection
    width_min : float
        Minimum peak width for fitting
    width_max : float
        Maximum peak width for fitting
    
    Returns:
    --------
    dict
        Dictionary with peak information (wavenumber, intensity, amplitude, width, area, fit_params)
    """
    try:
        # Filter to range
        range_mask = (wavenumbers >= wavenumber_min) & (wavenumbers <= wavenumber_max)
        if not np.any(range_mask):
            return {
                'wavenumber': np.nan,
                'intensity': np.nan,
                'amplitude': np.nan,
                'width': np.nan,
                'area': np.nan,
                'fit_params': None
            }
        
        wavenumbers_range = wavenumbers[range_mask]
        intensities_range = intensities[range_mask]
        
        if len(intensities_range) < 3:
            return {
                'wavenumber': np.nan,
                'intensity': np.nan,
                'amplitude': np.nan,
                'width': np.nan,
                'area': np.nan,
                'fit_params': None
            }
        
        # Find peak using scipy find_peaks
        prominence = np.max(intensities_range) * prominence_factor
        peaks, properties = find_peaks(intensities_range, prominence=prominence)
        
        if len(peaks) == 0:
            # Fallback to argmax
            peak_idx = np.argmax(intensities_range)
        else:
            peak_idx = peaks[np.argmax(intensities_range[peaks])]
        
        peak_wavenumber = wavenumbers_range[peak_idx]
        peak_intensity = intensities_range[peak_idx]
        initial_offset = np.min(intensities_range)
        
        # Fit Lorentzian
        try:
            # Initial guess
            p0 = [
                peak_intensity - initial_offset,  # amplitude
                peak_wavenumber,  # center
                (wavenumber_max - wavenumber_min) / 4,  # width (HWHM)
                initial_offset  # offset
            ]
            
            # Bounds
            bounds = (
                [0, wavenumber_min, width_min, -np.inf],
                [np.inf, wavenumber_max, width_max, np.inf]
            )
            
            popt, _ = curve_fit(lorentzian, wavenumbers_range, intensities_range, 
                              p0=p0, bounds=bounds, maxfev=5000)
            
            fitted_amplitude = popt[0]
            fitted_center = popt[1]
            fitted_width = popt[2]
            
            # Calculate area analytically
            peak_area = calculate_peak_area_analytical(fitted_amplitude, fitted_width, 'lorentzian')
            
            return {
                'wavenumber': fitted_center,
                'intensity': fitted_amplitude + popt[3],
                'amplitude': fitted_amplitude,
                'width': fitted_width,
                'area': peak_area,
                'fit_params': popt
            }
        except Exception:
            # If fitting fails, return simple peak
            return {
                'wavenumber': peak_wavenumber,
                'intensity': peak_intensity,
                'amplitude': peak_intensity - initial_offset,
                'width': None,
                'area': None,
                'fit_params': None
            }
    except Exception:
        # Fallback to argmax
        range_mask = (wavenumbers >= wavenumber_min) & (wavenumbers <= wavenumber_max)
        if not np.any(range_mask):
            return {
                'wavenumber': np.nan,
                'intensity': np.nan,
                'amplitude': np.nan,
                'width': None,
                'area': None,
                'fit_params': None
            }
        wavenumbers_range = wavenumbers[range_mask]
        intensities_range = intensities[range_mask]
        peak_idx = np.argmax(intensities_range)
        return {
            'wavenumber': wavenumbers_range[peak_idx],
            'intensity': intensities_range[peak_idx],
            'amplitude': intensities_range[peak_idx] - np.min(intensities_range),
            'width': None,
            'area': None,
            'fit_params': None
        }

