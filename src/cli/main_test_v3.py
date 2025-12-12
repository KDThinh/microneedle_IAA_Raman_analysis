"""
Entry point for processing v3 - continuation of v2 focusing on normalized data with ratios.
This version uses the same processing workflow as v2 but focuses on normalized data only
and includes fluorescence ratio calculations.

Workflow (same as v2):
1. Start at 250 cm-1
2. Savitzky-Golay smoothing for each scan
3. Keep original smoothed scan and normalized smoothed scan (normalized to average background value between 250 cm-1 to 1250 cm-1)
4. Lieberfit the whole spectrum for normalized smoothed scan
5. Detect the Raman peak at around 850 cm-1 and 1600 cm-1 (G-band) from the lieberfit-corrected scan
6. Perform Lorentzian fitting and calculate the AUC for each Raman peak
7. Calculate AUC fluorescence from 1250 cm-1 onward, subtracted the calculated area of G-band
8. Aggregate normalized data into batch_summary_data
9. Apply baseline correction to normalized fluorescence intensity, G-band, and Raman peak at 850 cm-1
10. Calculate fluorescence_to_g_band_ratio and fluorescence_to_raman_peak_850_ratio (before and after baseline correction)
11. Plot normalized data before/after baseline correction and ratio timeseries

Usage examples:
    # Batch processing (default skips first 500 scans)
    python main_test_v3.py --batch
    
    # Batch with baseline correction parameters
    python main_test_v3.py --batch --baseline-threshold 7.5 --baseline-window 15 --correct-smoothed
    
    # Batch with custom scan range
    python main_test_v3.py --batch --scans "100,200,300" --max-scans 10
"""
import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from scipy.signal import savgol_filter, find_peaks, medfilt
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d

# Add src to path
PROJECT_SRC_DIR = Path(__file__).resolve().parent.parent  # Go up from src/cli/ to src/
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.append(str(PROJECT_SRC_DIR))

PROJECT_ROOT = PROJECT_SRC_DIR.parent  # Go up from src/ to project root

from pipeline.config_loader import load_profile_config
from pipeline.ingestion import load_raman_dataset
from pipeline.utils import add_day_night_shading, create_dir_if_needed, lieberfit

# Import from scripts (need to add project root to path for scripts)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.fix_baseline_shifts_v2 import correct_baseline_shifts, correct_baseline_shifts_with_jump_indices
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

DEFAULT_CONFIG_PATH = str((PROJECT_ROOT / "config" / "pipeline.yml").resolve())
DEFAULT_PROFILE = "bok_choy_control_6to22_run1"


def parse_light_transition_config(config):
    """
    Parse light transition configuration from config dictionary.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary that may contain 'light_transition' key
        
    Returns:
    --------
    dict or None
        Dictionary with 'transition_datetime' (datetime object) and 'initial_light_cycle' (str),
        or None if no transition is configured
    """
    light_transition = config.get('light_transition')
    if not light_transition:
        return None
    
    transition_str = light_transition.get('transition_datetime')
    if not transition_str:
        return None
    
    # Parse datetime string (format: "YYYY-MM-DD HH:MM:SS")
    try:
        transition_datetime = datetime.strptime(transition_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Try alternative format without seconds
        try:
            transition_datetime = datetime.strptime(transition_str, "%Y-%m-%d %H:%M")
        except ValueError:
            print(f"Warning: Could not parse transition_datetime '{transition_str}'. Expected format: 'YYYY-MM-DD HH:MM:SS'")
            return None
    
    return {
        'transition_datetime': transition_datetime,
        'initial_light_cycle': light_transition.get('initial_light_cycle', '8to24')
    }


def parse_shade_transition_config(config):
    """
    Parse shade transition configuration from config dictionary.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary that may contain 'shade_transition' key
        
    Returns:
    --------
    dict or None
        Dictionary with 'transition_datetime' (datetime object), 'initial_light_quality' (str),
        'final_light_quality' (str), and optional 'ppfd' and 'r_fr_ratio',
        or None if no transition is configured
    """
    shade_transition = config.get('shade_transition')
    if not shade_transition:
        return None
    
    transition_str = shade_transition.get('transition_datetime')
    if not transition_str:
        return None
    
    # Parse datetime string (format: "YYYY-MM-DD HH:MM:SS")
    try:
        transition_datetime = datetime.strptime(transition_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Try alternative format without seconds
        try:
            transition_datetime = datetime.strptime(transition_str, "%Y-%m-%d %H:%M")
        except ValueError:
            print(f"Warning: Could not parse transition_datetime '{transition_str}'. Expected format: 'YYYY-MM-DD HH:MM:SS'")
            return None
    
    result = {
        'transition_datetime': transition_datetime,
        'initial_light_quality': shade_transition.get('initial_light_quality', 'white'),
        'final_light_quality': shade_transition.get('final_light_quality', 'shade')
    }
    
    # Optional parameters
    if 'ppfd' in shade_transition:
        result['ppfd'] = shade_transition['ppfd']
    if 'r_fr_ratio' in shade_transition:
        result['r_fr_ratio'] = shade_transition['r_fr_ratio']
    
    return result


def parse_treatment_events_config(config):
    """
    Parse treatment events configuration from config dictionary.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary that may contain 'treatment_events' key
        
    Returns:
    --------
    list or None
        List of dictionaries with event information, or None if no events are configured
        Each dictionary contains:
        - 'datetime': datetime object
        - 'event_type': str (e.g., 'drought_stress_observed')
        - 'description': str
        - 'marker_color': str (optional, default: 'red')
        - 'marker_style': str (optional, default: 'dotted')
    """
    treatment_events = config.get('treatment_events')
    if not treatment_events:
        return None
    
    if not isinstance(treatment_events, list):
        return None
    
    parsed_events = []
    for event in treatment_events:
        event_str = event.get('datetime')
        if not event_str:
            continue
        
        # Parse datetime string (format: "YYYY-MM-DD HH:MM:SS")
        try:
            event_datetime = datetime.strptime(event_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            # Try alternative format without seconds
            try:
                event_datetime = datetime.strptime(event_str, "%Y-%m-%d %H:%M")
            except ValueError:
                print(f"Warning: Could not parse event datetime '{event_str}'. Expected format: 'YYYY-MM-DD HH:MM:SS'")
                continue
        
        parsed_event = {
            'datetime': event_datetime,
            'event_type': event.get('event_type', 'unknown'),
            'description': event.get('description', ''),
            'marker_color': event.get('marker_color', 'red'),
            'marker_style': event.get('marker_style', 'dotted')
        }
        parsed_events.append(parsed_event)
    
    return parsed_events if parsed_events else None


def lorentzian(x, amplitude, center, width, offset):
    """
    Lorentzian function for peak fitting.
    
    Parameters:
    -----------
    width : float
        Half-width at half-maximum (HWHM) of the Lorentzian
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
        peak_idx = np.argmax(intensities_range)
        return {
            'wavenumber': wavenumbers_range[peak_idx],
            'intensity': intensities_range[peak_idx],
            'amplitude': intensities_range[peak_idx] - np.min(intensities_range),
            'width': None,
            'area': None,
            'fit_params': None
        }


def process_scan_v2(scan_number, raman_df, config, 
                   cached_wavenumbers_full=None, cached_wavenumber_filter=None, cached_wavenumbers=None,
                   verbose=False):
    """
    Process a single scan using the revised workflow v2 (same as v2).
    This function processes both raw and normalized data, but v3 will only use normalized data.
    
    Parameters:
    -----------
    scan_number : int
        Scan number to process
    raman_df : pandas.DataFrame
        Raman dataset
    config : dict
        Configuration dictionary
    cached_wavenumbers_full : array, optional
        Pre-computed full wavenumber array
    cached_wavenumber_filter : array, optional
        Pre-computed wavenumber filter mask
    cached_wavenumbers : array, optional
        Pre-computed filtered wavenumber array
    verbose : bool, optional
        Whether to print verbose output
    
    Returns:
    --------
    summary : dict or None
        Dictionary with processing results
    """
    # Extract scan data
    scan_data = raman_df[raman_df['Scan Number'] == scan_number]
    if scan_data.empty:
        if verbose:
            print(f"ERROR: Scan {scan_number} not found!")
        return None
    
    datetime_val = scan_data.index[0]
    seconds = scan_data['Seconds'].iloc[0]
    
    # Get wavenumber arrays
    if cached_wavenumbers_full is not None and cached_wavenumber_filter is not None and cached_wavenumbers is not None:
        wavenumbers_full = cached_wavenumbers_full
        wavenumber_filter = cached_wavenumber_filter
        wavenumbers = cached_wavenumbers
    else:
        wavenumbers_full = np.array([float(col) for col in raman_df.columns if col not in ['Scan Number', 'Seconds']])
        wavenumber_filter = wavenumbers_full >= 250
        wavenumbers = wavenumbers_full[wavenumber_filter]
    
    intensities_raw_full = scan_data.iloc[0, 2:].values
    intensities_raw = intensities_raw_full[wavenumber_filter]
    
    # Step 2: Savitzky-Golay smoothing
    window_size = config.get('window_size', 25)
    poly_order_sg = config.get('poly_order_sg', 2)
    if window_size >= len(intensities_raw):
        window_size = len(intensities_raw) - 1 if len(intensities_raw) % 2 == 0 else len(intensities_raw) - 2
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        window_size = 3
    
    intensities_smooth = savgol_filter(intensities_raw, window_size, poly_order_sg)
    
    # Step 3: Calculate normalization factor and create normalized smoothed scan
    baseline_bg_mask = (wavenumbers >= 250) & (wavenumbers <= 1250)
    if np.any(baseline_bg_mask):
        bg_intensity_avg_250_1250 = np.mean(intensities_smooth[baseline_bg_mask])
    else:
        bg_intensity_avg_250_1250 = np.mean(intensities_smooth)
    
    intensities_normalized_smooth = intensities_smooth / bg_intensity_avg_250_1250 if bg_intensity_avg_250_1250 > 0 else intensities_smooth
    
    # Step 4: Lieberfit the whole spectrum for normalized smoothed
    # Get poly_order and tot_iter from config or use defaults
    # Note: Command-line overrides are handled in main() and passed via config
    poly_order = config.get('poly_order', 5)
    tot_iter = config.get('tot_iter', 100)
    
    corrected_normalized, baseline_normalized = lieberfit(intensities_normalized_smooth, order=poly_order, tot_iter=tot_iter)
    
    # Step 5 & 6: Detect Raman peaks at ~850 cm-1 and ~1600 cm-1 (G-band) from lieberfit-corrected scan
    raman_peak_850_normalized = find_peak_lorentzian(wavenumbers, corrected_normalized, 800, 900, width_min=5, width_max=50)
    gband_peak_1600_normalized = find_peak_lorentzian(wavenumbers, corrected_normalized, 1550, 1650, width_min=5, width_max=50)
    
    # Step 7: Calculate AUC fluorescence from 1250 cm-1 onward from normalized smoothed scan (not lieberfit)
    # Subtract G-band area from Lorentzian fitting
    fluo_mask = wavenumbers >= 1250
    
    if np.any(fluo_mask):
        fluo_auc_normalized = trapezoid(intensities_normalized_smooth[fluo_mask], x=wavenumbers[fluo_mask])
        # Fix: Properly handle None and NaN values from peak fitting
        # The .get('area', 0) doesn't work when 'area' key exists but is None
        gband_area = gband_peak_1600_normalized.get('area') if gband_peak_1600_normalized else None
        if gband_area is None or (isinstance(gband_area, (int, float)) and np.isnan(gband_area)):
            gband_area_normalized = 0
        else:
            gband_area_normalized = float(gband_area)
        fluorescence_intensity_normalized = fluo_auc_normalized - gband_area_normalized
    else:
        fluorescence_intensity_normalized = np.nan
    
    # Build summary dictionary (same as v2, but v3 will only extract normalized columns)
    summary = {
        'scan_number': scan_number,
        'datetime': datetime_val,
        'seconds': seconds,
        
        # Average background intensity (250-1250 cm-1)
        'average_background_intensity': bg_intensity_avg_250_1250,
        
        # Normalized smoothed data
        'normalized_background_intensity': 1.0 if bg_intensity_avg_250_1250 > 0 else np.nan,
        'normalized_fluorescence_intensity': fluorescence_intensity_normalized,
        'normalized_gband_intensity': gband_peak_1600_normalized.get('intensity', np.nan) if gband_peak_1600_normalized else np.nan,
        'normalized_gband_area': gband_peak_1600_normalized.get('area', np.nan) if gband_peak_1600_normalized else np.nan,
        'normalized_gband_wavenumber': gband_peak_1600_normalized.get('wavenumber', np.nan) if gband_peak_1600_normalized else np.nan,
        'normalized_raman_peak_850_intensity': raman_peak_850_normalized.get('intensity', np.nan) if raman_peak_850_normalized else np.nan,
        'normalized_raman_peak_850_area': raman_peak_850_normalized.get('area', np.nan) if raman_peak_850_normalized else np.nan,
        'normalized_raman_peak_850_wavenumber': raman_peak_850_normalized.get('wavenumber', np.nan) if raman_peak_850_normalized else np.nan,
        
        # Store spectrum data for plotting (only for first scan)
        'wavenumbers': wavenumbers,
        'intensities_normalized_smooth': intensities_normalized_smooth,
        'baseline_normalized': baseline_normalized,
        'corrected_normalized': corrected_normalized,
        'raman_peak_850_normalized': raman_peak_850_normalized,
        'gband_peak_1600_normalized': gband_peak_1600_normalized,
    }
    
    return summary


def aggregate_summaries_to_dataframe_v3(summaries, raman_df, x_axis_type='datetime'):
    """
    Aggregate summaries from batch processing into a DataFrame (normalized data only).
    
    Parameters:
    -----------
    summaries : list of dict
        List of summary dictionaries from process_scan_v2()
    raman_df : pandas.DataFrame
        Original Raman dataframe with datetime index
    x_axis_type : str, optional
        Type of x-axis: 'datetime' or 'scan_number' (default: 'datetime')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with datetime or scan_number index and normalized metrics columns
    """
    if not summaries:
        return pd.DataFrame()
    
    # Get first datetime for each scan number (using explicit loop to avoid FutureWarning)
    scan_to_datetime = {}
    for scan_num, group in raman_df.groupby('Scan Number'):
        scan_to_datetime[scan_num] = group.index[0]
    
    rows = []
    for summary in summaries:
        if summary is None:
            continue
        
        scan_number = summary.get('scan_number')
        datetime_val = summary.get('datetime')
        
        row = {
            'Scan Number': scan_number,
            'Seconds': summary.get('seconds'),
            'Datetime': pd.to_datetime(datetime_val) if datetime_val else scan_to_datetime.get(scan_number),
            
            # Average background intensity
            'Average_Background_Intensity_250_1250_cm-1': summary.get('average_background_intensity', np.nan),
            
            # Normalized smoothed data only
            'Normalized_Background_Intensity': summary.get('normalized_background_intensity', np.nan),
            'Normalized_Fluorescence_Intensity': summary.get('normalized_fluorescence_intensity', np.nan),
            'Normalized_Gband_Intensity': summary.get('normalized_gband_intensity', np.nan),
            'Normalized_Gband_Area': summary.get('normalized_gband_area', np.nan),
            'Normalized_Gband_Wavenumber': summary.get('normalized_gband_wavenumber', np.nan),
            'Normalized_Raman_Peak_850_Intensity': summary.get('normalized_raman_peak_850_intensity', np.nan),
            'Normalized_Raman_Peak_850_Area': summary.get('normalized_raman_peak_850_area', np.nan),
            'Normalized_Raman_Peak_850_Wavenumber': summary.get('normalized_raman_peak_850_wavenumber', np.nan),
        }
        
        if row['Datetime'] is None:
            continue
        
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Set index based on x_axis_type
    if x_axis_type == 'scan_number':
        df.set_index('Scan Number', inplace=True)
        df.sort_index(inplace=True)
    else:  # datetime (default)
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True)
    
    return df


def remove_spikes_hampel(signal, window_size=5, threshold=3.0, min_spike_length=1, max_spike_length=5):
    """
    Remove spikes (outliers) from time series using Hampel filter (median-based outlier detection).
    
    Parameters:
    -----------
    signal : np.array
        Input signal (may contain NaN values)
    window_size : int
        Half-window size for median calculation (default: 5)
        Total window = 2 * window_size + 1
    threshold : float
        Threshold in units of MAD (Median Absolute Deviation) for outlier detection (default: 3.0)
    min_spike_length : int
        Minimum consecutive points to consider as spike (default: 1)
    max_spike_length : int
        Maximum consecutive points to replace (default: 5)
    
    Returns:
    --------
    cleaned_signal : np.array
        Signal with spikes replaced by median-filtered values
    spike_mask : np.array
        Boolean mask indicating which points were replaced (True = spike)
    """
    cleaned_signal = signal.copy().astype(float)
    spike_mask = np.zeros(len(signal), dtype=bool)
    
    # Handle NaN values
    valid_mask = ~np.isnan(signal)
    if np.sum(valid_mask) < window_size * 2 + 1:
        # Not enough data for spike detection
        return cleaned_signal, spike_mask
    
    signal_valid = signal[valid_mask]
    indices_valid = np.where(valid_mask)[0]
    
    # Calculate median and MAD for each point using rolling window
    n = len(signal_valid)
    median_values = np.zeros(n)
    mad_values = np.zeros(n)
    
    for i in range(n):
        # Define window around point i
        start_idx = max(0, i - window_size)
        end_idx = min(n, i + window_size + 1)
        
        window_data = signal_valid[start_idx:end_idx]
        
        # Calculate median
        median_values[i] = np.median(window_data)
        
        # Calculate MAD (Median Absolute Deviation)
        deviations = np.abs(window_data - median_values[i])
        mad_values[i] = np.median(deviations)
    
    # Avoid division by zero
    mad_values[mad_values == 0] = np.finfo(float).eps
    
    # Detect outliers: points where |signal - median| > threshold * MAD
    deviations_from_median = np.abs(signal_valid - median_values)
    outlier_mask = deviations_from_median > (threshold * mad_values)
    
    # Find consecutive outlier groups (spikes)
    spike_groups = []
    i = 0
    while i < n:
        if outlier_mask[i]:
            # Start of potential spike
            start = i
            while i < n and outlier_mask[i]:
                i += 1
            end = i
            
            spike_length = end - start
            # Only consider spikes within the specified length range
            if min_spike_length <= spike_length <= max_spike_length:
                spike_groups.append((start, end))
        else:
            i += 1
    
    # Replace spikes with median-filtered values
    if len(spike_groups) > 0:
        # Apply median filter to get replacement values
        # Use larger kernel for better replacement values
        replacement_kernel = min(2 * window_size + 1, n)
        if replacement_kernel % 2 == 0:
            replacement_kernel += 1
        if replacement_kernel >= 3:
            median_filtered = medfilt(signal_valid, kernel_size=replacement_kernel)
        else:
            median_filtered = signal_valid.copy()
        
        for start, end in spike_groups:
            # Replace spike with median-filtered values
            # Use median of surrounding points if available
            if start > 0 and end < n:
                replacement_value = np.median([median_filtered[start-1], median_filtered[end] if end < n else median_filtered[start-1]])
            elif start > 0:
                replacement_value = median_filtered[start-1]
            elif end < n:
                replacement_value = median_filtered[end]
            else:
                replacement_value = median_filtered[start]
            
            cleaned_signal[indices_valid[start:end]] = replacement_value
            spike_mask[indices_valid[start:end]] = True
    
    return cleaned_signal, spike_mask


def apply_baseline_correction_v3(df, columns_to_correct=None, 
                                threshold_multiplier=5.0, window_size=5,
                                smooth_first=True, smooth_window=11, smooth_poly_order=2,
                                correct_smoothed=False, remove_spikes=False, spike_window=5, spike_threshold=3.0,
                                detect_cumulative_jumps=True, cumulative_window=5):
    """
    Apply baseline correction to normalized data only (using fix_baseline_shifts_v2.py approach).
    
    This function performs two-stage jump detection:
    1. Individual jump detection: Detects sudden step discontinuities where adjacent point differences exceed threshold
    2. Cumulative jump detection (optional): Detects gradual jumps over N consecutive points where 
       cumulative change exceeds threshold, even if individual steps are small
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with normalized time series data
    columns_to_correct : list, optional
        List of column names to correct. If None, corrects normalized columns only.
    threshold_multiplier : float, optional
        MAD threshold multiplier for jump detection (default: 5.0)
    window_size : int, optional
        Number of points to average before/after jump for offset calculation (default: 5)
    smooth_first : bool, optional
        If True, smooth signal before jump detection (default: True)
    smooth_window : int, optional
        Window size for Savitzky-Golay smoothing (must be odd, default: 11)
    smooth_poly_order : int, optional
        Polynomial order for Savitzky-Golay smoothing (default: 2)
    correct_smoothed : bool, optional
        If True, apply correction to smoothed signal (offset calculated from smoothed).
        If False, apply correction to original signal (offset calculated from original) (default: False)
    remove_spikes : bool, optional
        If True, remove spikes before smoothing (default: False)
    spike_window : int, optional
        Half-window size for spike detection (default: 5)
    spike_threshold : float, optional
        Threshold in units of MAD for spike detection (default: 3.0)
    detect_cumulative_jumps : bool, optional
        If True, detect gradual jumps over multiple consecutive points (default: True).
        This catches jumps that occur over 4-5 points where each individual step is small
        but the cumulative change is significant.
    cumulative_window : int, optional
        Window size for detecting cumulative changes (default: 5).
        Detects jumps where total change over N consecutive points exceeds threshold.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with corrected columns added
    dict
        Dictionary mapping column names to jump information
    """
    if columns_to_correct is None:
        columns_to_correct = [
            'Normalized_Fluorescence_Intensity',
            'Normalized_Gband_Area',
            'Normalized_Raman_Peak_850_Area'
        ]
    
    corrected_df = df.copy()
    all_jump_info = {}
    
    # Step 1: Process fluorescence signal first to detect jump points
    # These jump points will be used as reference for G-band and Raman peak
    fluorescence_column = 'Normalized_Fluorescence_Intensity'
    
    # Store jump indices from fluorescence signal
    fluorescence_jump_indices = None
    
    if fluorescence_column in df.columns:
        # Convert to numeric, coercing errors (including None) to NaN
        signal = pd.to_numeric(df[fluorescence_column], errors='coerce').values
        mask = ~np.isnan(signal)
        
        if np.sum(mask) >= 3:
            # Pre-filter NaN values (like optimize_baseline_shift_parameters.py)
            signal_valid = signal[mask]
            valid_indices = np.where(mask)[0]  # Original DataFrame indices for valid points
            
            # Detect jumps in fluorescence signal (using clean array)
            corrected_valid, jump_indices_valid, jump_info, smoothed_valid = correct_baseline_shifts(
                signal_valid, threshold_multiplier=threshold_multiplier, window_size=window_size,
                smooth_first=smooth_first, smooth_window=smooth_window, 
                smooth_poly_order=smooth_poly_order, correct_smoothed=correct_smoothed,
                detect_cumulative_jumps=detect_cumulative_jumps, cumulative_window=cumulative_window
            )
            
            # Map jump indices back to original DataFrame indices
            jump_indices = valid_indices[jump_indices_valid] if len(jump_indices_valid) > 0 else np.array([], dtype=int)
            
            # Map jump_info indices back to original DataFrame indices
            for info in jump_info:
                if 'index' in info:
                    info['index'] = valid_indices[info['index']]
            
            # Store jump indices for use with related signals
            fluorescence_jump_indices = jump_indices
            
            # Map corrected values back to full array (preserving NaN positions)
            corrected = np.full_like(signal, np.nan)
            corrected[mask] = corrected_valid
            
            smoothed = np.full_like(signal, np.nan)
            smoothed[mask] = smoothed_valid
            
            # Store corrected fluorescence signal
            corrected_col_name = f'{fluorescence_column}_BaselineCorrected'
            corrected_df[corrected_col_name] = corrected
            
            # If correcting smoothed data, also store the smoothed signal
            if correct_smoothed:
                smoothed_col_name = f'{fluorescence_column}_Smoothed'
                corrected_df[smoothed_col_name] = smoothed
            
            # Store jump information
            all_jump_info[fluorescence_column] = {
                'jump_indices': jump_indices,
                'jump_info': jump_info,
                'n_jumps': len(jump_indices),
                'correct_smoothed': correct_smoothed,
                'is_reference': True  # Mark as reference signal
            }
    
    # Step 2: Process G-band and Raman peak signals using fluorescence jump points
    for col in columns_to_correct:
        if col not in df.columns:
            continue
        
        # Skip fluorescence column (already processed)
        if col == fluorescence_column:
            continue
        
        # Convert to numeric, coercing errors (including None) to NaN
        signal = pd.to_numeric(df[col], errors='coerce').values
        mask = ~np.isnan(signal)
        
        if np.sum(mask) < 3:
            continue
        
        # Remove spikes before smoothing if requested
        if remove_spikes:
            signal, spike_mask = remove_spikes_hampel(signal, window_size=spike_window, threshold=spike_threshold)
            n_spikes = np.sum(spike_mask)
            if n_spikes > 0:
                print(f"  Removed {n_spikes} spike points from {col}")
        
        # Recalculate mask after spike removal (spikes are replaced with NaN)
        mask = ~np.isnan(signal)
        
        if np.sum(mask) < 3:
            continue
        
        # Pre-filter NaN values (like optimize_baseline_shift_parameters.py)
        signal_valid = signal[mask]
        valid_indices = np.where(mask)[0]  # Original DataFrame indices for valid points
        
        # Use fluorescence jump indices if available
        if fluorescence_jump_indices is not None and len(fluorescence_jump_indices) > 0:
            # Map fluorescence jump indices to valid signal indices
            # Find which fluorescence jump indices correspond to valid points in this signal
            fluorescence_jumps_in_valid = []
            for fluo_jump_idx in fluorescence_jump_indices:
                if fluo_jump_idx < len(mask) and mask[fluo_jump_idx]:
                    # Find position in valid_indices array
                    valid_pos = np.where(valid_indices == fluo_jump_idx)[0]
                    if len(valid_pos) > 0:
                        fluorescence_jumps_in_valid.append(valid_pos[0])
            
            if len(fluorescence_jumps_in_valid) > 0:
                fluorescence_jumps_valid = np.array(fluorescence_jumps_in_valid)
                # Use jump points from fluorescence signal (mapped to valid indices)
                corrected_valid, jump_info, smoothed_valid = correct_baseline_shifts_with_jump_indices(
                    signal_valid, fluorescence_jumps_valid, window_size=window_size,
                    smooth_first=smooth_first, smooth_window=smooth_window,
                    smooth_poly_order=smooth_poly_order, correct_smoothed=correct_smoothed
                )
                jump_indices = fluorescence_jump_indices  # Keep original DataFrame indices
                use_fluorescence_jumps = True
            else:
                # No fluorescence jumps correspond to valid points, use signal-specific detection
                corrected_valid, jump_indices_valid, jump_info, smoothed_valid = correct_baseline_shifts(
                    signal_valid, threshold_multiplier=threshold_multiplier, window_size=window_size,
                    smooth_first=smooth_first, smooth_window=smooth_window, 
                    smooth_poly_order=smooth_poly_order, correct_smoothed=correct_smoothed,
                    detect_cumulative_jumps=detect_cumulative_jumps, cumulative_window=cumulative_window
                )
                jump_indices = valid_indices[jump_indices_valid] if len(jump_indices_valid) > 0 else np.array([], dtype=int)
                use_fluorescence_jumps = False
        else:
            # Apply baseline correction with signal-specific detection
            corrected_valid, jump_indices_valid, jump_info, smoothed_valid = correct_baseline_shifts(
                signal_valid, threshold_multiplier=threshold_multiplier, window_size=window_size,
                smooth_first=smooth_first, smooth_window=smooth_window, 
                smooth_poly_order=smooth_poly_order, correct_smoothed=correct_smoothed,
                detect_cumulative_jumps=detect_cumulative_jumps, cumulative_window=cumulative_window
            )
            jump_indices = valid_indices[jump_indices_valid] if len(jump_indices_valid) > 0 else np.array([], dtype=int)
            use_fluorescence_jumps = False
        
        # Map jump_info indices back to original DataFrame indices
        for info in jump_info:
            if 'index' in info:
                # Find original index
                if use_fluorescence_jumps:
                    info['index'] = info['index']  # Already in original indices
                else:
                    info['index'] = valid_indices[info['index']]
        
        # Map corrected values back to full array (preserving NaN positions)
        corrected = np.full_like(signal, np.nan)
        corrected[mask] = corrected_valid
        
        smoothed = np.full_like(signal, np.nan)
        smoothed[mask] = smoothed_valid
        
        corrected_col_name = f'{col}_BaselineCorrected'
        corrected_df[corrected_col_name] = corrected
        
        # If correcting smoothed data, also store the smoothed signal
        if correct_smoothed:
            smoothed_col_name = f'{col}_Smoothed'
            corrected_df[smoothed_col_name] = smoothed
        
        # Store jump information
        all_jump_info[col] = {
            'jump_indices': jump_indices,
            'jump_info': jump_info,
            'n_jumps': len(jump_indices),
            'used_fluorescence_jumps': use_fluorescence_jumps,
            'reference_fluorescence_column': fluorescence_column if use_fluorescence_jumps else None,
            'correct_smoothed': correct_smoothed,
            'is_reference': False
        }
    
    return corrected_df, all_jump_info


def calculate_ratios(df):
    """
    Calculate fluorescence ratios from normalized data (before and after baseline correction).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with normalized data and optionally baseline-corrected data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with ratio columns added
    """
    df = df.copy()
    
    # Calculate ratios from original normalized data
    if 'Normalized_Fluorescence_Intensity' in df.columns and 'Normalized_Gband_Area' in df.columns:
        df['Fluorescence_to_Gband_Ratio'] = df['Normalized_Fluorescence_Intensity'] / df['Normalized_Gband_Area']
        df['Fluorescence_to_Gband_Ratio'] = df['Fluorescence_to_Gband_Ratio'].replace([np.inf, -np.inf], np.nan)
    
    if 'Normalized_Fluorescence_Intensity' in df.columns and 'Normalized_Raman_Peak_850_Area' in df.columns:
        df['Fluorescence_to_Raman_Peak_850_Ratio'] = df['Normalized_Fluorescence_Intensity'] / df['Normalized_Raman_Peak_850_Area']
        df['Fluorescence_to_Raman_Peak_850_Ratio'] = df['Fluorescence_to_Raman_Peak_850_Ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate ratios from baseline-corrected data if available
    if 'Normalized_Fluorescence_Intensity_BaselineCorrected' in df.columns and 'Normalized_Gband_Area_BaselineCorrected' in df.columns:
        df['Fluorescence_to_Gband_Ratio_BaselineCorrected'] = df['Normalized_Fluorescence_Intensity_BaselineCorrected'] / df['Normalized_Gband_Area_BaselineCorrected']
        df['Fluorescence_to_Gband_Ratio_BaselineCorrected'] = df['Fluorescence_to_Gband_Ratio_BaselineCorrected'].replace([np.inf, -np.inf], np.nan)
    
    if 'Normalized_Fluorescence_Intensity_BaselineCorrected' in df.columns and 'Normalized_Raman_Peak_850_Area_BaselineCorrected' in df.columns:
        df['Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected'] = df['Normalized_Fluorescence_Intensity_BaselineCorrected'] / df['Normalized_Raman_Peak_850_Area_BaselineCorrected']
        df['Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected'] = df['Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected'].replace([np.inf, -np.inf], np.nan)
    
    return df


def plot_normalized_baseline_correction_comparison(df, output_dir, config, light_cycle=None, jump_info_dict=None):
    """
    Create combined plots showing normalized fluorescence intensity, G-band, and Raman peak 
    before and after baseline correction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with normalized data and corrected columns
    output_dir : Path
        Output directory
    config : dict
        Configuration dictionary
    light_cycle : str, optional
        Light cycle for shading
    jump_info_dict : dict, optional
        Dictionary mapping column names to jump information
    """
    if light_cycle is None:
        light_cycle = config.get('light_cycle', 'Constant')
    
    # Parse transition and event configs if available
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    output_dir_str = str(output_dir)
    create_dir_if_needed(output_dir_str)
    
    # Normalized data plots
    fig_norm, axes_norm = plt.subplots(3, 1, figsize=(14, 12))
    
    # Fluorescence
    if 'Normalized_Fluorescence_Intensity' in df.columns:
        # Determine what to plot as "Before Correction"
        before_col = 'Normalized_Fluorescence_Intensity_Smoothed' if ('Normalized_Fluorescence_Intensity_Smoothed' in df.columns and 
                                                               jump_info_dict and 
                                                               'Normalized_Fluorescence_Intensity' in jump_info_dict and
                                                               jump_info_dict['Normalized_Fluorescence_Intensity'].get('correct_smoothed', False)) else 'Normalized_Fluorescence_Intensity'
        
        if before_col in df.columns:
            before_label = 'Before Correction (Smoothed)' if before_col == 'Normalized_Fluorescence_Intensity_Smoothed' else 'Before Correction'
            axes_norm[0].plot(df.index, df[before_col], 'o-', 
                           label=before_label, alpha=0.7, markersize=3)
        
        if 'Normalized_Fluorescence_Intensity_BaselineCorrected' in df.columns:
            after_label = 'After Correction (Corrected Smoothed)' if (jump_info_dict and 
                                                                      'Normalized_Fluorescence_Intensity' in jump_info_dict and
                                                                      jump_info_dict['Normalized_Fluorescence_Intensity'].get('correct_smoothed', False)) else 'After Correction'
            axes_norm[0].plot(df.index, df['Normalized_Fluorescence_Intensity_BaselineCorrected'], 's-', 
                           label=after_label, alpha=0.7, markersize=3)
        
        # Mark jump points if available
        if jump_info_dict and 'Normalized_Fluorescence_Intensity' in jump_info_dict:
            jump_info = jump_info_dict['Normalized_Fluorescence_Intensity']['jump_info']
            if jump_info:
                jump_indices = [info['index'] for info in jump_info]
                jump_x = df.index[jump_indices]
                jump_y_before = df[before_col].iloc[jump_indices].values if before_col in df.columns else df['Normalized_Fluorescence_Intensity'].iloc[jump_indices].values
                jump_y_after = df['Normalized_Fluorescence_Intensity_BaselineCorrected'].iloc[jump_indices].values if 'Normalized_Fluorescence_Intensity_BaselineCorrected' in df.columns else jump_y_before
                axes_norm[0].scatter(jump_x, jump_y_before, color='red', s=150, 
                                  zorder=5, marker='x', linewidths=2, label=f'Jump points (n={len(jump_info)})')
                axes_norm[0].scatter(jump_x, jump_y_after, color='darkred', s=150, 
                                  zorder=5, marker='+', linewidths=2, alpha=0.7)
        
        axes_norm[0].set_ylabel('Normalized Fluorescence Intensity', fontsize=12)
        axes_norm[0].set_title('Normalized - Fluorescence Intensity', fontsize=13, fontweight='bold')
        axes_norm[0].legend()
        axes_norm[0].grid(True, alpha=0.3)
    
    # G-band (using Area)
    if 'Normalized_Gband_Area' in df.columns:
        # Determine what to plot as "Before Correction" (same logic as fluorescence)
        before_col = 'Normalized_Gband_Area_Smoothed' if ('Normalized_Gband_Area_Smoothed' in df.columns and 
                                                          jump_info_dict and 
                                                          'Normalized_Gband_Area' in jump_info_dict and
                                                          jump_info_dict['Normalized_Gband_Area'].get('correct_smoothed', False)) else 'Normalized_Gband_Area'
        
        if before_col in df.columns:
            before_label = 'Before Correction (Smoothed)' if before_col == 'Normalized_Gband_Area_Smoothed' else 'Before Correction'
            axes_norm[1].plot(df.index, df[before_col], 'o-', 
                           label=before_label, alpha=0.7, markersize=3)
        
        if 'Normalized_Gband_Area_BaselineCorrected' in df.columns:
            after_label = 'After Correction (Corrected Smoothed)' if (jump_info_dict and 
                                                                        'Normalized_Gband_Area' in jump_info_dict and
                                                                        jump_info_dict['Normalized_Gband_Area'].get('correct_smoothed', False)) else 'After Correction'
            axes_norm[1].plot(df.index, df['Normalized_Gband_Area_BaselineCorrected'], 's-', 
                           label=after_label, alpha=0.7, markersize=3)
        
        # Mark jump points if available
        if jump_info_dict and 'Normalized_Gband_Area' in jump_info_dict:
            jump_info = jump_info_dict['Normalized_Gband_Area']['jump_info']
            if jump_info:
                jump_indices = [info['index'] for info in jump_info]
                jump_x = df.index[jump_indices]
                jump_y_before = df[before_col].iloc[jump_indices].values if before_col in df.columns else df['Normalized_Gband_Area'].iloc[jump_indices].values
                jump_y_after = df['Normalized_Gband_Area_BaselineCorrected'].iloc[jump_indices].values if 'Normalized_Gband_Area_BaselineCorrected' in df.columns else jump_y_before
                axes_norm[1].scatter(jump_x, jump_y_before, color='red', s=150, 
                                  zorder=5, marker='x', linewidths=2, label=f'Jump points (n={len(jump_info)})')
                axes_norm[1].scatter(jump_x, jump_y_after, color='darkred', s=150, 
                                  zorder=5, marker='+', linewidths=2, alpha=0.7)
        
        axes_norm[1].set_ylabel('G-band Area (Lorentzian)', fontsize=12)
        axes_norm[1].set_title('Normalized - G-band Area', fontsize=13, fontweight='bold')
        axes_norm[1].legend()
        axes_norm[1].grid(True, alpha=0.3)
    
    # Raman peak 850 (using Area)
    if 'Normalized_Raman_Peak_850_Area' in df.columns:
        before_col = 'Normalized_Raman_Peak_850_Area_Smoothed' if ('Normalized_Raman_Peak_850_Area_Smoothed' in df.columns and 
                                                                    jump_info_dict and 
                                                                    'Normalized_Raman_Peak_850_Area' in jump_info_dict and
                                                                    jump_info_dict['Normalized_Raman_Peak_850_Area'].get('correct_smoothed', False)) else 'Normalized_Raman_Peak_850_Area'
        
        if before_col in df.columns:
            before_label = 'Before Correction (Smoothed)' if before_col == 'Normalized_Raman_Peak_850_Area_Smoothed' else 'Before Correction'
            axes_norm[2].plot(df.index, df[before_col], 'o-', 
                           label=before_label, alpha=0.7, markersize=3)
        
        if 'Normalized_Raman_Peak_850_Area_BaselineCorrected' in df.columns:
            after_label = 'After Correction (Corrected Smoothed)' if (jump_info_dict and 
                                                                      'Normalized_Raman_Peak_850_Area' in jump_info_dict and
                                                                      jump_info_dict['Normalized_Raman_Peak_850_Area'].get('correct_smoothed', False)) else 'After Correction'
            axes_norm[2].plot(df.index, df['Normalized_Raman_Peak_850_Area_BaselineCorrected'], 's-', 
                           label=after_label, alpha=0.7, markersize=3)
        
        # Mark jump points if available
        if jump_info_dict and 'Normalized_Raman_Peak_850_Area' in jump_info_dict:
            jump_info = jump_info_dict['Normalized_Raman_Peak_850_Area']['jump_info']
            if jump_info:
                jump_indices = [info['index'] for info in jump_info]
                jump_x = df.index[jump_indices]
                jump_y_before = df[before_col].iloc[jump_indices].values if before_col in df.columns else df['Normalized_Raman_Peak_850_Area'].iloc[jump_indices].values
                jump_y_after = df['Normalized_Raman_Peak_850_Area_BaselineCorrected'].iloc[jump_indices].values if 'Normalized_Raman_Peak_850_Area_BaselineCorrected' in df.columns else jump_y_before
                axes_norm[2].scatter(jump_x, jump_y_before, color='red', s=150, 
                                  zorder=5, marker='x', linewidths=2, label=f'Jump points (n={len(jump_info)})')
                axes_norm[2].scatter(jump_x, jump_y_after, color='darkred', s=150, 
                                  zorder=5, marker='+', linewidths=2, alpha=0.7)
        
        axes_norm[2].set_ylabel('Raman Peak 850 Area (Lorentzian)', fontsize=12)
        axes_norm[2].set_title('Normalized - Raman Peak 850 Area', fontsize=13, fontweight='bold')
        axes_norm[2].legend()
        axes_norm[2].grid(True, alpha=0.3)
    
    # Set x-axis formatting
    if x_axis_type == 'datetime':
        for ax in axes_norm:
            add_day_night_shading(ax, df.index.min(), df.index.max(), light_cycle=light_cycle, light_transition=light_transition)
            # Add vertical line to mark light transition if configured
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    ax.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
            # Add vertical line to mark shade transition if configured
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    label_text = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
            # Add markers for treatment events if configured
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if df.index.min() <= event_time <= df.index.max():
                        ax.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                 linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            ax.set_xlabel('Date time (MM-DD HH)', fontsize=12)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
        fig_norm.autofmt_xdate()
    else:  # scan_number
        for ax in axes_norm:
            ax.set_xlabel('Scan Number', fontsize=12)
    
    plt.tight_layout()
    save_path_norm = Path(output_dir_str) / f'norm_baseline_comp_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(str(save_path_norm), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Normalized baseline correction comparison plot saved to: {save_path_norm}")


def plot_normalized_smoothing_comparison(df, output_dir, config, light_cycle=None, jump_info_dict=None):
    """
    Create combined plots showing normalized fluorescence intensity, G-band, and Raman peak 
    before and after smoothing (same layout as norm_baseline_comp).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with normalized data and optionally smoothed columns
    output_dir : Path
        Output directory
    config : dict
        Configuration dictionary
    light_cycle : str, optional
        Light cycle for shading
    jump_info_dict : dict, optional
        Dictionary mapping column names to jump information (used to check if smoothing was applied)
    """
    if light_cycle is None:
        light_cycle = config.get('light_cycle', 'Constant')
    
    # Parse transition and event configs if available
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    output_dir_str = str(output_dir)
    create_dir_if_needed(output_dir_str)
    
    # Get smoothing parameters (same as used in baseline correction)
    smooth_window = config.get('baseline_correction_smooth_window', 11)
    smooth_poly_order = config.get('baseline_correction_smooth_poly_order', 2)
    
    # Normalized data plots (same layout as norm_baseline_comp)
    fig_norm, axes_norm = plt.subplots(3, 1, figsize=(14, 12))
    
    # Helper function to get smoothed data (either from stored column or compute on the fly)
    def get_smoothed_signal(original_col, smoothed_col, signal_values):
        """Get smoothed signal, either from stored column or compute on the fly."""
        if smoothed_col in df.columns:
            return df[smoothed_col].values
        else:
            # Apply smoothing on the fly using Savitzky-Golay
            mask = ~np.isnan(signal_values)
            if np.sum(mask) >= smooth_window:
                smoothed = signal_values.copy()
                # Ensure window_size is odd and valid
                window_size = smooth_window if smooth_window % 2 == 1 else smooth_window - 1
                if window_size < 3:
                    window_size = 3
                if window_size >= len(signal_values[mask]):
                    window_size = len(signal_values[mask]) - 1 if len(signal_values[mask]) % 2 == 0 else len(signal_values[mask]) - 2
                if window_size < 3:
                    return signal_values  # Can't smooth, return original
                smoothed[mask] = savgol_filter(signal_values[mask], window_size, smooth_poly_order)
                return smoothed
            else:
                return signal_values
    
    # Fluorescence Intensity
    if 'Normalized_Fluorescence_Intensity' in df.columns:
        original_signal = df['Normalized_Fluorescence_Intensity'].values
        smoothed_signal = get_smoothed_signal(
            'Normalized_Fluorescence_Intensity',
            'Normalized_Fluorescence_Intensity_Smoothed',
            original_signal
        )
        
        axes_norm[0].plot(df.index, original_signal, 'o-', 
                       label='Before Smoothing (Original)', alpha=0.7, markersize=3, color='blue')
        axes_norm[0].plot(df.index, smoothed_signal, 's-', 
                       label='After Smoothing', alpha=0.7, markersize=3, color='green')
        
        axes_norm[0].set_ylabel('Normalized Fluorescence Intensity', fontsize=12)
        axes_norm[0].set_title('Normalized - Fluorescence Intensity (Smoothing Comparison)', fontsize=13, fontweight='bold')
        axes_norm[0].legend()
        axes_norm[0].grid(True, alpha=0.3)
    
    # G-band Area
    if 'Normalized_Gband_Area' in df.columns:
        original_signal = df['Normalized_Gband_Area'].values
        smoothed_signal = get_smoothed_signal(
            'Normalized_Gband_Area',
            'Normalized_Gband_Area_Smoothed',
            original_signal
        )
        
        axes_norm[1].plot(df.index, original_signal, 'o-', 
                       label='Before Smoothing (Original)', alpha=0.7, markersize=3, color='blue')
        axes_norm[1].plot(df.index, smoothed_signal, 's-', 
                       label='After Smoothing', alpha=0.7, markersize=3, color='green')
        
        axes_norm[1].set_ylabel('G-band Area (Lorentzian)', fontsize=12)
        axes_norm[1].set_title('Normalized - G-band Area (Smoothing Comparison)', fontsize=13, fontweight='bold')
        axes_norm[1].legend()
        axes_norm[1].grid(True, alpha=0.3)
    
    # Raman peak 850 Area
    if 'Normalized_Raman_Peak_850_Area' in df.columns:
        original_signal = df['Normalized_Raman_Peak_850_Area'].values
        smoothed_signal = get_smoothed_signal(
            'Normalized_Raman_Peak_850_Area',
            'Normalized_Raman_Peak_850_Area_Smoothed',
            original_signal
        )
        
        axes_norm[2].plot(df.index, original_signal, 'o-', 
                       label='Before Smoothing (Original)', alpha=0.7, markersize=3, color='blue')
        axes_norm[2].plot(df.index, smoothed_signal, 's-', 
                       label='After Smoothing', alpha=0.7, markersize=3, color='green')
        
        axes_norm[2].set_ylabel('Raman Peak 850 Area (Lorentzian)', fontsize=12)
        axes_norm[2].set_title('Normalized - Raman Peak 850 Area (Smoothing Comparison)', fontsize=13, fontweight='bold')
        axes_norm[2].legend()
        axes_norm[2].grid(True, alpha=0.3)
    
    # Set x-axis formatting (same as norm_baseline_comp)
    if x_axis_type == 'datetime':
        for ax in axes_norm:
            add_day_night_shading(ax, df.index.min(), df.index.max(), light_cycle=light_cycle, light_transition=light_transition)
            # Add vertical line to mark light transition if configured
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    ax.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
            # Add vertical line to mark shade transition if configured
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    label_text = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
            # Add markers for treatment events if configured
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if df.index.min() <= event_time <= df.index.max():
                        ax.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                 linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            ax.set_xlabel('Date time (MM-DD HH)', fontsize=12)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
        fig_norm.autofmt_xdate()
    else:  # scan_number
        for ax in axes_norm:
            ax.set_xlabel('Scan Number', fontsize=12)
    
    plt.tight_layout()
    save_path_norm = Path(output_dir_str) / f'norm_smoothing_comp_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(str(save_path_norm), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Normalized smoothing comparison plot saved to: {save_path_norm}")


def plot_representative_raman_spectrum(summary, output_dir, config):
    """
    Plot representative Raman spectrum from first processed scan showing:
    - Raw normalized spectrum
    - Lieberfit baseline
    - Lorentzian fitting for G-band and Raman peak at 850 cm-1
    
    Parameters:
    -----------
    summary : dict
        Summary dictionary from process_scan_v2() containing spectrum data
    output_dir : Path
        Output directory for saving plot
    config : dict
        Configuration dictionary
    """
    if summary is None:
        return
    
    output_dir_str = str(output_dir)
    create_dir_if_needed(output_dir_str)
    
    # Extract spectrum data
    wavenumbers = summary.get('wavenumbers')
    intensities_normalized_smooth = summary.get('intensities_normalized_smooth')
    baseline_normalized = summary.get('baseline_normalized')
    corrected_normalized = summary.get('corrected_normalized')
    raman_peak_850 = summary.get('raman_peak_850_normalized')
    gband_peak_1600 = summary.get('gband_peak_1600_normalized')
    scan_number = summary.get('scan_number')
    
    if wavenumbers is None or intensities_normalized_smooth is None:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot normalized smoothed spectrum
    ax.plot(wavenumbers, intensities_normalized_smooth, '-', 
            color='black', label='Normalized Smoothed Spectrum', alpha=0.7, linewidth=1.5)
    
    # Plot Lieberfit baseline
    if baseline_normalized is not None:
        ax.plot(wavenumbers, baseline_normalized, '--', 
                color='red', label='Lieberfit Baseline', alpha=0.8, linewidth=2)
    
    # Plot corrected spectrum (after Lieberfit)
    if corrected_normalized is not None:
        ax.plot(wavenumbers, corrected_normalized, '-', 
                color='blue', label='Lieberfit-Corrected Spectrum', alpha=0.6, linewidth=1)
    
    # Plot Lorentzian fits for G-band
    if gband_peak_1600 and gband_peak_1600.get('fit_params') is not None:
        fit_params = gband_peak_1600['fit_params']
        center = gband_peak_1600.get('wavenumber', fit_params[1])
        width = gband_peak_1600.get('width', fit_params[2])
        
        # Create fine grid around peak for smooth Lorentzian curve
        peak_range_mask = (wavenumbers >= center - 3*width) & (wavenumbers <= center + 3*width)
        if np.any(peak_range_mask):
            wavenumbers_peak = wavenumbers[peak_range_mask]
            lorentzian_fit = lorentzian(wavenumbers_peak, fit_params[0], fit_params[1], fit_params[2], fit_params[3])
            ax.plot(wavenumbers_peak, lorentzian_fit, '-', 
                    color='green', label=f"G-band Lorentzian Fit (Area={gband_peak_1600.get('area', 0):.2f})", 
                    linewidth=2.5, alpha=0.9)
            # Mark peak center
            ax.axvline(center, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Plot Lorentzian fits for Raman peak 850
    if raman_peak_850 and raman_peak_850.get('fit_params') is not None:
        fit_params = raman_peak_850['fit_params']
        center = raman_peak_850.get('wavenumber', fit_params[1])
        width = raman_peak_850.get('width', fit_params[2])
        
        # Create fine grid around peak for smooth Lorentzian curve
        peak_range_mask = (wavenumbers >= center - 3*width) & (wavenumbers <= center + 3*width)
        if np.any(peak_range_mask):
            wavenumbers_peak = wavenumbers[peak_range_mask]
            lorentzian_fit = lorentzian(wavenumbers_peak, fit_params[0], fit_params[1], fit_params[2], fit_params[3])
            ax.plot(wavenumbers_peak, lorentzian_fit, '-', 
                    color='orange', label=f"Raman 850 cm⁻¹ Lorentzian Fit (Area={raman_peak_850.get('area', 0):.2f})", 
                    linewidth=2.5, alpha=0.9)
            # Mark peak center
            ax.axvline(center, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=14)
    ax.set_ylabel('Normalized Intensity', fontsize=14)
    ax.set_title(f'Representative Raman Spectrum - Scan {scan_number}\n(Normalized, Lieberfit Baseline, and Lorentzian Fits)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(250, max(wavenumbers))
    
    save_path = Path(output_dir_str) / f'rep_raman_s{scan_number}_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Representative Raman spectrum plot saved to: {save_path}")


def plot_ratio_timeseries(df, output_dir, config, light_cycle=None):
    """
    Plot timeseries for fluorescence ratios (baseline corrected only).
    Shows pre-smoothed data as markers with shade, and Gaussian smoothed as bold dashed line.
    Two plots stacked in one figure.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with ratio columns
    output_dir : Path
        Output directory for saving plots
    config : dict
        Configuration dictionary
    light_cycle : str, optional
        Light cycle for shading ('Constant', '8to24', '6to22')
    """
    if df.empty:
        print("Warning: No data to plot in time series.")
        return
    
    # Get light cycle from config if not provided
    if light_cycle is None:
        light_cycle = config.get('light_cycle', 'Constant')
    
    # Parse transition and event configs if available
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    # Get x-axis type from config (default: datetime)
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    if x_axis_type not in ['datetime', 'scan_number']:
        print(f"Warning: Invalid x_axis_type '{x_axis_type}', using 'datetime'")
        x_axis_type = 'datetime'
    
    # Parse transition and event configs if available
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    output_dir_str = str(output_dir)
    create_dir_if_needed(output_dir_str)
    
    # Get smoothing sigma from config (same as main.py - uses sigma_gaussian)
    # Default: 25 (matches config), fallback to 5 if not in config
    ratio_smooth_sigma = config.get('sigma_gaussian', 25)
    
    # Create stacked figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'hspace': 0.15})
    
    # Plot 1: Fluorescence to G-band ratio (top subplot)
    if 'Fluorescence_to_Gband_Ratio_BaselineCorrected' in df.columns:
        ratio_data = df['Fluorescence_to_Gband_Ratio_BaselineCorrected'].values
        mask = ~np.isnan(ratio_data)
        
        if np.sum(mask) > 0:
            # Plot pre-smoothed data as markers with transparency (no border)
            ax1.plot(df.index, ratio_data, 'o', 
                    color='m', label='Baseline Corrected Ratio', alpha=0.5, markersize=4,
                    markeredgewidth=0)
            
            # Calculate and plot Gaussian smoothed as bold dashed line
            ratio_smoothed = ratio_data.copy()
            if np.sum(mask) > 1:
                ratio_smoothed[mask] = gaussian_filter1d(ratio_data[mask], sigma=ratio_smooth_sigma)
            ax1.plot(df.index, ratio_smoothed, '--', 
                    color='black', label='Gaussian Smoothed', alpha=1.0, linewidth=3)
        
        # Only add shading for datetime-based plots
        if x_axis_type == 'datetime':
            add_day_night_shading(ax1, df.index.min(), df.index.max(), light_cycle=light_cycle, light_transition=light_transition)
            # Add vertical line to mark light transition if configured
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    ax1.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
            # Add vertical line to mark shade transition if configured
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    label_text = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax1.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
            # Add markers for treatment events if configured
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if df.index.min() <= event_time <= df.index.max():
                        ax1.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                  linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            ax1.set_xlabel('Date time (MM-DD HH)', fontsize=14)
            ax1.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax1.xaxis.set_major_locator(DayLocator())
            fig.autofmt_xdate()
        else:  # scan_number
            ax1.set_xlabel('Scan Number', fontsize=14)
        
        ax1.set_ylabel('Fluorescence / G-band Ratio', fontsize=14)
        ax1.set_title('Fluorescence to G-band Ratio Timeseries', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=12)
    
    # Plot 2: Fluorescence to Raman peak 850 ratio (bottom subplot)
    if 'Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected' in df.columns:
        ratio_data = df['Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected'].values
        mask = ~np.isnan(ratio_data)
        
        if np.sum(mask) > 0:
            # Plot pre-smoothed data as markers with transparency (no border)
            ax2.plot(df.index, ratio_data, 'o', 
                    color='green', label='Baseline Corrected Ratio', alpha=0.5, markersize=4,
                    markeredgewidth=0)
            
            # Calculate and plot Gaussian smoothed as bold dashed line
            ratio_smoothed = ratio_data.copy()
            if np.sum(mask) > 1:
                ratio_smoothed[mask] = gaussian_filter1d(ratio_data[mask], sigma=ratio_smooth_sigma)
            ax2.plot(df.index, ratio_smoothed, '--', 
                    color='black', label='Gaussian Smoothed', alpha=1.0, linewidth=3)
        
        # Only add shading for datetime-based plots
        if x_axis_type == 'datetime':
            add_day_night_shading(ax2, df.index.min(), df.index.max(), light_cycle=light_cycle, light_transition=light_transition)
            # Add vertical line to mark light transition if configured
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    ax2.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
            # Add vertical line to mark shade transition if configured
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    label_text = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax2.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
            # Add markers for treatment events if configured
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if df.index.min() <= event_time <= df.index.max():
                        ax2.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                  linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            ax2.set_xlabel('Date time (MM-DD HH)', fontsize=14)
            ax2.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax2.xaxis.set_major_locator(DayLocator())
        else:  # scan_number
            ax2.set_xlabel('Scan Number', fontsize=14)
        
        ax2.set_ylabel('Fluorescence / Raman Peak 850 Ratio', fontsize=14)
        ax2.set_title('Fluorescence to Raman Peak 850 Ratio Timeseries', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    save_path = Path(output_dir_str) / f'fluo_ratios_ts_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Stacked fluorescence ratios plot saved to: {save_path}")


def apply_als_and_gaussian_v3(results_df, config, processed_dir, start_datetime=None, end_datetime=None, light_cycle='Constant', light_transition=None, shade_transition=None, treatment_events=None):
    """
    Apply ALS baseline correction and Gaussian smoothing to baseline-corrected ratios (v3).
    Works with both Fluorescence_to_Gband_Ratio_BaselineCorrected and Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with baseline-corrected ratio columns and Datetime index
    config : dict
        Configuration dictionary containing parameters like 'lam_als', 'p_als', 'niter_als', 'sigma_gaussian'
    processed_dir : Path
        Output directory for saving plots and updated CSV
    start_datetime : datetime, optional
        Start datetime for filtering the data (inclusive)
    end_datetime : datetime, optional
        End datetime for filtering the data (inclusive)
    light_cycle : str
        Light cycle for shading ('Constant', '8to24', '6to22')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with ALS-corrected columns added
    """
    create_dir_if_needed(str(processed_dir))
    
    # Filter by datetime range if provided
    if start_datetime is not None and end_datetime is not None:
        results_df = results_df[(results_df.index >= start_datetime) & (results_df.index <= end_datetime)]
    
    # Get ALS parameters from config
    lam_als = config.get('lam_als', 1e7)
    p_als = config.get('p_als', 0.001)
    niter_als = config.get('niter_als', 20)
    sigma_gaussian = config.get('sigma_gaussian', 25)
    
    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(results_df.index):
        if 'Datetime' in results_df.columns:
            results_df['Datetime'] = pd.to_datetime(results_df['Datetime'], errors='coerce')
            results_df = results_df.set_index('Datetime')
        else:
            results_df.index = pd.to_datetime(results_df.index, errors='coerce')
    
    if results_df.index.hasnans or not pd.api.types.is_datetime64_any_dtype(results_df.index):
        print("Warning: Invalid or non-Datetime values detected in index. Dropping rows with NaT or converting.")
        results_df = results_df[results_df.index.notna()]
        if not pd.api.types.is_datetime64_any_dtype(results_df.index):
            results_df.index = pd.to_datetime(results_df.index, errors='coerce', origin='unix')
    
    datetimes = results_df.index
    
    # ALS baseline correction function
    def baseline_als(y, lam=lam_als, p=p_als, niter=niter_als):
        L = len(y)
        D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), format='csc')
        w = np.ones(L)
        for i in range(niter):
            W = diags(w, 0, shape=(L, L), format='csc')
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z
    
    # Process G-band ratio
    if 'Fluorescence_to_Gband_Ratio_BaselineCorrected' in results_df.columns:
        print("\n=== Applying ALS baseline correction to Fluorescence/G-band Ratio ===")
        ratios_gband = results_df['Fluorescence_to_Gband_Ratio_BaselineCorrected'].values
        mask_gband = ~np.isnan(ratios_gband)
        
        if np.sum(mask_gband) >= 3:
            ratios_gband_valid = ratios_gband[mask_gband]
            # Apply Gaussian smoothing FIRST (before ALS)
            ratios_gband_smoothed = gaussian_filter1d(ratios_gband_valid, sigma=sigma_gaussian)
            # Then apply ALS baseline correction
            baseline_als_vals_gband = baseline_als(ratios_gband_smoothed)
            corrected_als_gband = ratios_gband_smoothed - baseline_als_vals_gband
            # No additional smoothing needed (already smoothed before ALS)
            smoothed_corrected_gband = corrected_als_gband.copy()
            
            # Store results (with NaN preservation)
            corrected_gband_full = np.full_like(ratios_gband, np.nan)
            smoothed_gband_full = np.full_like(ratios_gband, np.nan)
            corrected_gband_full[mask_gband] = corrected_als_gband
            smoothed_gband_full[mask_gband] = smoothed_corrected_gband
            
            results_df['Fluorescence_to_Gband_Ratio_ALS'] = corrected_gband_full
            results_df['Fluorescence_to_Gband_Ratio_ALS_Smoothed'] = smoothed_gband_full
            
            # Create plot
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(datetimes, ratios_gband, label='Baseline Corrected Ratio', color='m', alpha=0.6, linewidth=1)
            # Plot smoothed signal (input to ALS)
            smoothed_gband_full_plot = np.full_like(ratios_gband, np.nan)
            smoothed_gband_full_plot[mask_gband] = ratios_gband_smoothed
            ax.plot(datetimes, smoothed_gband_full_plot, label='Gaussian Smoothed (before ALS)', color='blue', alpha=0.6, linewidth=1, linestyle='-.')
            ax.plot(datetimes[mask_gband], baseline_als_vals_gband, label='Baseline', color='red', linestyle='--', linewidth=2)
            ax.plot(datetimes, corrected_gband_full, label='Corrected', color='green', alpha=0.8, linewidth=1.5)
            
            # Add day/night shading
            if light_transition is None:
                light_transition = parse_light_transition_config(config)
            if shade_transition is None:
                shade_transition = parse_shade_transition_config(config)
            if treatment_events is None:
                treatment_events = parse_treatment_events_config(config)
            add_day_night_shading(ax, datetimes.min(), datetimes.max(), light_cycle=light_cycle, light_transition=light_transition)
            # Add vertical line to mark light transition if configured
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if datetimes.min() <= transition_time <= datetimes.max():
                    ax.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
            # Add vertical line to mark shade transition if configured
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if datetimes.min() <= transition_time <= datetimes.max():
                    label_text = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
            # Add markers for treatment events if configured
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if datetimes.min() <= event_time <= datetimes.max():
                        ax.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                 linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            
            ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
            ax.set_ylabel('Fluorescence / G-band Ratio', fontsize=14)
            ax.set_title('Baseline Correction - Fluorescence to G-band Ratio', fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=12)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
            fig.autofmt_xdate()
            
            save_path = Path(processed_dir) / f'baseline_gband_{datetime.now().strftime("%Y%m%d")}.png'
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Baseline correction plot saved to: {save_path}")
        else:
            print("Warning: Insufficient data for G-band ratio ALS correction")
    
    # Process Raman peak 850 ratio
    if 'Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected' in results_df.columns:
        print("\n=== Applying ALS baseline correction to Fluorescence/Raman Peak 850 Ratio ===")
        ratios_raman = results_df['Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected'].values
        mask_raman = ~np.isnan(ratios_raman)
        
        if np.sum(mask_raman) >= 3:
            ratios_raman_valid = ratios_raman[mask_raman]
            # Apply Gaussian smoothing FIRST (before ALS)
            ratios_raman_smoothed = gaussian_filter1d(ratios_raman_valid, sigma=sigma_gaussian)
            # Then apply ALS baseline correction
            baseline_als_vals_raman = baseline_als(ratios_raman_smoothed)
            corrected_als_raman = ratios_raman_smoothed - baseline_als_vals_raman
            # No additional smoothing needed (already smoothed before ALS)
            smoothed_corrected_raman = corrected_als_raman.copy()
            
            # Store results (with NaN preservation)
            corrected_raman_full = np.full_like(ratios_raman, np.nan)
            smoothed_raman_full = np.full_like(ratios_raman, np.nan)
            corrected_raman_full[mask_raman] = corrected_als_raman
            smoothed_raman_full[mask_raman] = smoothed_corrected_raman
            
            results_df['Fluorescence_to_Raman_Peak_850_Ratio_ALS'] = corrected_raman_full
            results_df['Fluorescence_to_Raman_Peak_850_Ratio_ALS_Smoothed'] = smoothed_raman_full
            
            # Create plot
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(datetimes, ratios_raman, label='Baseline Corrected Ratio', color='m', alpha=0.6, linewidth=1)
            # Plot smoothed signal (input to ALS)
            smoothed_raman_full_plot = np.full_like(ratios_raman, np.nan)
            smoothed_raman_full_plot[mask_raman] = ratios_raman_smoothed
            ax.plot(datetimes, smoothed_raman_full_plot, label='Gaussian Smoothed (before ALS)', color='blue', alpha=0.6, linewidth=1, linestyle='-.')
            ax.plot(datetimes[mask_raman], baseline_als_vals_raman, label='Baseline', color='red', linestyle='--', linewidth=2)
            ax.plot(datetimes, corrected_raman_full, label='Corrected', color='green', alpha=0.8, linewidth=1.5)
            
            # Add day/night shading
            if light_transition is None:
                light_transition = parse_light_transition_config(config)
            if shade_transition is None:
                shade_transition = parse_shade_transition_config(config)
            if treatment_events is None:
                treatment_events = parse_treatment_events_config(config)
            add_day_night_shading(ax, datetimes.min(), datetimes.max(), light_cycle=light_cycle, light_transition=light_transition)
            # Add vertical line to mark light transition if configured
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if datetimes.min() <= transition_time <= datetimes.max():
                    ax.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
            # Add vertical line to mark shade transition if configured
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if datetimes.min() <= transition_time <= datetimes.max():
                    label_text = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
            # Add markers for treatment events if configured
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if datetimes.min() <= event_time <= datetimes.max():
                        ax.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                 linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            
            ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
            ax.set_ylabel('Fluorescence / Raman Peak 850 Ratio', fontsize=14)
            ax.set_title('Baseline Correction - Fluorescence to Raman Peak 850 Ratio', fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=12)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
            fig.autofmt_xdate()
            
            save_path = Path(processed_dir) / f'baseline_raman850_{datetime.now().strftime("%Y%m%d")}.png'
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Baseline correction plot saved to: {save_path}")
        else:
            print("Warning: Insufficient data for Raman 850 ratio ALS correction")
    
    return results_df


def compute_fourier_transform_v3(results_df, config, processed_dir, start_datetime=None, end_datetime=None, light_cycle='Constant', light_transition=None, shade_transition=None, treatment_events=None):
    """
    Compute and plot the Fourier Transform of the final baseline-corrected ratios.
    Works with both Fluorescence_to_Gband_Ratio_ALS_Smoothed and Fluorescence_to_Raman_Peak_850_Ratio_ALS_Smoothed.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with ALS-corrected ratio columns and Datetime index
    config : dict
        Configuration dictionary containing parameters like 'fft_top_peaks'
    processed_dir : Path
        Output directory for saving plots and CSVs
    start_datetime : datetime, optional
        Start datetime for filtering the data (inclusive)
    end_datetime : datetime, optional
        End datetime for filtering the data (inclusive)
    light_cycle : str
        Light cycle for shading ('Constant', '8to24', '6to22')
    
    Returns:
    --------
    dict
        Dictionary with FFT results DataFrames for each ratio
    """
    create_dir_if_needed(str(processed_dir))
    
    # Filter by datetime range if provided
    if start_datetime is not None and end_datetime is not None:
        results_df = results_df[(results_df.index >= start_datetime) & (results_df.index <= end_datetime)]
    
    top_peaks = config.get('fft_top_peaks', 5)
    datetimes = results_df.index
    
    # Calculate time in hours from start (handle both datetime and timedelta64)
    if pd.api.types.is_datetime64_any_dtype(datetimes):
        time_diffs = datetimes - datetimes.min()
        # Convert timedelta64 to hours
        time_hours = np.array([pd.Timedelta(td).total_seconds() / 3600.0 for td in time_diffs])
    else:
        time_hours = np.array([(dt - datetimes.min()).total_seconds() / 3600.0 for dt in datetimes])
    
    sampling_interval = np.mean(np.diff(time_hours)) if len(time_hours) > 1 else 1.0
    
    # Parse transition and event configs if not provided
    if light_transition is None:
        light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    def compute_ft(signal, label_suffix, ratio_name):
        """Helper function to compute FFT for a signal."""
        mask = ~np.isnan(signal)
        if np.sum(mask) < 3:
            print(f"Warning: Insufficient data for FFT analysis of {ratio_name}")
            return None, None
        
        signal_valid = signal[mask]
        time_hours_valid = time_hours[mask]
        
        # Detrend signal (remove mean)
        signal_detrended = signal_valid - np.mean(signal_valid)
        
        # Calculate sampling interval from valid data
        if len(time_hours_valid) > 1:
            sampling_interval_valid = np.mean(np.diff(time_hours_valid))
        else:
            sampling_interval_valid = sampling_interval
        
        # Compute FFT
        fft_result = np.fft.fft(signal_detrended)
        frequencies = np.fft.fftfreq(len(signal_detrended), d=sampling_interval_valid)
        
        # Create complete FFT DataFrame
        complete_ft_df = pd.DataFrame({
            'Frequency (cycles/hour)': frequencies,
            'Real': np.real(fft_result),
            'Imag': np.imag(fft_result),
            'Magnitude': np.abs(fft_result),
            'Phase': np.angle(fft_result)
        })
        
        # Save complete FFT results
        complete_filename = f'fft_full_{label_suffix}_{datetime.now().strftime("%Y%m%d")}.csv'
        complete_path = Path(processed_dir) / complete_filename
        complete_ft_df.to_csv(complete_path, index=False)
        print(f"Complete FFT results for {ratio_name} saved to: {complete_path}")
        
        # Extract positive frequencies only
        positive_mask = frequencies > 0
        positive_freqs = frequencies[positive_mask]
        magnitude = np.abs(fft_result)[positive_mask]
        
        # Find top peaks
        if len(magnitude) > 0:
            peak_indices = np.argsort(magnitude)[-top_peaks:][::-1]
            peak_freqs = positive_freqs[peak_indices]
            peak_magnitudes = magnitude[peak_indices]
            peak_periods = 1 / peak_freqs
            
            fft_df = pd.DataFrame({
                'Frequency (cycles/hour)': peak_freqs,
                'Period (hours)': peak_periods,
                'Magnitude': peak_magnitudes
            })
            
            # Save peaks CSV
            peaks_filename = f'fft_peaks_{label_suffix}_{datetime.now().strftime("%Y%m%d")}.csv'
            peaks_path = Path(processed_dir) / peaks_filename
            fft_df.to_csv(peaks_path, index=False)
            print(f"FFT peaks for {ratio_name} saved to: {peaks_path}")
            print(f"\nTop {top_peaks} peaks for {ratio_name}:")
            print(fft_df)
            
            # Create combined plot: timeseries on top, FFT below
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False, gridspec_kw={'hspace': 0.3})
            
            # Top subplot: Timeseries of baseline-corrected ratio
            if pd.api.types.is_datetime64_any_dtype(results_df.index):
                datetimes_valid = results_df.index[mask]
                ax1.plot(datetimes_valid, signal_valid, color='m', alpha=0.7, linewidth=1.5, label=f'{ratio_name}')
                
                # Add day/night shading (light_transition, shade_transition, treatment_events are from outer scope)
                add_day_night_shading(ax1, datetimes_valid.min(), datetimes_valid.max(), light_cycle=light_cycle, light_transition=light_transition)
                # Add vertical line to mark light transition if configured
                if light_transition:
                    transition_time = light_transition['transition_datetime']
                    if datetimes_valid.min() <= transition_time <= datetimes_valid.max():
                        ax1.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
                # Add vertical line to mark shade transition if configured
                if shade_transition:
                    transition_time = shade_transition['transition_datetime']
                    if datetimes_valid.min() <= transition_time <= datetimes_valid.max():
                        label_text = 'Shade transition'
                        if 'ppfd' in shade_transition:
                            label_text += f" (PPFD: {shade_transition['ppfd']})"
                        ax1.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
                # Add markers for treatment events if configured
                if treatment_events:
                    for event in treatment_events:
                        event_time = event['datetime']
                        if datetimes_valid.min() <= event_time <= datetimes_valid.max():
                            ax1.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                      linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
                
                ax1.set_xlabel('Date time (MM-DD HH)', fontsize=12)
                ax1.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
                ax1.xaxis.set_major_locator(DayLocator())
                fig.autofmt_xdate()
            else:
                ax1.plot(time_hours_valid, signal_valid, color='m', alpha=0.7, linewidth=1.5, label=f'{ratio_name}')
                ax1.set_xlabel('Time (hours from start)', fontsize=12)
            
            ax1.set_ylabel(f'{ratio_name}', fontsize=12)
            ax1.set_title(f'Timeseries - {ratio_name}', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10, loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='both', labelsize=11)
            
            # Bottom subplot: FFT Spectrum
            ax2.plot(positive_freqs, magnitude, color='blue', linewidth=1.5, label='FFT Magnitude')
            ax2.scatter(peak_freqs, peak_magnitudes, color='red', s=100, zorder=5, 
                       label=f'Top {top_peaks} Peaks', marker='o', edgecolors='darkred', linewidths=1.5)
            
            # Highlight diurnal range (0.03-0.05 cycles/hour, ~20-33 hour period)
            ax2.axvspan(0.03, 0.05, alpha=0.2, color='yellow', label='Diurnal Range (0.03-0.05 cycles/hour)')
            
            ax2.set_xlabel('Frequency (cycles/hour)', fontsize=12)
            ax2.set_ylabel('Magnitude', fontsize=12)
            ax2.set_title(f'FFT Spectrum - {ratio_name}', fontsize=13, fontweight='bold')
            ax2.set_xlim(0, 0.5)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10, loc='best')
            ax2.tick_params(axis='both', labelsize=11)
            
            plot_filename = f'fft_combined_{label_suffix}_{datetime.now().strftime("%Y%m%d")}.png'
            plot_path = Path(processed_dir) / plot_filename
            plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Combined FFT plot for {ratio_name} saved to: {plot_path}")
            
            return fft_df, complete_ft_df
        else:
            return None, None
    
    # Process both ratios
    fft_results = {}
    
    # G-band ratio
    if 'Fluorescence_to_Gband_Ratio_ALS_Smoothed' in results_df.columns:
        print("\n=== Computing FFT for Fluorescence/G-band Ratio ===")
        signal = results_df['Fluorescence_to_Gband_Ratio_ALS_Smoothed'].values
        fft_df_gband, complete_ft_df_gband = compute_ft(signal, 'gband_ratio', 'Fluorescence to G-band Ratio')
        if fft_df_gband is not None:
            fft_results['gband'] = {
                'peaks': fft_df_gband,
                'complete': complete_ft_df_gband
            }
    elif 'Fluorescence_to_Gband_Ratio_ALS' in results_df.columns:
        print("\n=== Computing FFT for Fluorescence/G-band Ratio (using ALS-corrected) ===")
        signal = results_df['Fluorescence_to_Gband_Ratio_ALS'].values
        fft_df_gband, complete_ft_df_gband = compute_ft(signal, 'gband_ratio', 'Fluorescence to G-band Ratio')
        if fft_df_gband is not None:
            fft_results['gband'] = {
                'peaks': fft_df_gband,
                'complete': complete_ft_df_gband
            }
    
    # Raman peak 850 ratio
    if 'Fluorescence_to_Raman_Peak_850_Ratio_ALS_Smoothed' in results_df.columns:
        print("\n=== Computing FFT for Fluorescence/Raman Peak 850 Ratio ===")
        signal = results_df['Fluorescence_to_Raman_Peak_850_Ratio_ALS_Smoothed'].values
        fft_df_raman, complete_ft_df_raman = compute_ft(signal, 'raman_850_ratio', 'Fluorescence to Raman Peak 850 Ratio')
        if fft_df_raman is not None:
            fft_results['raman_850'] = {
                'peaks': fft_df_raman,
                'complete': complete_ft_df_raman
            }
    elif 'Fluorescence_to_Raman_Peak_850_Ratio_ALS' in results_df.columns:
        print("\n=== Computing FFT for Fluorescence/Raman Peak 850 Ratio (using ALS-corrected) ===")
        signal = results_df['Fluorescence_to_Raman_Peak_850_Ratio_ALS'].values
        fft_df_raman, complete_ft_df_raman = compute_ft(signal, 'raman_850_ratio', 'Fluorescence to Raman Peak 850 Ratio')
        if fft_df_raman is not None:
            fft_results['raman_850'] = {
                'peaks': fft_df_raman,
                'complete': complete_ft_df_raman
            }
    
    return fft_results


def compute_diurnal_average_v3(results_df, config, processed_dir, start_datetime=None, end_datetime=None, light_cycle='Constant'):
    """
    Compute and plot the average diurnal cycle from the final baseline-corrected ratios.
    Works with both Fluorescence_to_Gband_Ratio_ALS_Smoothed and Fluorescence_to_Raman_Peak_850_Ratio_ALS_Smoothed.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with ALS-corrected ratio columns and Datetime index
    config : dict
        Configuration dictionary containing parameters like 'bin_factor'
    processed_dir : Path
        Output directory for saving plots and CSVs
    start_datetime : datetime, optional
        Start datetime for filtering the data (inclusive)
    end_datetime : datetime, optional
        End datetime for filtering the data (inclusive)
    light_cycle : str
        Light cycle for shading ('Constant', '8to24', '6to22')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with diurnal average columns added
    """
    create_dir_if_needed(str(processed_dir))
    
    # Filter by datetime range if provided
    if start_datetime is not None and end_datetime is not None:
        results_df = results_df[(results_df.index >= start_datetime) & (results_df.index <= end_datetime)]
    
    bin_factor = config.get('bin_factor', 2)
    datetimes = results_df.index
    
    # Add hour of day columns
    results_df = results_df.copy()
    results_df['Hour of Day'] = datetimes.hour + datetimes.minute / 60.0
    results_df['Binned Hour'] = np.round(results_df['Hour of Day'] * bin_factor) / bin_factor
    
    def plot_diurnal_average(column_name, ratio_name, color='green'):
        """Helper function to compute and plot diurnal average for a ratio column."""
        if column_name not in results_df.columns:
            print(f"Warning: Column '{column_name}' not found for diurnal average")
            return
        
        # Create short abbreviation for column name
        col_abbrev = column_name.replace('Fluorescence_to_', 'F_').replace('_Ratio', '_R').replace('_BaselineCorrected', '_BC').replace('_ALS', '_ALS').replace('_Smoothed', '_S').replace('_Raman_Peak_850', '_R850').replace('_Gband', '_G')
        
        # Group by binned hour and compute statistics
        daily_avg = results_df.groupby('Binned Hour')[column_name].agg(['mean', 'std', 'count']).reset_index()
        daily_avg['sem'] = daily_avg['std'] / np.sqrt(daily_avg['count'])
        
        # Apply additional Gaussian smoothing to mean values
        daily_avg['Smoothed Avg'] = gaussian_filter1d(daily_avg['mean'], sigma=1)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.errorbar(daily_avg['Binned Hour'], daily_avg['Smoothed Avg'], yerr=daily_avg['sem'],
                    marker='o', color=color, ecolor=color, alpha=0.6, capsize=3, 
                    label='Mean ± SEM', linewidth=2, markersize=6)
        
        # Add day/night shading for hour-based x-axis (not datetime-based)
        if light_cycle == 'Constant':
            # Constant light - all day
            ax.axvspan(0, 24, facecolor='yellow', alpha=0.1, label='Daytime')
        elif light_cycle == '8to24':
            # 8:00 to 24:00 (next day 0:00) is day
            ax.axvspan(8, 24, facecolor='yellow', alpha=0.1, label='Daytime')
            ax.axvspan(0, 8, facecolor='blue', alpha=0.1, label='Nighttime')
        elif light_cycle == '6to22':
            # 6:00 to 22:00 is day
            ax.axvspan(6, 22, facecolor='yellow', alpha=0.1, label='Daytime')
            ax.axvspan(0, 6, facecolor='blue', alpha=0.1, label='Nighttime')
            ax.axvspan(22, 24, facecolor='blue', alpha=0.1)
        
        ax.set_xlabel('Hour of Day', fontsize=14)
        ax.set_ylabel(f'Average {ratio_name}', fontsize=14)
        ax.set_title(f'Average Diurnal Cycle - {ratio_name}', fontsize=15, fontweight='bold')
        ax.set_xticks(np.arange(0, 25, 2))
        ax.set_xlim(0, 24)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11, loc='best')
        ax.tick_params(axis='both', labelsize=12)
        
        plot_filename = f'diurnal_{col_abbrev}_{datetime.now().strftime("%Y%m%d")}.png'
        plot_path = Path(processed_dir) / plot_filename
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Diurnal average plot for {ratio_name} saved to: {plot_path}")
        
        # Save diurnal average data
        csv_filename = f'diurnal_{col_abbrev}_{datetime.now().strftime("%Y%m%d")}.csv'
        csv_path = Path(processed_dir) / csv_filename
        daily_avg.to_csv(csv_path, index=False)
        print(f"Diurnal average data for {ratio_name} saved to: {csv_path}")
    
    # Process both ratios
    # G-band ratio
    if 'Fluorescence_to_Gband_Ratio_ALS_Smoothed' in results_df.columns:
        print("\n=== Computing diurnal average for Fluorescence/G-band Ratio ===")
        plot_diurnal_average('Fluorescence_to_Gband_Ratio_ALS_Smoothed', 
                            'Fluorescence to G-band Ratio', color='green')
    elif 'Fluorescence_to_Gband_Ratio_ALS' in results_df.columns:
        print("\n=== Computing diurnal average for Fluorescence/G-band Ratio (using ALS-corrected) ===")
        plot_diurnal_average('Fluorescence_to_Gband_Ratio_ALS', 
                            'Fluorescence to G-band Ratio', color='green')
    
    # Raman peak 850 ratio
    if 'Fluorescence_to_Raman_Peak_850_Ratio_ALS_Smoothed' in results_df.columns:
        print("\n=== Computing diurnal average for Fluorescence/Raman Peak 850 Ratio ===")
        plot_diurnal_average('Fluorescence_to_Raman_Peak_850_Ratio_ALS_Smoothed', 
                            'Fluorescence to Raman Peak 850 Ratio', color='orange')
    elif 'Fluorescence_to_Raman_Peak_850_Ratio_ALS' in results_df.columns:
        print("\n=== Computing diurnal average for Fluorescence/Raman Peak 850 Ratio (using ALS-corrected) ===")
        plot_diurnal_average('Fluorescence_to_Raman_Peak_850_Ratio_ALS', 
                            'Fluorescence to Raman Peak 850 Ratio', color='orange')
    
    return results_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Processing v3 - continuation of v2 focusing on normalized data with ratios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch processing (default skips first 500 scans)
  python main_test_v3.py --batch

  # Batch with baseline correction parameters
  python main_test_v3.py --batch --baseline-threshold 7.5 --baseline-window 15 --correct-smoothed

  # Batch with custom scan range
  python main_test_v3.py --batch --scans "100,200,300" --max-scans 10
        """
    )
    
    parser.add_argument(
        "--config-file",
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML pipeline config."
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Profile name to load from the YAML config."
    )
    
    # Batch processing
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch processing on multiple scans instead of single scan."
    )
    parser.add_argument(
        "--scans",
        type=str,
        help="Comma-separated scan numbers for batch processing. Defaults to all scans."
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        help="Limit the number of scans processed in batch mode."
    )
    parser.add_argument(
        "--skip-scans",
        type=int,
        default=500,
        help="Number of scans to skip from the beginning (default: 500)."
    )
    
    # Lieberfit parameters
    parser.add_argument(
        "--order",
        type=int,
        default=None,
        help="Override polynomial order for Lieberfit."
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Override Lieberfit iterations."
    )
    
    # Baseline correction parameters (same as v2)
    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=None,
        help="MAD threshold multiplier for baseline correction jump detection. If not provided, uses value from config file."
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=None,
        help="Window size for baseline correction offset calculation. If not provided, uses value from config file."
    )
    parser.add_argument(
        "--baseline-smooth-window",
        type=int,
        default=None,
        help="Window size for Savitzky-Golay smoothing before baseline correction (default: 11)."
    )
    parser.add_argument(
        "--baseline-smooth-poly",
        type=int,
        default=None,
        help="Polynomial order for Savitzky-Golay smoothing before baseline correction (default: 2)."
    )
    parser.add_argument(
        "--no-baseline-smooth",
        action="store_true",
        help="Disable smoothing before baseline correction jump detection."
    )
    parser.add_argument(
        "--correct-smoothed",
        action="store_true",
        help="Apply baseline correction to smoothed timeseries data instead of original data (default: True)."
    )
    parser.add_argument(
        "--no-correct-smoothed",
        dest="correct_smoothed",
        action="store_false",
        help="Disable baseline correction on smoothed data (use original data instead)."
    )
    
    # Spike removal parameters
    parser.add_argument(
        "--remove-spikes",
        action="store_true",
        help="Remove spikes (1-5 data points) from time series before smoothing."
    )
    parser.add_argument(
        "--spike-window",
        type=int,
        default=5,
        help="Half-window size for spike detection (default: 5). Total window = 2*window + 1."
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=3.0,
        help="Threshold in units of MAD for spike detection (default: 3.0). Higher = less sensitive."
    )
    
    # Cumulative jump detection parameters
    parser.add_argument(
        "--no-cumulative-jumps",
        dest="detect_cumulative_jumps",
        action="store_false",
        help="Disable detection of gradual jumps over multiple consecutive points."
    )
    parser.add_argument(
        "--cumulative-window",
        type=int,
        default=5,
        help="Window size for detecting cumulative changes (default: 5). Detects gradual jumps over N consecutive points."
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. Defaults to scripts/test_outputs/"
    )
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_profile_config(args.config_file, args.profile)
    
    # Override Lieberfit parameters from command line if provided
    if args.order is not None:
        config['poly_order'] = args.order
    if args.iter is not None:
        config['tot_iter'] = args.iter
    
    # Change to parent directory if using relative paths (requires_google_drive: false)
    # This allows relative paths to be resolved from the parent directory where the data is located
    data_source = config.get("data_source", {})
    requires_drive = data_source.get("requires_google_drive", True)
    original_cwd = os.getcwd()
    if not requires_drive:
        # Change to parent directory (two levels up from Script/swnt_iaa_analysis_v2 to IAA-MN longitudinal)
        parent_dir = PROJECT_ROOT.parent.parent
        os.chdir(parent_dir)
        print(f"Changed working directory to: {os.getcwd()}")
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_raman_dataset(config)
    finally:
        # Restore original working directory
        if not requires_drive:
            os.chdir(original_cwd)
    raman_df = dataset.spectra
    
    print(f"Loaded {len(raman_df)} scans")
    print(f"Wavenumber range: {raman_df.columns[2]} to {raman_df.columns[-1]}")
    print(f"Raw data file: {dataset.source_path}")
    
    # Set output directory relative to raw data file location
    if args.output_dir is None:
        raw_data_dir = Path(dataset.source_path).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.batch:
            output_dir = raw_data_dir / f"test_outputs_v3_batch_{timestamp}"
        else:
            output_dir = raw_data_dir / f"test_outputs_v3_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    if args.batch:
        # Batch processing mode
        print("\n=== Running batch processing v3 (normalized data with ratios) ===")
        
        scan_numbers = None
        if args.scans:
            try:
                scan_numbers = [
                    int(val.strip())
                    for val in args.scans.split(",")
                    if val.strip()
                ]
            except ValueError:
                raise ValueError("--scans must be a comma-separated list of integers.")
        
        # Get available scans
        available_scans = sorted(raman_df["Scan Number"].unique())
        if not available_scans:
            print("[Batch] No scans available to process.")
            return
        
        # Parse scan list
        if scan_numbers:
            scan_list = [s for s in scan_numbers if s in available_scans]
        else:
            scan_list = list(available_scans)
        
        # Skip first N scans (default: 500)
        if args.skip_scans > 0 and len(scan_list) > args.skip_scans:
            scan_list = scan_list[args.skip_scans:]
            print(f"[Batch] Skipped first {args.skip_scans} scans")
        
        if args.max_scans:
            scan_list = scan_list[:args.max_scans]
        
        if not scan_list:
            print("[Batch] No scans left to process after filtering.")
            return
        
        print(f"[Batch] Processing {len(scan_list)} scans")
        print(f"[Batch] Scan list: {scan_list[:10]}{'...' if len(scan_list) > 10 else ''}")
        
        # Pre-compute wavenumber arrays once (cached for all scans)
        wavenumbers_full = np.array([float(col) for col in raman_df.columns if col not in ['Scan Number', 'Seconds']])
        wavenumber_filter = wavenumbers_full >= 250
        wavenumbers = wavenumbers_full[wavenumber_filter]
        
        # Process all scans
        summaries = []
        # Configure tqdm for Windows PowerShell - use longer update intervals
        # to prevent line repetition issues in PowerShell
        total_scans = len(scan_list)
        pbar = tqdm(total=total_scans, desc="Processing scans", unit="scan", 
                   ncols=100, mininterval=1.0, maxinterval=10.0,
                   file=sys.stderr, dynamic_ncols=False, leave=True,
                   ascii=True, disable=False,
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for scan in scan_list:
            summary = process_scan_v2(
                scan_number=scan,
                raman_df=raman_df,
                config=config,
                cached_wavenumbers_full=wavenumbers_full,
                cached_wavenumber_filter=wavenumber_filter,
                cached_wavenumbers=wavenumbers,
                verbose=False
            )
            if summary:
                summaries.append(summary)
            pbar.update(1)
        
        pbar.close()
        
        if not summaries:
            print("[Batch] No summaries generated.")
            return
        
        print(f"\n[Batch] Completed processing {len(summaries)} scans")
        
        # Aggregate summaries into DataFrame (normalized data only)
        print("\n=== Aggregating results ===")
        # Get x-axis type from config (default: datetime)
        x_axis_type = config.get('timeseries_x_axis', 'datetime')
        if x_axis_type not in ['datetime', 'scan_number']:
            print(f"Warning: Invalid x_axis_type '{x_axis_type}', using 'datetime'")
            x_axis_type = 'datetime'
        
        results_df = aggregate_summaries_to_dataframe_v3(summaries, raman_df, x_axis_type=x_axis_type)
        
        if results_df.empty:
            print("[Batch] Warning: No data to aggregate.")
            return
        
        # Apply baseline correction
        print("\n=== Applying baseline correction ===")
        # Debug: Print config values to verify they're being read
        print(f"DEBUG: Profile name being used: '{args.profile}'")
        print(f"DEBUG: All config keys containing 'baseline': {sorted([k for k in config.keys() if 'baseline' in k.lower()])}")
        config_threshold = config.get('baseline_correction_threshold', 'NOT FOUND')
        print(f"DEBUG: Config value for 'baseline_correction_threshold': {config_threshold} (type: {type(config_threshold).__name__})")
        print(f"DEBUG: Command-line arg '--baseline-threshold': {args.baseline_threshold}")
        
        # Get baseline correction parameters from command line args, config, or use defaults
        if args.baseline_threshold is not None:
            baseline_threshold = args.baseline_threshold
            print(f"DEBUG: Using command-line argument value: {baseline_threshold}")
        else:
            baseline_threshold = config.get('baseline_correction_threshold', 7.5)
            if config_threshold == 'NOT FOUND':
                print(f"DEBUG: Config key not found, using default: {baseline_threshold}")
            else:
                print(f"DEBUG: Using config value: {baseline_threshold}")
        
        print(f"DEBUG: Final baseline_threshold value being used: {baseline_threshold}")
        baseline_window = args.baseline_window if args.baseline_window is not None else config.get('baseline_correction_window', 15)
        baseline_smooth_first = False if args.no_baseline_smooth else config.get('baseline_correction_smooth_first', True)
        baseline_smooth_window = args.baseline_smooth_window if args.baseline_smooth_window is not None else config.get('baseline_correction_smooth_window', 11)
        baseline_smooth_poly = args.baseline_smooth_poly if args.baseline_smooth_poly is not None else config.get('baseline_correction_smooth_poly_order', 2)
        # Default to True if neither flag is provided, otherwise use the flag value
        # Check sys.argv to see which flag was explicitly provided
        if '--no-correct-smoothed' in sys.argv:
            baseline_correct_smoothed = False
        elif '--correct-smoothed' in sys.argv:
            baseline_correct_smoothed = True
        else:
            # Neither flag provided, default to True
            baseline_correct_smoothed = config.get('baseline_correction_correct_smoothed', True)
        
        # Get spike removal parameters
        remove_spikes = args.remove_spikes if hasattr(args, 'remove_spikes') else config.get('remove_spikes', False)
        spike_window = args.spike_window if hasattr(args, 'spike_window') else config.get('spike_window', 5)
        spike_threshold = args.spike_threshold if hasattr(args, 'spike_threshold') else config.get('spike_threshold', 3.0)
        
        # Get cumulative jump detection parameters
        detect_cumulative_jumps = args.detect_cumulative_jumps if hasattr(args, 'detect_cumulative_jumps') else config.get('detect_cumulative_jumps', True)
        cumulative_window = args.cumulative_window if hasattr(args, 'cumulative_window') else config.get('cumulative_window', 5)
        
        print(f"Baseline correction parameters:")
        print(f"  Threshold multiplier: {baseline_threshold}")
        print(f"  Window size: {baseline_window}")
        print(f"  Smooth first: {baseline_smooth_first}")
        if baseline_smooth_first:
            print(f"  Smooth window: {baseline_smooth_window}")
            print(f"  Smooth poly order: {baseline_smooth_poly}")
        print(f"  Correct smoothed data: {baseline_correct_smoothed}")
        print(f"  Remove spikes: {remove_spikes}")
        if remove_spikes:
            print(f"  Spike detection window: {spike_window}")
            print(f"  Spike detection threshold (MAD): {spike_threshold}")
        print(f"  Detect cumulative jumps: {detect_cumulative_jumps}")
        if detect_cumulative_jumps:
            print(f"  Cumulative window size: {cumulative_window}")
        
        results_df, jump_info_dict = apply_baseline_correction_v3(
            results_df,
            threshold_multiplier=baseline_threshold,
            window_size=baseline_window,
            smooth_first=baseline_smooth_first,
            smooth_window=baseline_smooth_window,
            smooth_poly_order=baseline_smooth_poly,
            correct_smoothed=baseline_correct_smoothed,
            remove_spikes=remove_spikes,
            spike_window=spike_window,
            spike_threshold=spike_threshold,
            detect_cumulative_jumps=detect_cumulative_jumps,
            cumulative_window=cumulative_window
        )
        
        # Print jump information
        print("\n=== Jump Detection Summary ===")
        if detect_cumulative_jumps:
            print(f"Jump detection mode: Individual + Cumulative (window={cumulative_window})")
        else:
            print("Jump detection mode: Individual only")
        print()
        
        for col, jump_info in jump_info_dict.items():
            n_jumps = jump_info['n_jumps']
            is_reference = jump_info.get('is_reference', False)
            used_fluorescence_jumps = jump_info.get('used_fluorescence_jumps', False)
            reference_fluo_col = jump_info.get('reference_fluorescence_column', None)
            
            # Only show jump details for fluorescence (reference signal)
            # Skip G-band and Raman peak jump details
            if not is_reference:
                if used_fluorescence_jumps and reference_fluo_col:
                    info_text = f" [Using jump points from {reference_fluo_col}]"
                else:
                    info_text = " [Signal-specific detection]"
                print(f"{col}: {n_jumps} jump(s) detected{info_text}")
                continue
            
            # For fluorescence (reference signal), show detailed jump information
            info_text = " [Reference signal - jump points used for related signals]"
            print(f"{col}: {n_jumps} jump(s) detected{info_text}")
            if jump_info['jump_info']:
                for info in jump_info['jump_info']:
                    idx = info['index']
                    scan_num = results_df.index[idx] if idx < len(results_df) else 'N/A'
                    jump_type = info.get('jump_type', 'unknown')
                    jump_type_label = 'Individual' if jump_type == 'individual' else 'Cumulative' if jump_type == 'cumulative' else 'Unknown'
                    print(f"  - Index {idx} (Scan: {scan_num}): step_change={info['step_change']:.4f} [{jump_type_label}]")
        
        # Calculate ratios
        print("\n=== Calculating ratios ===")
        results_df = calculate_ratios(results_df)
        
        # Apply baseline correction for long-term drift removal
        print("\n=== Applying baseline correction (long-term drift removal) ===")
        light_cycle = config.get('light_cycle', 'Constant')
        light_transition = parse_light_transition_config(config)
        shade_transition = parse_shade_transition_config(config)
        treatment_events = parse_treatment_events_config(config)
        results_df = apply_als_and_gaussian_v3(
            results_df, 
            config, 
            output_dir, 
            start_datetime=None, 
            end_datetime=None, 
            light_cycle=light_cycle,
            light_transition=light_transition,
            shade_transition=shade_transition,
            treatment_events=treatment_events
        )
        
        # Print results summary (CSV will be saved at the end)
        print(f"\n[Batch] Results summary:")
        print(results_df.head())
        print(f"\n[Batch] Total scans processed: {len(results_df)}")
        
        # Create plots
        print("\n=== Creating plots ===")
        
        # Plot representative Raman spectrum from first processed scan
        if summaries:
            first_summary = summaries[0]
            print("\n=== Creating representative Raman spectrum plot ===")
            plot_representative_raman_spectrum(first_summary, output_dir, config)
        
        # Plot normalized baseline correction comparison
        plot_normalized_baseline_correction_comparison(results_df, output_dir, config, light_cycle, jump_info_dict)
        
        # Plot normalized smoothing comparison (before/after smoothing)
        plot_normalized_smoothing_comparison(results_df, output_dir, config, light_cycle, jump_info_dict)
        
        # Plot ratio timeseries (stacked, baseline corrected only)
        plot_ratio_timeseries(results_df, output_dir, config, light_cycle)
        
        # Compute diurnal averages
        print("\n=== Computing diurnal averages ===")
        results_df = compute_diurnal_average_v3(
            results_df,
            config,
            output_dir,
            start_datetime=None,
            end_datetime=None,
            light_cycle=light_cycle
        )
        
        # Compute Fourier Transform analysis
        print("\n=== Computing Fourier Transform analysis ===")
        fft_results = compute_fourier_transform_v3(
            results_df,
            config,
            output_dir,
            start_datetime=None,
            end_datetime=None,
            light_cycle=light_cycle,
            light_transition=light_transition,
            shade_transition=shade_transition,
            treatment_events=treatment_events
        )
        
        # Save final results CSV (with all processing steps completed)
        csv_path_final = output_dir / "batch_summary_v3_final.csv"
        results_df.to_csv(csv_path_final)
        print(f"[Batch] Final results saved to: {csv_path_final}")
        
        print(f"\n[Batch] All outputs saved to: {output_dir}")
    else:
        # Single scan testing mode
        print("\n=== Running single scan test v3 ===")
        available_scans = sorted(raman_df["Scan Number"].unique())
        if not available_scans:
            print("ERROR: No scans available in dataset.")
            return
        
        # Use middle scan as default
        scan_number = available_scans[len(available_scans) // 2]
        
        if scan_number not in available_scans:
            print(f"ERROR: Scan {scan_number} not found! Available scans: {available_scans[:10]}{'...' if len(available_scans) > 10 else ''}")
            return
        
        summary = process_scan_v2(
            scan_number=scan_number,
            raman_df=raman_df,
            config=config,
            verbose=True
        )
        
        if summary:
            print("\n=== Processing Summary ===")
            for key, value in summary.items():
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()
