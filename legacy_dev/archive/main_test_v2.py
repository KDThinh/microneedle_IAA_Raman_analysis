"""
Entry point for exploratory testing and optimization of new processing methods v2.
This is separate from the main production workflow to keep main.py clean.

Revised Workflow:
1. Start at 250 cm-1
2. Savitzky-Golay smoothing for each scan
3. Keep original smoothed scan and normalized smoothed scan (normalized to average background value between 250 cm-1 to 1250 cm-1)
4. Lieberfit the whole spectrum for each scan (original smoothed and normalized smoothed)
5. Detect the Raman peak at around 850 cm-1 and 1600 cm-1 from the lieberfit-corrected scan (for both original smoothed and normalized smoothed)
6. Perform Lorentzian fitting and calculate the AUC for each Raman peak for each scan and each case (original smoothed and normalized smoothed)
7. Calculate AUC fluorescence (without background correction) from 1250 cm-1 onward from the original smoothed and normalized smoothed scan (not from the lieberfit), subtracted the calculated area of G-band from Lorentzian fitting
8. Aggregate data into batch_summary_data with specific columns
9. Add baseline_corrected for fluorescence intensity, G-band, and Raman peak at 850 cm-1 (for each case), using the approach from fix_baseline_shifts_v2.py
10. For smoothed data set, provide the combined plot of fluorescent intensity, G-band, and Raman peak before and after baseline corrected. Do the same for normalized plot.
11. For --batch, make the default to skip the first 500 scans.

Usage examples:
    # Single scan test with optimization
    python main_test_v2.py --optimize

    # Single scan test with custom parameters
    python main_test_v2.py --scan 100 --order 6 --iter 150

    # Batch processing (default skips first 500 scans)
    python main_test_v2.py --batch --scans "100,200,300" --max-scans 10

    # Batch with optimization
    python main_test_v2.py --batch --optimize --max-scans 5
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config_loader import load_profile_config
from pipeline.ingestion import load_raman_dataset
from pipeline.utils import add_day_night_shading, create_dir_if_needed, lieberfit
from scripts.fix_baseline_shifts_v2 import correct_baseline_shifts, correct_baseline_shifts_with_jump_indices
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.dates import DateFormatter, DayLocator

DEFAULT_CONFIG_PATH = str((PROJECT_ROOT / "config" / "pipeline.yml").resolve())
DEFAULT_PROFILE = "bok_choy_control_6to22_run1"


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
        Wavenumber values
    intensities : array
        Intensity values
    wavenumber_min : float
        Minimum wavenumber for search range
    wavenumber_max : float
        Maximum wavenumber for search range
    prominence_factor : float
        Minimum peak prominence as fraction of max intensity in range
    width_min : float
        Minimum peak width in cm^-1
    width_max : float
        Maximum peak width in cm^-1
    
    Returns:
    --------
    peak_info : dict or None
        Dictionary with peak information or None if not found
    """
    # Create mask for the search range
    mask = (wavenumbers >= wavenumber_min) & (wavenumbers <= wavenumber_max)
    
    if not np.any(mask):
        return None
    
    wavenumbers_range = wavenumbers[mask]
    intensities_range = intensities[mask]
    
    if len(intensities_range) < 3:
        return None
    
    # Find peaks using scipy.signal.find_peaks
    max_intensity = np.max(intensities_range)
    min_intensity = np.min(intensities_range)
    mean_intensity = np.mean(intensities_range)
    std_intensity = np.std(intensities_range)
    
    prominence_threshold_relative = (max_intensity - min_intensity) * prominence_factor
    prominence_threshold_absolute = mean_intensity + 2 * std_intensity
    prominence_threshold = max(prominence_threshold_relative, prominence_threshold_absolute)
    
    wavenumber_spacing = np.mean(np.diff(wavenumbers_range))
    width_min_idx = max(1, int(width_min / wavenumber_spacing))
    width_max_idx = min(len(intensities_range) // 2, int(width_max / wavenumber_spacing))
    
    try:
        peaks, properties = find_peaks(
            intensities_range,
            prominence=prominence_threshold,
            width=(width_min_idx, width_max_idx),
            height=min_intensity + prominence_threshold * 0.5
        )
        
        if len(peaks) == 0:
            # Fallback to argmax
            peak_idx = np.argmax(intensities_range)
        else:
            # Select peak with highest intensity
            peak_idx = peaks[np.argmax(intensities_range[peaks])]
        
        peak_wavenumber = wavenumbers_range[peak_idx]
        peak_intensity = intensities_range[peak_idx]
        
        # Fit Lorentzian around the peak
        estimated_width = 15.0
        fit_window = max(5, int(2.5 * estimated_width / wavenumber_spacing))
        fit_window = min(fit_window, int(40 / wavenumber_spacing))
        start_idx = max(0, peak_idx - fit_window)
        end_idx = min(len(intensities_range), peak_idx + fit_window + 1)
        
        fit_wavenumbers = wavenumbers_range[start_idx:end_idx]
        fit_intensities = intensities_range[start_idx:end_idx]
        
        initial_amplitude = peak_intensity - np.min(fit_intensities)
        initial_center = peak_wavenumber
        initial_width = 10.0
        initial_offset = np.min(fit_intensities)
        
        try:
            popt, _ = curve_fit(
                lorentzian, fit_wavenumbers, fit_intensities,
                p0=[initial_amplitude, initial_center, initial_width, initial_offset],
                bounds=([0, wavenumber_min, 2, -np.inf], 
                       [np.inf, wavenumber_max, 50, np.inf])
            )
            
            fitted_center = popt[1]
            fitted_amplitude = popt[0]
            fitted_width = abs(popt[2])
            
            # Calculate peak area
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
    Process a single scan using the revised workflow v2.
    
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
    
    # Step 4: Lieberfit the whole spectrum for both original smoothed and normalized smoothed
    poly_order = config.get('poly_order', 5)
    tot_iter = config.get('tot_iter', 100)
    
    corrected_smooth, baseline_smooth = lieberfit(intensities_smooth, order=poly_order, tot_iter=tot_iter)
    corrected_normalized, baseline_normalized = lieberfit(intensities_normalized_smooth, order=poly_order, tot_iter=tot_iter)
    
    # Step 5 & 6: Detect Raman peaks at ~850 cm-1 and ~1600 cm-1 (G-band) from lieberfit-corrected scans
    # For original smoothed
    raman_peak_850_raw = find_peak_lorentzian(wavenumbers, corrected_smooth, 800, 900, width_min=5, width_max=50)
    gband_peak_1600_raw = find_peak_lorentzian(wavenumbers, corrected_smooth, 1550, 1650, width_min=5, width_max=50)
    
    # For normalized smoothed
    raman_peak_850_normalized = find_peak_lorentzian(wavenumbers, corrected_normalized, 800, 900, width_min=5, width_max=50)
    gband_peak_1600_normalized = find_peak_lorentzian(wavenumbers, corrected_normalized, 1550, 1650, width_min=5, width_max=50)
    
    # Step 7: Calculate AUC fluorescence from 1250 cm-1 onward from smoothed scan (not lieberfit)
    # Subtract G-band area from Lorentzian fitting
    fluo_mask = wavenumbers >= 1250
    
    # For original smoothed
    if np.any(fluo_mask):
        fluo_auc_raw = trapezoid(intensities_smooth[fluo_mask], x=wavenumbers[fluo_mask])
        gband_area_raw = gband_peak_1600_raw.get('area', 0) if gband_peak_1600_raw else 0
        fluorescence_intensity_raw = fluo_auc_raw - gband_area_raw
    else:
        fluorescence_intensity_raw = np.nan
    
    # For normalized smoothed
    if np.any(fluo_mask):
        fluo_auc_normalized = trapezoid(intensities_normalized_smooth[fluo_mask], x=wavenumbers[fluo_mask])
        gband_area_normalized = gband_peak_1600_normalized.get('area', 0) if gband_peak_1600_normalized else 0
        fluorescence_intensity_normalized = fluo_auc_normalized - gband_area_normalized
    else:
        fluorescence_intensity_normalized = np.nan
    
    # Build summary dictionary
    summary = {
        'scan_number': scan_number,
        'datetime': datetime_val,
        'seconds': seconds,
        
        # Average background intensity (250-1250 cm-1)
        'average_background_intensity': bg_intensity_avg_250_1250,
        
        # Original smoothed data
        'raw_fluorescence_intensity': fluorescence_intensity_raw,
        'raw_gband_intensity': gband_peak_1600_raw.get('intensity', np.nan) if gband_peak_1600_raw else np.nan,
        'raw_gband_area': gband_peak_1600_raw.get('area', np.nan) if gband_peak_1600_raw else np.nan,
        'raw_gband_wavenumber': gband_peak_1600_raw.get('wavenumber', np.nan) if gband_peak_1600_raw else np.nan,
        'raw_raman_peak_850_intensity': raman_peak_850_raw.get('intensity', np.nan) if raman_peak_850_raw else np.nan,
        'raw_raman_peak_850_area': raman_peak_850_raw.get('area', np.nan) if raman_peak_850_raw else np.nan,
        'raw_raman_peak_850_wavenumber': raman_peak_850_raw.get('wavenumber', np.nan) if raman_peak_850_raw else np.nan,
        
        # Normalized smoothed data (normalized background should be ~1.0)
        'normalized_background_intensity': 1.0 if bg_intensity_avg_250_1250 > 0 else np.nan,
        'normalized_fluorescence_intensity': fluorescence_intensity_normalized,
        'normalized_gband_intensity': gband_peak_1600_normalized.get('intensity', np.nan) if gband_peak_1600_normalized else np.nan,
        'normalized_gband_area': gband_peak_1600_normalized.get('area', np.nan) if gband_peak_1600_normalized else np.nan,
        'normalized_gband_wavenumber': gband_peak_1600_normalized.get('wavenumber', np.nan) if gband_peak_1600_normalized else np.nan,
        'normalized_raman_peak_850_intensity': raman_peak_850_normalized.get('intensity', np.nan) if raman_peak_850_normalized else np.nan,
        'normalized_raman_peak_850_area': raman_peak_850_normalized.get('area', np.nan) if raman_peak_850_normalized else np.nan,
        'normalized_raman_peak_850_wavenumber': raman_peak_850_normalized.get('wavenumber', np.nan) if raman_peak_850_normalized else np.nan,
    }
    
    return summary


def aggregate_summaries_to_dataframe_v2(summaries, raman_df, x_axis_type='datetime'):
    """
    Aggregate summaries from batch processing into a DataFrame.
    
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
        DataFrame with datetime or scan_number index and metrics columns
    """
    if not summaries:
        return pd.DataFrame()
    
    scan_to_datetime = raman_df.groupby('Scan Number').apply(lambda x: x.index[0]).to_dict()
    
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
            
            # Original smoothed data
            'Raw_Fluorescence_Intensity': summary.get('raw_fluorescence_intensity', np.nan),
            'Raw_Gband_Intensity': summary.get('raw_gband_intensity', np.nan),
            'Raw_Gband_Area': summary.get('raw_gband_area', np.nan),
            'Raw_Gband_Wavenumber': summary.get('raw_gband_wavenumber', np.nan),
            'Raw_Raman_Peak_850_Intensity': summary.get('raw_raman_peak_850_intensity', np.nan),
            'Raw_Raman_Peak_850_Area': summary.get('raw_raman_peak_850_area', np.nan),
            'Raw_Raman_Peak_850_Wavenumber': summary.get('raw_raman_peak_850_wavenumber', np.nan),
            
            # Normalized smoothed data
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


def apply_baseline_correction_v2(df, columns_to_correct=None, 
                                threshold_multiplier=5.0, window_size=5,
                                smooth_first=True, smooth_window=11, smooth_poly_order=2,
                                correct_smoothed=False):
    """
    Apply baseline correction using fix_baseline_shifts_v2.py approach.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with time series data
    columns_to_correct : list, optional
        List of column names to correct. If None, corrects default columns.
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
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with corrected columns added
    dict
        Dictionary mapping column names to jump information
    """
    if columns_to_correct is None:
        columns_to_correct = [
            'Raw_Fluorescence_Intensity',
            'Raw_Gband_Area',  # Changed from Intensity to Area
            'Raw_Raman_Peak_850_Area',  # Changed from Intensity to Area
            'Normalized_Fluorescence_Intensity',
            'Normalized_Gband_Area',  # Changed from Intensity to Area
            'Normalized_Raman_Peak_850_Area'  # Changed from Intensity to Area
        ]
    
    corrected_df = df.copy()
    all_jump_info = {}
    
    # Step 1: Process fluorescence signals first to detect jump points
    # These jump points will be used as reference for G-band and Raman peak
    fluorescence_columns = [
        'Raw_Fluorescence_Intensity',
        'Normalized_Fluorescence_Intensity'
    ]
    
    # Store jump indices from fluorescence signals
    fluorescence_jump_indices = {}
    
    for fluo_col in fluorescence_columns:
        if fluo_col not in df.columns:
            continue
        
        signal = df[fluo_col].values
        mask = ~np.isnan(signal)
        
        if np.sum(mask) < 3:
            continue
        
        # Detect jumps in fluorescence signal
        corrected, jump_indices, jump_info, smoothed = correct_baseline_shifts(
            signal, threshold_multiplier=threshold_multiplier, window_size=window_size,
            smooth_first=smooth_first, smooth_window=smooth_window, 
            smooth_poly_order=smooth_poly_order, correct_smoothed=correct_smoothed
        )
        
        # Store jump indices for use with related signals
        fluorescence_jump_indices[fluo_col] = jump_indices
        
        # Store corrected fluorescence signal
        corrected_col_name = f'{fluo_col}_BaselineCorrected'
        corrected_df[corrected_col_name] = corrected
        
        # If correcting smoothed data, also store the smoothed signal
        if correct_smoothed:
            smoothed_col_name = f'{fluo_col}_Smoothed'
            corrected_df[smoothed_col_name] = smoothed
        
        # Store jump information
        all_jump_info[fluo_col] = {
            'jump_indices': jump_indices,
            'jump_info': jump_info,
            'n_jumps': len(jump_indices),
            'used_shared_mad': False,
            'shared_mad_value': None,
            'correct_smoothed': correct_smoothed,
            'is_reference': True  # Mark as reference signal
        }
    
    # Step 2: Process G-band and Raman peak signals using fluorescence jump points
    for col in columns_to_correct:
        if col not in df.columns:
            continue
        
        # Skip fluorescence columns (already processed)
        if col in fluorescence_columns:
            continue
        
        signal = df[col].values
        mask = ~np.isnan(signal)
        
        if np.sum(mask) < 3:
            continue
        
        # Check if this is a G-band or Raman peak signal
        use_fluorescence_jumps = False
        reference_fluo_col = None
        
        if 'Gband' in col or 'Raman_Peak_850' in col:
            # Find corresponding fluorescence signal (Raw or Normalized)
            if 'Raw_' in col:
                reference_fluo_col = 'Raw_Fluorescence_Intensity'
            elif 'Normalized_' in col:
                reference_fluo_col = 'Normalized_Fluorescence_Intensity'
            
            # Use fluorescence jump indices if available
            if reference_fluo_col and reference_fluo_col in fluorescence_jump_indices:
                use_fluorescence_jumps = True
                fluo_jump_indices = fluorescence_jump_indices[reference_fluo_col]
        
        # Apply baseline correction
        if use_fluorescence_jumps:
            # Use jump points from fluorescence signal
            corrected, jump_info, smoothed = correct_baseline_shifts_with_jump_indices(
                signal, fluo_jump_indices, window_size=window_size,
                smooth_first=smooth_first, smooth_window=smooth_window,
                smooth_poly_order=smooth_poly_order, correct_smoothed=correct_smoothed
            )
            jump_indices = fluo_jump_indices
        else:
            # Apply baseline correction with signal-specific detection
            corrected, jump_indices, jump_info, smoothed = correct_baseline_shifts(
                signal, threshold_multiplier=threshold_multiplier, window_size=window_size,
                smooth_first=smooth_first, smooth_window=smooth_window, 
                smooth_poly_order=smooth_poly_order, correct_smoothed=correct_smoothed
            )
        
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
            'reference_fluorescence_column': reference_fluo_col if use_fluorescence_jumps else None,
            'correct_smoothed': correct_smoothed,
            'is_reference': False
        }
    
    return corrected_df, all_jump_info


def plot_baseline_correction_comparison(df, output_dir, config, light_cycle=None, jump_info_dict=None):
    """
    Create combined plots showing fluorescence intensity, G-band, and Raman peak 
    before and after baseline correction for both raw and normalized data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with data and corrected columns
    output_dir : Path
        Output directory
    config : dict
        Configuration dictionary
    light_cycle : str, optional
        Light cycle for shading
    jump_info_dict : dict, optional
        Dictionary mapping column names to jump information (from apply_baseline_correction_v2)
    """
    if light_cycle is None:
        light_cycle = config.get('light_cycle', 'Constant')
    
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    output_dir_str = str(output_dir)
    create_dir_if_needed(output_dir_str)
    
    # Raw (Original Smoothed) data plots
    fig_raw, axes_raw = plt.subplots(3, 1, figsize=(14, 12))
    
    # Fluorescence
    if 'Raw_Fluorescence_Intensity' in df.columns:
        # Determine what to plot as "Before Correction"
        # If corrected_smoothed was used, show smoothed data; otherwise show original
        before_col = 'Raw_Fluorescence_Intensity_Smoothed' if ('Raw_Fluorescence_Intensity_Smoothed' in df.columns and 
                                                               jump_info_dict and 
                                                               'Raw_Fluorescence_Intensity' in jump_info_dict and
                                                               jump_info_dict['Raw_Fluorescence_Intensity'].get('correct_smoothed', False)) else 'Raw_Fluorescence_Intensity'
        
        if before_col in df.columns:
            before_label = 'Before Correction (Smoothed)' if before_col == 'Raw_Fluorescence_Intensity_Smoothed' else 'Before Correction'
            axes_raw[0].plot(df.index, df[before_col], 'o-', 
                           label=before_label, alpha=0.7, markersize=3)
        
        if 'Raw_Fluorescence_Intensity_BaselineCorrected' in df.columns:
            after_label = 'After Correction (Corrected Smoothed)' if (jump_info_dict and 
                                                                      'Raw_Fluorescence_Intensity' in jump_info_dict and
                                                                      jump_info_dict['Raw_Fluorescence_Intensity'].get('correct_smoothed', False)) else 'After Correction'
            axes_raw[0].plot(df.index, df['Raw_Fluorescence_Intensity_BaselineCorrected'], 's-', 
                           label=after_label, alpha=0.7, markersize=3)
        
        # Mark jump points if available
        if jump_info_dict and 'Raw_Fluorescence_Intensity' in jump_info_dict:
            jump_info = jump_info_dict['Raw_Fluorescence_Intensity']['jump_info']
            if jump_info:
                jump_indices = [info['index'] for info in jump_info]
                jump_x = df.index[jump_indices]
                jump_y_before = df[before_col].iloc[jump_indices].values if before_col in df.columns else df['Raw_Fluorescence_Intensity'].iloc[jump_indices].values
                jump_y_after = df['Raw_Fluorescence_Intensity_BaselineCorrected'].iloc[jump_indices].values if 'Raw_Fluorescence_Intensity_BaselineCorrected' in df.columns else jump_y_before
                axes_raw[0].scatter(jump_x, jump_y_before, color='red', s=150, 
                                  zorder=5, marker='x', linewidths=2, label=f'Jump points (n={len(jump_info)})')
                axes_raw[0].scatter(jump_x, jump_y_after, color='darkred', s=150, 
                                  zorder=5, marker='+', linewidths=2, alpha=0.7)
        
        axes_raw[0].set_ylabel('Fluorescence Intensity', fontsize=12)
        axes_raw[0].set_title('Raw (Original Smoothed) - Fluorescence Intensity', fontsize=13, fontweight='bold')
        axes_raw[0].legend()
        axes_raw[0].grid(True, alpha=0.3)
    
    # G-band (using Area instead of Intensity)
    if 'Raw_Gband_Area' in df.columns:
        axes_raw[1].plot(df.index, df['Raw_Gband_Area'], 'o-', 
                       label='Before Correction', alpha=0.7, markersize=3)
        if 'Raw_Gband_Area_BaselineCorrected' in df.columns:
            axes_raw[1].plot(df.index, df['Raw_Gband_Area_BaselineCorrected'], 's-', 
                           label='After Correction', alpha=0.7, markersize=3)
        
        # Mark jump points if available
        if jump_info_dict and 'Raw_Gband_Area' in jump_info_dict:
            jump_info = jump_info_dict['Raw_Gband_Area']['jump_info']
            if jump_info:
                jump_indices = [info['index'] for info in jump_info]
                jump_x = df.index[jump_indices]
                jump_y_before = df['Raw_Gband_Area'].iloc[jump_indices].values
                jump_y_after = df['Raw_Gband_Area_BaselineCorrected'].iloc[jump_indices].values if 'Raw_Gband_Area_BaselineCorrected' in df.columns else jump_y_before
                axes_raw[1].scatter(jump_x, jump_y_before, color='red', s=150, 
                                  zorder=5, marker='x', linewidths=2, label=f'Jump points (n={len(jump_info)})')
                axes_raw[1].scatter(jump_x, jump_y_after, color='darkred', s=150, 
                                  zorder=5, marker='+', linewidths=2, alpha=0.7)
        
        axes_raw[1].set_ylabel('G-band Area (Lorentzian)', fontsize=12)
        axes_raw[1].set_title('Raw (Original Smoothed) - G-band Area', fontsize=13, fontweight='bold')
        axes_raw[1].legend()
        axes_raw[1].grid(True, alpha=0.3)
    
    # Raman peak 850 (using Area instead of Intensity)
    if 'Raw_Raman_Peak_850_Area' in df.columns:
        # Determine what to plot as "Before Correction"
        before_col = 'Raw_Raman_Peak_850_Area_Smoothed' if ('Raw_Raman_Peak_850_Area_Smoothed' in df.columns and 
                                                             jump_info_dict and 
                                                             'Raw_Raman_Peak_850_Area' in jump_info_dict and
                                                             jump_info_dict['Raw_Raman_Peak_850_Area'].get('correct_smoothed', False)) else 'Raw_Raman_Peak_850_Area'
        
        if before_col in df.columns:
            before_label = 'Before Correction (Smoothed)' if before_col == 'Raw_Raman_Peak_850_Area_Smoothed' else 'Before Correction'
            axes_raw[2].plot(df.index, df[before_col], 'o-', 
                            label=before_label, alpha=0.7, markersize=3)
        
        if 'Raw_Raman_Peak_850_Area_BaselineCorrected' in df.columns:
            after_label = 'After Correction (Corrected Smoothed)' if (jump_info_dict and 
                                                                      'Raw_Raman_Peak_850_Area' in jump_info_dict and
                                                                      jump_info_dict['Raw_Raman_Peak_850_Area'].get('correct_smoothed', False)) else 'After Correction'
            axes_raw[2].plot(df.index, df['Raw_Raman_Peak_850_Area_BaselineCorrected'], 's-', 
                           label=after_label, alpha=0.7, markersize=3)
        
        # Mark jump points if available
        if jump_info_dict and 'Raw_Raman_Peak_850_Area' in jump_info_dict:
            jump_info = jump_info_dict['Raw_Raman_Peak_850_Area']['jump_info']
            if jump_info:
                jump_indices = [info['index'] for info in jump_info]
                jump_x = df.index[jump_indices]
                jump_y_before = df[before_col].iloc[jump_indices].values if before_col in df.columns else df['Raw_Raman_Peak_850_Area'].iloc[jump_indices].values
                jump_y_after = df['Raw_Raman_Peak_850_Area_BaselineCorrected'].iloc[jump_indices].values if 'Raw_Raman_Peak_850_Area_BaselineCorrected' in df.columns else jump_y_before
                axes_raw[2].scatter(jump_x, jump_y_before, color='red', s=150, 
                                  zorder=5, marker='x', linewidths=2, label=f'Jump points (n={len(jump_info)})')
                axes_raw[2].scatter(jump_x, jump_y_after, color='darkred', s=150, 
                                  zorder=5, marker='+', linewidths=2, alpha=0.7)
        
        axes_raw[2].set_ylabel('Raman Peak 850 Area (Lorentzian)', fontsize=12)
        axes_raw[2].set_title('Raw (Original Smoothed) - Raman Peak 850 Area', fontsize=13, fontweight='bold')
        axes_raw[2].legend()
        axes_raw[2].grid(True, alpha=0.3)
    
    if x_axis_type == 'datetime':
        for ax in axes_raw:
            add_day_night_shading(ax, df.index.min(), df.index.max(), light_cycle=light_cycle)
            ax.set_xlabel('Date time (MM-DD HH)', fontsize=12)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
        fig_raw.autofmt_xdate()
    else:
        for ax in axes_raw:
            ax.set_xlabel('Scan Number', fontsize=12)
    
    plt.tight_layout()
    save_path_raw = Path(output_dir_str) / f'raw_baseline_correction_comparison_{datetime.now().strftime("%Y-%m-%d")}.png'
    plt.savefig(str(save_path_raw), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Raw baseline correction comparison plot saved to: {save_path_raw}")
    
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
        
        axes_norm[0].set_ylabel('Fluorescence Intensity', fontsize=12)
        axes_norm[0].set_title('Normalized Smoothed - Fluorescence Intensity', fontsize=13, fontweight='bold')
        axes_norm[0].legend()
        axes_norm[0].grid(True, alpha=0.3)
    
    # G-band (using Area instead of Intensity)
    if 'Normalized_Gband_Area' in df.columns:
        # Determine what to plot as "Before Correction"
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
        axes_norm[1].set_title('Normalized Smoothed - G-band Area', fontsize=13, fontweight='bold')
        axes_norm[1].legend()
        axes_norm[1].grid(True, alpha=0.3)
    
    # Raman peak 850 (using Area instead of Intensity)
    if 'Normalized_Raman_Peak_850_Area' in df.columns:
        # Determine what to plot as "Before Correction"
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
        axes_norm[2].set_title('Normalized Smoothed - Raman Peak 850 Area', fontsize=13, fontweight='bold')
        axes_norm[2].legend()
        axes_norm[2].grid(True, alpha=0.3)
    
    if x_axis_type == 'datetime':
        for ax in axes_norm:
            add_day_night_shading(ax, df.index.min(), df.index.max(), light_cycle=light_cycle)
            ax.set_xlabel('Date time (MM-DD HH)', fontsize=12)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
        fig_norm.autofmt_xdate()
    else:
        for ax in axes_norm:
            ax.set_xlabel('Scan Number', fontsize=12)
    
    plt.tight_layout()
    save_path_norm = Path(output_dir_str) / f'normalized_baseline_correction_comparison_{datetime.now().strftime("%Y-%m-%d")}.png'
    plt.savefig(str(save_path_norm), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Normalized baseline correction comparison plot saved to: {save_path_norm}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exploratory testing and optimization for new Lieberfit processing methods v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single scan test with optimization
  python main_test_v2.py --optimize

  # Single scan test with custom parameters
  python main_test_v2.py --scan 100 --order 6 --iter 150

  # Batch processing on specific scans (default skips first 500 scans)
  python main_test_v2.py --batch --scans "100,200,300" --max-scans 10

  # Batch with optimization
  python main_test_v2.py --batch --optimize --max-scans 5
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
    
    # Single scan testing
    parser.add_argument(
        "--scan",
        type=int,
        default=None,
        help="Single scan number to test. If not provided, uses default (scan 100 or middle scan)."
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
        help="Comma-separated scan numbers for batch processing. Defaults to all scans (skipping first 500)."
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
        help="Number of scans to skip from the beginning in batch mode (default: 500)."
    )
    
    # Optimization
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Lieberfit parameter optimization (not implemented in v2 yet)."
    )
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
    
    # Baseline correction parameters
    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=7.5,
        help="MAD threshold multiplier for baseline correction jump detection (default: 7.5)."
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=15,
        help="Window size for baseline correction offset calculation (default: 15)."
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
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_raman_dataset(config)
    raman_df = dataset.spectra
    
    print(f"Loaded {len(raman_df)} scans")
    print(f"Wavenumber range: {raman_df.columns[2]} to {raman_df.columns[-1]}")
    print(f"Raw data file: {dataset.source_path}")
    
    # Set output directory
    if args.output_dir is None:
        raw_data_dir = Path(dataset.source_path).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.batch:
            output_dir = raw_data_dir / f"test_outputs_v2_batch_{timestamp}"
        else:
            output_dir = raw_data_dir / f"test_outputs_v2_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Pre-compute wavenumber arrays once
    wavenumbers_full = np.array([float(col) for col in raman_df.columns if col not in ['Scan Number', 'Seconds']])
    wavenumber_filter = wavenumbers_full >= 250
    wavenumbers = wavenumbers_full[wavenumber_filter]
    
    if args.batch:
        # Batch processing mode
        print("\n=== Running batch processing v2 ===")
        
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
            # Default: skip first 500 scans
            scan_list = list(available_scans[args.skip_scans:])
        
        if args.max_scans:
            scan_list = scan_list[:args.max_scans]
        
        if not scan_list:
            print("[Batch] No scans left to process after filtering.")
            return
        
        print(f"[Batch] Processing {len(scan_list)} scans")
        print(f"[Batch] Scan list: {scan_list[:10]}{'...' if len(scan_list) > 10 else ''}")
        
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
        
        # Aggregate summaries into DataFrame
        print("\n=== Aggregating results ===")
        x_axis_type = config.get('timeseries_x_axis', 'datetime')
        if x_axis_type not in ['datetime', 'scan_number']:
            print(f"Warning: Invalid x_axis_type '{x_axis_type}', using 'datetime'")
            x_axis_type = 'datetime'
        
        results_df = aggregate_summaries_to_dataframe_v2(summaries, raman_df, x_axis_type=x_axis_type)
        
        if results_df.empty:
            print("[Batch] Warning: No data to aggregate.")
            return
        
        # Apply baseline correction
        print("\n=== Applying baseline correction ===")
        # Get baseline correction parameters from command line args, config, or use defaults
        # Use command line args (which have defaults), or fall back to config if args are None
        baseline_threshold = args.baseline_threshold if args.baseline_threshold is not None else config.get('baseline_correction_threshold', 7.5)
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
        
        print(f"Baseline correction parameters:")
        print(f"  Threshold multiplier: {baseline_threshold}")
        print(f"  Window size: {baseline_window}")
        print(f"  Smooth first: {baseline_smooth_first}")
        if baseline_smooth_first:
            print(f"  Smooth window: {baseline_smooth_window}")
            print(f"  Smooth poly order: {baseline_smooth_poly}")
        print(f"  Correct smoothed data: {baseline_correct_smoothed}")
        
        results_df, jump_info_dict = apply_baseline_correction_v2(
            results_df,
            threshold_multiplier=baseline_threshold,
            window_size=baseline_window,
            smooth_first=baseline_smooth_first,
            smooth_window=baseline_smooth_window,
            smooth_poly_order=baseline_smooth_poly,
            correct_smoothed=baseline_correct_smoothed
        )
        
        # Print jump information
        print("\n=== Jump Detection Summary ===")
        for col, jump_info in jump_info_dict.items():
            n_jumps = jump_info['n_jumps']
            is_reference = jump_info.get('is_reference', False)
            used_fluorescence_jumps = jump_info.get('used_fluorescence_jumps', False)
            reference_fluo_col = jump_info.get('reference_fluorescence_column', None)
            
            if is_reference:
                info_text = " [Reference signal - jump points used for related signals]"
            elif used_fluorescence_jumps and reference_fluo_col:
                info_text = f" [Using jump points from {reference_fluo_col}]"
            else:
                info_text = " [Signal-specific detection]"
            
            print(f"{col}: {n_jumps} jump(s) detected{info_text}")
            if jump_info['jump_info']:
                for info in jump_info['jump_info']:
                    idx = info['index']
                    scan_num = results_df.index[idx] if idx < len(results_df) else 'N/A'
                    print(f"  - Index {idx} (Scan: {scan_num}): step_change={info['step_change']:.4f}")
        
        # Save aggregated results to CSV
        csv_path = output_dir / "batch_summary_data_v2.csv"
        results_df.to_csv(csv_path)
        print(f"[Batch] Results saved to: {csv_path}")
        print(f"\n[Batch] Results summary:")
        print(results_df.head())
        print(f"\n[Batch] Total scans processed: {len(results_df)}")
        
        # Create baseline correction comparison plots
        print("\n=== Creating baseline correction comparison plots ===")
        light_cycle = config.get('light_cycle', 'Constant')
        plot_baseline_correction_comparison(results_df, output_dir, config, light_cycle, jump_info_dict)
        
        print(f"\n[Batch] All outputs saved to: {output_dir}")
    else:
        # Single scan testing mode
        print("\n=== Running single scan test v2 ===")
        available_scans = sorted(raman_df['Scan Number'].unique())
        if not available_scans:
            print("ERROR: No scans available in dataset.")
            return
        
        scan_number = args.scan
        if scan_number is None:
            scan_number = 100 if 100 in available_scans else available_scans[len(available_scans) // 2]
        
        if scan_number not in available_scans:
            print(f"ERROR: Scan {scan_number} not found! Available scans: {available_scans[:10]}{'...' if len(available_scans) > 10 else ''}")
            return
        
        summary = process_scan_v2(
            scan_number=scan_number,
            raman_df=raman_df,
            config=config,
            cached_wavenumbers_full=wavenumbers_full,
            cached_wavenumber_filter=wavenumber_filter,
            cached_wavenumbers=wavenumbers,
            verbose=True
        )
        
        if summary:
            print("\n=== Processing Summary ===")
            for key, value in summary.items():
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()

