import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter1d
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from matplotlib.ticker import MultipleLocator, MaxNLocator, NullFormatter, FuncFormatter
from datetime import datetime, timedelta
from .utils import create_dir_if_needed, add_day_night_shading, remove_spikes_hampel, parse_light_transition_config, parse_shade_transition_config, parse_treatment_events_config

# Import from scripts (need to add project root to path)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # Go up from src/pipeline/ to project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from scripts.fix_baseline_shifts_v2 import correct_baseline_shifts, correct_baseline_shifts_with_jump_indices

def apply_als_and_gaussian(results_df, config, processed_dir, start_datetime=None, end_datetime=None, light_cycle='Constant'):
    """
    Apply ALS baseline correction and Gaussian smoothing to the Fluorescence to G-band Ratio.
    
    Parameters:
    results_df : pandas.DataFrame
        DataFrame with results including 'Fluorescence to G-band Ratio' and Datetime index.
    config : dict
        Configuration dictionary containing parameters like 'lam_als', 'p_als', 'niter_als', 'sigma_gaussian'.
    processed_dir : str
        Output directory for saving plots and updated CSV.
    start_datetime : datetime, optional
        Start datetime for filtering the data (inclusive).
    end_datetime : datetime, optional
        End datetime for filtering the data (inclusive).
    light_cycle : str
        Light cycle for shading ('Constant', '8to24', '6to22').
    """
    create_dir_if_needed(processed_dir)
    if start_datetime is not None and end_datetime is not None:
        results_df = results_df[(results_df.index >= start_datetime) & (results_df.index <= end_datetime)]
    lam_als = config.get('lam_als', 1e7)
    p_als = config.get('p_als', 0.001)
    niter_als = config.get('niter_als', 20)
    sigma_gaussian = config.get('sigma_gaussian', 5)
    print("Available columns:", results_df.columns.tolist())
    print("Index name:", results_df.index.name)
    print("Index sample:", results_df.index[:5])
    print("Index type and range:", type(results_df.index[0]), results_df.index.min(), results_df.index.max())
    results_df.columns = results_df.columns.str.strip()
    if 'Datetime' not in results_df.columns:
        results_df.index.name = 'Datetime'
        results_df.index = pd.to_datetime(results_df.index, errors='coerce')
    else:
        results_df['Datetime'] = pd.to_datetime(results_df['Datetime'], errors='coerce')
        results_df = results_df.set_index('Datetime')
    if results_df.index.hasnans or not pd.api.types.is_datetime64_any_dtype(results_df.index):
        print("Warning: Invalid or non-Datetime values detected in index. Dropping rows with NaT or converting.")
        results_df = results_df[results_df.index.notna()]
        if not pd.api.types.is_datetime64_any_dtype(results_df.index):
            results_df.index = pd.to_datetime(results_df.index, errors='coerce', origin='unix')
    results_df = results_df.dropna(subset=['Fluorescence to G-band Ratio'])
    datetimes = results_df.index
    ratios = results_df['Fluorescence to G-band Ratio'].values
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
    baseline_als_vals = baseline_als(ratios)
    corrected_als = ratios - baseline_als_vals
    results_df['Corrected Ratio (ALS)'] = corrected_als
    results_df['Smoothed Corrected (ALS)'] = gaussian_filter1d(results_df['Corrected Ratio (ALS)'], sigma=sigma_gaussian)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(datetimes, ratios, label='Original PL/G', color='m')
    ax.plot(datetimes, baseline_als_vals, label='ALS Baseline', color='red', linestyle='--')
    ax.plot(datetimes, corrected_als, label='Corrected (ALS)', color='green')
    ax.plot(datetimes, results_df['Smoothed Corrected (ALS)'], label='Smoothed Corrected (ALS)', color='black', linestyle=':')
    add_day_night_shading(ax, datetimes.min(), datetimes.max(), light_cycle=light_cycle)
    ax.set_xlabel('Datetime', fontsize=14)
    ax.set_ylabel('Fluorescence to G-band Ratio', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=12)
    fig.autofmt_xdate()
    save_path = os.path.join(processed_dir, f'ALS_Baseline_and_Gaussian_Smoothed_Ratio_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show(block=False)
    return results_df

def compute_diurnal_average(results_df, config, processed_dir, start_datetime=None, end_datetime=None):
    """
    Compute and plot the average diurnal cycle from the smoothed corrected ratio.
    """
    create_dir_if_needed(processed_dir)
    if start_datetime is not None and end_datetime is not None:
        results_df = results_df[(results_df.index >= start_datetime) & (results_df.index <= end_datetime)]
    bin_factor = config.get('bin_factor', 2)
    datetimes = results_df.index
    results_df['Hour of Day'] = datetimes.hour + datetimes.minute / 60
    results_df['Binned Hour'] = np.round(results_df['Hour of Day'] * bin_factor) / bin_factor
    daily_avg = results_df.groupby('Binned Hour')['Smoothed Corrected (ALS)'].agg(['mean', 'std', 'count']).reset_index()
    daily_avg['sem'] = daily_avg['std'] / np.sqrt(daily_avg['count'])
    daily_avg['Smoothed Avg'] = gaussian_filter1d(daily_avg['mean'], sigma=1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(daily_avg['Binned Hour'], daily_avg['Smoothed Avg'], yerr=daily_avg['sem'],
                marker='o', color='green', ecolor='lightgreen', capsize=3, label='Mean ± SEM')
    ax.set_xlabel('Hour of Day', fontsize=14)
    ax.set_ylabel('Average Gaussian-Smoothed Corrected Ratio', fontsize=14)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.grid(True)
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', labelsize=12)
    save_path = os.path.join(processed_dir, f'Average_Diurnal_Cycle_PL_to_G_Ratio_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show(block=False)
    results_df.reset_index().to_csv(os.path.join(processed_dir, f'corrected_raman_results_gaussian_{datetime.now().strftime("%Y-%m-%d")}.csv'), index=False)
    print(f"Analysis complete with Gaussian smoothing. Updated DataFrame saved as '{os.path.join(processed_dir, f'corrected_raman_results_gaussian_{datetime.now().strftime("%Y-%m-%d")}.csv')}'.")
    return results_df

def compute_fourier_transform(results_df, config, processed_dir, start_datetime=None, end_datetime=None):
    """
    Compute and plot the Fourier Transform of both the detrended original Fluorescence to G-band Ratio
    and the Smoothed Corrected Fluorescence to G-band Ratio.
    """
    create_dir_if_needed(processed_dir)
    if start_datetime is not None and end_datetime is not None:
        results_df = results_df[(results_df.index >= start_datetime) & (results_df.index <= end_datetime)]
    top_peaks = config.get('fft_top_peaks', 5)
    datetimes = results_df.index
    time_hours = (datetimes - datetimes.min()).total_seconds() / 3600.0
    sampling_interval = np.mean(np.diff(time_hours))
    sampling_rate = 1 / sampling_interval
    def compute_ft(signal, label_suffix):
        signal_detrended = signal - np.mean(signal)
        fft_result = np.fft.fft(signal_detrended)
        frequencies = np.fft.fftfreq(len(time_hours), d=sampling_interval)
        complete_ft_df = pd.DataFrame({
            'Frequency (cycles/hour)': frequencies,
            'Real': np.real(fft_result),
            'Imag': np.imag(fft_result),
            'Magnitude': np.abs(fft_result),
            'Phase': np.angle(fft_result)
        })
        complete_filename = f'complete_fft_{label_suffix}_{datetime.now().strftime("%Y-%m-%d")}.csv'
        complete_ft_df.to_csv(os.path.join(processed_dir, complete_filename), index=False)
        print(f"Complete FFT results for {label_suffix} saved to '{os.path.join(processed_dir, complete_filename)}'.")
        positive_mask = frequencies > 0
        positive_freqs = frequencies[positive_mask]
        magnitude = np.abs(fft_result)[positive_mask]
        peak_indices = np.argsort(magnitude)[-top_peaks:][::-1]
        peak_freqs = positive_freqs[peak_indices]
        peak_magnitudes = magnitude[peak_indices]
        peak_periods = 1 / peak_freqs
        fft_df = pd.DataFrame({
            'Frequency (cycles/hour)': peak_freqs,
            'Period (hours)': peak_periods,
            'Magnitude': peak_magnitudes
        })
        peaks_filename = f'fft_peaks_{label_suffix}_{datetime.now().strftime("%Y-%m-%d")}.csv'
        fft_df.to_csv(os.path.join(processed_dir, peaks_filename), index=False)
        print(f"FFT analysis for {label_suffix} complete. Peaks saved to '{os.path.join(processed_dir, peaks_filename)}'.")
        print(fft_df)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(positive_freqs, magnitude, color='blue')
        ax.scatter(peak_freqs, peak_magnitudes, color='red', label='Dominant Peaks')
        ax.set_xlabel('Frequency (cycles/hour)', fontsize=14)
        ax.set_ylabel('Magnitude', fontsize=14)
        ax.set_xlim(0, 0.5)
        ax.grid(True)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', labelsize=12)
        plot_filename = f'FFT_Spectrum_{label_suffix}_{datetime.now().strftime("%Y-%m-%d")}.png'
        save_path = os.path.join(processed_dir, plot_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved successfully to {save_path}")
        plt.show(block=False)
        return fft_df, complete_ft_df
    original_signal = results_df['Fluorescence to G-band Ratio'].values
    smoothed_signal = results_df['Smoothed Corrected (ALS)'].values
    fft_df_original, complete_ft_df_original = compute_ft(original_signal, 'original')
    fft_df_smoothed, complete_ft_df_smoothed = compute_ft(smoothed_signal, 'smoothed')
    return fft_df_original, fft_df_smoothed, complete_ft_df_smoothed


# ============================================================================
# Post-Processing Functions for v4 (moved from main_test_v3.py)
# ============================================================================

def apply_baseline_correction_v4(df, columns_to_correct=None, 
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


# ============================================================================
# Additional Post-Processing Functions for v4 (moved from main_test_v3.py)
# ============================================================================
def apply_als_and_gaussian_v4(results_df, config, processed_dir, start_datetime=None, end_datetime=None, light_cycle='Constant', light_transition=None, shade_transition=None, treatment_events=None):
    """
    Apply ALS baseline correction and Gaussian smoothing to baseline-corrected ratios (v4).
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
    
    # Check x_axis_type to determine if we should preserve scan number index
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    if x_axis_type not in ['datetime', 'scan_number']:
        x_axis_type = 'datetime'
    
    # Store original index info for scan_number mode
    is_scan_number_mode = (x_axis_type == 'scan_number')
    original_index = None
    original_index_name = None
    
    # If in scan_number mode, try to preserve or restore scan number index
    if is_scan_number_mode:
        if results_df.index.name == 'Scan Number' and pd.api.types.is_integer_dtype(results_df.index):
            # Index is already scan numbers - preserve it
            original_index = results_df.index.copy()
            original_index_name = results_df.index.name
            x_values = results_df.index.values
        elif 'Scan Number' in results_df.columns:
            # Index was converted but 'Scan Number' column exists - restore it
            original_index = results_df['Scan Number'].values.copy()
            original_index_name = 'Scan Number'
            results_df.index = original_index
            results_df.index.name = original_index_name
            x_values = results_df.index.values
        else:
            # Can't restore scan numbers - fall back to datetime mode
            print("Warning: scan_number mode requested but 'Scan Number' column not found. Using datetime index.")
            is_scan_number_mode = False
    
    if not is_scan_number_mode:
        # Ensure datetime index for datetime mode
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
        
        x_values = results_df.index
    
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
            ax.plot(x_values, ratios_gband, label='Baseline Corrected Ratio', color='m', alpha=0.6, linewidth=1)
            # Plot smoothed signal (input to ALS)
            smoothed_gband_full_plot = np.full_like(ratios_gband, np.nan)
            smoothed_gband_full_plot[mask_gband] = ratios_gband_smoothed
            ax.plot(x_values, smoothed_gband_full_plot, label='Gaussian Smoothed (before ALS)', color='blue', alpha=0.6, linewidth=1, linestyle='-.')
            ax.plot(x_values[mask_gband], baseline_als_vals_gband, label='Baseline', color='red', linestyle='--', linewidth=2)
            ax.plot(x_values, corrected_gband_full, label='Corrected', color='green', alpha=0.8, linewidth=1.5)
            
            # Add day/night shading and transitions only for datetime mode
            if not is_scan_number_mode:
                if light_transition is None:
                    light_transition = parse_light_transition_config(config)
                if shade_transition is None:
                    shade_transition = parse_shade_transition_config(config)
                if treatment_events is None:
                    treatment_events = parse_treatment_events_config(config)
                add_day_night_shading(ax, x_values.min(), x_values.max(), light_cycle=light_cycle, light_transition=light_transition)
                # Add vertical line to mark light transition if configured
                if light_transition:
                    transition_time = light_transition['transition_datetime']
                    if x_values.min() <= transition_time <= x_values.max():
                        ax.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
                # Add vertical line to mark shade transition if configured
                if shade_transition:
                    transition_time = shade_transition['transition_datetime']
                    if x_values.min() <= transition_time <= x_values.max():
                        label_text = 'Shade transition'
                        if 'ppfd' in shade_transition:
                            label_text += f" (PPFD: {shade_transition['ppfd']})"
                        ax.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
                # Add markers for treatment events if configured
                if treatment_events:
                    for event in treatment_events:
                        event_time = event['datetime']
                        if x_values.min() <= event_time <= x_values.max():
                            ax.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                     linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            
            if is_scan_number_mode:
                ax.set_xlabel('Scan Number', fontsize=14)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
            else:
                ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
                ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
                ax.xaxis.set_major_locator(DayLocator())
                fig.autofmt_xdate()
            ax.set_ylabel('Fluorescence / G-band Ratio', fontsize=14)
            ax.set_title('Baseline Correction - Fluorescence to G-band Ratio', fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=12)
            
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
            ax.plot(x_values, ratios_raman, label='Baseline Corrected Ratio', color='m', alpha=0.6, linewidth=1)
            # Plot smoothed signal (input to ALS)
            smoothed_raman_full_plot = np.full_like(ratios_raman, np.nan)
            smoothed_raman_full_plot[mask_raman] = ratios_raman_smoothed
            ax.plot(x_values, smoothed_raman_full_plot, label='Gaussian Smoothed (before ALS)', color='blue', alpha=0.6, linewidth=1, linestyle='-.')
            ax.plot(x_values[mask_raman], baseline_als_vals_raman, label='Baseline', color='red', linestyle='--', linewidth=2)
            ax.plot(x_values, corrected_raman_full, label='Corrected', color='green', alpha=0.8, linewidth=1.5)
            
            # Add day/night shading and transitions only for datetime mode
            if not is_scan_number_mode:
                if light_transition is None:
                    light_transition = parse_light_transition_config(config)
                if shade_transition is None:
                    shade_transition = parse_shade_transition_config(config)
                if treatment_events is None:
                    treatment_events = parse_treatment_events_config(config)
                add_day_night_shading(ax, x_values.min(), x_values.max(), light_cycle=light_cycle, light_transition=light_transition)
                # Add vertical line to mark light transition if configured
                if light_transition:
                    transition_time = light_transition['transition_datetime']
                    if x_values.min() <= transition_time <= x_values.max():
                        ax.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
                # Add vertical line to mark shade transition if configured
                if shade_transition:
                    transition_time = shade_transition['transition_datetime']
                    if x_values.min() <= transition_time <= x_values.max():
                        label_text = 'Shade transition'
                        if 'ppfd' in shade_transition:
                            label_text += f" (PPFD: {shade_transition['ppfd']})"
                        ax.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
                # Add markers for treatment events if configured
                if treatment_events:
                    for event in treatment_events:
                        event_time = event['datetime']
                        if x_values.min() <= event_time <= x_values.max():
                            ax.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                     linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            
            if is_scan_number_mode:
                ax.set_xlabel('Scan Number', fontsize=14)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
            else:
                ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
                ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
                ax.xaxis.set_major_locator(DayLocator())
                fig.autofmt_xdate()
            ax.set_ylabel('Fluorescence / Raman Peak 850 Ratio', fontsize=14)
            ax.set_title('Baseline Correction - Fluorescence to Raman Peak 850 Ratio', fontsize=15, fontweight='bold')
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=12)
            
            save_path = Path(processed_dir) / f'baseline_raman850_{datetime.now().strftime("%Y%m%d")}.png'
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Baseline correction plot saved to: {save_path}")
        else:
            print("Warning: Insufficient data for Raman 850 ratio ALS correction")
    
    # Restore original scan number index if we were in scan_number mode
    if is_scan_number_mode and original_index is not None:
        results_df.index = original_index
        results_df.index.name = original_index_name
    
    return results_df


def compute_fourier_transform_v4(results_df, config, processed_dir, start_datetime=None, end_datetime=None, light_cycle='Constant', light_transition=None, shade_transition=None, treatment_events=None):
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
    
    # Check x_axis_type - FFT analysis only makes sense for datetime mode
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    if x_axis_type not in ['datetime', 'scan_number']:
        x_axis_type = 'datetime'
    
    if x_axis_type == 'scan_number':
        print("Skipping Fourier Transform computation (not applicable for scan_number mode)")
        return {}
    
    # Filter by datetime range if provided
    if start_datetime is not None and end_datetime is not None:
        results_df = results_df[(results_df.index >= start_datetime) & (results_df.index <= end_datetime)]
    
    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(results_df.index):
        if 'Datetime' in results_df.columns:
            results_df['Datetime'] = pd.to_datetime(results_df['Datetime'], errors='coerce')
            results_df = results_df.set_index('Datetime')
        else:
            results_df.index = pd.to_datetime(results_df.index, errors='coerce')
    
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


def compute_diurnal_average_v4(results_df, config, processed_dir, start_datetime=None, end_datetime=None, light_cycle='Constant'):
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
    
    # Check x_axis_type - diurnal averages only make sense for datetime mode
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    if x_axis_type not in ['datetime', 'scan_number']:
        x_axis_type = 'datetime'
    
    if x_axis_type == 'scan_number':
        print("Skipping diurnal average computation (not applicable for scan_number mode)")
        return results_df
    
    # Filter by datetime range if provided
    if start_datetime is not None and end_datetime is not None:
        results_df = results_df[(results_df.index >= start_datetime) & (results_df.index <= end_datetime)]
    
    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(results_df.index):
        if 'Datetime' in results_df.columns:
            results_df['Datetime'] = pd.to_datetime(results_df['Datetime'], errors='coerce')
            results_df = results_df.set_index('Datetime')
        else:
            results_df.index = pd.to_datetime(results_df.index, errors='coerce')
    
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


