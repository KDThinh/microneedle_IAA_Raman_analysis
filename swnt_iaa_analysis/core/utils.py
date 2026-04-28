"""Utility functions for core processing."""

import os
import json
import numpy as np
from datetime import timedelta, datetime
from scipy.signal import medfilt, savgol_filter
from pathlib import Path


def find_google_drive():
    """Find Google Drive mount point on Windows."""
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        drive = f"{letter}:\\"
        if os.path.exists(drive):
            if os.path.exists(os.path.join(drive, "My Drive")):
                return drive
    raise Exception("Google Drive not found on any drive letter!")


def raman_wavenumber_to_emission_nm(wavenumber_cm, excitation_nm=830):
    """
    Convert Raman wavenumber (cm^-1) to emission wavelength (nm).
    """
    wavenumber_cm = np.asarray(wavenumber_cm)
    emission_nm = excitation_nm / (1 - (wavenumber_cm * excitation_nm / 1e7))
    return emission_nm


def emission_nm_to_raman_wavenumber(emission_nm, excitation_nm=830):
    """
    Convert emission wavelength (nm) to Raman wavenumber (cm^-1).
    Inverse of raman_wavenumber_to_emission_nm.
    """
    emission_nm = np.asarray(emission_nm)
    wavenumber_cm = 1e7 * (emission_nm - excitation_nm) / (emission_nm * excitation_nm)
    return wavenumber_cm


def correct_wavenumber_for_excitation(wavenumber_cm, recorded_excitation_nm, actual_excitation_nm):
    """
    Correct Raman wavenumber when the excitation wavelength was incorrectly recorded.
    
    This function corrects wavenumbers by first converting them to the actual emission wavelength
    (which is invariant), then converting back to wavenumber using the correct excitation wavelength.
    
    Parameters:
    -----------
    wavenumber_cm : array-like
        Wavenumbers (cm^-1) that were calculated assuming `recorded_excitation_nm` excitation.
    recorded_excitation_nm : float
        The excitation wavelength (nm) that was incorrectly recorded/used.
    actual_excitation_nm : float
        The actual excitation wavelength (nm) that was used.
    
    Returns:
    --------
    corrected_wavenumber_cm : array
        Corrected wavenumbers (cm^-1) for the actual excitation wavelength.
    """
    wavenumber_cm = np.asarray(wavenumber_cm)
    
    # Convert recorded wavenumber to actual emission wavelength
    # The emission wavelength is physically invariant regardless of recorded excitation
    emission_nm = raman_wavenumber_to_emission_nm(wavenumber_cm, excitation_nm=recorded_excitation_nm)
    
    # Convert emission wavelength back to wavenumber using the correct excitation wavelength
    corrected_wavenumber_cm = emission_nm_to_raman_wavenumber(emission_nm, excitation_nm=actual_excitation_nm)
    
    return corrected_wavenumber_cm


def create_dir_if_needed(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def remove_spikes_hampel(signal, window_size=5, threshold=3.0, min_spike_length=1, max_spike_length=5):
    """
    Remove spikes (outliers) from time series using Hampel filter (median-based outlier detection).
    
    Parameters:
    -----------
    signal : np.array
        Input signal (may contain NaN values)
    window_size : int
        Half-window size for median calculation (default: 5)
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
    # Use a robust approach that (when possible) excludes the current point
    # from the median estimate so that large spikes do not bias the window.
    n = len(signal_valid)
    median_values = np.zeros(n)
    mad_values = np.zeros(n)
    
    for i in range(n):
        # Define window around point i
        start_idx = max(0, i - window_size)
        end_idx = min(n, i + window_size + 1)
        
        window_data = signal_valid[start_idx:end_idx]
        
        # Calculate median, preferring to exclude the current point i from the window
        # when there are enough neighbouring points. This makes the estimator
        # more robust to large spikes that otherwise dominate the local window.
        if len(window_data) > 1:
            # Build window without the current point (if it lies inside the window)
            window_without_i = np.concatenate(
                [signal_valid[start_idx:i], signal_valid[i + 1:end_idx]]
            )
            if len(window_without_i) > 0:
                median_values[i] = np.median(window_without_i)
            else:
                median_values[i] = np.median(window_data)
        else:
            median_values[i] = np.median(window_data)
        
        # Calculate MAD (Median Absolute Deviation) relative to this robust median
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
        replacement_kernel = min(2 * window_size + 1, n)
        if replacement_kernel % 2 == 0:
            replacement_kernel += 1
        if replacement_kernel >= 3:
            median_filtered = medfilt(signal_valid, kernel_size=replacement_kernel)
        else:
            median_filtered = signal_valid.copy()
        
        for start, end in spike_groups:
            # Replace spike with median-filtered values
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


def smooth_signal(signal, window_size=11, poly_order=2):
    """
    Smooth signal using Savitzky-Golay filter.
    
    Parameters
    ----------
    signal : np.array
        Array of signal values
    window_size : int
        Window size for Savitzky-Golay filter (must be odd, default: 11)
    poly_order : int
        Polynomial order for Savitzky-Golay filter (default: 2)
    
    Returns
    -------
    smoothed : np.array
        Smoothed signal
    """
    # Ensure window_size is odd and valid
    if window_size % 2 == 0:
        window_size += 1
    if window_size > len(signal):
        window_size = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    if window_size < 3:
        return signal  # Too short to smooth
    
    # Ensure poly_order is less than window_size
    if poly_order >= window_size:
        poly_order = window_size - 1
    
    try:
        smoothed = savgol_filter(signal, window_size, poly_order)
        return smoothed
    except Exception as e:
        print(f"Warning: Smoothing failed ({e}), using original signal")
        return signal


def correct_baseline_shifts(signal, threshold_multiplier=5, window_size=5,
                           smooth_first=True, smooth_window=11, smooth_poly_order=2,
                           correct_smoothed=False, shared_threshold=None,
                           detect_cumulative_jumps=True, cumulative_window=5):
    """
    Detect and correct baseline shifts using MAD-based detection with median offset calculation.
    
    This algorithm:
    1. Calculates derivative with prepend to maintain array length
    2. Uses MAD to detect outliers (individual jumps)
    3. Optionally detects cumulative changes over a window (gradual jumps over multiple points)
    4. Filters consecutive jumps to avoid detecting ramps
    5. Uses median (not mean) for robust offset calculation
    6. Applies cumulative corrections
    
    Parameters
    ----------
    signal : np.array
        Array of signal values
    threshold_multiplier : float
        How many times larger than the noise floor a jump must be to be corrected (default: 5)
    window_size : int
        Number of points to average before/after a jump to calculate precise offset (default: 5)
    smooth_first : bool
        If True, smooth the signal before jump detection (default: True)
    smooth_window : int
        Window size for smoothing if smooth_first=True (default: 11)
    smooth_poly_order : int
        Polynomial order for smoothing if smooth_first=True (default: 2)
    correct_smoothed : bool
        If True, apply correction to smoothed signal (offset calculated from smoothed).
        If False, apply correction to original signal (offset calculated from original) (default: False)
    shared_threshold : float, optional
        If provided, use this absolute threshold instead of calculating MAD-based threshold.
        This allows using a shared MAD from a reference signal (default: None)
    detect_cumulative_jumps : bool
        If True, also detect gradual jumps over multiple consecutive points (default: True)
    cumulative_window : int
        Window size for detecting cumulative changes (default: 5)
        Detects jumps where total change over N consecutive points exceeds threshold
    
    Returns
    -------
    corrected : np.array
        Corrected signal with jumps removed
    jump_indices : np.array
        Indices where jumps were detected
    jump_info : list
        List of dicts with jump information
    smoothed_signal : np.array
        Smoothed version of input signal (if smooth_first=True) or original signal
    """
    # Smooth signal first if requested
    if smooth_first:
        smoothed_signal = smooth_signal(signal, window_size=smooth_window, poly_order=smooth_poly_order)
    else:
        smoothed_signal = np.array(signal).copy()
    
    # Calculate the first difference (derivative) with prepend to maintain length
    diffs = np.diff(smoothed_signal, prepend=smoothed_signal[0])
    
    # Define threshold: use shared threshold if provided, otherwise calculate from MAD
    if shared_threshold is not None:
        threshold = shared_threshold
    else:
        # Estimate the "noise floor" using Median Absolute Deviation (MAD)
        # This is more robust than Standard Deviation for data with outliers/steps
        median_diff = np.median(diffs)
        mad = np.median(np.abs(diffs - median_diff))
        
        # Handle edge case where MAD is zero
        if mad == 0:
            mad = np.std(diffs) if np.std(diffs) > 0 else 1.0
        
        # Define a threshold for what constitutes a "Shift" vs just "Noise"
        threshold = mad * threshold_multiplier
    
    # Find indices where the jump exceeds the threshold (individual jumps)
    jump_indices_individual = np.where(np.abs(diffs) > threshold)[0]
    
    # Optionally detect cumulative changes over a window (gradual jumps)
    jump_indices_cumulative = np.array([], dtype=int)
    if detect_cumulative_jumps and len(smoothed_signal) > cumulative_window:
        cumulative_changes = []
        cumulative_indices = []
        
        for i in range(len(smoothed_signal) - cumulative_window):
            # Calculate total change over the window
            total_change = smoothed_signal[i + cumulative_window] - smoothed_signal[i]
            cumulative_changes.append(total_change)
            cumulative_indices.append(i + cumulative_window)  # Mark end of window
        
        # Calculate MAD for cumulative changes
        if len(cumulative_changes) > 0:
            median_cumulative = np.median(cumulative_changes)
            mad_cumulative = np.median(np.abs(np.array(cumulative_changes) - median_cumulative))
            
            if mad_cumulative == 0:
                mad_cumulative = np.std(cumulative_changes) if np.std(cumulative_changes) > 0 else 1.0
            
            cumulative_threshold = mad_cumulative * threshold_multiplier
            
            # Find where cumulative change exceeds threshold
            cumulative_array = np.array(cumulative_changes)
            cumulative_mask = np.abs(cumulative_array) > cumulative_threshold
            cumulative_detected = np.where(cumulative_mask)[0]
            
            # Only consider cumulative jumps that don't overlap with individual jumps
            # (to avoid double-detection)
            for idx in cumulative_detected:
                jump_idx = cumulative_indices[idx]
                # Check if this index is far enough from individual jumps
                if len(jump_indices_individual) == 0 or np.min(np.abs(jump_indices_individual - jump_idx)) > window_size:
                    jump_indices_cumulative = np.append(jump_indices_cumulative, jump_idx)
    
    # Combine both types of jump detections
    jump_indices_combined = np.unique(np.concatenate([jump_indices_individual, jump_indices_cumulative]))
    jump_indices_combined = np.sort(jump_indices_combined)
    
    # Track which jumps are individual vs cumulative
    jump_type_map = {}
    for idx in jump_indices_individual:
        jump_type_map[idx] = 'individual'
    for idx in jump_indices_cumulative:
        jump_type_map[idx] = 'cumulative'
    
    # Filter indices: ensure we don't pick up consecutive points (ramp) as multiple jumps
    # We only take the peak of the jump
    clean_indices = []
    clean_jump_types = []
    if len(jump_indices_combined) > 0:
        clean_indices.append(jump_indices_combined[0])
        clean_jump_types.append(jump_type_map.get(jump_indices_combined[0], 'unknown'))
        for i in range(1, len(jump_indices_combined)):
            if jump_indices_combined[i] - jump_indices_combined[i-1] > window_size:
                clean_indices.append(jump_indices_combined[i])
                clean_jump_types.append(jump_type_map.get(jump_indices_combined[i], 'unknown'))
    
    jump_indices = np.array(clean_indices, dtype=int)
    
    # Apply Correction
    # Choose which signal to correct based on correct_smoothed parameter
    if correct_smoothed:
        # Correct smoothed signal: use smoothed signal for both offset calculation and correction
        target_signal = smoothed_signal.copy().astype(float)
        y_corrected = smoothed_signal.copy().astype(float)
    else:
        # Correct original signal: use original signal for offset calculation and correction
        target_signal = signal.copy().astype(float)
        y_corrected = signal.copy().astype(float)
    
    cumulative_offset = 0.0
    jump_info = []
    
    # #region agent log
    _log_path = Path(__file__).parent.parent.parent / '.cursor' / 'debug.log'
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_log_path, 'a') as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1,H2", "location": f"{__file__}:379", "message": "correct_baseline_shifts: starting correction loop", "data": {"num_jumps": len(clean_indices), "correct_smoothed": correct_smoothed, "signal_len": len(signal), "y_corrected_initial": y_corrected[:5].tolist() if len(y_corrected) >= 5 else y_corrected.tolist()}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
    # #endregion
    
    # Iterate through the detected jumps and "stitch" the segments
    # Calculate all offsets from original signal first, then apply corrections cumulatively
    for idx in clean_indices:
        # Define a small window before and after the jump index
        start_idx = max(0, idx - window_size)
        end_idx = min(len(target_signal), idx + window_size)
        
        # Calculate the median level before and after the jump using ORIGINAL target signal
        # This ensures consistent offset calculation regardless of previous corrections
        # We exclude the jump point from both windows to avoid bias
        
        # Before jump: exclude jump point (idx is not included)
        val_before = np.median(target_signal[start_idx:idx])
        
        # After jump: exclude jump point (use idx+1 to start after the jump)
        # This prevents the jump point itself from biasing the median
        after_start = idx + 1
        if after_start < end_idx:
            val_after = np.median(target_signal[after_start:end_idx])
        else:
            # Edge case: if jump is very close to the end, use a smaller window
            # or use the value right after the jump if available
            if idx + 1 < len(target_signal):
                val_after = np.median(target_signal[idx+1:min(len(target_signal), idx+1+window_size)])
            else:
                # Last point: use the value before as fallback
                val_after = val_before
        
        # The jump magnitude (calculated from original target signal)
        step_change = val_after - val_before
        
        # #region agent log
        _log_path = Path(__file__).parent.parent.parent / '.cursor' / 'debug.log'
        _y_before_corr = y_corrected[idx:min(idx+3, len(y_corrected))].copy() if len(y_corrected) > idx else []
        # #endregion
        
        # We accumulate this offset
        cumulative_offset += step_change
        
        # Apply correction to the target signal
        # (We subtract the jump to bring the new baseline down/up to the old one)
        # First apply a hard step correction starting at the adjusted jump index `idx`.
        # This makes the baseline continuous in an average sense.
        y_corrected[idx:] -= step_change
        
        # #region agent log
        _y_after_corr = y_corrected[idx:min(idx+3, len(y_corrected))].copy() if len(y_corrected) > idx else []
        with open(_log_path, 'a') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": f"{__file__}:428", "message": "correct_baseline_shifts: applied correction", "data": {"jump_idx": int(idx), "val_before": float(val_before), "val_after": float(val_after), "step_change": float(step_change), "y_before_corr": _y_before_corr.tolist() if hasattr(_y_before_corr, 'tolist') else _y_before_corr, "y_after_corr": _y_after_corr.tolist() if hasattr(_y_after_corr, 'tolist') else _y_after_corr, "correction_applied": not np.allclose(_y_before_corr, _y_after_corr, atol=1e-10) if len(_y_before_corr) > 0 and len(_y_after_corr) > 0 else False}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
        # #endregion
        
        jump_info.append({
            'index': idx,
            'val_before': val_before,
            'val_after': val_after,
            'step_change': step_change,
            'cumulative_offset': cumulative_offset,
            'jump_type': jump_type_map.get(idx, 'unknown')
        })
    
    # #region agent log
    _log_path = Path(__file__).parent.parent.parent / '.cursor' / 'debug.log'
    _max_diff = np.nanmax(np.abs(y_corrected - target_signal)) if len(y_corrected) == len(target_signal) else -1
    _are_identical = np.allclose(y_corrected, target_signal, atol=1e-10) if len(y_corrected) == len(target_signal) else False
    with open(_log_path, 'a') as f:
        f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": f"{__file__}:445", "message": "correct_baseline_shifts: returning corrected signal", "data": {"num_jumps": len(clean_indices), "max_diff_target_vs_corrected": float(_max_diff), "are_identical": bool(_are_identical), "y_corrected_sample": y_corrected[:5].tolist() if len(y_corrected) >= 5 else y_corrected.tolist(), "target_sample": target_signal[:5].tolist() if len(target_signal) >= 5 else target_signal.tolist()}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
    # #endregion
    
    # Optional: local Gaussian smoothing around each jump to reduce residual kinks
    try:
        from scipy.ndimage import gaussian_filter1d
        if len(clean_indices) > 0 and window_size > 0:
            n = len(y_corrected)
            half_width = max(1, window_size)
            sigma_local = max(1.0, window_size / 3.0)
            for idx in clean_indices:
                start_s = max(0, idx - half_width)
                end_s = min(n, idx + half_width + 1)
                segment = y_corrected[start_s:end_s]
                if len(segment) >= 3:
                    smoothed_seg = gaussian_filter1d(segment, sigma=sigma_local, mode='nearest')
                    y_corrected[start_s:end_s] = smoothed_seg
    except Exception:
        # If scipy is unavailable or smoothing fails, fall back to unsmoothed result
        pass
    
    return y_corrected, np.array(clean_indices), jump_info, smoothed_signal


def apply_corrections_at_jump_indices(signal, jump_indices_valid, window_size=5, 
                                      smooth_first=True, smooth_window=11, smooth_poly_order=2,
                                      correct_smoothed=False):
    """
    Apply baseline corrections at pre-detected jump indices.
    
    This function applies corrections at specified jump locations without detecting jumps.
    Useful when jump detection is performed on one signal and the same jump locations
    should be used for other signals.
    
    Parameters
    ----------
    signal : np.array
        Array of signal values to correct
    jump_indices_valid : np.array
        Jump indices in valid signal space (must be sorted, correspond to positions in signal array)
    window_size : int
        Number of points to average before/after jump for offset calculation (default: 5)
    smooth_first : bool
        If True, smooth the signal before calculating offsets (default: True)
    smooth_window : int
        Window size for smoothing if smooth_first=True (default: 11)
    smooth_poly_order : int
        Polynomial order for smoothing if smooth_first=True (default: 2)
    correct_smoothed : bool
        If True, apply correction to smoothed signal; if False, to original signal (default: False)
    
    Returns
    -------
    corrected_signal : np.array
        Corrected signal
    jump_info : list
        List of dictionaries with jump information (index, val_before, val_after, step_change, cumulative_offset)
    smoothed_signal : np.array
        Smoothed signal (if smooth_first=True) or original signal (if smooth_first=False)
    """
    signal = np.array(signal).copy().astype(float)
    jump_indices_valid = np.array(jump_indices_valid)
    
    if len(jump_indices_valid) == 0:
        # No jumps to correct
        if smooth_first:
            from scipy.signal import savgol_filter
            smoothed_signal = savgol_filter(signal, smooth_window, smooth_poly_order) if len(signal) >= smooth_window else signal.copy()
        else:
            smoothed_signal = signal.copy()
        return signal.copy(), [], smoothed_signal
    
    # Sort jump indices
    jump_indices_valid = np.sort(jump_indices_valid)
    # Filter to valid range
    jump_indices_valid = jump_indices_valid[(jump_indices_valid >= 0) & (jump_indices_valid < len(signal))]
    
    # Smooth signal if requested
    if smooth_first and len(signal) >= smooth_window:
        from scipy.signal import savgol_filter
        smoothed_signal = savgol_filter(signal, smooth_window, smooth_poly_order)
    else:
        smoothed_signal = signal.copy()
    
    # Choose which signal to correct based on correct_smoothed parameter
    if correct_smoothed:
        target_signal = smoothed_signal.copy().astype(float)
        y_corrected = smoothed_signal.copy().astype(float)
    else:
        target_signal = signal.copy().astype(float)
        y_corrected = signal.copy().astype(float)
    
    cumulative_offset = 0.0
    jump_info = []
    
    # Apply corrections at each jump index
    for idx in jump_indices_valid:
        # Define a small window before and after the jump index
        start_idx = max(0, idx - window_size)
        end_idx = min(len(target_signal), idx + window_size)
        
        # Calculate the median level before and after the jump
        val_before = np.median(target_signal[start_idx:idx]) if idx > start_idx else target_signal[idx]
        
        # After jump: exclude jump point
        after_start = idx + 1
        if after_start < end_idx:
            val_after = np.median(target_signal[after_start:end_idx])
        else:
            if idx + 1 < len(target_signal):
                val_after = np.median(target_signal[idx+1:min(len(target_signal), idx+1+window_size)])
            else:
                val_after = val_before
        
        # The jump magnitude
        step_change = val_after - val_before
        
        # Accumulate offset
        cumulative_offset += step_change
        
        # Apply correction cumulatively
        y_corrected[idx:] -= step_change
        
        jump_info.append({
            'index': int(idx),
            'val_before': float(val_before),
            'val_after': float(val_after),
            'step_change': float(step_change),
            'cumulative_offset': float(cumulative_offset),
            'jump_type': 'shared'  # Indicates this jump was shared from another signal
        })
    
    # Optional: local Gaussian smoothing around each jump to reduce residual kinks
    try:
        from scipy.ndimage import gaussian_filter1d
        if len(jump_indices_valid) > 0 and window_size > 0:
            n = len(y_corrected)
            half_width = max(1, window_size)
            sigma_local = max(1.0, window_size / 3.0)
            for idx in jump_indices_valid:
                start_s = max(0, idx - half_width)
                end_s = min(n, idx + half_width + 1)
                segment = y_corrected[start_s:end_s]
                if len(segment) >= 3:
                    smoothed_seg = gaussian_filter1d(segment, sigma=sigma_local, mode='nearest')
                    y_corrected[start_s:end_s] = smoothed_seg
    except Exception:
        # If scipy is unavailable or smoothing fails, fall back to unsmoothed result
        pass
    
    return y_corrected, jump_info, smoothed_signal


def parse_light_transition_config(config):
    """Parse light transition configuration from config dictionary."""
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
    """Parse shade transition configuration from config dictionary."""
    shade_transition = config.get('shade_transition')
    if not shade_transition:
        return None
    
    transition_str = shade_transition.get('transition_datetime')
    if not transition_str:
        return None
    
    try:
        transition_datetime = datetime.strptime(transition_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
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
    
    if 'ppfd' in shade_transition:
        result['ppfd'] = shade_transition['ppfd']
    if 'r_fr_ratio' in shade_transition:
        result['r_fr_ratio'] = shade_transition['r_fr_ratio']
    
    return result


def parse_treatment_events_config(config):
    """Parse treatment events configuration from config dictionary."""
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
        
        try:
            event_datetime = datetime.strptime(event_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
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


def add_day_night_shading(ax, start_time, end_time, light_cycle='Constant', light_transition=None):
    """
    Add shading to a Matplotlib axis for day/night cycles in time-series plots.
    Supports light cycle transitions (e.g., from 8to24 to Constant).
    """
    if light_transition and 'transition_datetime' in light_transition:
        transition_time = light_transition['transition_datetime']
        initial_cycle = light_transition.get('initial_light_cycle', '8to24')
        
        if start_time < transition_time:
            period_end = min(transition_time, end_time)
            _add_shading_for_cycle(ax, start_time, period_end, initial_cycle, is_first=True)
        
        if transition_time < end_time:
            period_start = max(transition_time, start_time)
            _add_shading_for_cycle(ax, period_start, end_time, light_cycle, is_first=(transition_time >= start_time))
    else:
        _add_shading_for_cycle(ax, start_time, end_time, light_cycle, is_first=True)


def _add_shading_for_cycle(ax, start_time, end_time, light_cycle, is_first=False):
    """Helper function to add shading for a specific light cycle period."""
    current_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    
    while current_time < end_time:
        if light_cycle == 'Constant':
            day_start = current_time.replace(hour=0, minute=0)
            day_end = (current_time + timedelta(days=1)).replace(hour=0, minute=0)
            if day_start < end_time and day_end > start_time:
                ax.axvspan(max(day_start, start_time), min(day_end, end_time),
                           facecolor='yellow', alpha=0.1,
                           label='Daytime (Constant)' if is_first and current_time == start_time.replace(hour=0, minute=0) else None)
        else:
            day_start_hour = 8 if light_cycle == '8to24' else 6
            day_end_hour = 0 if light_cycle == '8to24' else 22
            day_end_day_offset = 1 if light_cycle == '8to24' else 0
            
            day_start = current_time.replace(hour=day_start_hour, minute=0)
            day_end = (current_time + timedelta(days=day_end_day_offset)).replace(hour=day_end_hour, minute=0)
            if day_start < end_time and day_end > start_time:
                ax.axvspan(max(day_start, start_time), min(day_end, end_time),
                           facecolor='yellow', alpha=0.1,
                           label='Daytime' if is_first and current_time == start_time.replace(hour=0, minute=0) else None)
            
            night_start1 = current_time.replace(hour=0, minute=0)
            night_end1 = current_time.replace(hour=day_start_hour, minute=0)
            if night_start1 < end_time and night_end1 > start_time:
                ax.axvspan(max(night_start1, start_time), min(night_end1, end_time),
                           facecolor='blue', alpha=0.1,
                           label='Nighttime' if is_first and current_time == start_time.replace(hour=0, minute=0) else None)
            
            if day_end_hour != 0:
                night_start2 = current_time.replace(hour=day_end_hour, minute=0)
                night_end2 = (current_time + timedelta(days=1)).replace(hour=0, minute=0)
                if night_start2 < end_time and night_end2 > start_time:
                    ax.axvspan(max(night_start2, start_time), min(night_end2, end_time),
                               facecolor='blue', alpha=0.1)
        
        current_time += timedelta(days=1)

