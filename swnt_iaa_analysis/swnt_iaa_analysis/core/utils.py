"""Utility functions for core processing."""

import os
import numpy as np
from datetime import timedelta, datetime
from scipy.signal import medfilt


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

