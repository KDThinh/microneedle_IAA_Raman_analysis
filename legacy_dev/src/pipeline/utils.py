import os
import numpy as np
import logging
import re
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from datetime import timedelta, datetime
from scipy.signal import medfilt

logging.basicConfig(level=logging.INFO)

def find_google_drive():
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
    
    Example:
    --------
    If data was recorded with 840 nm excitation but should be 830 nm:
    >>> wavenumbers_recorded = np.array([100, 200, 300])
    >>> wavenumbers_corrected = correct_wavenumber_for_excitation(
    ...     wavenumbers_recorded, recorded_excitation_nm=840, actual_excitation_nm=830
    ... )
    """
    wavenumber_cm = np.asarray(wavenumber_cm)
    
    # Convert recorded wavenumber to actual emission wavelength
    # The emission wavelength is physically invariant regardless of recorded excitation
    emission_nm = raman_wavenumber_to_emission_nm(wavenumber_cm, excitation_nm=recorded_excitation_nm)
    
    # Convert emission wavelength back to wavenumber using the correct excitation wavelength
    corrected_wavenumber_cm = emission_nm_to_raman_wavenumber(emission_nm, excitation_nm=actual_excitation_nm)
    
    return corrected_wavenumber_cm

def lieberfit(spectrum, order=5, tot_iter=100):
    """
    Perform iterative polynomial baseline correction (lieberfit).
    """
    pix_size = len(spectrum)
    polyspec_iter = np.array(spectrum)
    x = np.arange(pix_size)
    
    for _ in range(tot_iter):
        p_order = np.polyfit(x, polyspec_iter, order)
        polyspec_order = np.polyval(p_order, x)
        polyspec_iter = np.minimum(polyspec_order, polyspec_iter)
    
    baseline = polyspec_iter
    corrected_spectrum = spectrum - baseline
    return corrected_spectrum, baseline

def add_day_night_shading(ax, start_time, end_time, light_cycle='Constant', light_transition=None):
    """
    Add shading to a Matplotlib axis for day/night cycles in time-series plots.
    Supports light cycle transitions (e.g., from 8to24 to Constant).
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to add shading to.
    start_time : datetime
        Start of the time range.
    end_time : datetime
        End of the time range.
    light_cycle : str, optional
        Light cycle: 'Constant' (0:00-24:00), '8to24' (08:00-00:00 next day), or '6to22' (06:00-22:00).
        This is the final/current light cycle if a transition is specified.
    light_transition : dict, optional
        Dictionary with keys:
            - 'transition_datetime': datetime when light cycle changes
            - 'initial_light_cycle': str, light cycle before transition (e.g., '8to24')
        Example: {'transition_datetime': datetime(2025, 11, 17, 17, 0), 'initial_light_cycle': '8to24'}
    """
    # Handle light cycle transition
    if light_transition and 'transition_datetime' in light_transition:
        transition_time = light_transition['transition_datetime']
        initial_cycle = light_transition.get('initial_light_cycle', '8to24')
        
        # Shade period before transition
        if start_time < transition_time:
            period_end = min(transition_time, end_time)
            _add_shading_for_cycle(ax, start_time, period_end, initial_cycle, is_first=True)
        
        # Shade period after transition
        if transition_time < end_time:
            period_start = max(transition_time, start_time)
            _add_shading_for_cycle(ax, period_start, end_time, light_cycle, is_first=(transition_time >= start_time))
    else:
        # No transition - use single light cycle
        _add_shading_for_cycle(ax, start_time, end_time, light_cycle, is_first=True)

def _add_shading_for_cycle(ax, start_time, end_time, light_cycle, is_first=False):
    """
    Helper function to add shading for a specific light cycle period.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to add shading to.
    start_time : datetime
        Start of the time range.
    end_time : datetime
        End of the time range.
    light_cycle : str
        Light cycle: 'Constant', '8to24', or '6to22'.
    is_first : bool
        Whether this is the first period (for legend labels).
    """
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

def create_dir_if_needed(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logging.info(f"Created directory: {path}")


# ============================================================================
# Configuration Parsing Functions (moved from main_test_v3.py for v4)
# ============================================================================

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


# ============================================================================
# Signal Processing Functions (moved from main_test_v3.py for v4)
# ============================================================================

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
