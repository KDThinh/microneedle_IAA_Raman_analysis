import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
from scipy.integrate import trapezoid

from .utils import raman_wavenumber_to_emission_nm, emission_nm_to_raman_wavenumber, lieberfit

def process_raman_data(raman_df, scan_number, config):
    """
    Process a single Raman spectrum with Savitzky-Golay smoothing, background correction, and lieberfit for G-band.
    
    Parameters:
    raman_df : pandas.DataFrame
        DataFrame with Raman spectra.
    scan_number : int
        Scan number to process.
    config : dict
        Configuration dictionary containing parameters like 'poly_order', 'window_size', 'poly_order_sg',
        'tot_iter', 'excitation_nm', 'bg_emission_min', 'bg_emission_max'.
    
    Returns:
    wavenumbers : array
        Wavenumber values.
    emission_nm : array
        Emission wavelengths (nm).
    intensities_raw : array
        Raw intensities.
    intensities_smooth : array
        Smoothed intensities.
    bg_intensity : float
        Average background intensity.
    intensities_bg_corrected : array
        Background-corrected intensities (smoothed - background).
    baseline_gband : array
        Fitted G-band baseline (interpolated to full range).
    corrected_spectrum_gband : array
        G-band baseline-corrected spectrum.
    datetime : datetime
        Datetime of the scan.
    seconds : float
        Seconds of the scan.
    """
    # Extract config parameters
    poly_order = config.get('poly_order', 5)
    window_size = config.get('window_size', 25)
    poly_order_sg = config.get('poly_order_sg', 2)
    tot_iter = config.get('tot_iter', 100)
    excitation_nm = config.get('excitation_nm', 830)
    
    # Get background emission wavelength ranges from config (backward compatibility)
    bg_emission_min = config.get('bg_emission_min', 850)
    bg_emission_max = config.get('bg_emission_max', 925)
    
    # Extract scan data
    scan_data = raman_df[raman_df['Scan Number'] == scan_number]
    if scan_data.empty:
        print(f"Scan Number {scan_number} not found in the data!")
        return None
    
    # Extract time and intensities
    datetime = scan_data.index[0]
    seconds = scan_data['Seconds'].iloc[0]
    wavenumbers = np.array([float(col) for col in raman_df.columns if col not in ['Scan Number', 'Seconds']])
    intensities_raw = scan_data.iloc[0, 2:].values
    emission_nm = raman_wavenumber_to_emission_nm(wavenumbers, excitation_nm)
    
    # Apply Savitzky-Golay smoothing
    intensities_smooth = savgol_filter(intensities_raw, window_size, poly_order_sg)
    
    # Convert background emission wavelength range to wavenumber range
    bg_wavenumber_min = emission_nm_to_raman_wavenumber(bg_emission_min, excitation_nm)
    bg_wavenumber_max = emission_nm_to_raman_wavenumber(bg_emission_max, excitation_nm)
    bg_wavenumber_range = (wavenumbers >= bg_wavenumber_min) & (wavenumbers <= bg_wavenumber_max)
    
    # Calculate average background intensity
    bg_intensity = np.mean(intensities_smooth[bg_wavenumber_range])
    
    # Subtract background from smoothed spectrum
    intensities_bg_corrected = intensities_smooth - bg_intensity
    
    # Perform lieberfit for G-band region (1500-2100 cm^-1) on background-corrected spectrum
    gband_mask = (wavenumbers >= 1500) & (wavenumbers <= 2100)
    gband_spectrum = intensities_bg_corrected[gband_mask]
    corrected_gband, baseline_gband_partial = lieberfit(gband_spectrum, order=poly_order, tot_iter=tot_iter)
    # Interpolate G-band baseline to full spectrum (set to 0 outside range)
    baseline_gband = np.zeros_like(wavenumbers)
    baseline_gband[gband_mask] = np.interp(wavenumbers[gband_mask], wavenumbers[gband_mask], baseline_gband_partial)
    corrected_spectrum_gband = intensities_bg_corrected - baseline_gband
    
    return wavenumbers, emission_nm, intensities_raw, intensities_smooth, bg_intensity, intensities_bg_corrected, baseline_gband, corrected_spectrum_gband, datetime, seconds

def process_all_raman_data(raman_df, config):
    """
    Process all Raman spectra in raman_df with Savitzky-Golay smoothing, background correction, and lieberfit.
    Processing sequence:
    1. Apply Savitzky-Golay smoothing
    2. Calculate and subtract background intensity
    3. Apply Lieberfit baseline correction to background-corrected fluorescence region
    4. Extract G-band metrics and calculate fluorescence ratios
    
    Extract datetime (as index), Scan Number, G-band height, G-band area, background intensity, 
    initial fluorescence, background-subtracted and G-band-subtracted fluorescence, 
    and fluorescence-to-G-band ratio.
    
    All processing is performed using wavenumbers (cm^-1) for consistency with raw data.
    
    Parameters:
    raman_df : pandas.DataFrame
        DataFrame with Raman spectra.
    config : dict
        Configuration dictionary containing parameters like 'poly_order', 'window_size', 'poly_order_sg',
        'tot_iter', 'excitation_nm', 'bg_emission_min', 'bg_emission_max', 'fluo_emission_min',
        'fluo_emission_max', 'gband_emission_min', 'gband_emission_max'.
        Note: Emission wavelength parameters are converted to wavenumbers internally.
    
    Returns:
    results_df : pandas.DataFrame
        DataFrame containing results for all scans with Datetime as index and Scan Number as column.
    """
    # Extract config parameters
    poly_order = config.get('poly_order', 3)
    window_size = config.get('window_size', 25)
    poly_order_sg = config.get('poly_order_sg', 2)
    tot_iter = config.get('tot_iter', 10)
    excitation_nm = config.get('excitation_nm', 830)
    
    # Get emission wavelength ranges from config (backward compatibility)
    bg_emission_min = config.get('bg_emission_min', 850)
    bg_emission_max = config.get('bg_emission_max', 925)
    fluo_emission_min = config.get('fluo_emission_min', 925)
    fluo_emission_max = config.get('fluo_emission_max', 1000)
    gband_emission_min = config.get('gband_emission_min', 952)
    gband_emission_max = config.get('gband_emission_max', 958)
    
    # Convert emission wavelength ranges to wavenumber ranges
    # Note: Smaller emission wavelengths correspond to smaller wavenumber shifts
    bg_wavenumber_min = emission_nm_to_raman_wavenumber(bg_emission_min, excitation_nm)
    bg_wavenumber_max = emission_nm_to_raman_wavenumber(bg_emission_max, excitation_nm)
    fluo_wavenumber_min = emission_nm_to_raman_wavenumber(fluo_emission_min, excitation_nm)
    fluo_wavenumber_max = emission_nm_to_raman_wavenumber(fluo_emission_max, excitation_nm)
    gband_wavenumber_min = emission_nm_to_raman_wavenumber(gband_emission_min, excitation_nm)
    gband_wavenumber_max = emission_nm_to_raman_wavenumber(gband_emission_max, excitation_nm)
    
    # Extract wavenumbers from DataFrame columns
    wavenumbers = np.array([float(col) for col in raman_df.columns if col not in ['Scan Number', 'Seconds']])

    # Initialize lists to store results
    scan_numbers = []
    gband_heights = []
    gband_areas = []
    bg_intensities = []
    fluo_initials = []
    fluo_bg_subtracted = []
    fluo_final = []
    fluo_to_gband_ratios = []

    # Define region masks using wavenumbers
    bg_wavenumber_range = (wavenumbers >= bg_wavenumber_min) & (wavenumbers <= bg_wavenumber_max)
    fluo_wavenumber_range = (wavenumbers >= fluo_wavenumber_min) & (wavenumbers <= fluo_wavenumber_max)
    gband_wavenumber_window = (wavenumbers >= gband_wavenumber_min) & (wavenumbers <= gband_wavenumber_max)

    # Iterate over all scan numbers
    for scan_number in raman_df['Scan Number']:
        # Extract scan data
        scan_data = raman_df[raman_df['Scan Number'] == scan_number]
        if scan_data.empty:
            print(f"Scan Number {scan_number} not found in the data! Skipping...")
            continue
        
        # Extract time and intensities
        datetime = scan_data.index[0]
        intensities_raw = scan_data.iloc[0, 2:].values
        
        # Apply Savitzky-Golay smoothing
        intensities_smooth = savgol_filter(intensities_raw, window_size, poly_order_sg)
        
        # Calculate average background intensity (using wavenumber range)
        bg_intensity = np.mean(intensities_smooth[bg_wavenumber_range])
        
        # Subtract background from smoothed spectrum (background correction)
        intensities_bg_corrected = intensities_smooth - bg_intensity
        
        # Perform lieberfit for fluorescence region on background-corrected spectrum (using wavenumber range)
        fluo_spectrum = intensities_bg_corrected[fluo_wavenumber_range]
        corrected_fluo, baseline_fluo_partial = lieberfit(fluo_spectrum, order=poly_order, tot_iter=tot_iter)
        # Interpolate baseline to full spectrum (set to 0 outside range)
        baseline_fluo = np.zeros_like(wavenumbers)
        wavenumbers_fluo_range = wavenumbers[fluo_wavenumber_range]
        baseline_fluo[fluo_wavenumber_range] = np.interp(wavenumbers_fluo_range, wavenumbers_fluo_range, baseline_fluo_partial)
        corrected_spectrum_fluo = intensities_bg_corrected - baseline_fluo
        
        # Extract G-band height and area from corrected spectrum (using wavenumber window)
        gband_peak_idx = np.argmax(corrected_spectrum_fluo[gband_wavenumber_window])
        gband_height = corrected_spectrum_fluo[gband_wavenumber_window][gband_peak_idx]
        gband_area = np.trapezoid(corrected_spectrum_fluo[gband_wavenumber_window], x=wavenumbers[gband_wavenumber_window])
        
        # Calculate initial SWNT fluorescence (using wavenumber range)
        fluo_initial = np.trapezoid(intensities_smooth[fluo_wavenumber_range], x=wavenumbers[fluo_wavenumber_range])
        
        # Calculate background-subtracted SWNT fluorescence (using background-corrected spectrum)
        fluo_bg_subtracted_area = np.trapezoid(intensities_bg_corrected[fluo_wavenumber_range], x=wavenumbers[fluo_wavenumber_range])
        
        # Calculate final SWNT fluorescence (subtract G-band area)
        fluo_final_area = fluo_bg_subtracted_area - gband_area
        
        # Calculate ratio of corrected fluorescence to G-band area
        fluo_to_gband_ratio = fluo_bg_subtracted_area / gband_area if gband_area != 0 else 0
        
        # Append results
        scan_numbers.append(scan_number)
        gband_heights.append(gband_height)
        gband_areas.append(gband_area)
        bg_intensities.append(bg_intensity)
        fluo_initials.append(fluo_initial)
        fluo_bg_subtracted.append(fluo_bg_subtracted_area)
        fluo_final.append(fluo_final_area)
        fluo_to_gband_ratios.append(fluo_to_gband_ratio)

    # Create results DataFrame with Datetime as index and Scan Number as column
    results_df = pd.DataFrame({
        'Scan Number': scan_numbers,
        'G-band Height': gband_heights,
        'G-band Area': gband_areas,
        'Average Background Intensity': bg_intensities,
        'Initial SWNT Fluorescence': fluo_initials,
        'Background-Subtracted SWNT Fluorescence': fluo_bg_subtracted,
        'Final SWNT Fluorescence': fluo_final,
        'Fluorescence to G-band Ratio': fluo_to_gband_ratios
    }, index=raman_df.index)

    return results_df


# ============================================================================
# Peak Fitting Functions (moved from main_test_v3.py for v4)
# ============================================================================

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


# ============================================================================
# Core Processing Functions for v4 (moved from main_test_v3.py)
# ============================================================================

def process_scan_v4(scan_number, raman_df, config, 
                   cached_wavenumbers_full=None, cached_wavenumber_filter=None, cached_wavenumbers=None,
                   verbose=False):
    """
    Process a single scan using the v4 workflow (normalized data with ratios).
    This function processes normalized data and uses Lorentzian peak fitting.
    
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
        gband_area = gband_peak_1600_normalized.get('area') if gband_peak_1600_normalized else None
        if gband_area is None or (isinstance(gband_area, (int, float)) and np.isnan(gband_area)):
            gband_area_normalized = 0
        else:
            gband_area_normalized = float(gband_area)
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


def aggregate_summaries_to_dataframe_v4(summaries, raman_df, x_axis_type='datetime'):
    """
    Aggregate summaries from batch processing into a DataFrame (normalized data only).
    
    Parameters:
    -----------
    summaries : list of dict
        List of summary dictionaries from process_scan_v4()
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
        # Keep 'Scan Number' as a column even when it's the index (for CSV export)
        scan_numbers = df['Scan Number'].copy()
        df.set_index('Scan Number', inplace=True)
        df['Scan Number'] = scan_numbers  # Add it back as a column
        df.sort_index(inplace=True)
    else:  # datetime (default)
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True)
    
    return df
