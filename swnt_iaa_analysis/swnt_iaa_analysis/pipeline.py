"""High-level pipeline for SWNT IAA Raman analysis."""

import os
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

import numpy as np
import pandas as pd

from .core.loader import load_raman_dataset, load_temperature_data
from .core.preprocessing import apply_savgol_filter
from .core.baseline import lieberfit, apply_gaussian_smoothing
from .core.utils import remove_spikes_hampel, correct_baseline_shifts, apply_corrections_at_jump_indices, smooth_signal
from .analysis.peaks import find_peak_lorentzian
from .analysis.ratios import calculate_ratios
from .analysis.fourier import compute_fourier_transform, compute_diurnal_average
from .io.config import load_profile_config
from .io.exporter import export_results, export_fft_results, load_processed_data
from .visualization.plotting import plot_ratio_timeseries, plot_signal_correction_comparison, plot_representative_raman_spectrum, plot_fft_analysis, plot_spike_removal_comparison, plot_processing_stages

logger = logging.getLogger(__name__)


def add_timestamp_to_output_dir(output_dir: str) -> str:
    """Add a timestamp suffix to the output directory name."""
    if not output_dir:
        return output_dir
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    normalized = os.path.normpath(output_dir)
    dir_path = os.path.dirname(normalized)
    base_name = os.path.basename(normalized)
    base_name_with_timestamp = f"{base_name}_{timestamp}"
    
    if dir_path and dir_path != '.':
        result = os.path.join(dir_path, base_name_with_timestamp)
    else:
        result = base_name_with_timestamp
    
    if output_dir.startswith('./') and not result.startswith('./'):
        result = f"./{result}"
    
    return result


class RamanPipeline:
    """High-level pipeline for Raman analysis."""
    
    def __init__(
        self,
        config_path: str,
        profile_name: str,
        output_dir: Optional[str] = None,
        algorithm: str = 'v4',
    ):
        """
        Initialize pipeline.
        
        Parameters:
        -----------
        config_path : str
            Path to YAML config file
        profile_name : str
            Profile name to use
        output_dir : str, optional
            Output directory (will have timestamp added)
        algorithm : str
            Algorithm version ('v3' or 'v4', default: 'v4')
        """
        self.config = load_profile_config(config_path, profile_name)
        self.profile_name = profile_name
        self.algorithm = algorithm
        self._output_dir_override = output_dir  # Store override if provided
        
        # Output directory will be set after loading data (to use data file's directory)
        self.output_dir = None
        
        # Initialize results
        self.dataset = None
        self.temp_dataset = None
        self.results = None
        self.fft_results = None
        self.representative_spectrum_data = None  # Store spectrum data for representative plot
        
        logger.info(f"Initialized RamanPipeline with profile: {profile_name}")
    
    def run(self):
        """Run the complete analysis pipeline."""
        start_time = time.time()
        logger.info("Starting pipeline execution...")
        
        # 1. Load data
        print("Loading Raman dataset...")
        logger.info("Loading Raman dataset...")
        self.dataset = load_raman_dataset(self.config)
        
        # 1a. Setup output directory (after we know where the data file is)
        if self.output_dir is None:
            if self._output_dir_override is not None:
                # Use user-specified output directory
                output_dir = self._output_dir_override
            else:
                # Use the same directory as the Raman data file
                data_file_path = Path(self.dataset.source_path)
                data_dir = data_file_path.parent
                # Use results_v4_{timestamp} format for v4 algorithm
                if self.algorithm == 'v4':
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_dir = str(data_dir / f"results_v4_{timestamp}")
                else:
                    output_dir = str(data_dir / f"results_{self.profile_name}")
            
            if self.algorithm != 'v4':  # Only add timestamp if not v4 (v4 already has timestamp)
                self.output_dir = Path(add_timestamp_to_output_dir(output_dir))
            else:
                self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {self.output_dir}")
        
        # 1b. Load Temperature data (optional)
        logger.info("Loading temperature data...")
        self.temp_dataset = load_temperature_data(self.config)
        if self.temp_dataset is not None:
            logger.info(f"Loaded {len(self.temp_dataset)} temperature records")
        else:
            logger.info("No temperature data configured or found")
        
        # 2. Process scans
        print(f"Processing scans using algorithm {self.algorithm}...")
        logger.info(f"Processing scans using algorithm {self.algorithm}...")
        if self.algorithm == 'v4':
            self.results = self._process_v4()
        else:
            self.results = self._process_v3()
        print(f"  Processed {len(self.results)} scans")
        
        # 2a. Apply signal corrections (spike removal and baseline correction) for v4
        if self.algorithm == 'v4':
            print("Applying signal corrections (spike removal and baseline correction)...")
            logger.info("Applying signal corrections (spike removal and baseline correction)...")
            # Store original data for comparison plots (original columns are preserved, corrected ones created with _BaselineCorrected suffix)
            results_original = self.results.copy()
            self.results, jump_info_dict, spike_info_dict, results_after_spikes = self._apply_signal_corrections(self.results)
            # Plot before/after comparison
            print("Generating spike removal comparison plots...")
            logger.info("Generating spike removal comparison plots...")
            plot_spike_removal_comparison(results_original, results_after_spikes, self.output_dir, self.config, spike_info_dict)
            print("Generating signal correction comparison plots...")
            logger.info("Generating signal correction comparison plots...")
            plot_signal_correction_comparison(results_original, self.results, self.output_dir, self.config, jump_info_dict)
            print("Generating processing stages plots...")
            logger.info("Generating processing stages plots...")
            plot_processing_stages(results_original, results_after_spikes, self.results, self.output_dir, self.config)
        
        # 3. Calculate ratios
        print("Calculating ratios...")
        logger.info("Calculating ratios...")
        self.results = calculate_ratios(self.results)
        
        # 4. Analyze (FFT)
        print("Computing Fourier transforms...")
        logger.info("Computing Fourier transforms...")
        self.fft_results = self._compute_fft()
        
        # 5. Export and plot
        print("Exporting results...")
        logger.info("Exporting results...")
        self._export_results()
        print("Generating plots...")
        logger.info("Generating plots...")
        self._plot_results()
        
        # 5a. Plot representative Raman spectrum (for v4)
        if self.algorithm == 'v4' and self.representative_spectrum_data is not None:
            print("Generating representative Raman spectrum plot...")
            logger.info("Generating representative Raman spectrum plot...")
            plot_representative_raman_spectrum(self.representative_spectrum_data, self.output_dir, self.config)
        
        # 5b. Plot FFT analysis (for v4)
        if self.algorithm == 'v4' and self.fft_results:
            print("Generating FFT analysis plots...")
            logger.info("Generating FFT analysis plots...")
            for key, fft_result in self.fft_results.items():
                ratio_name = 'Fluorescence to G-band Ratio' if key == 'gband' else f'Ratio ({key})'
                plot_fft_analysis(self.results, fft_result, self.output_dir, self.config, ratio_name=ratio_name)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline execution complete! Elapsed time: {elapsed_time:.2f} seconds")
        return self.results
    
    def _process_v4(self) -> pd.DataFrame:
        """Process scans using v4 algorithm."""
        from scipy.integrate import trapezoid
        
        raman_df = self.dataset.spectra
        config = self.config
        
        # Get raw wavenumbers from file (usually based on device's assumed excitation wavelength)
        wavenumbers_full = np.array([float(col) for col in raman_df.columns 
                                    if col not in ['Scan Number', 'Seconds']])
        
        # Get excitation wavelength parameters with defaults
        # "excitation_nm": The laser you ACTUALLY used (e.g. 830)
        # "recorded_excitation_nm": The laser the file metadata implies (e.g. 785)
        processing_cfg = config.get('processing', {})
        actual_excitation = processing_cfg.get('excitation_nm', config.get('excitation_nm', 830))
        assumed_excitation = processing_cfg.get('recorded_excitation_nm', config.get('recorded_excitation_nm', actual_excitation))
        
        # Apply wavenumber correction if excitation wavelengths differ
        if actual_excitation != assumed_excitation:
            logger.info(f"Correcting wavenumbers: {assumed_excitation}nm -> {actual_excitation}nm")
            shift_val = (1e7 / assumed_excitation) - (1e7 / actual_excitation)
            wavenumbers_full = wavenumbers_full - shift_val
            logger.info(f"Applied wavenumber shift: {shift_val:.2f} cm^-1")
        
        # Filter wavenumbers (start at 250 cm^-1)
        wavenumber_filter = wavenumbers_full >= 250
        wavenumbers = wavenumbers_full[wavenumber_filter]
        
        # Get skip_scans parameter from config (default: 500)
        skip_scans = processing_cfg.get('skip_scans', 500)
        
        summaries = []
        scan_numbers = sorted(raman_df['Scan Number'].unique())
        
        # Skip first N scans
        if skip_scans > 0 and len(scan_numbers) > skip_scans:
            scan_numbers = scan_numbers[skip_scans:]
            logger.info(f"Skipped first {skip_scans} scans. Processing {len(scan_numbers)} scans.")
        
        # Track if we've stored representative spectrum data
        representative_stored = False
        
        for scan_number in scan_numbers:
            scan_data = raman_df[raman_df['Scan Number'] == scan_number]
            if scan_data.empty:
                continue
            
            datetime_val = scan_data.index[0]
            seconds = scan_data['Seconds'].iloc[0]
            intensities_raw_full = scan_data.iloc[0, 2:].values
            intensities_raw = intensities_raw_full[wavenumber_filter]
            
            # Savitzky-Golay smoothing
            window_size = config.get('window_size', 25)
            poly_order_sg = config.get('poly_order_sg', 2)
            intensities_smooth = apply_savgol_filter(intensities_raw, window_size, poly_order_sg)
            
            # Normalization
            baseline_bg_mask = (wavenumbers >= 250) & (wavenumbers <= 1250)
            if np.any(baseline_bg_mask):
                bg_intensity_avg = np.mean(intensities_smooth[baseline_bg_mask])
            else:
                bg_intensity_avg = np.mean(intensities_smooth)
            
            intensities_normalized = intensities_smooth / bg_intensity_avg if bg_intensity_avg > 0 else intensities_smooth
            
            # Lieberfit
            poly_order = config.get('poly_order', 5)
            tot_iter = config.get('tot_iter', 100)
            corrected_normalized, baseline_normalized = lieberfit(intensities_normalized, 
                                                                  order=poly_order, tot_iter=tot_iter)
            
            # Find peaks
            raman_peak_850 = find_peak_lorentzian(wavenumbers, corrected_normalized, 800, 900)
            gband_peak_1600 = find_peak_lorentzian(wavenumbers, corrected_normalized, 1550, 1650)
            
            # Store representative spectrum data (from first scan after skipping)
            if not representative_stored:
                self.representative_spectrum_data = {
                    'wavenumbers': wavenumbers,
                    'intensities_normalized': intensities_normalized,
                    'baseline_normalized': baseline_normalized,
                    'corrected_normalized': corrected_normalized,
                    'raman_peak_850': raman_peak_850,
                    'gband_peak_1600': gband_peak_1600,
                    'scan_number': scan_number
                }
                representative_stored = True
                logger.info(f"Stored representative spectrum data from scan {scan_number}")
            
            # Calculate fluorescence
            fluo_mask = wavenumbers >= 1250
            if np.any(fluo_mask):
                fluo_auc = trapezoid(intensities_normalized[fluo_mask], x=wavenumbers[fluo_mask])
                gband_area = gband_peak_1600.get('area', 0) if gband_peak_1600 else 0
                if gband_area is None or (isinstance(gband_area, (int, float)) and np.isnan(gband_area)):
                    gband_area = 0
                fluorescence_intensity = fluo_auc - float(gband_area)
            else:
                fluorescence_intensity = np.nan
            
            summary = {
                'scan_number': scan_number,
                'datetime': datetime_val,
                'seconds': seconds,
                'average_background_intensity': bg_intensity_avg,
                'normalized_background_intensity': 1.0 if bg_intensity_avg > 0 else np.nan,
                'normalized_fluorescence_intensity': fluorescence_intensity,
                'normalized_gband_intensity': gband_peak_1600.get('intensity', np.nan) if gband_peak_1600 else np.nan,
                'normalized_gband_area': gband_peak_1600.get('area', np.nan) if gband_peak_1600 else np.nan,
                'normalized_gband_wavenumber': gband_peak_1600.get('wavenumber', np.nan) if gband_peak_1600 else np.nan,
                'normalized_raman_peak_850_intensity': raman_peak_850.get('intensity', np.nan) if raman_peak_850 else np.nan,
                'normalized_raman_peak_850_area': raman_peak_850.get('area', np.nan) if raman_peak_850 else np.nan,
                'normalized_raman_peak_850_wavenumber': raman_peak_850.get('wavenumber', np.nan) if raman_peak_850 else np.nan,
            }
            summaries.append(summary)
        
        # Convert to DataFrame
        rows = []
        for summary in summaries:
            row = {
                'Scan Number': summary['scan_number'],
                'Seconds': summary['seconds'],
                'Datetime': pd.to_datetime(summary['datetime']),
                'Average_Background_Intensity_250_1250_cm-1': summary['average_background_intensity'],
                'Normalized_Background_Intensity': summary['normalized_background_intensity'],
                'Normalized_Fluorescence_Intensity': summary['normalized_fluorescence_intensity'],
                'Normalized_Gband_Intensity': summary['normalized_gband_intensity'],
                'Normalized_Gband_Area': summary['normalized_gband_area'],
                'Normalized_Gband_Wavenumber': summary['normalized_gband_wavenumber'],
                'Normalized_Raman_Peak_850_Intensity': summary['normalized_raman_peak_850_intensity'],
                'Normalized_Raman_Peak_850_Area': summary['normalized_raman_peak_850_area'],
                'Normalized_Raman_Peak_850_Wavenumber': summary['normalized_raman_peak_850_wavenumber'],
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        x_axis_type = self.config.get('timeseries_x_axis', 'datetime')
        if x_axis_type == 'scan_number':
            df.set_index('Scan Number', inplace=True)
        else:
            df.set_index('Datetime', inplace=True)
        
        return df
    
    def _apply_signal_corrections(self, df: pd.DataFrame) -> tuple:
        """
        Apply spike removal and baseline correction to specified columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with normalized intensity columns
            
        Returns:
        --------
        tuple
            (DataFrame with corrected columns, dict of jump information per column)
        """
        config = self.config
        # After flattening, processing params are at top level, but nested structure may be in sections
        processing_cfg = config.get('processing', {}) or config.get('sections', {}).get('processing', {})
        
        # Get parameters from config with defaults
        # Note: After flattening by _flatten_profile, processing params are at top level (in config),
        # but nested structure is also preserved in config['sections']['processing']
        # So we check both: processing_cfg (nested) and config (flattened top-level)
        # Also supports baseline_correction_* prefix (new) and baseline_* (old) for backwards compatibility
        def get_param(nested_key, top_key=None, default=None):
            """Helper to get parameter from nested or flattened location."""
            if top_key is None:
                top_key = nested_key
            # Check nested processing section first, then flattened top level, then use default
            if nested_key in processing_cfg:
                return processing_cfg[nested_key]
            if top_key in config:
                return config[top_key]
            return default
        
        spike_window = get_param('spike_window', default=5)
        spike_threshold = get_param('spike_threshold', default=3.0)
        spike_min_length = get_param('spike_min_length', default=1)
        spike_max_length = get_param('spike_max_length', default=10)
        timeseries_sg_window = get_param('timeseries_sg_window', default=5)
        timeseries_sg_poly_order = get_param('timeseries_sg_poly_order', default=2)
        timeseries_smoothing_method = get_param('timeseries_smoothing_method', default='savitzky').lower()
        timeseries_gaussian_sigma = get_param('timeseries_gaussian_sigma', default=10)
        baseline_threshold_multiplier = (get_param('baseline_correction_threshold') or
                                        get_param('baseline_threshold_multiplier', default=5.0))
        baseline_window_size = (get_param('baseline_correction_window') or
                               get_param('baseline_window_size', default=5))
        baseline_smooth_first = (get_param('baseline_correction_smooth_first') if 
                                'baseline_correction_smooth_first' in processing_cfg or 'baseline_correction_smooth_first' in config else
                                get_param('baseline_smooth_first', default=True))
        baseline_smooth_window = (get_param('baseline_correction_smooth_window') or
                                 get_param('baseline_smooth_window', default=11))
        baseline_smooth_poly_order = (get_param('baseline_correction_smooth_poly_order') or
                                     get_param('baseline_smooth_poly_order', default=2))
        baseline_correct_smoothed = (get_param('baseline_correction_correct_smoothed') if
                                    'baseline_correction_correct_smoothed' in processing_cfg or 'baseline_correction_correct_smoothed' in config else
                                    get_param('baseline_correct_smoothed', default=False))
        baseline_detect_cumulative_jumps = (get_param('baseline_detect_cumulative_jumps') if
                                           'baseline_detect_cumulative_jumps' in processing_cfg or 'baseline_detect_cumulative_jumps' in config else
                                           True)
        baseline_cumulative_window = get_param('baseline_cumulative_window', default=5)
        
        # Columns to process
        reference_column = 'Normalized_Fluorescence_Intensity'
        columns_to_correct = [
            'Normalized_Fluorescence_Intensity',
            'Normalized_Gband_Area',
            'Normalized_Raman_Peak_850_Area'
        ]
        
        df_corrected = df.copy()
        df_after_spikes = df.copy()  # Store data after spike removal but before baseline correction
        jump_info_dict = {}  # Store jump information for each column
        spike_info_dict = {}  # Store spike information for each column
        df_jump_indices_shared = np.array([], dtype=int)  # Shared jump indices in DataFrame space
        
        # Process reference column first to detect jumps
        if reference_column in df.columns:
            logger.info(f"Processing reference column '{reference_column}' to detect baseline jumps...")
            signal_ref = df[reference_column].values
            
            # Handle NaN values
            mask_ref = ~np.isnan(signal_ref)
            if np.sum(mask_ref) >= 3:
                # Extract valid signal
                signal_valid_ref = signal_ref[mask_ref]
                # Apply chosen smoothing for spike detection
                if timeseries_smoothing_method == 'gaussian':
                    signal_valid_ref_sg = apply_gaussian_smoothing(
                        signal_valid_ref,
                        sigma=timeseries_gaussian_sigma,
                    )
                else:
                    signal_valid_ref_sg = smooth_signal(
                        signal_valid_ref,
                        window_size=timeseries_sg_window,
                        poly_order=timeseries_sg_poly_order,
                    )
                valid_indices_ref = np.where(mask_ref)[0]
                
                # Remove spikes from reference
                logger.info(f"Removing spikes from {reference_column}...")
                cleaned_signal_ref, spike_mask_ref = remove_spikes_hampel(
                    signal_valid_ref_sg,
                    window_size=spike_window,
                    threshold=spike_threshold,
                    min_spike_length=spike_min_length,
                    max_spike_length=spike_max_length,
                )
                n_spikes_ref = np.sum(spike_mask_ref)
                if n_spikes_ref > 0:
                    logger.info(f"  Removed {n_spikes_ref} spike(s) from {reference_column}")
                
                # Detect jumps on reference signal
                logger.info(f"Detecting baseline jumps in {reference_column}...")
                corrected_signal_ref, jump_indices_valid_ref, jump_info_ref, smoothed_signal_ref = correct_baseline_shifts(
                    cleaned_signal_ref,
                    threshold_multiplier=baseline_threshold_multiplier,
                    window_size=baseline_window_size,
                    smooth_first=baseline_smooth_first,
                    smooth_window=baseline_smooth_window,
                    smooth_poly_order=baseline_smooth_poly_order,
                    correct_smoothed=baseline_correct_smoothed,
                    detect_cumulative_jumps=baseline_detect_cumulative_jumps,
                    cumulative_window=baseline_cumulative_window
                )
                
                # Map jump indices from valid signal space to DataFrame indices
                if len(jump_indices_valid_ref) > 0:
                    df_jump_indices_shared = valid_indices_ref[jump_indices_valid_ref]
                    logger.info(f"  Detected {len(df_jump_indices_shared)} baseline jump(s) on {reference_column}")
                else:
                    df_jump_indices_shared = np.array([], dtype=int)
                    logger.info(f"  No baseline jumps detected on {reference_column}")
                
                # Store spike-removed version (before baseline correction) for comparison plots
                cleaned_full_ref = np.full_like(signal_ref, np.nan)
                cleaned_full_ref[mask_ref] = cleaned_signal_ref
                df_after_spikes[reference_column] = cleaned_full_ref
                
                # Store results for reference column (after baseline correction)
                corrected_full_ref = np.full_like(signal_ref, np.nan)
                corrected_full_ref[mask_ref] = corrected_signal_ref
                corrected_col_name_ref = reference_column + '_BaselineCorrected'
                df_corrected[corrected_col_name_ref] = corrected_full_ref
                
                # Map spike indices from valid signal space to DataFrame indices
                df_spike_indices_ref = valid_indices_ref[spike_mask_ref] if np.sum(spike_mask_ref) > 0 else np.array([], dtype=int)
                jump_info_dict[reference_column] = {
                    'jump_indices': df_jump_indices_shared,
                    'jump_info': jump_info_ref
                }
                spike_info_dict[reference_column] = {
                    'spike_indices': df_spike_indices_ref
                }
        
        # Process all columns (reference column is skipped if already processed)
        for col in columns_to_correct:
            # Skip reference column if already processed
            if col == reference_column and col in jump_info_dict:
                continue
            
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping...")
                continue
            
            signal = df[col].values
            
            # Handle NaN values
            mask = ~np.isnan(signal)
            if np.sum(mask) < 3:
                logger.warning(f"Column '{col}' has insufficient data, skipping...")
                continue
            
            # Extract valid signal
            signal_valid = signal[mask]
            # Apply chosen smoothing for spike detection
            if timeseries_smoothing_method == 'gaussian':
                signal_valid_sg = apply_gaussian_smoothing(
                    signal_valid,
                    sigma=timeseries_gaussian_sigma,
                )
            else:
                signal_valid_sg = smooth_signal(
                    signal_valid,
                    window_size=timeseries_sg_window,
                    poly_order=timeseries_sg_poly_order,
                )
            valid_indices = np.where(mask)[0]
            
            # Step 1: Remove spikes
            logger.info(f"Removing spikes from {col}...")
            cleaned_signal, spike_mask = remove_spikes_hampel(
                signal_valid_sg,
                window_size=spike_window,
                threshold=spike_threshold,
                min_spike_length=spike_min_length,
                max_spike_length=spike_max_length,
            )
            n_spikes = np.sum(spike_mask)
            if n_spikes > 0:
                logger.info(f"  Removed {n_spikes} spike(s) from {col}")
            
            # Map spike indices from valid signal space to DataFrame indices
            df_spike_indices = valid_indices[spike_mask] if np.sum(spike_mask) > 0 else np.array([], dtype=int)
            spike_info_dict[col] = {
                'spike_indices': df_spike_indices
            }
            
            # Store spike-removed version (before baseline correction) for comparison plots
            cleaned_full = np.full_like(signal, np.nan)
            cleaned_full[mask] = cleaned_signal
            df_after_spikes[col] = cleaned_full
            
            # Step 2: Correct baseline shifts
            if col == reference_column:
                # For reference column, use full detection
                logger.info(f"Detecting baseline jumps in {col}...")
                corrected_signal, jump_indices, jump_info, smoothed_signal = correct_baseline_shifts(
                    cleaned_signal,
                    threshold_multiplier=baseline_threshold_multiplier,
                    window_size=baseline_window_size,
                    smooth_first=baseline_smooth_first,
                    smooth_window=baseline_smooth_window,
                    smooth_poly_order=baseline_smooth_poly_order,
                    correct_smoothed=baseline_correct_smoothed,
                    detect_cumulative_jumps=baseline_detect_cumulative_jumps,
                    cumulative_window=baseline_cumulative_window
                )
                # Update shared jump indices
                if len(jump_indices) > 0:
                    df_jump_indices_shared = valid_indices[jump_indices]
            else:
                # For other columns, apply corrections at shared jump indices from reference column
                logger.info(f"Applying baseline corrections to {col} using jump indices from {reference_column}...")
                # Map DataFrame jump indices to valid signal indices for this column
                # Only include jumps that fall within valid data points for this column
                jump_indices_valid_this_col = []
                for df_idx in df_jump_indices_shared:
                    # Find where this DataFrame index appears in valid_indices
                    valid_pos = np.where(valid_indices == df_idx)[0]
                    if len(valid_pos) > 0:
                        jump_indices_valid_this_col.append(valid_pos[0])
                
                jump_indices_valid_this_col = np.array(jump_indices_valid_this_col, dtype=int)
                
                if len(jump_indices_valid_this_col) > 0:
                    logger.info(f"  Applying corrections at {len(jump_indices_valid_this_col)} jump location(s) in {col}")
                else:
                    logger.info(f"  No valid jump locations found in {col} (all jumps fall on NaN values)")
                
                # Apply corrections at these jump indices
                corrected_signal, jump_info, smoothed_signal = apply_corrections_at_jump_indices(
                    cleaned_signal,
                    jump_indices_valid_this_col,
                    window_size=baseline_window_size,
                    smooth_first=baseline_smooth_first,
                    smooth_window=baseline_smooth_window,
                    smooth_poly_order=baseline_smooth_poly_order,
                    correct_smoothed=baseline_correct_smoothed
                )
                jump_indices = jump_indices_valid_this_col
            
            # #region agent log
            _log_path = Path(__file__).parent.parent.parent / '.cursor' / 'debug.log'
            _log_path.parent.mkdir(parents=True, exist_ok=True)
            _max_diff_cleaned_corrected = np.nanmax(np.abs(corrected_signal - cleaned_signal)) if len(corrected_signal) == len(cleaned_signal) else -1
            _are_identical_cleaned_corrected = np.allclose(corrected_signal, cleaned_signal, atol=1e-10) if len(corrected_signal) == len(cleaned_signal) else False
            _step_changes = [j.get('step_change', 0) for j in jump_info] if jump_info else []
            with open(_log_path, 'a') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1,H3", "location": f"{__file__}:405", "message": "pipeline: after correct_baseline_shifts", "data": {"column": col, "num_jumps": len(jump_indices), "max_diff_cleaned_vs_corrected": float(_max_diff_cleaned_corrected), "are_identical": bool(_are_identical_cleaned_corrected), "step_changes": [float(s) for s in _step_changes], "correct_smoothed": baseline_correct_smoothed, "cleaned_sample": cleaned_signal[:5].tolist() if len(cleaned_signal) >= 5 else cleaned_signal.tolist(), "corrected_sample": corrected_signal[:5].tolist() if len(corrected_signal) >= 5 else corrected_signal.tolist()}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
            # #endregion
            
            if len(jump_indices) > 0:
                logger.info(f"  Detected {len(jump_indices)} baseline jump(s) in {col}")
            
            # Map jump indices back to original DataFrame indices
            if len(jump_indices) > 0:
                # jump_indices are relative to valid_indices, need to map to DataFrame index
                df_jump_indices = valid_indices[jump_indices]
                jump_info_dict[col] = {
                    'jump_indices': df_jump_indices,
                    'jump_info': jump_info
                }
            else:
                jump_info_dict[col] = {
                    'jump_indices': np.array([], dtype=int),
                    'jump_info': []
                }
            
            # Map corrected values back to full array (preserving NaN positions)
            corrected_full = np.full_like(signal, np.nan)
            corrected_full[mask] = corrected_signal
            
            # Create new column with _BaselineCorrected suffix (preserve original column)
            corrected_col_name = col + '_BaselineCorrected'
            df_corrected[corrected_col_name] = corrected_full
            
            # #region agent log
            _log_path = Path(__file__).parent.parent.parent / '.cursor' / 'debug.log'
            _orig_vals = df_corrected[col].values[~np.isnan(df_corrected[col].values)]
            _corr_vals = df_corrected[corrected_col_name].values[~np.isnan(df_corrected[corrected_col_name].values)]
            _min_len = min(len(_orig_vals), len(_corr_vals)) if len(_orig_vals) > 0 and len(_corr_vals) > 0 else 0
            _max_diff_df = np.nanmax(np.abs(_orig_vals[:_min_len] - _corr_vals[:_min_len])) if _min_len > 0 else -1
            _are_identical_df = np.allclose(_orig_vals[:_min_len], _corr_vals[:_min_len], atol=1e-10) if _min_len > 0 else False
            with open(_log_path, 'a') as f:
                f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H3", "location": f"{__file__}:430", "message": "pipeline: after creating corrected column", "data": {"column": col, "corrected_col_name": corrected_col_name, "max_diff_df_columns": float(_max_diff_df), "are_identical_df": bool(_are_identical_df), "orig_sample": _orig_vals[:5].tolist() if len(_orig_vals) >= 5 else _orig_vals.tolist(), "corr_sample": _corr_vals[:5].tolist() if len(_corr_vals) >= 5 else _corr_vals.tolist()}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
            # #endregion
            
            # Original column remains unchanged (will be included in CSV export)
        
        return df_corrected, jump_info_dict, spike_info_dict, df_after_spikes
    
    def _process_v3(self) -> pd.DataFrame:
        """
        Process scans using v3 algorithm.
        
        Note: V3 algorithm is not yet implemented in the refactored package.
        To use V3, you can either:
        1. Implement the V3 logic here (based on the original main_test_v3.py)
        2. Use algorithm='v4' instead
        
        Raises:
        --------
        NotImplementedError
            V3 algorithm is not yet implemented in the refactored package
        """
        raise NotImplementedError(
            "V3 algorithm is not yet implemented in the refactored package. "
            "Please use algorithm='v4' or implement the V3 logic based on the original main_test_v3.py"
        )
    
    def _compute_fft(self) -> dict:
        """
        Compute Fourier transforms on processed signal.
        
        Processing pipeline:
        1. Extract baseline-corrected ratio
        2. Apply Gaussian smoothing
        3. Apply ALS baseline correction (to remove long-term trends)
        4. Mean detrending (in compute_fourier_transform)
        5. Compute FFT
        """
        if self.results is None or len(self.results) == 0:
            return {}
        
        x_axis_type = self.config.get('timeseries_x_axis', 'datetime')
        if x_axis_type == 'scan_number':
            return {}  # FFT only makes sense for datetime
        
        # Ensure datetime index
        if not pd.api.types.is_datetime64_any_dtype(self.results.index):
            if 'Datetime' in self.results.columns:
                self.results['Datetime'] = pd.to_datetime(self.results['Datetime'])
                self.results.set_index('Datetime', inplace=True)
        
        datetimes = self.results.index
        time_hours = np.array([(dt - datetimes.min()).total_seconds() / 3600.0 for dt in datetimes])
        
        fft_results = {}
        
        # Import required functions
        from .core.baseline import apply_gaussian_smoothing, apply_als_baseline
        
        # Process G-band ratio if available
        ratio_col = 'Fluorescence_to_Gband_Ratio_BaselineCorrected'
        if ratio_col not in self.results.columns:
            ratio_col = 'Fluorescence_to_Gband_Ratio'
        
        if ratio_col in self.results.columns:
            signal = self.results[ratio_col].values
            mask = ~np.isnan(signal)
            if np.sum(mask) >= 3:
                # Step 1: Extract valid signal (baseline-corrected ratio)
                signal_valid = signal[mask]
                time_hours_valid = time_hours[mask]
                
                # Step 2: Apply Gaussian smoothing
                processing_cfg = self.config.get('processing', {})
                sigma_gaussian = processing_cfg.get('sigma_gaussian', self.config.get('sigma_gaussian', 50))
                logger.info(f"Applying Gaussian smoothing (sigma={sigma_gaussian}) to {ratio_col} for FFT...")
                signal_smoothed = apply_gaussian_smoothing(signal_valid, sigma=sigma_gaussian)
                
                # Step 3: Apply ALS baseline correction (to remove long-term trends)
                lam_als = processing_cfg.get('lam_als', self.config.get('lam_als', 100000000))
                p_als = processing_cfg.get('p_als', self.config.get('p_als', 0.000100))
                niter_als = processing_cfg.get('niter_als', self.config.get('niter_als', 20))
                logger.info(f"Applying ALS baseline correction (lam={lam_als}, p={p_als}, niter={niter_als})...")
                als_baseline = apply_als_baseline(signal_smoothed, lam=lam_als, p=p_als, niter=niter_als)
                signal_als_corrected = signal_smoothed - als_baseline
                
                # Step 4 & 5: Mean detrending and FFT computation (done inside compute_fourier_transform)
                logger.info("Computing FFT on processed signal...")
                fft_result = compute_fourier_transform(
                    signal_als_corrected, 
                    time_hours_valid, 
                    top_peaks=self.config.get('fft_top_peaks', 10)
                )
                fft_results['gband'] = fft_result
        
        return fft_results
    
    def reprocess_from_csv(
        self,
        csv_path: Path,
        steps: List[str] = None,
        output_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Re-process existing processed_data.csv with selective steps.
        
        This method allows re-running time-series signal processing steps without
        re-running the time-consuming spectral processing.
        
        Parameters:
        -----------
        csv_path : Path
            Path to processed_data.csv file
        steps : list of str, optional
            List of steps to run. Valid steps:
            - 'spike_removal': Apply spike removal to normalized intensity columns
            - 'baseline_correction': Apply baseline correction
            - 'ratios': Recalculate ratios (requires spike_removal + baseline_correction)
            - 'fft': Recompute FFT analysis (requires ratios)
            If None, runs all steps.
        output_dir : str, optional
            Output directory. If None, creates new timestamped folder.
        
        Returns:
        --------
        pd.DataFrame
            Re-processed results DataFrame
        """
        if steps is None:
            steps = ['spike_removal', 'baseline_correction', 'ratios', 'fft']
        
        start_time = time.time()
        logger.info(f"Starting re-processing from CSV: {csv_path}")
        
        # Load processed data (do this once for validation and processing)
        print("Loading processed data from CSV...")
        logger.info("Loading processed data from CSV...")
        self.results = load_processed_data(csv_path)
        
        # Validate step dependencies after loading
        if 'ratios' in steps and ('spike_removal' not in steps or 'baseline_correction' not in steps):
            # Check if baseline-corrected columns already exist
            has_baseline_corrected = any('_BaselineCorrected' in col for col in self.results.columns)
            if not has_baseline_corrected:
                raise ValueError(
                    "Step 'ratios' requires 'spike_removal' and 'baseline_correction'. "
                    "Either include those steps or ensure baseline-corrected columns exist in the CSV."
                )
        
        if 'fft' in steps and 'ratios' not in steps:
            has_ratios = any('Ratio' in col for col in self.results.columns)
            if not has_ratios:
                raise ValueError(
                    "Step 'fft' requires 'ratios'. "
                    "Either include 'ratios' step or ensure ratio columns exist in the CSV."
                )
        
        # Ensure index type matches config preference (if needed)
        # Note: The CSV might have been saved with a different index type than the config
        # For re-processing, we'll use whatever index is in the CSV
        # But we need to ensure we have the necessary columns for FFT (requires datetime)
        # FFT will handle this by checking x_axis_type in _compute_fft
        
        # Extract original columns if re-processing signal corrections
        if 'spike_removal' in steps or 'baseline_correction' in steps:
            # Remove existing _BaselineCorrected columns to start fresh
            baseline_corrected_cols = [col for col in self.results.columns if col.endswith('_BaselineCorrected')]
            if baseline_corrected_cols:
                logger.info(f"Removing existing baseline-corrected columns: {baseline_corrected_cols}")
                self.results = self.results.drop(columns=baseline_corrected_cols)
        
        # Setup output directory
        if output_dir is None:
            # Create new timestamped folder in same location as source CSV
            source_dir = csv_path.parent
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = str(source_dir / f"results_v4_{timestamp}_reprocess")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")
        
        # Store original for comparison plots
        results_original = self.results.copy() if ('spike_removal' in steps or 'baseline_correction' in steps) else None
        
        # Apply selected processing steps
        jump_info_dict = None
        spike_info_dict = None
        
        if 'spike_removal' in steps or 'baseline_correction' in steps:
            print("Applying signal corrections (spike removal and baseline correction)...")
            logger.info("Applying signal corrections...")
            self.results, jump_info_dict, spike_info_dict, results_after_spikes = self._apply_signal_corrections(self.results)
            
            if results_original is not None:
                # Generate spike removal comparison plot if spike removal was performed
                if 'spike_removal' in steps and spike_info_dict:
                    print("Generating spike removal comparison plots...")
                    logger.info("Generating spike removal comparison plots...")
                    plot_spike_removal_comparison(results_original, results_after_spikes, self.output_dir, self.config, spike_info_dict)
                
                # Generate signal correction comparison plot (includes baseline correction)
                if 'baseline_correction' in steps:
                    print("Generating signal correction comparison plots...")
                    logger.info("Generating signal correction comparison plots...")
                    plot_signal_correction_comparison(results_original, self.results, self.output_dir, self.config, jump_info_dict)
                # Always generate processing stages plot when we have spike & baseline info
                print("Generating processing stages plots...")
                logger.info("Generating processing stages plots...")
                plot_processing_stages(results_original, results_after_spikes, self.results, self.output_dir, self.config)
        
        if 'ratios' in steps:
            print("Calculating ratios...")
            logger.info("Calculating ratios...")
            # Remove existing ratio columns if re-calculating
            ratio_cols = [col for col in self.results.columns if 'Ratio' in col]
            if ratio_cols:
                logger.info(f"Removing existing ratio columns: {ratio_cols}")
                self.results = self.results.drop(columns=ratio_cols)
            self.results = calculate_ratios(self.results)
        
        if 'fft' in steps:
            print("Computing Fourier transforms...")
            logger.info("Computing Fourier transforms...")
            self.fft_results = self._compute_fft()
        else:
            self.fft_results = None
        
        # Export and plot
        print("Exporting results...")
        logger.info("Exporting results...")
        self._export_results()
        
        print("Generating plots...")
        logger.info("Generating plots...")
        self._plot_results()
        
        if self.fft_results:
            print("Generating FFT analysis plots...")
            logger.info("Generating FFT analysis plots...")
            for key, fft_result in self.fft_results.items():
                ratio_name = 'Fluorescence to G-band Ratio' if key == 'gband' else f'Ratio ({key})'
                plot_fft_analysis(self.results, fft_result, self.output_dir, self.config, ratio_name=ratio_name)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Re-processing complete! Elapsed time: {elapsed_time:.2f} seconds")
        
        return self.results
    
    def _export_results(self):
        """Export results to files."""
        if self.results is not None:
            export_results(self.results, self.output_dir)
        
        if self.fft_results:
            export_fft_results(self.fft_results, self.output_dir)
    
    def _plot_results(self):
        """Generate plots."""
        if self.results is None:
            return
        
        # #region agent log
        _log_path = Path(__file__).parent.parent.parent / '.cursor' / 'debug.log'
        _log_path.parent.mkdir(parents=True, exist_ok=True)
        _ratio_cols = [col for col in self.results.columns if 'Ratio' in col]
        _baseline_corrected_cols = [col for col in self.results.columns if 'BaselineCorrected' in col]
        with open(_log_path, 'a') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": f"{__file__}:561", "message": "pipeline: _plot_results - checking columns", "data": {"all_columns": list(self.results.columns), "ratio_columns": _ratio_cols, "baseline_corrected_columns": _baseline_corrected_cols, "num_ratio_cols": len(_ratio_cols), "num_baseline_corrected_cols": len(_baseline_corrected_cols)}, "timestamp": int(datetime.now().timestamp() * 1000)}) + '\n')
        # #endregion
        
        # Plot ratio timeseries
        ratio_cols = [col for col in self.results.columns if 'Ratio' in col]
        for col in ratio_cols:
            plot_ratio_timeseries(self.results, col, self.output_dir, self.config)
        
        # Plot normalized timeseries (for v4 algorithm)
        # Use Area-based columns for G-band and Raman peak 850 (based on Lorentzian fit area)
        if self.algorithm == 'v4':
            timeseries_cols = [
                'Normalized_Fluorescence_Intensity',  # Area-based (AUC calculation)
                'Normalized_Gband_Area',  # Area from Lorentzian fit
                'Normalized_Raman_Peak_850_Area'  # Area from Lorentzian fit
            ]
            for col in timeseries_cols:
                if col in self.results.columns:
                    plot_ratio_timeseries(self.results, col, self.output_dir, self.config)

