"""High-level pipeline for SWNT IAA Raman analysis."""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from .core.loader import load_raman_dataset, load_temperature_data
from .core.preprocessing import apply_savgol_filter
from .core.baseline import lieberfit
from .core.utils import remove_spikes_hampel, correct_baseline_shifts
from .analysis.peaks import find_peak_lorentzian
from .analysis.ratios import calculate_ratios
from .analysis.fourier import compute_fourier_transform, compute_diurnal_average
from .io.config import load_profile_config
from .io.exporter import export_results, export_fft_results
from .visualization.plotting import plot_ratio_timeseries, plot_signal_correction_comparison, plot_representative_raman_spectrum, plot_fft_analysis

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
        logger.info("Starting pipeline execution...")
        
        # 1. Load data
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
        logger.info(f"Processing scans using algorithm {self.algorithm}...")
        if self.algorithm == 'v4':
            self.results = self._process_v4()
        else:
            self.results = self._process_v3()
        
        # 2a. Apply signal corrections (spike removal and baseline correction) for v4
        if self.algorithm == 'v4':
            logger.info("Applying signal corrections (spike removal and baseline correction)...")
            # Store original data for comparison plots (original columns are preserved, corrected ones created with _BaselineCorrected suffix)
            results_original = self.results.copy()
            self.results, jump_info_dict = self._apply_signal_corrections(self.results)
            # Plot before/after comparison
            logger.info("Generating signal correction comparison plots...")
            plot_signal_correction_comparison(results_original, self.results, self.output_dir, self.config, jump_info_dict)
        
        # 3. Calculate ratios
        logger.info("Calculating ratios...")
        self.results = calculate_ratios(self.results)
        
        # 4. Analyze (FFT)
        logger.info("Computing Fourier transforms...")
        self.fft_results = self._compute_fft()
        
        # 5. Export and plot
        logger.info("Exporting results...")
        self._export_results()
        self._plot_results()
        
        # 5a. Plot representative Raman spectrum (for v4)
        if self.algorithm == 'v4' and self.representative_spectrum_data is not None:
            logger.info("Generating representative Raman spectrum plot...")
            plot_representative_raman_spectrum(self.representative_spectrum_data, self.output_dir, self.config)
        
        # 5b. Plot FFT analysis (for v4)
        if self.algorithm == 'v4' and self.fft_results:
            logger.info("Generating FFT analysis plots...")
            for key, fft_result in self.fft_results.items():
                ratio_name = 'Fluorescence to G-band Ratio' if key == 'gband' else f'Ratio ({key})'
                plot_fft_analysis(self.results, fft_result, self.output_dir, self.config, ratio_name=ratio_name)
        
        logger.info("Pipeline execution complete!")
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
        processing_cfg = config.get('processing', {})
        
        # Get parameters from config with defaults
        spike_window = processing_cfg.get('spike_window', 5)
        spike_threshold = processing_cfg.get('spike_threshold', 3.0)
        baseline_threshold_multiplier = processing_cfg.get('baseline_threshold_multiplier', 5.0)
        baseline_window_size = processing_cfg.get('baseline_window_size', 5)
        baseline_smooth_first = processing_cfg.get('baseline_smooth_first', True)
        baseline_smooth_window = processing_cfg.get('baseline_smooth_window', 11)
        baseline_smooth_poly_order = processing_cfg.get('baseline_smooth_poly_order', 2)
        baseline_correct_smoothed = processing_cfg.get('baseline_correct_smoothed', False)
        baseline_detect_cumulative_jumps = processing_cfg.get('baseline_detect_cumulative_jumps', True)
        baseline_cumulative_window = processing_cfg.get('baseline_cumulative_window', 5)
        
        # Columns to process
        columns_to_correct = [
            'Normalized_Fluorescence_Intensity',
            'Normalized_Gband_Area',
            'Normalized_Raman_Peak_850_Area'
        ]
        
        df_corrected = df.copy()
        jump_info_dict = {}  # Store jump information for each column
        
        for col in columns_to_correct:
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
            valid_indices = np.where(mask)[0]
            
            # Step 1: Remove spikes
            logger.info(f"Removing spikes from {col}...")
            cleaned_signal, spike_mask = remove_spikes_hampel(
                signal_valid,
                window_size=spike_window,
                threshold=spike_threshold
            )
            n_spikes = np.sum(spike_mask)
            if n_spikes > 0:
                logger.info(f"  Removed {n_spikes} spike(s) from {col}")
            
            # Step 2: Correct baseline shifts
            logger.info(f"Correcting baseline shifts in {col}...")
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
            
            # Original column remains unchanged (will be included in CSV export)
        
        return df_corrected, jump_info_dict
    
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

