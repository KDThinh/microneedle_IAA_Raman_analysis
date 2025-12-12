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
from .analysis.peaks import find_peak_lorentzian
from .analysis.ratios import calculate_ratios
from .analysis.fourier import compute_fourier_transform, compute_diurnal_average
from .io.config import load_profile_config
from .io.exporter import export_results, export_fft_results
from .visualization.plotting import plot_ratio_timeseries

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
                output_dir = str(data_dir / f"results_{self.profile_name}")
            
            self.output_dir = Path(add_timestamp_to_output_dir(output_dir))
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
        
        summaries = []
        scan_numbers = sorted(raman_df['Scan Number'].unique())
        
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
        """Compute Fourier transforms."""
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
        
        # Process G-band ratio if available
        ratio_col = 'Fluorescence_to_Gband_Ratio_BaselineCorrected'
        if ratio_col not in self.results.columns:
            ratio_col = 'Fluorescence_to_Gband_Ratio'
        
        if ratio_col in self.results.columns:
            signal = self.results[ratio_col].values
            mask = ~np.isnan(signal)
            if np.sum(mask) >= 3:
                fft_result = compute_fourier_transform(signal[mask], time_hours[mask], 
                                                      top_peaks=self.config.get('fft_top_peaks', 10))
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

