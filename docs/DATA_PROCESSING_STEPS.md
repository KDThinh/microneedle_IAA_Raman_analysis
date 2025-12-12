# Data Processing Steps

This document outlines the complete data processing pipeline for Raman/fluorescence spectroscopy analysis of SWNT-IAA nanosensor data.

## Overview

The pipeline processes Raman spectroscopy data to extract fluorescence-to-G-band ratios, correct for baselines, smooth the data, and perform time-series analysis including diurnal averaging and Fourier transform analysis.

## Processing Flow

### 1. Configuration and Data Loading

#### 1.1 Configuration Loading
- Load profile configuration from `pipeline.yml`
- Support for profile inheritance and command-line overrides
- Extract processing parameters (window sizes, polynomial orders, etc.)
- Configure plotting axes/output paths (e.g., `outputs.timeseries_x_axis`)

#### 1.2 Data Ingestion
- **Input**: Raw Raman data file (TSV format with wavenumber columns)
- Load Raman spectra data into DataFrame
- Parse datetime from filename or metadata
- Validate data structure (Scan Number, Seconds, wavenumber columns)
- Extract metadata (plant_type, treatment, temp_hum_control, light_cycle, replicate_number)
- Handle optional temperature/humidity data file

#### 1.3 Output Directory Setup
- Create timestamped output directory: `processed_results_YYYYMMDD_HHMMSS`
- All outputs saved to this directory to prevent overwrites

#### 1.4 Data Range Selection
- Prompt user for start/end datetime (with defaults)
- Filter DataFrame to selected time range

---

### 2. Spectral Processing (Single Scan)

**Function**: `process_raman_data()` in `src/pipeline/processing.py`

#### 2.1 Raw Data Extraction
- Extract spectrum for specified scan number
- Get wavenumber array (excluding 'Scan Number' and 'Seconds' columns)
- Convert wavenumbers to emission wavelengths (nm) using excitation wavelength (default: 830 nm)

#### 2.2 Savitzky-Golay Smoothing
- Apply Savitzky-Golay filter to raw intensities
- **Parameters**: `window_size` (default: 25), `poly_order_sg` (default: 2)
- Purpose: Reduce noise in spectral data

#### 2.3 G-band Baseline Correction (Lieberfit)
- Focus on G-band region: 1500-2100 cm⁻¹ (~952-958 nm emission)
- Apply Lieberfit algorithm (iterative polynomial baseline correction)
- **Parameters**: `poly_order` (default: 5), `tot_iter` (default: 100)
- Interpolate baseline to full spectrum range
- Calculate baseline-corrected spectrum

#### 2.4 Output
- Wavenumbers, emission wavelengths (nm)
- Raw intensities, smoothed intensities
- G-band baseline, corrected spectrum
- Datetime and seconds for the scan

---

### 3. Batch Spectral Processing (All Scans)

**Function**: `process_all_raman_data()` in `src/pipeline/processing.py`

#### 3.1 Per-Scan Processing
For each scan in the dataset:

##### 3.1.1 Spectral Smoothing
- Apply Savitzky-Golay filter (same as single scan)
- **Parameters**: `window_size` (default: 25), `poly_order_sg` (default: 2)

##### 3.1.2 Fluorescence Region Baseline Correction
- Focus on fluorescence region: ~925-1000 nm emission
- Apply Lieberfit algorithm
- **Parameters**: `poly_order` (default: 3), `tot_iter` (default: 10)

##### 3.1.3 G-band Peak Detection
- Extract G-band peak from corrected spectrum
- G-band window: 952-958 nm emission
- Calculate G-band **height** (peak value) and **area** (trapezoidal integration)

##### 3.1.4 Background Intensity Calculation
- Calculate average background intensity
- Background window: 850-925 nm emission

##### 3.1.5 Fluorescence Calculations
- **Initial SWNT Fluorescence**: Trapezoidal area of smoothed spectrum in fluorescence region (925-1000 nm)
- **Background-Subtracted Fluorescence**: Initial fluorescence minus average background
- **Final SWNT Fluorescence**: Background-subtracted minus G-band area

##### 3.1.6 Ratio Calculation
- **Fluorescence to G-band Ratio**: Background-subtracted fluorescence area / G-band area

#### 3.2 Results Compilation
- Create DataFrame with Datetime as index
- Columns:
  - Scan Number
  - G-band Height
  - G-band Area
  - Average Background Intensity
  - Initial SWNT Fluorescence
  - Background-Subtracted SWNT Fluorescence
  - Final SWNT Fluorescence
  - Fluorescence to G-band Ratio

#### 3.3 Export Results
- Save to CSV: `raman_analysis_results_emission_nm_YYYY-MM-DD.csv`

---

### 4. Plotting: Individual Spectrum

**Function**: `plot_raman_spectrum()` in `src/pipeline/plotting.py`

- Generate single spectrum plot showing:
  - Raw intensities
  - Smoothed intensities
  - G-band baseline
  - Baseline-corrected spectrum
- Save as PNG in processed directory

---

### 5. Plotting: Time Series

#### 5.1 Combined Time Series Plot
**Function**: `plot_combined_time_series()` in `src/pipeline/plotting.py`

- Filter data: Scan Number > 10, exclude first 2 hours
- Plot multiple metrics over time:
  - G-band Height
  - G-band Area
  - Background-Subtracted SWNT Fluorescence
  - Fluorescence to G-band Ratio
- Apply Gaussian smoothing to ratio (sigma from config)
- Add day/night shading based on light cycle

#### 5.2 Stacked Metrics Plot (with Temperature)
**Function**: `plot_stacked_metrics()` in `src/pipeline/plotting.py`

- If temperature/humidity file available:
  - Load temperature/humidity data
  - Create stacked subplot with:
    - Temperature and humidity
    - Fluorescence metrics
- Add day/night shading

#### 5.3 Extrema Plot
**Function**: `plot_extrema()` in `src/pipeline/plotting.py`

- Identify and plot maximum/minimum points for key metrics
- Highlight extrema with markers

---

### 6. Post-Processing: ALS Baseline Correction

**Function**: `apply_als_and_gaussian()` in `src/pipeline/post_processing.py`

#### 6.1 ALS (Asymmetric Least Squares) Baseline Correction
- Apply ALS algorithm to "Fluorescence to G-band Ratio" time series
- **Parameters**: 
  - `lam_als` (default: 1e7): Smoothness parameter
  - `p_als` (default: 0.001): Asymmetry parameter
  - `niter_als` (default: 20): Number of iterations
- Calculate baseline and subtract from original ratio

#### 6.2 Gaussian Smoothing
- Apply 1D Gaussian filter to ALS-corrected ratio
- **Parameter**: `sigma_gaussian` (default: 5)
- Creates "Smoothed Corrected (ALS)" column

#### 6.3 Plotting
- Plot original ratio, ALS baseline, corrected ratio, and smoothed corrected ratio
- Add day/night shading
- Save plot: `ALS_Baseline_and_Gaussian_Smoothed_Ratio_YYYY-MM-DD.png`

---

### 7. Post-Processing: Diurnal Average

**Function**: `compute_diurnal_average()` in `src/pipeline/post_processing.py`

#### 7.1 Hour Binning
- Extract hour of day (including fractional hours)
- Bin hours using `bin_factor` (default: 2) - rounds to nearest 0.5 hours

#### 7.2 Statistical Aggregation
- Group by binned hour
- Calculate:
  - Mean of smoothed corrected ratio
  - Standard deviation
  - Standard error of mean (SEM)
  - Sample count
- Apply additional Gaussian smoothing (sigma=1) to mean values

#### 7.3 Plotting
- Error bar plot showing mean ± SEM across 24 hours
- Save plot: `Average_Diurnal_Cycle_PL_to_G_Ratio_YYYY-MM-DD.png`

#### 7.4 Export
- Save updated DataFrame to CSV: `corrected_raman_results_gaussian_YYYY-MM-DD.csv`
- Includes all original columns plus:
  - Corrected Ratio (ALS)
  - Smoothed Corrected (ALS)
  - Hour of Day
  - Binned Hour

---

### 8. Post-Processing: Fourier Transform Analysis

**Function**: `compute_fourier_transform()` in `src/pipeline/post_processing.py`

#### 8.1 Signal Preparation
- Extract two signals:
  1. Original "Fluorescence to G-band Ratio" (detrended)
  2. "Smoothed Corrected (ALS)" (detrended)
- Convert datetime index to hours from start
- Calculate sampling rate from mean interval

#### 8.2 FFT Computation
- Apply Fast Fourier Transform to detrended signals
- Calculate:
  - Frequencies (cycles/hour)
  - Real and imaginary components
  - Magnitude (amplitude)
  - Phase

#### 8.3 Peak Detection
- Identify top N peaks (default: 5) from `fft_top_peaks` config
- Extract:
  - Peak frequencies
  - Peak magnitudes
  - Corresponding periods (hours)

#### 8.4 Plotting and Export
- Generate FFT spectrum plots (magnitude vs frequency)
- Mark dominant peaks
- Save plots:
  - `FFT_Spectrum_original_YYYY-MM-DD.png`
  - `FFT_Spectrum_smoothed_YYYY-MM-DD.png`
- Export complete FFT results:
  - `complete_fft_original_YYYY-MM-DD.csv`
  - `complete_fft_smoothed_YYYY-MM-DD.csv`
- Export peak summaries:
  - `fft_peaks_original_YYYY-MM-DD.csv`
  - `fft_peaks_smoothed_YYYY-MM-DD.csv`

---

### 9. Aggregated Output Utilities

Instead of appending directly to "master" CSVs during every run, aggregated views are now produced with standalone utility scripts:

- `compile_batch_summaries.py`: Scans each profile's latest batch folder, grabs `batch_summary_v*_final.csv`, injects metadata, and produces a combined CSV/Parquet file (with optional compression and dtype optimization).
- `compile_fft_data.py`: Scans the latest batch folder for each profile, loads `fft_full_gband_ratio_*.csv` and `fft_full_raman_850_ratio_*.csv`, filters frequencies (default 0–0.5 cycles/hour), and compiles them into one dataset.

These utilities replace the previous `append_master` workflow and keep aggregation steps explicit and reproducible.

---

## Key Parameters Summary

### Spectral Processing
- **Savitzky-Golay**: `window_size=25`, `poly_order_sg=2`
- **Lieberfit**: `poly_order=3-5`, `tot_iter=10-100`
- **G-band window**: 952-958 nm emission (1500-2100 cm⁻¹)
- **Fluorescence window**: 925-1000 nm emission
- **Background window**: 850-925 nm emission

### Time-Series Processing
- **ALS**: `lam_als=1e7`, `p_als=0.001`, `niter_als=20`
- **Gaussian smoothing**: `sigma_gaussian=5-25` (varies by plot)
- **Diurnal binning**: `bin_factor=2` (0.5-hour bins)

### Analysis
- **FFT top peaks**: `fft_top_peaks=5`
- **Data filtering**: Exclude Scan Number ≤ 10, exclude first 2 hours

---

## Output Files Generated

1. `raman_analysis_results_emission_nm_YYYY-MM-DD.csv` - Raw processing results
2. `corrected_raman_results_gaussian_YYYY-MM-DD.csv` - ALS-corrected and smoothed results
3. `complete_fft_original_YYYY-MM-DD.csv` - Complete FFT of original signal
4. `complete_fft_smoothed_YYYY-MM-DD.csv` - Complete FFT of smoothed signal
5. `fft_peaks_original_YYYY-MM-DD.csv` - FFT peaks (original)
6. `fft_peaks_smoothed_YYYY-MM-DD.csv` - FFT peaks (smoothed)
7. Various PNG plots (spectrum, time series, diurnal, FFT, etc.)
8. Compiled summaries produced via `compile_batch_summaries.py` (optional utility)
9. Compiled FFT tables produced via `compile_fft_data.py` (optional utility)

---

## Notes

- All processing is controlled by configuration profiles in `pipeline.yml`
- Aggregation is handled via dedicated utility scripts rather than auto-appending master files
- Timestamped output directories prevent data overwrites
- Temperature/humidity data is optional and gracefully handled if missing
- Day/night shading in plots adapts to light cycle (Constant, 8to24, 6to22)

