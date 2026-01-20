# SWNT IAA Raman Analysis

Professional Python package for analyzing Single-Walled Carbon Nanotube (SWNT) Indole-3-Acetic Acid (IAA) Raman spectroscopy data. This package processes Raman spectroscopy measurements to extract fluorescence-to-G-band ratios, perform baseline corrections, and analyze time-series data for monitoring IAA levels in plant systems.

## Overview

The `swnt_iaa_analysis` package provides a complete pipeline for processing Raman spectroscopy data from SWNT-IAA nanosensor experiments. It handles spectral preprocessing, peak detection, ratio calculations, signal corrections, and frequency-domain analysis (FFT) to extract meaningful biological signals from raw spectroscopic measurements.

Key features:
- **Spectral Processing**: Savitzky-Golay smoothing and Lieberfit baseline correction
- **Peak Detection**: Lorentzian fitting for G-band and Raman peak identification
- **Signal Correction**: Spike removal and baseline shift correction for time-series data
- **Ratio Calculations**: Fluorescence-to-G-band and fluorescence-to-Raman peak ratios
- **Frequency Analysis**: Fourier transform analysis with diurnal averaging
- **Visualization**: Comprehensive plotting of spectra, timeseries, and FFT results
- **Profile-Based Configuration**: YAML-based configuration with inheritance support

## Installation

Install the package in development mode:

```bash
cd swnt_iaa_analysis
pip install -e .
```

This will install the package and make it available as `swnt_iaa_analysis` command.

## Usage

### Command Line Interface

The package provides three main commands for analyzing Raman spectroscopy data:

#### 1. Analyze a Single Profile

```bash
swnt_iaa_analysis analyze <profile_name> [options]
```

**Process Flow:**

When the `analyze` command is invoked, the following detailed process occurs:

1. **Configuration Loading**
   - Loads the profile configuration from `config.yaml` (searches current directory, then package directory)
   - Resolves profile inheritance to merge base configuration with profile-specific settings
   - Validates configuration parameters

2. **Data Loading**
   - Loads Raman spectroscopy dataset from the path specified in the profile configuration
   - Parses TSV file with columns: `Scan Number`, `Seconds`, and wavenumber columns (cm⁻¹)
   - Converts datetime information from the dataset
   - Optionally loads temperature/humidity data if configured

3. **Output Directory Setup**
   - Creates timestamped output directory: `results_v4_YYYYMMDD_HHMMSS` (for v4 algorithm)
   - Output directory is created in the same location as the input data file
   - Can be overridden with `--output` option

4. **Spectral Processing (V4 Algorithm)**
   For each scan in the dataset (after skipping initial scans):
   
   a. **Wavenumber Correction**
      - Applies excitation wavelength correction if `excitation_nm` differs from `recorded_excitation_nm`
      - Corrects wavenumbers based on the actual laser wavelength used
   
   b. **Wavenumber Filtering**
      - Filters spectrum to start at 250 cm⁻¹ (removes low-frequency noise)
   
   c. **Savitzky-Golay Smoothing**
      - Applies Savitzky-Golay filter to reduce noise
      - Parameters: `window_size` (default: 25), `poly_order_sg` (default: 2)
   
   d. **Normalization**
      - Normalizes intensities by average background (250-1250 cm⁻¹ range)
      - Creates normalized spectrum for comparison
   
   e. **Lieberfit Baseline Correction**
      - Applies iterative polynomial baseline correction (Lieberfit algorithm)
      - Parameters: `poly_order` (default: 5), `tot_iter` (default: 100)
      - Removes fluorescence background from spectrum
   
   f. **Peak Detection**
      - Finds Raman peak at ~850 cm⁻¹ using Lorentzian fitting (800-900 cm⁻¹ window)
      - Finds G-band peak at ~1600 cm⁻¹ using Lorentzian fitting (1550-1650 cm⁻¹ window)
      - Extracts peak intensity, area, and wavenumber for each peak
   
   g. **Fluorescence Calculation**
      - Calculates fluorescence as area under curve (AUC) for wavenumbers ≥1250 cm⁻¹
      - Subtracts G-band area to isolate pure fluorescence signal
   
   h. **Summary Compilation**
      - Stores scan number, datetime, background intensity, normalized intensities, peak parameters
      - Creates DataFrame indexed by datetime (or scan number if configured)

5. **Signal Corrections (V4 Only)**
   - Applies Hampel filter for spike removal from time-series columns:
     - `Normalized_Fluorescence_Intensity`
     - `Normalized_Gband_Area`
     - `Normalized_Raman_Peak_850_Area`
   - Corrects baseline shifts using MAD-based jump detection
   - Creates baseline-corrected columns with `_BaselineCorrected` suffix
   - Generates signal correction comparison plots (before/after)

6. **Ratio Calculations**
   - Calculates fluorescence-to-G-band ratio: `Fluorescence_Intensity / Gband_Area`
   - Calculates fluorescence-to-Raman-peak-850 ratio: `Fluorescence_Intensity / Raman_Peak_850_Area`
   - Computes both original and baseline-corrected ratios

7. **Fourier Transform Analysis**
   - Extracts baseline-corrected ratio time-series
   - Applies Gaussian smoothing (sigma from config)
   - Applies ALS (Asymmetric Least Squares) baseline correction to remove long-term trends
   - Computes FFT to identify dominant frequencies
   - Extracts top N peaks from frequency spectrum

8. **Export Results**
   - Exports processed data to CSV: `processed_data.csv`
   - Exports FFT results to CSV: `fft_results.csv`
   - Includes both original and baseline-corrected columns

9. **Visualization**
   - Generates ratio timeseries plots (original and baseline-corrected)
   - Generates normalized intensity timeseries plots
   - Generates representative Raman spectrum plot (from first processed scan)
   - Generates FFT analysis plots (frequency spectrum and power spectral density)
   - Generates signal correction comparison plots
   - All plots saved to output directory

**Options:**
- `--config, -c`: Path to config.yaml file (default: ./config.yaml)
- `--output, -o`: Output directory (default: auto-generated with timestamp)
- `--algorithm, -a`: Algorithm version: 'v3' or 'v4' (default: v4)

**Example:**
```bash
swnt_iaa_analysis analyze bokchoy_control_6to22_Temp_Hum_Variable_Run1
swnt_iaa_analysis analyze bokchoy_control_6to22_Temp_Hum_Variable_Run1 --output ./my_results --algorithm v4
```

#### 2. List Available Profiles

```bash
swnt_iaa_analysis list-profiles-cmd [options]
```

**Process Flow:**

1. **Configuration Loading**
   - Loads `config.yaml` from current directory or package directory
   - Parses YAML structure to extract profile definitions

2. **Profile Extraction**
   - Extracts all profile names from the `profiles` section
   - Optionally filters by experiment type if `--experiment` is specified

3. **Display**
   - Lists all matching profile names to the console
   - Formatted for easy selection

**Options:**
- `--config, -c`: Path to config.yaml file (default: ./config.yaml)
- `--experiment, -e`: Filter by experiment type (e.g., 'in planta', 'in vitro')

**Example:**
```bash
swnt_iaa_analysis list-profiles-cmd
swnt_iaa_analysis list-profiles-cmd --experiment "in planta"
```

#### 3. Batch Process Multiple Profiles

```bash
swnt_iaa_analysis batch-process [options]
```

**Process Flow:**

1. **Configuration Loading**
   - Loads configuration file as in the analyze command

2. **Profile Selection**
   - If `--profiles` specified: Uses comma-separated list of profile names
   - If `--experiment` specified: Lists all profiles matching experiment type
   - Validates that profiles exist in configuration

3. **Iterative Processing**
   For each selected profile:
   - Creates a new `RamanPipeline` instance
   - Executes the complete analysis pipeline (same as `analyze` command)
   - Creates separate output directory for each profile (auto-generated)
   - Continues to next profile even if one fails (errors are logged)

4. **Completion Summary**
   - Reports success/failure for each profile
   - Displays final batch processing status

**Options:**
- `--config, -c`: Path to config.yaml file
- `--experiment, -e`: Process all profiles of this experiment type
- `--profiles, -p`: Comma-separated list of profile names to process
- `--algorithm, -a`: Algorithm version: 'v3' or 'v4' (default: v4)

**Note:** Must specify either `--experiment` or `--profiles`

**Example:**
```bash
swnt_iaa_analysis batch-process --experiment "in planta"
swnt_iaa_analysis batch-process --profiles "profile1,profile2,profile3" --algorithm v4
```

### Python API

For programmatic use, the package provides a Python API:

```python
from swnt_iaa_analysis.pipeline import RamanPipeline

# Initialize pipeline
pipeline = RamanPipeline(
    config_path="config.yaml",
    profile_name="bokchoy_control_6to22_Temp_Hum_Variable_Run1",
    output_dir="./results",  # Optional
    algorithm="v4"  # 'v3' or 'v4'
)

# Run complete analysis
results = pipeline.run()

# Access results
print(results.head())  # Processed DataFrame
print(pipeline.fft_results)  # FFT analysis results
print(pipeline.output_dir)  # Output directory path
```

The `RamanPipeline.run()` method executes the same processing pipeline as the CLI `analyze` command, returning a pandas DataFrame with all processed data.

## Ratio Calculations: Original vs Baseline-Corrected

The pipeline calculates two sets of ratios to provide both raw and cleaned data for analysis. Understanding the difference between these ratios is crucial for interpreting results correctly.

### Overview

The pipeline computes:

1. **Original Ratios** - Calculated from normalized intensity columns (before time-series signal corrections)
2. **Baseline-Corrected Ratios** - Calculated from signal-corrected columns (after spike removal and baseline shift correction)

### Processing Pipeline Order

The calculation order is important:

```
Step 1: Spectral Processing (per scan)
  ↓
  Creates: Normalized_Fluorescence_Intensity
           Normalized_Gband_Area
           Normalized_Raman_Peak_850_Area

Step 2: Signal Corrections (time-series processing)
  ↓
  Creates: Normalized_Fluorescence_Intensity_BaselineCorrected
           Normalized_Gband_Area_BaselineCorrected
           Normalized_Raman_Peak_850_Area_BaselineCorrected

Step 3: Ratio Calculations
  ↓
  Calculates: Fluorescence_to_Gband_Ratio (from original columns)
              Fluorescence_to_Gband_Ratio_BaselineCorrected (from corrected columns)
```

### Original Ratios

**Source Columns:**
- `Normalized_Fluorescence_Intensity`
- `Normalized_Gband_Area`
- `Normalized_Raman_Peak_850_Area`

**Calculation:**
```python
Fluorescence_to_Gband_Ratio = Normalized_Fluorescence_Intensity / Normalized_Gband_Area
Fluorescence_to_Raman_Peak_850_Ratio = Normalized_Fluorescence_Intensity / Normalized_Raman_Peak_850_Area
```

**Characteristics:**
- Reflect raw normalized intensities directly from spectral processing
- May contain:
  - **Spikes**: Sudden outliers from detector artifacts or electrical noise
  - **Baseline shifts**: Step discontinuities from instrument drift, temperature changes, or sample movement
  - **Short-term instabilities**: Instrument fluctuations affecting signal quality

**Use Cases:**
- Data quality assessment (comparing with corrected ratios to evaluate correction effectiveness)
- Detecting problematic regions in the time-series
- Understanding raw signal behavior before corrections

### Baseline-Corrected Ratios

**Source Columns:**
- `Normalized_Fluorescence_Intensity_BaselineCorrected`
- `Normalized_Gband_Area_BaselineCorrected`
- `Normalized_Raman_Peak_850_Area_BaselineCorrected`

**Pre-Processing Applied:**

Before ratio calculation, these columns undergo two correction steps:

1. **Spike Removal (Hampel Filter)**
   - Detects statistical outliers using Median Absolute Deviation (MAD)
   - Parameters: `spike_window` (default: 5), `spike_threshold` (default: 3.0)
   - Replaces detected spikes with interpolated values

2. **Baseline Shift Correction**
   - Detects sudden jumps using MAD-based thresholding
   - Also detects gradual shifts over a window (cumulative jumps)
   - Calculates offset using median of points before/after each jump
   - Applies cumulative corrections to remove all discontinuities
   - Key parameters:
     - `baseline_threshold_multiplier` (default: 5.0) - sensitivity of jump detection
     - `baseline_window_size` (default: 5) - window for offset calculation
     - `baseline_smooth_first` (default: True) - smooth before detection

**Calculation:**
```python
Fluorescence_to_Gband_Ratio_BaselineCorrected = 
    Normalized_Fluorescence_Intensity_BaselineCorrected / 
    Normalized_Gband_Area_BaselineCorrected

Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected = 
    Normalized_Fluorescence_Intensity_BaselineCorrected / 
    Normalized_Raman_Peak_850_Area_BaselineCorrected
```

**Characteristics:**
- Cleaned time-series signals with instrumental artifacts removed
- Spikes removed to prevent false signals
- Baseline discontinuities corrected to maintain signal continuity
- More suitable for biological signal analysis and interpretation

**Use Cases:**
- **Primary scientific analysis** (recommended for most applications)
- Time-series analysis and trend detection
- FFT analysis (used by default in frequency domain analysis)
- Publication-quality plots and figures
- Quantitative comparisons across experiments

### Key Differences Summary

| Aspect | Original Ratios | Baseline-Corrected Ratios |
|--------|----------------|---------------------------|
| **Data Source** | Raw normalized intensities | Signal-corrected intensities |
| **Spikes** | Present | Removed (Hampel filter) |
| **Baseline Shifts** | Present | Corrected (MAD-based detection) |
| **Signal Quality** | Raw/uncorrected | Cleaned/corrected |
| **Primary Use** | Quality assessment | Scientific analysis |
| **FFT Analysis** | Not recommended | Used by default |

### Why Both Are Calculated?

1. **Quality Control**: Compare original vs corrected to assess the impact of corrections
2. **Transparency**: Both datasets are exported for full inspection
3. **Flexibility**: Choose based on specific analysis needs
4. **Validation**: Verify that corrections don't remove legitimate biological signals

### Output Files

Both ratio types are included in the exported `processed_data.csv`:

**Original Ratios:**
- `Fluorescence_to_Gband_Ratio`
- `Fluorescence_to_Raman_Peak_850_Ratio`

**Baseline-Corrected Ratios:**
- `Fluorescence_to_Gband_Ratio_BaselineCorrected`
- `Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected`

### Visualization

The pipeline generates separate plots for both types:
- Timeseries plots for both original and baseline-corrected ratios
- Signal correction comparison plots showing before/after corrections with jump locations marked
- FFT analysis uses baseline-corrected ratios by default

### Recommendation

**For scientific analysis, use the baseline-corrected ratios** (`*_BaselineCorrected`) as they:
- Remove instrumental artifacts that can obscure biological signals
- Provide cleaner signals for time-series analysis
- Enable more reliable frequency domain analysis (FFT)
- Are used by default in all downstream analysis steps

Use original ratios for quality control and validation to ensure corrections are working properly.

## Package Structure

```
swnt_iaa_analysis/
├── setup.py              # Package setup and entry points
├── config.yaml           # Default configuration with profile definitions
├── swnt_iaa_analysis/    # Source code package
│   ├── __init__.py
│   ├── core/             # Core processing modules
│   │   ├── loader.py     # Data loading (Raman data, temperature data)
│   │   ├── preprocessing.py  # Spectral preprocessing (Savitzky-Golay)
│   │   ├── baseline.py   # Baseline correction (Lieberfit, ALS)
│   │   └── utils.py      # Utility functions (spike removal, baseline shifts)
│   ├── analysis/         # Scientific analysis modules
│   │   ├── peaks.py      # Peak fitting (Lorentzian)
│   │   ├── ratios.py     # Ratio calculations
│   │   └── fourier.py    # Fourier transforms and diurnal analysis
│   ├── visualization/    # Plotting modules
│   │   └── plotting.py   # Plotting functions (timeseries, spectra, FFT)
│   ├── io/               # I/O operations
│   │   ├── config.py     # Configuration management (YAML parsing, inheritance)
│   │   └── exporter.py  # Data export (CSV export)
│   ├── pipeline.py       # Main pipeline class (RamanPipeline)
│   └── cli.py            # Command-line interface (Typer-based)
└── tests/                # Unit tests
```

## Configuration

The package uses YAML configuration files with profile-based inheritance. Each profile specifies:

- **Data Source**: Path to Raman data file and optional temperature data
- **Processing Parameters**: Window sizes, polynomial orders, excitation wavelengths
- **Metadata**: Experiment type, plant type, treatment, replicate information
- **Output Settings**: X-axis type (datetime vs scan_number), plotting preferences

Profiles can inherit from base profiles to share common processing parameters. See `config.yaml` for detailed examples and parameter descriptions.

Key configuration sections:
- `processing`: Spectral processing parameters (Savitzky-Golay, Lieberfit, ALS)
- `data_source`: File paths and data source settings
- `metadata`: Experiment metadata
- `outputs`: Visualization and export settings

### Signal Correction Parameters

The package provides comprehensive parameters for controlling spike removal and baseline jump detection/correction. These parameters are critical for cleaning time-series data and removing instrumental artifacts.

#### Spike Removal Parameters (Hampel Filter)

Applied **before** baseline correction to remove statistical outliers:

- **`spike_window`** (default: 5)
  - Half-window size for median calculation in spike detection
  - **Effect**: Larger values = more smoothing (may miss short spikes), smaller values = more sensitivity
  - **Typical range**: 3-10

- **`spike_threshold`** (default: 3.0)
  - Threshold in units of MAD (Median Absolute Deviation) for outlier detection
  - **Effect**: Higher values = fewer spikes detected (more conservative), lower values = more spikes detected (more aggressive)
  - **Typical range**: 2.0-5.0

#### Baseline Jump Detection Parameters

- **`baseline_correction_threshold`** (default: 15 in config, 5.0 in code)
  - Multiplier for MAD-based threshold calculation: `threshold = MAD × threshold_multiplier`
  - **Effect**: Controls sensitivity of jump detection
    - **Higher values (10-20)**: Fewer jumps detected, more conservative (only detects large jumps)
    - **Lower values (3-7)**: More jumps detected, more aggressive (detects smaller jumps)
  - **Recommended**: Start with 10-15, adjust based on your data characteristics

- **`baseline_correction_window`** (default: 5)
  - Number of points averaged before/after each jump to calculate precise offset
  - Also used to filter consecutive jumps (jumps closer than this distance are treated as one)
  - **Effect**: 
    - Larger values (10-15): More robust offset calculation, may miss closely-spaced jumps
    - Smaller values (3-5): More precise offset, but more sensitive to noise
  - **Recommended**: 5-10 for most datasets

- **`baseline_detect_cumulative_jumps`** (default: True)
  - Whether to detect gradual jumps over multiple consecutive points
  - **Effect**: 
    - `True`: Detects both sudden jumps and gradual shifts over time
    - `False`: Only detects sudden, instantaneous jumps
  - **Recommended**: Keep `True` for comprehensive correction

- **`baseline_cumulative_window`** (default: 5)
  - Window size for detecting cumulative/gradual jumps
  - Total change over `cumulative_window` consecutive points must exceed threshold
  - **Effect**: 
    - Larger values (10-20): Detects slower trends as jumps
    - Smaller values (3-5): Only detects faster changes
  - **Recommended**: 5-10 to catch gradual baseline shifts

#### Pre-Processing Parameters (Before Jump Detection)

- **`baseline_correction_smooth_first`** (default: True)
  - Whether to smooth the signal before jump detection
  - **Effect**: 
    - `True`: Reduces noise, provides more stable jump detection (recommended)
    - `False`: Uses raw signal, more sensitive to noise fluctuations
  - **Recommended**: Keep `True` for most applications

- **`baseline_correction_smooth_window`** (default: 5)
  - Window size for Savitzky-Golay smoothing (must be odd: 5, 7, 9, 11, etc.)
  - **Effect**: 
    - Larger values (11-21): More smoothing, less sensitive to noise
    - Smaller values (5-7): Less smoothing, preserves sharp features
  - **Recommended**: 5-11 depending on noise level

- **`baseline_correction_smooth_poly_order`** (default: 2)
  - Polynomial order for Savitzky-Golay smoothing
  - **Effect**: Higher orders preserve more features but may introduce artifacts
  - **Recommended**: 2-3 (quadratic or cubic)

#### Correction Application Parameters

- **`baseline_correction_correct_smoothed`** (default: True in config, False in code)
  - Whether to apply corrections to the smoothed signal or original signal
  - **Effect**: 
    - `True`: Corrections applied to smoothed signal (may appear less dramatic in plots)
    - `False`: Corrections applied to original signal (corrections more visible, recommended for inspection)
  - **Recommended**: `False` for better visibility of corrections

#### Parameter Recommendations by Data Characteristics

**For Noisy Data:**
```yaml
baseline_correction_threshold: 10-15        # More conservative
baseline_correction_smooth_window: 11-15    # More smoothing
baseline_correction_window: 7-10            # Larger window for stability
```

**For Clean Data (High Signal-to-Noise):**
```yaml
baseline_correction_threshold: 5-7          # More sensitive
baseline_correction_smooth_window: 5-7      # Less smoothing
baseline_correction_window: 3-5             # Smaller window for precision
```

**For Data with Many Small Jumps:**
```yaml
baseline_correction_threshold: 3-5          # Aggressive detection
baseline_detect_cumulative_jumps: true      # Detect gradual shifts
baseline_cumulative_window: 5-10            # Catch gradual changes
```

**For Data with Few Large Jumps:**
```yaml
baseline_correction_threshold: 15-20        # Conservative, only large jumps
baseline_correction_window: 10-15           # Robust offset calculation
```

#### How Parameters Work Together

The correction pipeline executes in this order:

1. **Spike Removal**: Hampel filter removes outliers (`spike_window`, `spike_threshold`)
2. **Smoothing** (if enabled): Savitzky-Golay smoothing (`smooth_window`, `smooth_poly_order`)
3. **Jump Detection**: 
   - Calculate signal derivative
   - Detect sudden jumps using MAD threshold (`baseline_correction_threshold`)
   - Detect gradual jumps over window (`baseline_cumulative_window`, if enabled)
4. **Offset Calculation**: For each jump, calculate offset using median of `window_size` points before/after
5. **Correction Application**: Apply cumulative corrections to remove all discontinuities

#### Adjusting Parameters

1. Start with default values from your config
2. Run analysis and inspect `signal_correction_comparison_*.png` plots
3. If too many jumps detected: Increase `baseline_correction_threshold`
4. If jumps missed: Decrease `baseline_correction_threshold` or adjust `baseline_cumulative_window`
5. If corrections look incorrect: Adjust `baseline_correction_window` for better offset calculation

## Algorithm Versions

The package currently supports:

- **V4 Algorithm** (default): 
  - Normalized intensity-based processing
  - Area-based fluorescence calculation
  - Lorentzian peak fitting
  - Signal corrections (spike removal, baseline shift correction)
  - Comprehensive visualization

- **V3 Algorithm**: 
  - Not yet implemented in the refactored package
  - Will raise `NotImplementedError` if selected

## Output Files

Each analysis run generates:

- `processed_data.csv`: Complete processed dataset with all calculated intensities, peaks, and ratios
- `fft_results.csv`: Fourier transform analysis results (if applicable)
- Multiple plot files (PNG):
  - Ratio timeseries plots
  - Normalized intensity timeseries plots
  - Representative Raman spectrum
  - FFT analysis plots
  - Signal correction comparison plots

All outputs are saved to a timestamped directory to prevent overwrites.

## Development

This package follows the same architecture as `microneedle_nir_imaging_analysis` for consistency and maintainability. The codebase is organized into logical modules with clear separation of concerns:

- **Core modules**: Low-level data processing algorithms
- **Analysis modules**: Scientific calculations and analysis
- **IO modules**: Configuration and data export
- **Visualization modules**: Plotting and visualization
- **Pipeline**: High-level orchestration
- **CLI**: User-facing command-line interface

