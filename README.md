# SWNT IAA Raman Analysis

Repository root: **`microneedle_IAA_Raman_analysis`** — install and configure from this folder (`setup.py` and `config.yaml` live here).


Professional Python package for analyzing Single-Walled Carbon Nanotube (SWNT) Indole-3-Acetic Acid (IAA) Raman spectroscopy data. This package processes Raman spectroscopy measurements to extract fluorescence-to-G-band ratios, perform baseline corrections, and analyze time-series data for monitoring IAA levels in plant systems.

## Overview

The `swnt_iaa_analysis` package provides a complete pipeline for processing Raman spectroscopy data from SWNT-IAA nanosensor experiments. It handles spectral preprocessing, peak detection, ratio calculations, signal corrections, and frequency-domain analysis (FFT) to extract meaningful biological signals from raw spectroscopic measurements.

Key features:
- **Spectral Processing**: Savitzky-Golay smoothing and Lieberfit baseline correction
- **Peak Detection**: Lorentzian fitting for G-band and Raman peak identification
- **Timeseries Baseline Correction**: Automatic detection of lighting-transition ramps by rate of change, with drift-free step removal that preserves the daily biological signal
- **Raw and Normalized Channels**: Both the normalized and the raw (un-normalized, physical) fluorescence and peak areas are exported
- **Ratio Calculations**: Fluorescence-to-G-band and fluorescence-to-Raman peak ratios
- **Frequency Analysis**: Fourier transform analysis with diurnal averaging
- **Visualization**: Comprehensive plotting of spectra, timeseries, and FFT results
- **Profile-Based Configuration**: YAML-based configuration with inheritance support
- **Cross-Profile Aggregation**: Script to collect the latest results across profiles into one QC summary

## Installation

Install the package in development mode from the repository root:

```bash
python -m pip install -e .
```

This installs the package and exposes the `swnt_iaa_analysis` CLI; run these commands from the directory that contains `setup.py` and `config.yaml` (repository root).

If the `swnt_iaa_analysis` command is not found in your shell, run the CLI module directly:

```bash
python -m swnt_iaa_analysis.cli analyze <profile_name>
```

## Usage

### Command Line Interface

The package provides four main commands for analyzing Raman spectroscopy data:

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
      - Parameters: `spectral_sg_window` (default: 25), `spectral_sg_poly_order` (default: 2)
   
   d. **Normalization**
      - Normalizes intensities by average background (250-1250 cm⁻¹ range)
      - Creates normalized spectrum for comparison
   
   e. **Lieberfit Baseline Correction**
      - Applies iterative polynomial baseline correction (Lieberfit algorithm)
      - Parameters: `spectral_lieberfit_poly_order` (default: 5), `spectral_lieberfit_iterations` (default: 100)
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

5. **Raw Channel Recovery**
   - Undoes the per-scan normalization to recover the physical quantities:
     `Raw_X = Normalized_X × Average_Background_Intensity_250_1250_cm-1`
   - Adds `Raw_Fluorescence_Intensity`, `Raw_Gband_Area`, `Raw_Gband_Intensity`,
     `Raw_Raman_Peak_850_Area`

6. **Timeseries Baseline Correction (V4 Only)**
   - Runs the method selected by `baseline_correction_method` (default `transition_stitch`)
   - Detects lighting-transition ramps on the normalized fluorescence by rate of change
   - Removes the transition steps and the accumulated drift from the fluorescence,
     leaving the daily biological variation intact
   - Despikes and lightly smooths the Raman peak areas (never stitched — they carry
     no lighting artifact), keeping the ratio denominator strictly positive
   - Creates baseline-corrected columns with the `_BaselineCorrected` suffix
   - Generates processing-stages plots (raw / smoothed / corrected, with transitions marked)
   - See [Timeseries Baseline Correction](#timeseries-baseline-correction) for details

7. **Ratio Calculations**
   - Calculates fluorescence-to-G-band ratio: `Fluorescence_Intensity / Gband_Area`
   - Calculates fluorescence-to-Raman-peak-850 ratio: `Fluorescence_Intensity / Raman_Peak_850_Area`
   - Computes both original and baseline-corrected ratios

8. **Fourier Transform Analysis**
   - Extracts baseline-corrected ratio time-series
   - Applies Gaussian smoothing (sigma from config)
   - Applies ALS (Asymmetric Least Squares) baseline correction to remove long-term trends
   - Computes FFT to identify dominant frequencies
   - Extracts top N peaks from frequency spectrum

9. **Export Results**
   - Exports processed data to CSV: `processed_data.csv`
   - Exports FFT results to CSV: `fft_results.csv`
   - Includes both original and baseline-corrected columns

10. **Visualization**
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

#### 4. Re-process Existing Results

```bash
swnt_iaa_analysis reprocess <profile_name> [options]
```

Re-runs the time-series stages on an existing `processed_data.csv` **without re-fitting the
spectra**, which is far faster when you only want to change baseline-correction settings. It
finds the profile's most recent results folder, drops the existing `_BaselineCorrected` and
ratio columns, and recomputes the steps you select.

**Options:**
- `--steps`: Comma-separated subset of `spike_removal,baseline_correction,ratios,fft`
  (default: all). `ratios` requires baseline-corrected columns to exist or be recomputed;
  `fft` requires `ratios`.
- `--config, -c`: Path to config.yaml file
- `--output, -o`: Output directory (default: new timestamped folder beside the source CSV)

**Example:**
```bash
# Try new baseline settings without re-fitting every spectrum
swnt_iaa_analysis reprocess Nb_Control_6to22_Temp_Hum_Variable_Run5 --steps baseline_correction,ratios,fft
```

> **Note:** each `reprocess` writes a new `results_v4_*_reprocess` folder *inside* the results
> folder it read from, so repeated runs nest. Tooling that looks for "the latest results" should
> search recursively (see [Aggregating Results](#aggregating-results-across-profiles)).

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

## Timeseries Baseline Correction

After the per-scan spectral processing, the pipeline corrects the extracted time-series for
lighting artifacts. The method is chosen with `baseline_correction_method`.

### The problem

The fluorescence is an area-under-curve over wavenumbers ≥1250 cm⁻¹, so it **includes the
broadband background**. When the grow lights switch on or off, that background steps, adding a
sharp artifact to the fluorescence (~20% of signal level). The step is not instantaneous — it
ramps over several scans (roughly 20–90 minutes).

The G-band and 850 cm⁻¹ areas behave differently. They come from a Lorentzian fit whose `offset`
term absorbs the local baseline, computed on the lieberfit-corrected spectrum, and the area is
`π × amplitude × width` — which **excludes the offset**. Additive lamp light therefore cannot
enter those peak areas, and they show no transition step.

The discriminator between artifact and biology is the **rate of change**: a lighting transition
is a fast, sustained, net-displacing move, whereas the biology varies gradually.

### Methods

| Method | Reconstruction | Use |
|---|---|---|
| `transition_stitch` *(default)* | Removes the transition steps **and** the accumulated drift, preserving the daily biology | Recommended |
| `transition_ramp` | Rebuilds the signal as its slow component (rolling median over ~one light cycle) | Removes all sub-daily variation |
| `jump_stitch` | Legacy MAD-based cumulative jump correction | Reproducing older results only |

`transition_stitch` runs:

1. **Despike** the fluorescence (Hampel filter, MAD-based).
2. **Detect** transition ramps: local slope `median(next k) − median(prev k)`; contiguous
   same-sign candidates above `transition_candidate_sigma × MAD` are merged into regions, and a
   region is kept only if its net displacement exceeds `transition_min_displacement` **and** its
   rate exceeds `transition_min_rate_per_scan`. The rate test (rather than a duration cap) is
   what lets long early-experiment ramps still be recognised.
3. **Stitch**: subtract each transition's step from all later scans, then subtract the offset
   staircase's own slow component. Without that second step the offset ratchets — morning steps
   outweigh evening ones — and the signal drifts below zero.
4. **Mask** the ramp scans (untrustworthy during a transition) and interpolate across them.
5. **Denominator**: the Raman peak areas are only despiked and lightly smoothed — never
   stitched — so the ratio denominator stays strictly positive.

### `stitch_space`: raw vs normalized

Each spectrum is normalized per scan by its 250–1250 cm⁻¹ background, and **that background has
its own day/night pattern**. Correcting the normalized fluorescence while dividing by an
un-stitched normalized denominator breaks the background's cancellation and injects a spurious
daytime dip into the ratio.

- `raw` *(default)* — correct the raw (physical) channels. The background never enters
  inconsistently, and the resulting ratio is **exactly scale-invariant** across datasets, since a
  dataset-wide brightness factor cancels in a ratio. Cross-dataset comparison therefore needs no
  normalization at all.
- `normalized` — previous behaviour, retained for comparison.

Detection always runs on the *normalized* fluorescence, whose scale is dataset-independent, so
the detection thresholds mean the same thing in either mode.

> **Note on the normalized channels.** Because the normalization divisor varies day/night, the
> individual `Normalized_*` series carry that pattern. For example `Normalized_Gband_Area` looks
> ~74% higher by day, while the raw G-band area is essentially flat (day/night ≈ 1.01). Read the
> `Raw_*` columns when you want the physical behaviour of a single channel; the ratio is
> unaffected either way, because the background cancels.

### Ratios

```python
Fluorescence_to_Gband_Ratio                  = Fluorescence / Gband_Area          # uncorrected
Fluorescence_to_Gband_Ratio_BaselineCorrected = corrected_Fluor / corrected_Gband  # corrected
```

The **uncorrected** ratio is identical in either space (the background cancels:
`Normalized_fluor / Normalized_gband == Raw_fluor / Raw_gband`). Only the corrected ratio depends
on `stitch_space`. Use the uncorrected ratio for QC and the corrected one for analysis and FFT.

## Package Structure

```
microneedle_IAA_Raman_analysis/           # Repository root (your clone path may differ)
├── setup.py                                       # Packaging and console entry points
├── CHANGELOG.md
├── README.md                                      # This file
├── config.yaml                                     # Profile definitions (`config.yaml`)
├── swnt_iaa_analysis/                               # Importable Python package
│   ├── __init__.py
│   ├── core/                                       # Core processing modules
│   │   ├── loader.py                               # Raman/temperature loads, path resolution
│   │   ├── preprocessing.py
│   │   ├── baseline.py                             # Spectral: lieberfit, ALS, smoothing
│   │   ├── transition_baseline.py                  # Timeseries: transition detection + stitch
│   │   └── utils.py                                # Hampel despike, legacy jump_stitch
│   ├── analysis/
│   │   ├── peaks.py
│   │   ├── ratios.py                               # Ratios + raw-channel recovery
│   │   └── fourier.py
│   ├── visualization/
│   │   └── plotting.py
│   ├── io/
│   │   ├── config.py
│   │   └── exporter.py
│   ├── pipeline.py
│   └── cli.py
├── scripts/
│   ├── aggregate_results.py                        # Cross-profile results aggregation + QC
│   ├── condition_report.py                         # PL/G-band aggregated by condition
│   └── diurnal_reproducibility.py                  # Diurnal cycle reproducibility across runs
└── tests/                                          # Tests (reserved)
```

## Configuration

The package uses YAML configuration files with profile-based inheritance. Each profile specifies:

- **Data Source**: Path to Raman data file and optional temperature data
- **Processing Parameters**: Window sizes, polynomial orders, excitation wavelengths
- **Metadata**: Experiment type, plant type, treatment, replicate information
- **Output Settings**: X-axis type (datetime vs scan_number), plotting preferences

Profiles can inherit from base profiles to share common processing parameters. See `config.yaml` for detailed examples and parameter descriptions.

### Configuration Mental Model

Tiny mental diagram:

```text
config.yaml
└── profiles
    ├── processing_default              # base profile (shared processing defaults)
    │   ├── data_source → { requires_google_drive }
    │   └── processing → { excitation_nm, spectral_sg_window, spectral_lieberfit_poly_order, fft params, ... }
    │
    ├── Nb_petiole_feeding_run1         # concrete runnable profile
    │   ├── inherits: processing_default
    │   ├── data_source → { raman_relative_path, temp_relative_path? }
    │   ├── metadata → { experiment, plant_type, treatment, replicate_number, ... }
    │   └── outputs → { timeseries_x_axis, ... }
    │
    ├── Nb_petiole_feeding_run2
    │   └── (same pattern: inherits + per-run overrides)
    │
    └── ... many other experiment profiles ...
```

Runtime mental model:

```text
selected_profile
    + inherits chain (usually processing_default)
    -> deep merge
    -> resolved effective config used by pipeline
```

Key configuration sections:
- `processing`: Spectral processing parameters (Savitzky-Golay, Lieberfit, ALS)
- `data_source`: File paths and data source settings
- `metadata`: Experiment metadata
- `outputs`: Visualization and export settings

### Baseline Correction Parameters

All of these live under `processing:` in `config.yaml` and are inherited from
`processing_default` unless a profile overrides them.

#### Method selection

| Parameter | Default | Meaning |
|---|---|---|
| `baseline_correction_method` | `transition_stitch` | `transition_stitch`, `transition_ramp`, or legacy `jump_stitch` |
| `stitch_space` | `raw` | Correct the `raw` (physical) or `normalized` channels |

#### Spike removal (Hampel filter)

Applied to every channel before correction. The threshold is in MAD units, so it is
scale-invariant across datasets.

| Parameter | Default | Effect |
|---|---|---|
| `spike_window` | 5 | Half-window for the rolling median |
| `spike_threshold` | 3.0 | Outlier cutoff in MADs. **Higher = fewer spikes removed**; ≥10 effectively disables it |
| `spike_max_length` | 20 | Longest run of consecutive points treated as one spike |

#### Transition detection (shared by `transition_stitch` and `transition_ramp`)

| Parameter | Default | Effect |
|---|---|---|
| `transition_slope_halfwindow_scans` | 5 | `k` in the local slope `median(next k) − median(prev k)` |
| `transition_candidate_sigma` | 6 | Candidate slope threshold, in MADs of the slope distribution |
| `transition_min_displacement` | 250 | Minimum net step for a region to count (normalized fluorescence units) |
| `transition_min_rate_per_scan` | 30 | Minimum sustained rate (net ÷ duration). Rejects slow biology without vetoing long ramps |

#### `transition_stitch` reconstruction

| Parameter | Default | Effect |
|---|---|---|
| `stitch_drift_window_hours` | 24 | Window for estimating and removing the accumulated stitch drift (~one light cycle) |
| `denominator_smooth_scans` | 7 | Light median smoothing of the G-band / 850 denominator |

#### `transition_ramp` reconstruction

| Parameter | Default | Effect |
|---|---|---|
| `slow_window_hours` | 24 | Rolling-median window. This is the artifact/biology boundary: shorter keeps more sub-daily structure, ~24 h removes the daily cycle entirely |

#### Legacy `jump_stitch` parameters

Only read when `baseline_correction_method: jump_stitch`. Retained for reproducing older
results; they have no effect under the default method.

`baseline_correction_threshold`, `baseline_correction_window`,
`baseline_detect_cumulative_jumps`, `baseline_cumulative_window`

#### Tuning guidance

- **Transitions missed** → lower `transition_min_displacement` or `transition_min_rate_per_scan`.
- **Biology wrongly flagged as a transition** → raise `transition_candidate_sigma` or
  `transition_min_rate_per_scan`.
- **Spikes still visible in the corrected output** → lower `spike_threshold` (a per-profile
  override of 10 disables the filter in practice).
- Inspect `processing_stages_*.png` after any change; detected transitions are marked.

## Algorithm Versions

The package currently supports:

- **V4 Algorithm** (default): 
  - Normalized intensity-based processing, with raw (un-normalized) channels also recovered
  - Area-based fluorescence calculation
  - Lorentzian peak fitting
  - Timeseries baseline correction (despiking + transition-ramp stitch; see
    [Timeseries Baseline Correction](#timeseries-baseline-correction))
  - Comprehensive visualization

- **V3 Algorithm**: 
  - Not yet implemented in the refactored package
  - Will raise `NotImplementedError` if selected

## Output Files

Each analysis run generates:

- `processed_data.csv`: Complete processed dataset (columns below)
- `fft_results.csv` / `fft_peaks_*.csv`: Fourier transform analysis results (if applicable)
- `effective_config.yaml`, `run_summary.txt`: The resolved settings and a run report
- Plot files (PNG, with SVG copies in `svg/`):
  - Ratio timeseries (original and baseline-corrected)
  - Normalized intensity timeseries
  - Representative Raman spectrum with the lieberfit baseline and Lorentzian fits
  - FFT analysis plots
  - Processing-stages plots (raw / smoothed / baseline-corrected, transitions marked)

All outputs are saved to a timestamped directory to prevent overwrites.

### `processed_data.csv` columns

| Group | Columns |
|---|---|
| Identifiers | `Datetime`, `Scan Number`, `Seconds` |
| Normalization reference | `Average_Background_Intensity_250_1250_cm-1`, `Normalized_Background_Intensity` |
| Normalized per-scan | `Normalized_Fluorescence_Intensity`, `Normalized_Gband_Area` / `_Intensity` / `_Wavenumber`, `Normalized_Raman_Peak_850_Area` / `_Intensity` / `_Wavenumber` |
| Raw (physical) per-scan | `Raw_Fluorescence_Intensity`, `Raw_Gband_Area`, `Raw_Gband_Intensity`, `Raw_Raman_Peak_850_Area` |
| Baseline-corrected | `<channel>_BaselineCorrected` — prefixed `Raw_` or `Normalized_` depending on `stitch_space` |
| Ratios | `Fluorescence_to_Gband_Ratio`, `Fluorescence_to_Raman_Peak_850_Ratio`, and their `_BaselineCorrected` counterparts |

The `Raw_*` columns undo the per-scan background normalization and are the ones to read when you
want a single channel's physical behaviour; see the note under
[`stitch_space`](#stitch_space-raw-vs-normalized).

## Aggregating Results Across Profiles

`scripts/aggregate_results.py` collects the **latest** results for each profile into one place, so
you do not have to open each output folder:

```bash
python scripts/aggregate_results.py                       # all in-planta profiles
python scripts/aggregate_results.py --downsample 5        # thin the combined timeseries
python scripts/aggregate_results.py --profiles A,B        # specific profiles
python scripts/aggregate_results.py --experiment "in vitro" --output "D:/somewhere"
```

Outputs (default `aggregated_results/`):

| File | Contents |
|---|---|
| `summary.csv` | One QC row per profile: metadata, results folder used, run timestamp, scan count, duration, ratio stats, negative counts, and a `flags` column |
| `combined_timeseries.csv` | Every profile's timeseries stacked long-format with `plant_type` / `treatment` / `light_cycle` / `temp_hum_control` / `replicate_number` columns |
| `overview.png` | PL / G-band ratio (baseline-corrected) |
| `overview_ratio_PL_850.png` | PL / 850 cm⁻¹ Raman ratio (baseline-corrected) |
| `overview_corrected_PL.png` | Corrected PL (fluorescence) |
| `overview_corrected_gband.png` | Corrected G-band area |
| `overview_corrected_850.png` | Corrected 850 cm⁻¹ Raman peak area |

Each overview is a small-multiple grid, one panel per profile, ordered by plant type →
treatment → light cycle → replicate and coloured by treatment (blue Control, red Drought,
green Shade). Panels prefer the `Raw_*` corrected channels and fall back to `Normalized_*`,
marking any panel that fell back so the differing scale is obvious.

> **Panels have independent y-axes.** Profiles differ in level by more than an order of
> magnitude, so a shared axis would flatten most of them. Compare magnitudes using
> `summary.csv`, not by eye across panels.

"Latest" is the results folder whose **name timestamp** is greatest, searched recursively (so
nested `_reprocess` folders are included) and requiring a real `processed_data.csv` so aborted
runs are skipped. Folder-name timestamps are used rather than mtime, which cloud sync can rewrite.
The script only reads results and writes to its output directory, so re-running it is safe.

## Condition-Level Analysis

Two further scripts turn the per-run results into condition-level statements for reporting. Both
read the latest results per profile (same rule as above) and only write to their output directory.
They follow the same plotting conventions as `swnt_iaa_analysis/visualization/plotting.py`:
matching fonts, dashed grid, hidden top/right spines, day/night shading, and 300 dpi PNGs with an
SVG copy in an `svg/` subfolder.

### Aggregating by condition

`scripts/condition_report.py` — one figure per experimental condition, showing the PL/G-band
ratio with every replicate drawn individually plus the mean ± SD.

```bash
python scripts/condition_report.py --output "G:/.../_aggregate/conditions"
python scripts/condition_report.py --bin-minutes 5 --baseline-hours 24 --min-n 2
python scripts/condition_report.py --align-hour 6 --length-mode truncate
```

Four things have to be handled before runs can be averaged, and the script does all four:

| problem | handling |
|---|---|
| Absolute level differs ~6× between runs | each run normalised to **its own baseline** (median of its first `--baseline-hours`), reported as % of baseline |
| Runs start on different calendar dates | aligned on **elapsed time**, not date |
| Runs start at different **times of day** | `--align clock` (default) trims each run forward to the first `--align-hour` |
| Runs differ in sampling interval and duration | resampled to a common grid (`--bin-minutes`, use the **coarsest** native interval); length handled by `--length-mode` |

Without normalisation an average simply tracks whichever run sits highest rather than the biology.

#### Aligning the time of day

Aligning only on elapsed time is not enough. In this dataset the start time-of-day spans up to
**12 hours within one condition** — two runs effectively in antiphase — so "day 1.5" is midnight
in one run and noon in another, and averaging mixes opposite points of the daily cycle.

`--align clock` (default) trims each run forward to the first `--align-hour` (default 00:00) so a
given elapsed time is the same clock time in every run. It costs up to 24 h at the head of each
run and measurably recovers rhythm that was previously cancelled out: diurnal amplitude surviving
in the condition mean rose **+32%** for Bok Choy Control and **+14%** for Nb Control 6to22 — the
two conditions with the largest phase spread — and was unchanged for constant-light runs, as
expected. Use `--align start` to measure from each run's own first sample instead.

#### Handling different run lengths

| `--length-mode` | balanced n? | uses all data? | trend preserved? |
|---|---|---|---|
| `ragged` (default) | no, n falls over time | yes | yes |
| `truncate` | yes | no, discards the tail | yes |
| `fold` | yes | yes | **no** |

`fold` wraps longer runs onto the shortest and averages their segments **within each run**, so a
run still contributes exactly one series and `n` is unchanged. It never manufactures replicates.
But it averages together points that are days apart, so **any trend slower than the fold period is
averaged away** — do not use it when the multi-day trend is the signal (a drought response
building over a week, for example). The active mode is printed on each figure.

> **Do not fold a long run into several pseudo-replicates.** Segments from one plant share the
> plant, sensor, and batch, so treating them as independent replicates is pseudoreplication — it
> shrinks the SD for a reason that is not biological.

#### Reading the figures

Each condition figure is **vertically stacked**: one row per replicate, then the mean ± SD, then
n(t). Replicates are stacked rather than overlaid because at 5–6 runs an overlay is unreadable;
a shared x and y axis keeps the rows directly comparable.

- **Individual replicate traces** are always drawn. At n of 2–6 they are more informative than
  any error band, and show whether the mean reflects consistent behaviour or one outlier.
- **± SD** (spread among plants) is shaded only where n ≥ 3; at n = 2 an SD from one degree of
  freedom would mislead. Use SD for "how variable are plants", SEM for "how well do I know the
  mean" — `condition_summary.csv` reports both.
- A shaded **balanced window** marks where *every* run in the condition still contributes, and an
  n(t) panel shows when replicates drop out. **Quote headline numbers from the balanced window**:
  beyond it the mean shifts partly because the set of runs changes, not the biology.
- **Day/night shading** (yellow lights-on, blue night) follows the same 6to22 / 8to24 / Constant
  convention as the pipeline plots. Each replicate row is shaded from **its own** start time. The
  mean panel is shaded only when every run starts within 2 h of the same time of day — otherwise
  the phase is not shared, and the panel says so rather than showing a misleading overlay.

Outputs: `condition_<condition>.png` per condition (plus an SVG copy under `svg/`), and
`condition_summary.csv` with mean/SD/SEM/n at days 1, 2, 3, 5, 7, 10, 14 plus a `balanced` flag.

### Diurnal reproducibility

`scripts/diurnal_reproducibility.py` — does the daily cycle replicate across runs of the same
condition?

```bash
python scripts/diurnal_reproducibility.py --output "G:/.../_aggregate/diurnal"
python scripts/diurnal_reproducibility.py --agree-threshold 0.6 --min-days 3
```

Method: detrend each run with a 24 h rolling median (so a shared multi-day decline cannot inflate
the correlations), fold the residual onto time-of-day, average the per-day profiles, then compare.
Two measures are reported — **within-run** (day-to-day inside one run: is the plant's own cycle
stable?) and **between-run** (across runs: does the cycle replicate?).

The observed lighting-transition hours are detected from the data and reported alongside, because
runs sharing a config `light_cycle` label do not always share an actual schedule — and
reproducibility tracks the real schedule, not the label. A `schedules_agree` column flags
conditions whose runs transition at different hours; a low correlation there is expected and says
nothing about the measurement.

Each figure has three panels: mean diurnal profiles, the same z-scored (shape only), and a
between-run correlation heatmap. Day/night shading follows the **configured** `light_cycle`, and
each run's **detected** transitions are drawn as dotted vertical lines in its own colour — so a
marker sitting away from a shading edge means the metadata and the data disagree for that run.

Outputs: `diurnal_<condition>.png` (plus an SVG copy under `svg/`), `diurnal_runs.csv`,
`diurnal_reproducibility.csv`, and `agreeing_runs.csv`.

### Restricting the aggregate to reproducible runs

`agreeing_runs.csv` lists, per condition, the largest subset of runs in which **every pair**
correlates at or above `--agree-threshold`. Feed it back in to aggregate only those runs:

```bash
python scripts/diurnal_reproducibility.py --output ".../diurnal"
python scripts/condition_report.py --include ".../diurnal/agreeing_runs.csv" \
    --output ".../conditions_agreeing"
```

It is a plain table, so it can be hand-edited before re-running if you disagree with a call.

> Treat the filtered version as a **robustness check, not a replacement**. Conditions whose runs
> do not agree drop out entirely, so a comparison that depends on them will disappear from the
> filtered set. Report the full set as the primary result and the filtered set as evidence that
> the conclusion survives restriction to reproducible runs.

## Development

This package follows the same architecture as `microneedle_nir_imaging_analysis` for consistency and maintainability. The codebase is organized into logical modules with clear separation of concerns:

- **Core modules**: Low-level data processing algorithms
- **Analysis modules**: Scientific calculations and analysis
- **IO modules**: Configuration and data export
- **Visualization modules**: Plotting and visualization
- **Pipeline**: High-level orchestration
- **CLI**: User-facing command-line interface

