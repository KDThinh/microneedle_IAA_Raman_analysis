# Main Test Guide: Exploratory Testing and Optimization

## Overview

`main_test.py` is a dedicated entry point for **exploratory testing and optimization** of new Lieberfit-based processing methods. It is separate from the production workflow (`main.py`) to keep the production pipeline clean and focused.

### Purpose

- **Test new processing methods** before integrating into production
- **Optimize Lieberfit parameters** automatically or manually
- **Process single scans** for detailed analysis and visualization
- **Process multiple scans in batch** with time series analysis
- **Compare different processing approaches** and parameters

### Key Differences from `main.py`

| Feature | `main.py` (Production) | `main_test.py` (Exploratory) |
|---------|------------------------|------------------------------|
| Purpose | Production data processing | Testing and optimization |
| Workflow | Standard pipeline | New Lieberfit method |
| Outputs | Full analysis suite | Focused test outputs |
| Optimization | Manual parameter tuning | Automatic optimization available |
| Time Series | Always generated | Only in batch mode |
| Master CSV | Can append to master files | No master CSV updates |

---

## Installation and Setup

### Prerequisites

- Python environment with required packages (numpy, pandas, matplotlib, scipy, etc.)
- Access to `config/pipeline.yml` configuration file
- Raman spectroscopy data files

### Configuration

`main_test.py` uses the same configuration system as `main.py`:

- **Default config file**: `config/pipeline.yml`
- **Default profile**: `bok_choy_control_6to22_run1`
- **Configurable via command-line**: `--config-file` and `--profile`

The configuration file provides:
- Data source paths (Raman data, temperature/humidity files)
- Processing parameters (excitation wavelength, smoothing windows, etc.)
- Metadata (plant type, treatment, light cycle, etc.)

---

## Usage Modes

### 1. Single Scan Testing Mode (Default)

Process and visualize a single scan with detailed analysis.

**Basic Usage:**
```bash
python main_test.py
```

**With Optimization:**
```bash
python main_test.py --optimize
```

**With Custom Parameters:**
```bash
python main_test.py --scan 100 --order 6 --iter 150
```

**With Custom Profile:**
```bash
python main_test.py --profile in_planta_default
```

### 2. Batch Processing Mode

Process multiple scans, aggregate results, and generate time series plots.

**Basic Batch:**
```bash
python main_test.py --batch
```

**Batch with Specific Scans:**
```bash
python main_test.py --batch --scans "100,200,300,400"
```

**Batch with Limit:**
```bash
python main_test.py --batch --max-scans 10
```

**Batch with Optimization:**
```bash
python main_test.py --batch --optimize --max-scans 5
```

**Complete Example:**
```bash
python main_test.py --batch --scans "100,150,200" --optimize --order 5 --iter 100 --output-dir ./my_results
```

---

## Command-Line Arguments

### Configuration Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config-file` | str | `config/pipeline.yml` | Path to YAML pipeline config |
| `--profile` | str | `bok_choy_control_6to22_run1` | Profile name from config file |

### Single Scan Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--scan` | int | None | Single scan number to test (uses scan 100 or middle scan if not specified) |

### Batch Processing Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--batch` | flag | False | Enable batch processing mode |
| `--scans` | str | None | Comma-separated scan numbers (e.g., "100,200,300") |
| `--max-scans` | int | None | Limit number of scans processed |

### Optimization Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--optimize` | flag | False | Run Lieberfit parameter optimization |
| `--order` | int | None | Override polynomial order for Lieberfit |
| `--iter` | int | None | Override Lieberfit iterations |

### Output Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | `scripts/test_outputs` | Output directory for results |

---

## Processing Workflow

### Single Scan Mode Workflow

1. **Load Configuration**
   - Reads profile from `config/pipeline.yml`
   - Loads processing parameters (excitation wavelength, smoothing, etc.)

2. **Load Dataset**
   - Loads Raman spectroscopy data based on config paths
   - Selects scan number (default: scan 100 or middle scan)

3. **Data Processing Steps**
   - **Step 1**: Filter wavenumber to 250 cm⁻¹ onward
   - **Step 2**: Smooth with Savitzky-Golay filter
   - **Step 3**: Apply Lieberfit baseline correction
   - **Step 4**: Detect Raman peak (800-950 cm⁻¹) with Lorentzian fitting
   - **Step 5**: Detect G-band peak (1500-1750 cm⁻¹) with Lorentzian fitting
   - **Step 6**: Calculate peak areas (analytical formula)
   - **Step 7**: Calculate fluorescence (two methods)

4. **Optimization (if `--optimize` is used)**
   - Tests multiple parameter combinations:
     - Polynomial orders: 3, 4, 5, 6, 7, 8
     - Iterations: 50, 100, 150, 200, 300
   - Evaluates each combination using:
     - Residual sum of squares (RSS)
     - Mean absolute error (MAE)
     - Baseline smoothness
     - Peak preservation
     - Negative fraction
   - Selects best parameters based on composite score
   - Generates optimization visualization

5. **Visualization**
   - Creates comprehensive 3×2 subplot figure showing:
     - Raw vs Smoothed spectrum
     - Lieberfit baseline correction
     - Peak detection on corrected spectrum
     - Fitted peaks with Lorentzian curves
     - Fluorescence calculation comparison
     - Summary overlay

6. **Output**
   - Saves plot: `test_scan_{scan_number}_new_method.png`
   - Prints comprehensive statistics
   - Returns summary dictionary

### Batch Mode Workflow

1. **First Scan Processing**
   - Processes first scan with full visualization
   - If `--optimize` is used:
     - Runs optimization on first scan
     - Shows optimization plot
     - Extracts optimized parameters
     - Applies optimized parameters to remaining scans

2. **Remaining Scans Processing**
   - Processes each scan independently
   - Suppresses plots (only first scan gets visualization)
   - Uses optimized parameters if optimization was run
   - Collects summary data from each scan

3. **Aggregation**
   - Combines all scan summaries into DataFrame
   - Extracts metrics:
     - Fluorescence (Method 1 and Method 2)
     - Raman peak area, wavenumber, intensity
     - G-band peak area, wavenumber, intensity
     - Fluorescence to G-band ratio
   - Creates datetime-indexed DataFrame

4. **Time Series Analysis**
   - Generates time series plots for:
     - Fluorescence (Method 1)
     - G-band Peak Area
     - Raman Peak Area
     - Fluorescence to G-band Ratio
   - Uses functions from `src/pipeline/plotting.py`
   - Includes day/night shading based on light cycle

5. **Output**
   - Saves CSV: `batch_results_summary.csv`
   - Generates time series plots
   - Prints summary statistics

---

## Outputs

### Single Scan Mode Outputs

**Files Generated:**
- `test_scan_{scan_number}_new_method.png` - Comprehensive visualization (3×2 subplots)
- `lieberfit_optimization.png` - Optimization plot (if `--optimize` is used)

**Console Output:**
- Processing steps summary
- Peak detection results (Raman and G-band)
- Fluorescence values (two methods)
- Comprehensive statistics
- Optimization tips (if applicable)

**Return Value:**
Dictionary containing:
- `scan_number`, `datetime`, `seconds`
- `raman_peak` (dict with area, wavenumber, intensity, etc.)
- `gband_peak` (dict with area, wavenumber, intensity, etc.)
- `fluorescence_method1`, `fluorescence_method2`
- `poly_order`, `tot_iter` (parameters used)
- `output_dir`, `figure_path`

### Batch Mode Outputs

**Files Generated:**
- `test_scan_{first_scan}_new_method.png` - First scan visualization
- `lieberfit_optimization.png` - Optimization plot (if `--optimize` is used)
- `batch_results_summary.csv` - Aggregated results with all metrics
- `fluorescence_method1_timeseries_{date}.png` - Fluorescence time series
- `gband_area_timeseries_{date}.png` - G-band area time series
- `raman_area_timeseries_{date}.png` - Raman area time series
- `fluorescence_gband_ratio_timeseries_{date}.png` - Ratio time series

**Console Output:**
- Batch processing progress
- Optimization results (if used)
- Aggregation summary
- Time series plot generation status

**Output Directory:**
- Default: `scripts/test_outputs_batch_{timestamp}/`
- Customizable via `--output-dir`

---

## Optimization Feature

### When to Use Optimization

Use `--optimize` when:
- You're working with a new dataset
- Default parameters don't give good results
- You want to find optimal parameters automatically
- You're comparing different datasets

### How Optimization Works

1. **Parameter Space Exploration**
   - Tests 30 combinations (6 orders × 5 iterations)
   - Each combination processes the spectrum

2. **Evaluation Metrics**
   - **RSS (Residual Sum of Squares)**: Lower is better
   - **MAE (Mean Absolute Error)**: Lower is better
   - **Baseline Smoothness**: Lower is better (smoother baseline)
   - **Peak Preservation**: Higher is better (preserves Raman/G-band peaks)
   - **Negative Fraction**: Lower is better (corrected spectrum should be positive)

3. **Composite Score**
   - Weighted combination of normalized metrics
   - Penalizes baselines that go above spectrum
   - Selects best overall combination

4. **Visualization**
   - Shows top 5 parameter combinations
   - Parameter space heatmap
   - Best fit detailed view
   - Metrics comparison

5. **Parameter Application**
   - In batch mode: optimized parameters applied to all scans
   - In single scan mode: parameters used for that scan only

### Optimization Tips

- **Run optimization on representative scan**: Use a scan that represents your dataset
- **Check optimization plot**: Verify the baseline looks reasonable
- **Compare methods**: Try both optimized and manual parameters
- **Consider data quality**: Optimization works best with clean, representative data

---

## Examples and Use Cases

### Example 1: Quick Single Scan Test

```bash
python main_test.py --scan 100
```

**Use Case**: Quickly check how the new method processes a specific scan.

**Output**: Single visualization showing all processing steps.

---

### Example 2: Optimize Parameters for Dataset

```bash
python main_test.py --scan 100 --optimize
```

**Use Case**: Find optimal Lieberfit parameters for your dataset.

**Output**: 
- Optimization plot showing parameter space
- Best parameters identified
- Single scan processed with optimized parameters

---

### Example 3: Batch Process with Optimization

```bash
python main_test.py --batch --optimize --max-scans 20
```

**Use Case**: Process multiple scans with automatically optimized parameters.

**Output**:
- Optimization on first scan
- All scans processed with optimized parameters
- Time series plots showing trends
- CSV with all metrics

---

### Example 4: Compare Different Parameters

```bash
# Test with order=5, iter=100
python main_test.py --scan 100 --order 5 --iter 100 --output-dir ./test_order5_iter100

# Test with order=6, iter=150
python main_test.py --scan 100 --order 6 --iter 150 --output-dir ./test_order6_iter150
```

**Use Case**: Compare how different parameters affect processing.

**Output**: Separate visualizations for each parameter set.

---

### Example 5: Process Specific Scans

```bash
python main_test.py --batch --scans "50,100,150,200,250" --output-dir ./specific_scans
```

**Use Case**: Process only specific scans of interest.

**Output**: Time series for selected scans only.

---

## Understanding the Outputs

### Peak Detection

**Raman Peak (800-950 cm⁻¹)**:
- Detected using Lorentzian fitting
- Area calculated analytically: `π × amplitude × HWHM`
- Position, intensity, width reported

**G-band Peak (1500-1750 cm⁻¹)**:
- Detected using Lorentzian fitting
- Area calculated analytically and numerically (trapezoidal)
- Position, intensity, width reported

### Fluorescence Calculation

**Method 1**: AUC of background-subtracted Lieberfit baseline (≥1250 cm⁻¹)
- Uses Lieberfit baseline as fluorescence signal
- Subtracts average background (250-1250 cm⁻¹)
- Integrates remaining signal

**Method 2**: Raw AUC - G-band area - Background AUC
- Calculates raw AUC of fluorescence region
- Subtracts G-band peak area
- Subtracts background AUC

**Comparison**: Both methods are reported for validation.

### Time Series Metrics

When processing multiple scans, the following metrics are tracked over time:
- **Fluorescence**: Background-corrected fluorescence signal
- **G-band Area**: Peak area of G-band
- **Raman Peak Area**: Peak area of Raman peak
- **Fluorescence to G-band Ratio**: Key metric for analysis

---

## Tips and Best Practices

### 1. Start with Single Scan Mode

Before running batch processing, test with a single scan:
```bash
python main_test.py --scan 100 --optimize
```

This helps you:
- Understand the processing workflow
- Verify data quality
- Find optimal parameters

### 2. Use Optimization for New Datasets

When working with a new dataset, always run optimization:
```bash
python main_test.py --batch --optimize --max-scans 5
```

This finds optimal parameters for your specific data.

### 3. Check Optimization Results

After optimization, review:
- Optimization plot (`lieberfit_optimization.png`)
- Best parameters reported
- First scan visualization

If results look poor, try:
- Different scan for optimization
- Manual parameter tuning
- Checking data quality

### 4. Batch Processing Considerations

- **First scan gets full visualization**: Use a representative scan
- **Remaining scans processed silently**: Faster processing
- **Time series requires multiple scans**: At least 2-3 scans recommended
- **Output directory**: Use timestamped directories to avoid overwriting

### 5. Parameter Guidelines

**Polynomial Order (`--order`)**:
- Lower (3-5): Smoother baseline, may underfit complex backgrounds
- Higher (6-8): More flexible, may overfit and remove signal
- Recommended: 5-6 for most cases

**Iterations (`--iter`)**:
- Fewer (50-100): Faster, may not fully converge
- More (200-300): Better convergence, diminishing returns
- Recommended: 100-150 for most cases

### 6. Output Management

- Use `--output-dir` to organize results
- Batch mode creates timestamped directories automatically
- Keep optimization plots for reference
- Compare results across different parameter sets

---

## Troubleshooting

### Issue: No peaks detected

**Possible Causes**:
- Peak search range incorrect
- Signal too weak
- Baseline correction too aggressive

**Solutions**:
- Check peak search ranges in code
- Try different scan
- Adjust optimization parameters
- Check raw data quality

### Issue: Optimization fails

**Possible Causes**:
- Data quality issues
- Insufficient data points
- Parameter ranges too wide

**Solutions**:
- Check data quality
- Try different scan for optimization
- Narrow parameter ranges manually
- Use manual parameters instead

### Issue: Time series plots empty

**Possible Causes**:
- No scans processed successfully
- Datetime extraction failed
- Empty summaries

**Solutions**:
- Check batch processing output
- Verify scan numbers exist
- Check datetime in data
- Review summary aggregation

### Issue: Configuration not found

**Possible Causes**:
- Config file path incorrect
- Profile name misspelled
- Config file missing

**Solutions**:
- Check `--config-file` path
- Verify profile name in `pipeline.yml`
- Use absolute path if needed
- Check file permissions

---

## Integration with Production Pipeline

### When to Use `main_test.py` vs `main.py`

**Use `main_test.py` when**:
- Testing new processing methods
- Optimizing parameters
- Exploring data quality
- Comparing different approaches
- Developing new features

**Use `main.py` when**:
- Processing production data
- Generating final results
- Appending to master CSV files
- Running standard analysis pipeline
- Generating publication-ready outputs

### Migrating from Test to Production

Once you've validated a new method in `main_test.py`:

1. **Document optimal parameters**: Note the best `poly_order` and `tot_iter`
2. **Update config file**: Add new profile or update existing one
3. **Test in production**: Use `main.py` with new parameters
4. **Compare results**: Verify consistency between test and production

---

## Advanced Usage

### Custom Configuration

Create a custom config file:
```yaml
profiles:
  my_custom_profile:
    inherits: bok_choy_control_6to22_run1
    processing:
      poly_order: 6
      tot_iter: 150
    data_source:
      raman_relative_path: path/to/my/data.txt
```

Use it:
```bash
python main_test.py --config-file my_config.yml --profile my_custom_profile
```

### Scripting Multiple Tests

Create a script to test multiple parameter combinations:

```bash
#!/bin/bash
for order in 5 6 7; do
    for iter in 100 150 200; do
        python main_test.py --scan 100 --order $order --iter $iter \
            --output-dir "./results_order${order}_iter${iter}"
    done
done
```

### Extracting Metrics Programmatically

```python
from scripts.test_single_scan_new_method import test_new_method
from pipeline.config_loader import load_profile_config
from pipeline.ingestion import load_raman_dataset

config = load_profile_config("config/pipeline.yml", "bok_choy_control_6to22_run1")
dataset = load_raman_dataset(config)

summary = test_new_method(
    scan_number=100,
    optimize_params=False,
    raman_df=dataset.spectra,
    config=config,
    suppress_plot=True  # Skip visualization
)

print(f"Fluorescence: {summary['fluorescence_method1']}")
print(f"G-band area: {summary['gband_peak']['area']}")
```

---

## Summary

`main_test.py` is a powerful tool for:
- ✅ Testing new processing methods
- ✅ Optimizing Lieberfit parameters
- ✅ Analyzing single scans in detail
- ✅ Processing multiple scans with time series
- ✅ Comparing different parameter sets

Key features:
- Separate from production pipeline
- Automatic parameter optimization
- Comprehensive visualizations
- Time series analysis
- Flexible configuration

Remember:
- Start with single scan mode
- Use optimization for new datasets
- Check optimization results carefully
- Use batch mode for time series analysis
- Keep test outputs organized

For production data processing, use `main.py` instead.

