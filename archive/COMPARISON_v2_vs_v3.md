# Comparison: main_test_v2.py vs main_test_v3.py

## Overview

This document compares the outputs and functionality between `main_test_v2.py` (full processing with baseline correction) and `main_test_v3.py` (simplified normalized data with ratios only).

---

## 1. Data Processing

### main_test_v2.py
- **Processes BOTH raw and normalized data**
  - Raw (original smoothed) data processing
  - Normalized smoothed data processing
- **Full workflow**: Smoothing → Normalization → Lieberfit → Peak Detection → Fluorescence Calculation

### main_test_v3.py
- **Processes ONLY normalized data**
  - Normalized smoothed data processing only
  - Skips raw data processing entirely
- **Simplified workflow**: Smoothing → Normalization → Lieberfit → Peak Detection → Fluorescence Calculation → Ratio Calculation

---

## 2. CSV Output Columns

### main_test_v2.py (`batch_summary_data_v2.csv`)

**Common Columns:**
- `Scan Number`
- `Seconds`
- `Datetime` (index)
- `Average_Background_Intensity_250_1250_cm-1`

**Raw (Original Smoothed) Data Columns:**
- `Raw_Fluorescence_Intensity`
- `Raw_Gband_Intensity`
- `Raw_Gband_Area`
- `Raw_Gband_Wavenumber`
- `Raw_Raman_Peak_850_Intensity`
- `Raw_Raman_Peak_850_Area`
- `Raw_Raman_Peak_850_Wavenumber`

**Normalized Smoothed Data Columns:**
- `Normalized_Background_Intensity`
- `Normalized_Fluorescence_Intensity`
- `Normalized_Gband_Intensity`
- `Normalized_Gband_Area`
- `Normalized_Gband_Wavenumber`
- `Normalized_Raman_Peak_850_Intensity`
- `Normalized_Raman_Peak_850_Area`
- `Normalized_Raman_Peak_850_Wavenumber`

**Baseline Corrected Columns (if baseline correction applied):**
- `Raw_Fluorescence_Intensity_BaselineCorrected`
- `Raw_Gband_Area_BaselineCorrected`
- `Raw_Raman_Peak_850_Area_BaselineCorrected`
- `Normalized_Fluorescence_Intensity_BaselineCorrected`
- `Normalized_Gband_Area_BaselineCorrected`
- `Normalized_Raman_Peak_850_Area_BaselineCorrected`
- `Raw_Fluorescence_Intensity_Smoothed` (if `--correct-smoothed` used)
- `Normalized_Fluorescence_Intensity_Smoothed` (if `--correct-smoothed` used)
- ... (similar for other signals)

**Total Columns:** ~20-30 columns (depending on baseline correction settings)

---

### main_test_v3.py (`batch_summary_data_v3.csv`)

**Common Columns:**
- `Scan Number`
- `Seconds`
- `Datetime` (index)
- `Average_Background_Intensity_250_1250_cm-1`

**Normalized Smoothed Data Columns:**
- `Normalized_Fluorescence_Intensity`
- `Normalized_Gband_Area`
- `Normalized_Raman_Peak_850_Area`

**Ratio Columns:**
- `Fluorescence_to_Gband_Ratio` = Normalized_Fluorescence_Intensity / Normalized_Gband_Area
- `Fluorescence_to_Raman_Peak_850_Ratio` = Normalized_Fluorescence_Intensity / Normalized_Raman_Peak_850_Area

**Total Columns:** ~8 columns

---

## 3. Plot Outputs

### main_test_v2.py

**Baseline Correction Comparison Plots:**

1. **`raw_baseline_correction_comparison_YYYY-MM-DD.png`**
   - 3 subplots (stacked vertically):
     - Raw Fluorescence Intensity (before/after correction)
     - Raw G-band Area (before/after correction)
     - Raw Raman Peak 850 Area (before/after correction)
   - Shows jump points marked with 'x' and '+' markers
   - Includes day/night shading if datetime-based

2. **`normalized_baseline_correction_comparison_YYYY-MM-DD.png`**
   - 3 subplots (stacked vertically):
     - Normalized Fluorescence Intensity (before/after correction)
     - Normalized G-band Area (before/after correction)
     - Normalized Raman Peak 850 Area (before/after correction)
   - Shows jump points marked with 'x' and '+' markers
   - Includes day/night shading if datetime-based

**Total Plots:** 2 plots (6 subplots total)

---

### main_test_v3.py

**Ratio Timeseries Plots:**

1. **`fluorescence_to_gband_ratio_timeseries_YYYY-MM-DD.png`**
   - Single plot showing Fluorescence / G-band Ratio over time
   - Includes day/night shading if datetime-based

2. **`fluorescence_to_raman_peak_850_ratio_timeseries_YYYY-MM-DD.png`**
   - Single plot showing Fluorescence / Raman Peak 850 Ratio over time
   - Includes day/night shading if datetime-based

3. **`fluorescence_ratios_comparison_timeseries_YYYY-MM-DD.png`**
   - Combined plot with both ratios overlaid
   - Blue line: Fluorescence / G-band Ratio
   - Green line: Fluorescence / Raman Peak 850 Ratio
   - Includes day/night shading if datetime-based

**Total Plots:** 3 plots

---

## 4. Baseline Correction

### main_test_v2.py
- **Includes baseline correction** using `fix_baseline_shifts_v2.py`
- Uses fluorescence jump points as reference for G-band and Raman peak correction
- Configurable parameters:
  - `--baseline-threshold` (default: 7.5)
  - `--baseline-window` (default: 15)
  - `--baseline-smooth-window` (default: 11)
  - `--baseline-smooth-poly` (default: 2)
  - `--no-baseline-smooth` (disable smoothing)
  - `--correct-smoothed` (default: True)
- Generates jump detection summary
- Marks jump points on plots

### main_test_v3.py
- **NO baseline correction**
- Focuses on ratio calculations only
- Simpler, faster processing

---

## 5. Command-Line Arguments

### main_test_v2.py

**Processing:**
- `--batch` - Batch processing mode
- `--scans` - Comma-separated scan numbers
- `--max-scans` - Limit number of scans
- `--skip-scans` - Skip first N scans (default: 500)
- `--optimize` - Run Lieberfit optimization (not implemented)
- `--order` - Override polynomial order
- `--iter` - Override iterations

**Baseline Correction:**
- `--baseline-threshold` (default: 7.5)
- `--baseline-window` (default: 15)
- `--baseline-smooth-window` (default: 11)
- `--baseline-smooth-poly` (default: 2)
- `--no-baseline-smooth`
- `--correct-smoothed` (default: True)
- `--no-correct-smoothed`

**Output:**
- `--output-dir` - Custom output directory
- `--config-file` - Custom config file
- `--profile` - Profile name

---

### main_test_v3.py

**Processing:**
- `--batch` - Batch processing mode
- `--scans` - Comma-separated scan numbers
- `--max-scans` - Limit number of scans
- `--skip-scans` - Skip first N scans (default: 500)

**Output:**
- `--output-dir` - Custom output directory
- `--config-file` - Custom config file
- `--profile` - Profile name

**Note:** No baseline correction arguments (not applicable)

---

## 6. Use Cases

### Use main_test_v2.py when:
- You need **both raw and normalized data** analysis
- You want to **apply baseline correction** to remove step discontinuities
- You need **detailed peak information** (intensity, area, wavenumber) for both raw and normalized
- You want to **compare raw vs normalized** processing results
- You need to **investigate baseline shifts** and jump points

### Use main_test_v3.py when:
- You only need **normalized data** (most common use case)
- You want to **calculate fluorescence ratios** (G-band and Raman 850)
- You need **simpler, faster processing** without baseline correction
- You want **clean ratio timeseries plots** for visualization
- You're doing **exploratory analysis** or **quick checks**

---

## 7. Performance Comparison

### main_test_v2.py
- **Slower** due to:
  - Processing both raw and normalized data
  - Baseline correction calculations
  - Jump detection and offset calculations
  - More complex plotting with jump point markers

### main_test_v3.py
- **Faster** due to:
  - Processing only normalized data
  - No baseline correction
  - Simpler plotting
  - Fewer calculations

**Estimated speed difference:** v3 is approximately **2-3x faster** than v2 for the same number of scans.

---

## 8. Output File Structure

### main_test_v2.py
```
test_outputs_v2_batch_YYYYMMDD_HHMMSS/
├── batch_summary_data_v2.csv
├── raw_baseline_correction_comparison_YYYY-MM-DD.png
└── normalized_baseline_correction_comparison_YYYY-MM-DD.png
```

### main_test_v3.py
```
test_outputs_v3_batch_YYYYMMDD_HHMMSS/
├── batch_summary_data_v3.csv
├── fluorescence_to_gband_ratio_timeseries_YYYY-MM-DD.png
├── fluorescence_to_raman_peak_850_ratio_timeseries_YYYY-MM-DD.png
└── fluorescence_ratios_comparison_timeseries_YYYY-MM-DD.png
```

---

## 9. Summary Table

| Feature | main_test_v2.py | main_test_v3.py |
|---------|----------------|-----------------|
| **Raw Data Processing** | ✅ Yes | ❌ No |
| **Normalized Data Processing** | ✅ Yes | ✅ Yes |
| **Baseline Correction** | ✅ Yes | ❌ No |
| **Ratio Calculations** | ❌ No | ✅ Yes |
| **CSV Columns** | ~20-30 | ~8 |
| **Plots Generated** | 2 (6 subplots) | 3 |
| **Processing Speed** | Slower | Faster (2-3x) |
| **Jump Point Detection** | ✅ Yes | ❌ No |
| **Peak Wavenumber Info** | ✅ Yes | ❌ No |
| **Peak Intensity Info** | ✅ Yes | ❌ No |
| **Peak Area Info** | ✅ Yes | ✅ Yes (normalized only) |
| **Use Case** | Full analysis | Quick ratio analysis |

---

## 10. Recommendations

1. **For initial exploration**: Use `main_test_v3.py` to quickly calculate ratios and visualize trends
2. **For publication/analysis**: Use `main_test_v2.py` for comprehensive analysis with baseline correction
3. **For ratio-focused studies**: Use `main_test_v3.py` for cleaner, focused output
4. **For baseline shift investigation**: Use `main_test_v2.py` to identify and correct discontinuities

---

## Example Usage

### main_test_v2.py
```bash
# Full processing with baseline correction
python main_test_v2.py --batch --skip-scans 500 \
    --baseline-threshold 7.5 --baseline-window 15 --correct-smoothed
```

### main_test_v3.py
```bash
# Quick ratio calculation
python main_test_v3.py --batch --skip-scans 500
```

