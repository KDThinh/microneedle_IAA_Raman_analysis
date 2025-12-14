# Analysis Scripts

This directory contains standalone analysis and comparison scripts for evaluating processing methods and results.

## Scripts

### `analyze_fluorescence_jumps.py`
Compares fluorescence stability between different processing methods by:
- Calculating differences between consecutive scans
- Detecting sudden jumps (>3 standard deviations)
- Comparing Method 1 (Lieberfit baseline AUC) vs Method 2 (Raw - G-band - Background) vs Main.py method
- Analyzing correlation between background jumps and fluorescence jumps
- Calculating coefficient of variation (CV) for stability assessment

**Note:** This script has hardcoded paths to specific experiment data. Update the paths in the script before running.

### `create_overlay_plots.py`
Creates overlay visualization plots comparing different processing methods:
- Normalized time series plots for each method
- Combined comparison plots with all methods
- Overlays background, fluorescence, Raman peak area, and G-band area metrics

**Note:** This script has hardcoded paths to specific experiment data. Update the paths in the script before running.

## Usage

These scripts are standalone and don't require imports from the main project. They can be run directly:

```bash
cd Script/swnt_iaa_analysis_v2/scripts/analysis
python analyze_fluorescence_jumps.py
python create_overlay_plots.py
```

**Important:** Update the hardcoded file paths in both scripts to point to your specific experiment data before running.

