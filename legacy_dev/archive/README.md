# Archive Directory

This directory contains old and unused code files that are no longer actively used in the project.

## Archived Files

### Entry Point Scripts (Root Level)
These were old entry point scripts that imported from `src/cli/`:

- `main.py` - Original main entry point
- `main_wo_append.py` - Main without append functionality
- `main_wo_append_vertical_plot.py` - Main without append and vertical plot
- `main_in_vitro.py` - In vitro experiment entry point
- `main_updated_v1.py` - Updated version 1 entry point

### Analysis Scripts
- `Fourier_comparison/` - Old Fourier analysis script for Bok Choy only (superseded by `src/analysis/fourier_aggregated.py`)

## Current Active Versions

The current active entry points are:
- `main_test_v3.py` - Latest version (simplified normalized data with ratios)
- `main_test_v2.py` - Previous version (full processing with baseline correction)
- `main_test.py` - Original test version

## Note

If you need to reference any of these archived files, they are preserved here for historical purposes. The corresponding CLI modules may still exist in `src/cli/` if they are still referenced elsewhere.

