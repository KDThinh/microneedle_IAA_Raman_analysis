r"""
This script processes Raman/fluorescence spectroscopy data from SWNT-based fluorescence nanosensors for IAA response in cuvette tests, computing and plotting the Pl/G ratio versus scan number.
- Processes raw spectral data from 'run 1.txt' (in vitro test, 830 nm excitation) and generates results in a 'processed_results' subfolder.
- Plots Pl/G ratio versus scan number, reflecting sensor response to IAA.
- Designed for cuvette data: no datetime range selection or light cycle considerations.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_SRC_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.append(str(PROJECT_SRC_DIR))

PROJECT_ROOT = PROJECT_SRC_DIR.parent

from pipeline.config_loader import load_profile_config
from pipeline.ingestion import load_raman_dataset
from pipeline.plotting import plot_raman_spectrum
from pipeline.processing import process_all_raman_data, process_raman_data
from pipeline.utils import create_dir_if_needed

plt.ion()

DEFAULT_CONFIG_PATH = str((PROJECT_ROOT / "config" / "pipeline.yml").resolve())
DEFAULT_PROFILE = "in_vitro_default"

def plot_plg_ratio(plot_df, processed_dir):
    """
    Plot Pl/G ratio versus scan number and save to processed_results folder.
    
    Parameters:
    plot_df : pd.DataFrame
        DataFrame with 'Scan Number' and Pl/G ratio columns.
    processed_dir : str
        Directory to save the plot.
    
    Raises:
    KeyError: If the Pl/G ratio column is not found in plot_df.
    """
    plg_column = 'Fluorescence to G-band Ratio'  # Updated based on CSV columns
    if plg_column not in plot_df.columns:
        print(f"Error: '{plg_column}' column not found in plot_df. Available columns: {list(plot_df.columns)}")
        print("Please update the plg_column variable in plot_plg_ratio with the correct Pl/G ratio column name.")
        raise KeyError(f"'{plg_column}' not found in DataFrame columns")
    
    plt.figure(figsize=(10, 6))
    plt.plot(plot_df['Scan Number'], plot_df[plg_column], color='blue', label='Pl/G Ratio')
    plt.xlabel('Scan Number')
    plt.ylabel('Pl/G Ratio')
    plt.title('SWNT Nanosensor Pl/G Ratio vs Scan Number (In Vitro IAA Test)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(processed_dir, 'plg_ratio_vs_scan.png')
    plt.savefig(output_path)
    plt.close()

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process in-vitro Raman data using shared config profiles.")
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_PATH, help="Path to YAML pipeline config.")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="Profile name to load from the YAML config.")
    parser.add_argument(
        "--config-override",
        action="append",
        default=[],
        help="Override config values using dotted paths, e.g. processing.window_size=31",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    runtime_config = load_profile_config(args.config_file, args.profile, args.config_override)
    dataset = load_raman_dataset(runtime_config)
    if dataset.warnings:
        for warning in dataset.warnings:
            print(f"[INGESTION WARNING] {warning}")
    raman_df = dataset.spectra
    current_date = datetime.now().strftime('%Y-%m-%d')

    # Use the actual source path from the dataset (resolved absolute path)
    file_path = dataset.source_path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # Set base_path to the directory containing the raw data file
    base_path = os.path.dirname(file_path)
    print(f"Base directory: {base_path}")
    print(f"Raman file path: {file_path}")

    # Create timestamped output folder in the same directory as the raw data
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    processed_dir = os.path.join(base_path, f"processed_results_{timestamp}")
    create_dir_if_needed(processed_dir)
    print(f"Output directory: {processed_dir}")

    print(f"Loaded Raman data: {raman_df.shape} (scans x wavenumbers + Scan Number + Seconds)")
    print("\nFirst 5 rows:")
    print(raman_df.head())

    result = process_raman_data(raman_df, scan_number=100, config=runtime_config)
    if result is not None:
        wavenumbers, emission_nm, intensities_raw, intensities_smooth, bg_intensity, intensities_bg_corrected, baseline_gband, corrected_spectrum_gband, scan_datetime, seconds = result
        plot_raman_spectrum(wavenumbers, emission_nm, intensities_raw, intensities_smooth, baseline_gband,
                            corrected_spectrum_gband, 100, None, seconds, processed_dir, runtime_config)
        print("Plot generation completed successfully.")
    else:
        print("No plot generated due to an error or invalid scan number.")

    results_df = process_all_raman_data(raman_df, runtime_config)
    results_df.to_csv(os.path.join(processed_dir, f'raman_analysis_results_emission_nm_{current_date}.csv'), index=False)
    print(f"Results exported to {os.path.join(processed_dir, f'raman_analysis_results_emission_nm_{current_date}.csv')}")
    print("\nFirst 5 rows of results:")
    print(results_df.head())
    print("\nColumns in results_df:")
    print(list(results_df.columns))

    plot_df = results_df[results_df['Scan Number'] > 10].copy()
    plot_plg_ratio(plot_df, processed_dir)
    print("Pl/G ratio versus scan number plot generated.")

if __name__ == "__main__":
    main()
