r"""
This script serves as the main entry point for processing and analyzing Raman/fluorescence spectroscopy data from SWNT-based fluorescence nanosensors designed for continuous in planta monitoring of indole-3-acetic acid (IAA) levels in plants. As an expert in Raman and fluorescence spectroscopy, this code:
- Infers parameters (plant_type, treatment, temp_hum_control, light_cycle, replicate_number) from raman_relative_path, with command-line overrides.
- Processes raw spectral data and generates results in a 'processed_results' subfolder.
- Appends diurnal cycle averages to master_diurnal.csv and Fourier transform data to master_fourier.csv in G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\Master.
"""
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from matplotlib.ticker import MultipleLocator

PROJECT_SRC_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.append(str(PROJECT_SRC_DIR))

PROJECT_ROOT = PROJECT_SRC_DIR.parent

from pipeline.config_loader import load_profile_config
from pipeline.ingestion import load_raman_dataset
from pipeline.master_csv_utils import update_master_diurnal, update_master_fourier
from pipeline.plotting import (
    plot_combined_time_series,
    plot_extrema,
    plot_raman_spectrum,
    plot_stacked_metrics,
    plot_time_series,
)
from pipeline.post_processing import (
    apply_als_and_gaussian,
    compute_diurnal_average,
    compute_fourier_transform,
)
from pipeline.processing import process_all_raman_data, process_raman_data
from pipeline.utils import create_dir_if_needed, find_google_drive

plt.ion()

DEFAULT_CONFIG_PATH = str((PROJECT_ROOT / "config" / "pipeline.yml").resolve())
DEFAULT_PROFILE = "in_planta_default"

def infer_config_from_path(raman_path):
    """
    Infer parameters from raman_relative_path, handling paths with or without Run directory.
    
    Parameters:
    raman_path : str
        Relative path to the Raman data file.
    
    Returns:
    dict : Inferred parameters (plant_type, treatment, temp_hum_control, light_cycle, replicate_number).
    
    Raises:
    ValueError : If path format is invalid or parsing fails.
    """
    path_parts = raman_path.split(os.sep)
    print(f"Path parts: {path_parts}")
    
    if len(path_parts) < 9 or path_parts[-2] != 'Raw data':
        raise ValueError(f"Invalid raman_relative_path format: {raman_path}. Expected structure: ...\\<plant_type>\\Treatment <treatment>\\Light_<light_cycle>\\Temp_Hum_<condition>[\Run <replicate_number>]\\Raw data\\<filename>.txt")
    
    try:
        plant_type = path_parts[5]
        print(f"Parsed plant_type: {plant_type}")
        treatment = path_parts[6].replace('Treatment ', '')
        print(f"Parsed treatment: {treatment}")
        light_cycle = path_parts[7].replace('Light_', '')
        print(f"Parsed light_cycle: {light_cycle}")
        temp_hum = path_parts[8].split('_')[-1]
        temp_hum_control = 'Yes' if temp_hum == 'Constant' else 'No'
        print(f"Parsed temp_hum_control: {temp_hum_control}")
        
        replicate_number = 1
        if len(path_parts) >= 11 and path_parts[9].startswith('Run '):
            replicate_number = int(path_parts[9].replace('Run ', ''))
        print(f"Parsed replicate_number: {replicate_number}")
        
        valid_light_cycles = ['Constant', '8to24', '6to22']
        if light_cycle not in valid_light_cycles:
            raise ValueError(f"Invalid light_cycle '{light_cycle}' inferred from path. Must be one of {valid_light_cycles}")
        
        inferred = {
            'plant_type': plant_type,
            'treatment': treatment,
            'temp_hum_control': temp_hum_control,
            'light_cycle': light_cycle,
            'replicate_number': replicate_number
        }
        print(f"Inferred parameters: {inferred}")
        return inferred
    except Exception as e:
        raise ValueError(f"Failed to parse raman_relative_path: {e}")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process Raman/fluorescence spectroscopy data with profile-based configuration."
    )
    parser.add_argument("--plant_type", type=str, help="Plant type (e.g., Bok_choy)")
    parser.add_argument("--treatment", type=str, help="Treatment (e.g., Control, Shade)")
    parser.add_argument("--temp_hum_control", type=str, choices=["Yes", "No"], help="Temperature/humidity controlled (Yes/No)")
    parser.add_argument("--light_cycle", type=str, choices=["Constant", "8to24", "6to22"], help="Light cycle (Constant, 8to24, 6to22)")
    parser.add_argument("--replicate_number", type=int, help="Replicate number (e.g., 1, 2)")
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
    metadata_overrides = {
        "plant_type": args.plant_type,
        "treatment": args.treatment,
        "temp_hum_control": args.temp_hum_control,
        "light_cycle": args.light_cycle,
        "replicate_number": args.replicate_number,
    }
    dataset = load_raman_dataset(runtime_config, metadata_overrides=metadata_overrides)
    if dataset.warnings:
        for warning in dataset.warnings:
            print(f"[INGESTION WARNING] {warning}")

    plant_type = dataset.metadata["plant_type"]
    treatment = dataset.metadata["treatment"]
    temp_hum_control = dataset.metadata["temp_hum_control"]
    light_cycle = dataset.metadata["light_cycle"]
    replicate_number = dataset.metadata["replicate_number"]
    raman_df = dataset.spectra

    current_date = datetime.now().strftime('%Y-%m-%d')
    data_source_cfg = runtime_config.get('data_source', {})
    requires_drive = data_source_cfg.get('requires_google_drive', True)
    if requires_drive:
        google_drive_root = find_google_drive()
        print(f"Google Drive root: {google_drive_root}")
        base_path = os.path.join(google_drive_root, "My Drive", "Work")
        os.chdir(base_path)
    else:
        base_path = os.path.dirname(dataset.source_path)
    print(f"Current working directory: {os.getcwd()}")
    print(f"Raman file path: {dataset.source_path}")

    temp_rel = runtime_config.get('temp_relative_path')
    if temp_rel:
        temp_file_abs = temp_rel if os.path.isabs(temp_rel) else os.path.join(base_path, temp_rel)
        print(f"Temp/humidity file path: {temp_file_abs}")
    else:
        temp_file_abs = None

    file_path = dataset.source_path
    processed_dir = os.path.join(os.path.dirname(file_path), "processed_results")
    create_dir_if_needed(processed_dir)

    print(f"Loaded Raman data: {raman_df.shape} (scans x wavenumbers + Scan Number + Seconds)")
    print("\nFirst 5 rows:")
    print(raman_df.head())

    default_start = raman_df.index.min()
    default_end = raman_df.index.max()
    start_input = input(f"Enter start datetime (YYYY-MM-DD HH:MM) or press Enter for default ({default_start}): ") or str(default_start)
    end_input = input(f"Enter end datetime (YYYY-MM-DD HH:MM) or press Enter for default ({default_end}): ") or str(default_end)

    try:
        start_datetime = pd.to_datetime(start_input)
        end_datetime = pd.to_datetime(end_input)
    except ValueError:
        print("Invalid datetime format. Using default data range.")
        start_datetime = default_start
        end_datetime = default_end

    raman_df = raman_df[(raman_df.index >= start_datetime) & (raman_df.index <= end_datetime)]

    result = process_raman_data(raman_df, scan_number=100, config=runtime_config)
    if result is not None:
        wavenumbers, emission_nm, intensities_raw, intensities_smooth, baseline_gband, corrected_spectrum_gband, datetime_val, seconds = result
        plot_raman_spectrum(wavenumbers, emission_nm, intensities_raw, intensities_smooth, baseline_gband,
                            corrected_spectrum_gband, 100, datetime_val, seconds, processed_dir, runtime_config)
        print("Plot generation completed successfully.")
    else:
        print("No plot generated due to an error or invalid scan number.")

    results_df = process_all_raman_data(raman_df, runtime_config)
    results_df.to_csv(os.path.join(processed_dir, f'raman_analysis_results_emission_nm_{current_date}.csv'), index_label='Datetime')
    print(f"Results exported to {os.path.join(processed_dir, f'raman_analysis_results_emission_nm_{current_date}.csv')}")
    print("\nFirst 5 rows of results:")
    print(results_df.head())

    plot_df = results_df[results_df['Scan Number'] > 10].copy()
    start_time = plot_df.index.min()
    two_hours_later = start_time + timedelta(hours=2)
    plot_df = plot_df[plot_df.index >= two_hours_later]

    plot_combined_time_series(plot_df, processed_dir, runtime_config, light_cycle=light_cycle)

    if not temp_file_abs:
        raise FileNotFoundError("Temperature/humidity file path not configured for this profile.")
    temp_file_path = temp_file_abs
    if not os.path.exists(temp_file_path):
        raise FileNotFoundError(f"File not found: {temp_file_path}")

    plot_df_temp = pd.read_csv(temp_file_path, parse_dates=['Timestamp'], dayfirst=True)
    plot_df_temp.set_index('Timestamp', inplace=True)
    plot_df_temp = plot_df_temp[(plot_df_temp.index >= start_datetime) & (plot_df_temp.index <= end_datetime)]

    plot_stacked_metrics(plot_df, plot_df_temp, processed_dir, runtime_config, light_cycle=light_cycle)
    plot_extrema(results_df, processed_dir, runtime_config)
    results_df = apply_als_and_gaussian(results_df, runtime_config, processed_dir, start_datetime, end_datetime, light_cycle=light_cycle)
    results_df = compute_diurnal_average(results_df, runtime_config, processed_dir, start_datetime, end_datetime)

    master_diurnal_path = os.path.join(base_path, "DiSTAP", "Research", "Auxin IAA", "IAA-MN longitudinal", "IAA Nanosensor Experiment", "Master", "master_diurnal.csv")
    print(f"Updating master diurnal CSV at: {master_diurnal_path}")
    update_master_diurnal(
        master_diurnal_path, results_df, processed_dir,
        plant_type, treatment, temp_hum_control, light_cycle, replicate_number, current_date
    )

    fft_df_original, fft_df_smoothed, complete_ft_df_smoothed = compute_fourier_transform(results_df, runtime_config, processed_dir, start_datetime, end_datetime)
    print("Pre-update complete_ft_df_smoothed:\n", complete_ft_df_smoothed)
    master_fourier_path = os.path.join(base_path, "DiSTAP", "Research", "Auxin IAA", "IAA-MN longitudinal", "IAA Nanosensor Experiment", "Master", "master_fourier.csv")
    print(f"Updating master Fourier CSV at: {master_fourier_path}")
    update_master_fourier(
        master_fourier_path, complete_ft_df_smoothed, processed_dir,
        plant_type, treatment, temp_hum_control, light_cycle, replicate_number, current_date
    )

if __name__ == "__main__":
    main()
