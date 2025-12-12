"""
Entry point for processing v4 - refactored version of v3 with modular architecture.
This version uses the same processing workflow as v3 but with all functions organized
into pipeline modules (utils, processing, post_processing, plotting).

Workflow (same as v3):
1. Start at 250 cm-1
2. Savitzky-Golay smoothing for each scan
3. Keep original smoothed scan and normalized smoothed scan (normalized to average background value between 250 cm-1 to 1250 cm-1)
4. Lieberfit the whole spectrum for normalized smoothed scan
5. Detect the Raman peak at around 850 cm-1 and 1600 cm-1 (G-band) from the lieberfit-corrected scan
6. Perform Lorentzian fitting and calculate the AUC for each Raman peak
7. Calculate AUC fluorescence from 1250 cm-1 onward, subtracted the calculated area of G-band
8. Aggregate normalized data into batch_summary_data
9. Apply baseline correction to normalized fluorescence intensity, G-band, and Raman peak at 850 cm-1
10. Calculate fluorescence_to_g_band_ratio and fluorescence_to_raman_peak_850_ratio (before and after baseline correction)
11. Plot normalized data before/after baseline correction and ratio timeseries

Usage examples:
    # Batch processing (default skips first 500 scans)
    python main_test_v4.py --batch
    
    # Batch with baseline correction parameters
    python main_test_v4.py --batch --baseline-threshold 7.5 --baseline-window 15 --correct-smoothed
    
    # Batch with custom scan range
    python main_test_v4.py --batch --scans "100,200,300" --max-scans 10
"""
import argparse
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Add src to path
PROJECT_SRC_DIR = Path(__file__).resolve().parent.parent  # Go up from src/cli/ to src/
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.append(str(PROJECT_SRC_DIR))

PROJECT_ROOT = PROJECT_SRC_DIR.parent  # Go up from src/ to project root

from pipeline.config_loader import load_profile_config
from pipeline.ingestion import load_raman_dataset
from pipeline.utils import (
    add_day_night_shading, 
    create_dir_if_needed, 
    lieberfit,
    parse_light_transition_config,
    parse_shade_transition_config,
    parse_treatment_events_config
)
from pipeline.processing import (
    process_scan_v4,
    aggregate_summaries_to_dataframe_v4
)
from pipeline.post_processing import (
    apply_baseline_correction_v4,
    calculate_ratios,
    apply_als_and_gaussian_v4,
    compute_fourier_transform_v4,
    compute_diurnal_average_v4
)
from pipeline.plotting import (
    plot_normalized_baseline_correction_comparison,
    plot_normalized_smoothing_comparison,
    plot_representative_raman_spectrum,
    plot_overlaid_normalized_spectra,
    plot_ratio_timeseries
)

# Import from scripts (need to add project root to path for scripts)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG_PATH = str((PROJECT_ROOT / "config" / "pipeline.yml").resolve())
DEFAULT_PROFILE = "bok_choy_control_6to22_run1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Processing v4 - refactored version of v3 with modular architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Batch processing (default skips first 500 scans)
  python main_test_v4.py --batch

  # Batch with baseline correction parameters
  python main_test_v4.py --batch --baseline-threshold 7.5 --baseline-window 15 --correct-smoothed

  # Batch with custom scan range
  python main_test_v4.py --batch --scans "100,200,300" --max-scans 10
        """
    )
    
    parser.add_argument(
        "--config-file",
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML pipeline config."
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Profile name to load from the YAML config."
    )
    
    # Batch processing
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch processing on multiple scans instead of single scan."
    )
    parser.add_argument(
        "--scans",
        type=str,
        help="Comma-separated scan numbers for batch processing. Defaults to all scans."
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        help="Limit the number of scans processed in batch mode."
    )
    parser.add_argument(
        "--skip-scans",
        type=int,
        default=500,
        help="Number of scans to skip from the beginning (default: 500)."
    )
    
    # Lieberfit parameters
    parser.add_argument(
        "--order",
        type=int,
        default=None,
        help="Override polynomial order for Lieberfit."
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=None,
        help="Override Lieberfit iterations."
    )
    
    # Baseline correction parameters (same as v3)
    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=None,
        help="MAD threshold multiplier for baseline correction jump detection. If not provided, uses value from config file."
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=None,
        help="Window size for baseline correction offset calculation. If not provided, uses value from config file."
    )
    parser.add_argument(
        "--baseline-smooth-window",
        type=int,
        default=None,
        help="Window size for Savitzky-Golay smoothing before baseline correction (default: 11)."
    )
    parser.add_argument(
        "--baseline-smooth-poly",
        type=int,
        default=None,
        help="Polynomial order for Savitzky-Golay smoothing before baseline correction (default: 2)."
    )
    parser.add_argument(
        "--no-baseline-smooth",
        action="store_true",
        help="Disable smoothing before baseline correction jump detection."
    )
    parser.add_argument(
        "--correct-smoothed",
        action="store_true",
        help="Apply baseline correction to smoothed timeseries data instead of original data (default: True)."
    )
    parser.add_argument(
        "--no-correct-smoothed",
        dest="correct_smoothed",
        action="store_false",
        help="Disable baseline correction on smoothed data (use original data instead)."
    )
    parser.add_argument(
        "--no-baseline-correction",
        action="store_true",
        help="Skip baseline shift correction entirely (no jump detection or correction applied)."
    )
    
    # Spike removal parameters
    parser.add_argument(
        "--remove-spikes",
        action="store_true",
        help="Remove spikes (1-5 data points) from time series before smoothing."
    )
    parser.add_argument(
        "--spike-window",
        type=int,
        default=5,
        help="Half-window size for spike detection (default: 5). Total window = 2*window + 1."
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=3.0,
        help="Threshold in units of MAD for spike detection (default: 3.0). Higher = less sensitive."
    )
    
    # Cumulative jump detection parameters
    parser.add_argument(
        "--no-cumulative-jumps",
        dest="detect_cumulative_jumps",
        action="store_false",
        help="Disable detection of gradual jumps over multiple consecutive points."
    )
    parser.add_argument(
        "--cumulative-window",
        type=int,
        default=5,
        help="Window size for detecting cumulative changes (default: 5). Detects gradual jumps over N consecutive points."
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. Defaults to scripts/test_outputs/"
    )
    
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_profile_config(args.config_file, args.profile)
    
    # Override Lieberfit parameters from command line if provided
    if args.order is not None:
        config['poly_order'] = args.order
    if args.iter is not None:
        config['tot_iter'] = args.iter
    
    # Change to parent directory if using relative paths (requires_google_drive: false)
    # This allows relative paths to be resolved from the parent directory where the data is located
    data_source = config.get("data_source", {})
    requires_drive = data_source.get("requires_google_drive", True)
    original_cwd = os.getcwd()
    if not requires_drive:
        # Change to parent directory (two levels up from Script/swnt_iaa_analysis_v2 to IAA-MN longitudinal)
        parent_dir = PROJECT_ROOT.parent.parent
        os.chdir(parent_dir)
        print(f"Changed working directory to: {os.getcwd()}")
    
    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_raman_dataset(config)
    finally:
        # Restore original working directory
        if not requires_drive:
            os.chdir(original_cwd)
    raman_df = dataset.spectra
    
    print(f"Loaded {len(raman_df)} scans")
    print(f"Wavenumber range: {raman_df.columns[2]} to {raman_df.columns[-1]}")
    print(f"Raw data file: {dataset.source_path}")
    
    # Set output directory relative to raw data file location
    if args.output_dir is None:
        raw_data_dir = Path(dataset.source_path).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.batch:
            output_dir = raw_data_dir / f"test_outputs_v4_batch_{timestamp}"
        else:
            output_dir = raw_data_dir / f"test_outputs_v4_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    if args.batch:
        # Batch processing mode
        print("\n=== Running batch processing v4 (refactored modular architecture) ===")
        
        scan_numbers = None
        multiple_scans_specified = False
        if args.scans:
            try:
                scan_numbers = [
                    int(val.strip())
                    for val in args.scans.split(",")
                    if val.strip()
                ]
                multiple_scans_specified = len(scan_numbers) > 1
            except ValueError:
                raise ValueError("--scans must be a comma-separated list of integers.")
        
        # Get available scans
        available_scans = sorted(raman_df["Scan Number"].unique())
        if not available_scans:
            print("[Batch] No scans available to process.")
            return
        
        # Parse scan list
        if scan_numbers:
            scan_list = [s for s in scan_numbers if s in available_scans]
        else:
            scan_list = list(available_scans)
        
        # Skip first N scans (default: 500)
        if args.skip_scans > 0 and len(scan_list) > args.skip_scans:
            scan_list = scan_list[args.skip_scans:]
            print(f"[Batch] Skipped first {args.skip_scans} scans")
        
        if args.max_scans:
            scan_list = scan_list[:args.max_scans]
        
        if not scan_list:
            print("[Batch] No scans left to process after filtering.")
            return
        
        print(f"[Batch] Processing {len(scan_list)} scans")
        print(f"[Batch] Scan list: {scan_list[:10]}{'...' if len(scan_list) > 10 else ''}")
        
        # Pre-compute wavenumber arrays once (cached for all scans)
        wavenumbers_full = np.array([float(col) for col in raman_df.columns if col not in ['Scan Number', 'Seconds']])
        wavenumber_filter = wavenumbers_full >= 250
        wavenumbers = wavenumbers_full[wavenumber_filter]
        
        # Process all scans
        summaries = []
        # Configure tqdm for Windows PowerShell - use longer update intervals
        # to prevent line repetition issues in PowerShell
        total_scans = len(scan_list)
        pbar = tqdm(total=total_scans, desc="Processing scans", unit="scan", 
                   ncols=100, mininterval=1.0, maxinterval=10.0,
                   file=sys.stderr, dynamic_ncols=False, leave=True,
                   ascii=True, disable=False,
                   bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for scan in scan_list:
            summary = process_scan_v4(
                scan_number=scan,
                raman_df=raman_df,
                config=config,
                cached_wavenumbers_full=wavenumbers_full,
                cached_wavenumber_filter=wavenumber_filter,
                cached_wavenumbers=wavenumbers,
                verbose=False
            )
            if summary:
                summaries.append(summary)
            pbar.update(1)
        
        pbar.close()
        
        if not summaries:
            print("[Batch] No summaries generated.")
            return
        
        print(f"\n[Batch] Completed processing {len(summaries)} scans")
        
        # Aggregate summaries into DataFrame (normalized data only)
        print("\n=== Aggregating results ===")
        # Get x-axis type from config (default: datetime)
        x_axis_type = config.get('timeseries_x_axis', 'datetime')
        if x_axis_type not in ['datetime', 'scan_number']:
            print(f"Warning: Invalid x_axis_type '{x_axis_type}', using 'datetime'")
            x_axis_type = 'datetime'
        
        results_df = aggregate_summaries_to_dataframe_v4(summaries, raman_df, x_axis_type=x_axis_type)
        
        if results_df.empty:
            print("[Batch] Warning: No data to aggregate.")
            return
        
        # Apply baseline correction (skip if --no-baseline-correction flag is set)
        if args.no_baseline_correction:
            print("\n=== Skipping baseline correction (--no-baseline-correction flag set) ===")
            jump_info_dict = {}
        else:
            print("\n=== Applying baseline correction ===")
            # Debug: Print config values to verify they're being read
            print(f"DEBUG: Profile name being used: '{args.profile}'")
            print(f"DEBUG: All config keys containing 'baseline': {sorted([k for k in config.keys() if 'baseline' in k.lower()])}")
            config_threshold = config.get('baseline_correction_threshold', 'NOT FOUND')
            print(f"DEBUG: Config value for 'baseline_correction_threshold': {config_threshold} (type: {type(config_threshold).__name__})")
            print(f"DEBUG: Command-line arg '--baseline-threshold': {args.baseline_threshold}")
            
            # Get baseline correction parameters from command line args, config, or use defaults
            if args.baseline_threshold is not None:
                baseline_threshold = args.baseline_threshold
                print(f"DEBUG: Using command-line argument value: {baseline_threshold}")
            else:
                baseline_threshold = config.get('baseline_correction_threshold', 7.5)
                if config_threshold == 'NOT FOUND':
                    print(f"DEBUG: Config key not found, using default: {baseline_threshold}")
                else:
                    print(f"DEBUG: Using config value: {baseline_threshold}")
            
            print(f"DEBUG: Final baseline_threshold value being used: {baseline_threshold}")
            baseline_window = args.baseline_window if args.baseline_window is not None else config.get('baseline_correction_window', 15)
            baseline_smooth_first = False if args.no_baseline_smooth else config.get('baseline_correction_smooth_first', True)
            baseline_smooth_window = args.baseline_smooth_window if args.baseline_smooth_window is not None else config.get('baseline_correction_smooth_window', 11)
            baseline_smooth_poly = args.baseline_smooth_poly if args.baseline_smooth_poly is not None else config.get('baseline_correction_smooth_poly_order', 2)
            # Default to True if neither flag is provided, otherwise use the flag value
            # Check sys.argv to see which flag was explicitly provided
            if '--no-correct-smoothed' in sys.argv:
                baseline_correct_smoothed = False
            elif '--correct-smoothed' in sys.argv:
                baseline_correct_smoothed = True
            else:
                # Neither flag provided, default to True
                baseline_correct_smoothed = config.get('baseline_correction_correct_smoothed', True)
            
            # Get spike removal parameters
            remove_spikes = args.remove_spikes if hasattr(args, 'remove_spikes') else config.get('remove_spikes', False)
            spike_window = args.spike_window if hasattr(args, 'spike_window') else config.get('spike_window', 5)
            spike_threshold = args.spike_threshold if hasattr(args, 'spike_threshold') else config.get('spike_threshold', 3.0)
            
            # Get cumulative jump detection parameters
            detect_cumulative_jumps = args.detect_cumulative_jumps if hasattr(args, 'detect_cumulative_jumps') else config.get('detect_cumulative_jumps', True)
            cumulative_window = args.cumulative_window if hasattr(args, 'cumulative_window') else config.get('cumulative_window', 5)
            
            print(f"Baseline correction parameters:")
            print(f"  Threshold multiplier: {baseline_threshold}")
            print(f"  Window size: {baseline_window}")
            print(f"  Smooth first: {baseline_smooth_first}")
            if baseline_smooth_first:
                print(f"  Smooth window: {baseline_smooth_window}")
                print(f"  Smooth poly order: {baseline_smooth_poly}")
            print(f"  Correct smoothed data: {baseline_correct_smoothed}")
            print(f"  Remove spikes: {remove_spikes}")
            if remove_spikes:
                print(f"  Spike detection window: {spike_window}")
                print(f"  Spike detection threshold (MAD): {spike_threshold}")
            print(f"  Detect cumulative jumps: {detect_cumulative_jumps}")
            if detect_cumulative_jumps:
                print(f"  Cumulative window size: {cumulative_window}")
            
            results_df, jump_info_dict = apply_baseline_correction_v4(
                results_df,
                threshold_multiplier=baseline_threshold,
                window_size=baseline_window,
                smooth_first=baseline_smooth_first,
                smooth_window=baseline_smooth_window,
                smooth_poly_order=baseline_smooth_poly,
                correct_smoothed=baseline_correct_smoothed,
                remove_spikes=remove_spikes,
                spike_window=spike_window,
                spike_threshold=spike_threshold,
                detect_cumulative_jumps=detect_cumulative_jumps,
                cumulative_window=cumulative_window
            )
            
            # Print jump information
            print("\n=== Jump Detection Summary ===")
            if detect_cumulative_jumps:
                print(f"Jump detection mode: Individual + Cumulative (window={cumulative_window})")
            else:
                print("Jump detection mode: Individual only")
            print()
            
            for col, jump_info in jump_info_dict.items():
                n_jumps = jump_info['n_jumps']
                is_reference = jump_info.get('is_reference', False)
                used_fluorescence_jumps = jump_info.get('used_fluorescence_jumps', False)
                reference_fluo_col = jump_info.get('reference_fluorescence_column', None)
                
                # Only show jump details for fluorescence (reference signal)
                # Skip G-band and Raman peak jump details
                if not is_reference:
                    if used_fluorescence_jumps and reference_fluo_col:
                        info_text = f" [Using jump points from {reference_fluo_col}]"
                    else:
                        info_text = " [Signal-specific detection]"
                    print(f"{col}: {n_jumps} jump(s) detected{info_text}")
                    continue
                
                # For fluorescence (reference signal), show detailed jump information
                info_text = " [Reference signal - jump points used for related signals]"
                print(f"{col}: {n_jumps} jump(s) detected{info_text}")
                if jump_info['jump_info']:
                    for info in jump_info['jump_info']:
                        idx = info['index']
                        scan_num = results_df.index[idx] if idx < len(results_df) else 'N/A'
                        jump_type = info.get('jump_type', 'unknown')
                        jump_type_label = 'Individual' if jump_type == 'individual' else 'Cumulative' if jump_type == 'cumulative' else 'Unknown'
                        print(f"  - Index {idx} (Scan: {scan_num}): step_change={info['step_change']:.4f} [{jump_type_label}]")
        
        # Calculate ratios
        print("\n=== Calculating ratios ===")
        results_df = calculate_ratios(results_df)
        
        # Apply baseline correction for long-term drift removal
        print("\n=== Applying baseline correction (long-term drift removal) ===")
        light_cycle = config.get('light_cycle', 'Constant')
        light_transition = parse_light_transition_config(config)
        shade_transition = parse_shade_transition_config(config)
        treatment_events = parse_treatment_events_config(config)
        results_df = apply_als_and_gaussian_v4(
            results_df, 
            config, 
            output_dir, 
            start_datetime=None, 
            end_datetime=None, 
            light_cycle=light_cycle,
            light_transition=light_transition,
            shade_transition=shade_transition,
            treatment_events=treatment_events
        )
        
        # Print results summary (CSV will be saved at the end)
        print(f"\n[Batch] Results summary:")
        print(results_df.head())
        print(f"\n[Batch] Total scans processed: {len(results_df)}")
        
        # Create plots
        print("\n=== Creating plots ===")
        
        # Plot representative Raman spectrum from first processed scan
        if summaries:
            first_summary = summaries[0]
            print("\n=== Creating representative Raman spectrum plot ===")
            plot_representative_raman_spectrum(first_summary, output_dir, config)
            
            # Plot overlaid normalized spectra if multiple scans were specified via --scans
            if multiple_scans_specified and len(summaries) > 1:
                print("\n=== Creating overlaid normalized Raman spectra plot ===")
                plot_overlaid_normalized_spectra(summaries, output_dir, config)
        
        # Plot normalized baseline correction comparison
        plot_normalized_baseline_correction_comparison(results_df, output_dir, config, light_cycle, jump_info_dict)
        
        # Plot normalized smoothing comparison (before/after smoothing)
        plot_normalized_smoothing_comparison(results_df, output_dir, config, light_cycle, jump_info_dict)
        
        # Plot ratio timeseries (use baseline corrected if available, otherwise use non-corrected)
        use_baseline_corrected = not args.no_baseline_correction
        plot_ratio_timeseries(results_df, output_dir, config, light_cycle, use_baseline_corrected=use_baseline_corrected)
        
        # Compute diurnal averages
        print("\n=== Computing diurnal averages ===")
        results_df = compute_diurnal_average_v4(
            results_df,
            config,
            output_dir,
            start_datetime=None,
            end_datetime=None,
            light_cycle=light_cycle
        )
        
        # Compute Fourier Transform analysis
        print("\n=== Computing Fourier Transform analysis ===")
        fft_results = compute_fourier_transform_v4(
            results_df,
            config,
            output_dir,
            start_datetime=None,
            end_datetime=None,
            light_cycle=light_cycle,
            light_transition=light_transition,
            shade_transition=shade_transition,
            treatment_events=treatment_events
        )
        
        # Save final results CSV (with all processing steps completed)
        csv_path_final = output_dir / "batch_summary_v4_final.csv"
        # Ensure 'Scan Number' is included as a column for CSV export
        csv_df = results_df.copy()
        # Check x_axis_type config to determine if we need to reset index
        if x_axis_type == 'scan_number':
            # If 'Scan Number' is in columns, we're good (it should be preserved from aggregate_summaries_to_dataframe_v4)
            # If it's only in index, reset it
            if csv_df.index.name == 'Scan Number' and 'Scan Number' not in csv_df.columns:
                csv_df.reset_index(inplace=True)
            # If index was converted to datetime but we have 'Scan Number' column, keep it as is
            # If index is integer (scan numbers), make sure 'Scan Number' column exists
            elif pd.api.types.is_integer_dtype(csv_df.index) and 'Scan Number' not in csv_df.columns:
                csv_df['Scan Number'] = csv_df.index.values
                csv_df.reset_index(drop=True, inplace=True)
        csv_df.to_csv(csv_path_final, index=False)
        print(f"[Batch] Final results saved to: {csv_path_final}")
        
        print(f"\n[Batch] All outputs saved to: {output_dir}")
    else:
        # Single scan testing mode
        print("\n=== Running single scan test v4 ===")
        available_scans = sorted(raman_df["Scan Number"].unique())
        if not available_scans:
            print("ERROR: No scans available in dataset.")
            return
        
        # Use middle scan as default
        scan_number = available_scans[len(available_scans) // 2]
        
        if scan_number not in available_scans:
            print(f"ERROR: Scan {scan_number} not found! Available scans: {available_scans[:10]}{'...' if len(available_scans) > 10 else ''}")
            return
        
        summary = process_scan_v4(
            scan_number=scan_number,
            raman_df=raman_df,
            config=config,
            verbose=True
        )
        
        if summary:
            print("\n=== Processing Summary ===")
            for key, value in summary.items():
                print(f"{key}: {value}")


if __name__ == "__main__":
    main()

