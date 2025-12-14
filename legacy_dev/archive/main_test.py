"""
Entry point for exploratory testing and optimization of new processing methods.
This is separate from the main production workflow to keep main.py clean.

Usage examples:
    # Single scan test with optimization
    python main_test.py --optimize

    # Single scan test with custom parameters
    python main_test.py --scan 100 --order 6 --iter 150

    # Batch processing
    python main_test.py --batch --scans "100,200,300" --max-scans 10

    # Batch with optimization
    python main_test.py --batch --optimize --max-scans 5
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline.config_loader import load_profile_config
from pipeline.ingestion import load_raman_dataset
from pipeline.plotting import plot_combined_time_series, plot_time_series
from pipeline.utils import add_day_night_shading, create_dir_if_needed
from scripts.test_single_scan_new_method import test_new_method
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from matplotlib.dates import DateFormatter, DayLocator

DEFAULT_CONFIG_PATH = str((PROJECT_ROOT / "config" / "pipeline.yml").resolve())
DEFAULT_PROFILE = "bok_choy_control_6to22_run1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exploratory testing and optimization for new Lieberfit processing methods.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single scan test with optimization
  python main_test.py --optimize

  # Single scan test with custom parameters
  python main_test.py --scan 100 --order 6 --iter 150

  # Batch processing on specific scans
  python main_test.py --batch --scans "100,200,300" --max-scans 10

  # Batch with optimization
  python main_test.py --batch --optimize --max-scans 5
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
    
    # Single scan testing
    parser.add_argument(
        "--scan",
        type=int,
        default=None,
        help="Single scan number to test. If not provided, uses default (scan 100 or middle scan)."
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
    
    # Optimization
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run Lieberfit parameter optimization."
    )
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
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results. Defaults to scripts/test_outputs/"
    )
    
    return parser


def aggregate_summaries_to_dataframe(summaries, raman_df, x_axis_type='datetime'):
    """
    Aggregate summaries from batch processing into a DataFrame for time series plotting.
    
    Parameters
    ----------
    summaries : list of dict
        List of summary dictionaries returned from test_new_method()
    raman_df : pandas.DataFrame
        Original Raman dataframe with datetime index
    x_axis_type : str, optional
        Type of x-axis to use: 'datetime' or 'scan_number' (default: 'datetime')
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with datetime or scan_number index and metrics columns
    """
    if not summaries:
        return pd.DataFrame()
    
    # Pre-compute scan_number to datetime lookup dictionary efficiently using groupby
    # Much faster than filtering: O(n) instead of O(n²)
    # groupby returns first occurrence of each scan number, then extract datetime index
    scan_to_datetime = raman_df.groupby('Scan Number').apply(lambda x: x.index[0]).to_dict()
    
    rows = []
    for summary in summaries:
        if summary is None:
            continue
            
        scan_number = summary.get('scan_number')
        datetime_val = summary.get('datetime')
        seconds = summary.get('seconds')
        
        # Get metrics from summary - handle new nested structure
        raman_peak = summary.get('raman_peak', {})
        
        # Raw (Original) Fluorescent Intensity Analysis
        raw_data = summary.get('raw', {})
        fluorescence_method1_raw = raw_data.get('fluorescence_method1', np.nan)
        fluorescence_method2_raw = raw_data.get('fluorescence_method2', np.nan)
        fluorescence_raw_auc = raw_data.get('fluorescence_raw_auc', np.nan)
        gband_peak_raw = raw_data.get('gband_peak', {})
        
        # Normalized Fluorescent Intensity Analysis
        normalized_data = summary.get('normalized', {})
        fluorescence_method1_normalized = normalized_data.get('fluorescence_method1', np.nan)
        fluorescence_method2_normalized = normalized_data.get('fluorescence_method2', np.nan)
        fluorescence_raw_auc_normalized = normalized_data.get('fluorescence_raw_auc', np.nan)
        gband_peak_normalized = normalized_data.get('gband_peak', {})
        
        # Backward compatibility: also check old flat structure
        if np.isnan(fluorescence_method1_raw):
            fluorescence_method1_raw = summary.get('fluorescence_method1', np.nan)
        if np.isnan(fluorescence_method2_raw):
            fluorescence_method2_raw = summary.get('fluorescence_method2', np.nan)
        if np.isnan(fluorescence_raw_auc):
            fluorescence_raw_auc = summary.get('fluorescence_raw_auc', np.nan)
        if not gband_peak_raw:
            gband_peak_raw = summary.get('gband_peak', {})
        
        baseline_bg_intensity = summary.get('baseline_bg_intensity', np.nan)  # Background average (250-1250 cm^-1) for raw spectrum
        
        # Extract peak metrics
        raman_area = raman_peak.get('area') if raman_peak else np.nan
        raman_wavenumber = raman_peak.get('wavenumber') if raman_peak else np.nan
        raman_intensity = raman_peak.get('intensity') if raman_peak else np.nan
        
        # Raw G-band metrics
        gband_area_raw = gband_peak_raw.get('area') if gband_peak_raw else np.nan
        gband_wavenumber_raw = gband_peak_raw.get('wavenumber') if gband_peak_raw else np.nan
        gband_intensity_raw = gband_peak_raw.get('intensity') if gband_peak_raw else np.nan
        
        # Normalized G-band metrics
        gband_area_normalized = gband_peak_normalized.get('area') if gband_peak_normalized else np.nan
        gband_wavenumber_normalized = gband_peak_normalized.get('wavenumber') if gband_peak_normalized else np.nan
        gband_intensity_normalized = gband_peak_normalized.get('intensity') if gband_peak_normalized else np.nan
        
        # Calculate ratios for Raw Method 1
        fluorescence_to_gband_ratio_method1_raw = fluorescence_method1_raw / gband_area_raw if (fluorescence_method1_raw and gband_area_raw and gband_area_raw > 0 and not np.isnan(gband_area_raw)) else np.nan
        fluorescence_to_raman_ratio_method1_raw = fluorescence_method1_raw / raman_area if (fluorescence_method1_raw and raman_area and raman_area > 0 and not np.isnan(raman_area)) else np.nan
        
        # Calculate ratios for Raw Method 2
        fluorescence_to_gband_ratio_method2_raw = fluorescence_method2_raw / gband_area_raw if (fluorescence_method2_raw and gband_area_raw and gband_area_raw > 0 and not np.isnan(gband_area_raw)) else np.nan
        fluorescence_to_raman_ratio_method2_raw = fluorescence_method2_raw / raman_area if (fluorescence_method2_raw and raman_area and raman_area > 0 and not np.isnan(raman_area)) else np.nan
        
        # Calculate ratios for Normalized Method 1
        fluorescence_to_gband_ratio_method1_normalized = fluorescence_method1_normalized / gband_area_normalized if (fluorescence_method1_normalized and gband_area_normalized and gband_area_normalized > 0 and not np.isnan(gband_area_normalized)) else np.nan
        fluorescence_to_raman_ratio_method1_normalized = fluorescence_method1_normalized / raman_area if (fluorescence_method1_normalized and raman_area and raman_area > 0 and not np.isnan(raman_area)) else np.nan
        
        # Calculate ratios for Normalized Method 2
        fluorescence_to_gband_ratio_method2_normalized = fluorescence_method2_normalized / gband_area_normalized if (fluorescence_method2_normalized and gband_area_normalized and gband_area_normalized > 0 and not np.isnan(gband_area_normalized)) else np.nan
        fluorescence_to_raman_ratio_method2_normalized = fluorescence_method2_normalized / raman_area if (fluorescence_method2_normalized and raman_area and raman_area > 0 and not np.isnan(raman_area)) else np.nan
        
        # Use fixed column names organized by Raw and Normalized analyses
        row = {
            'Scan Number': scan_number,
            'Seconds': seconds,
            
            # Raw (Original) Fluorescent Intensity Analysis
            'Raw_Fluorescence_Method1': fluorescence_method1_raw,
            'Raw_Fluorescence_Method2': fluorescence_method2_raw,
            'Raw_Fluorescence_RawAUC': fluorescence_raw_auc,
            'Raw_Gband_area': gband_area_raw,
            'Raw_Gband_wavenumber': gband_wavenumber_raw,
            'Raw_Gband_intensity': gband_intensity_raw,
            'Raw_Fluorescence_to_Gband_Ratio_Method1': fluorescence_to_gband_ratio_method1_raw,
            'Raw_Fluorescence_to_Raman_Ratio_Method1': fluorescence_to_raman_ratio_method1_raw,
            'Raw_Fluorescence_to_Gband_Ratio_Method2': fluorescence_to_gband_ratio_method2_raw,
            'Raw_Fluorescence_to_Raman_Ratio_Method2': fluorescence_to_raman_ratio_method2_raw,
            
            # Normalized Fluorescent Intensity Analysis
            'Normalized_Fluorescence_Method1': fluorescence_method1_normalized,
            'Normalized_Fluorescence_Method2': fluorescence_method2_normalized,
            'Normalized_Fluorescence_RawAUC': fluorescence_raw_auc_normalized,
            'Normalized_Gband_area': gband_area_normalized,
            'Normalized_Gband_wavenumber': gband_wavenumber_normalized,
            'Normalized_Gband_intensity': gband_intensity_normalized,
            'Normalized_Fluorescence_to_Gband_Ratio_Method1': fluorescence_to_gband_ratio_method1_normalized,
            'Normalized_Fluorescence_to_Raman_Ratio_Method1': fluorescence_to_raman_ratio_method1_normalized,
            'Normalized_Fluorescence_to_Gband_Ratio_Method2': fluorescence_to_gband_ratio_method2_normalized,
            'Normalized_Fluorescence_to_Raman_Ratio_Method2': fluorescence_to_raman_ratio_method2_normalized,
            
            # Common metrics
            'Raman_peak_area': raman_area,
            'Raman_peak_wavenumber': raman_wavenumber,
            'Raman_peak_intensity': raman_intensity,
            'Background_Average_250_1250_cm-1': baseline_bg_intensity,
        }
        
        # Store both datetime and scan_number for flexibility
        if datetime_val:
            row['Datetime'] = pd.to_datetime(datetime_val)
        else:
            # Use pre-computed lookup dictionary (O(1) instead of O(n) filtering)
            if scan_number in scan_to_datetime:
                row['Datetime'] = scan_to_datetime[scan_number]
            else:
                continue  # Skip if we can't find datetime
        
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    
    # Set index based on x_axis_type
    if x_axis_type == 'scan_number':
        df.set_index('Scan Number', inplace=True)
        df.sort_index(inplace=True)
    else:  # datetime (default)
        df.set_index('Datetime', inplace=True)
        df.sort_index(inplace=True)
    
    return df


def plot_comparison_time_series(df, columns, labels, colors, processed_dir, filename, config, light_cycle, shading=True, smooth_sigma=None, x_axis_type='datetime'):
    """
    Plot multiple time-series metrics on the same plot for comparison.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with datetime or scan_number index and metric columns
    columns : list of str
        List of column names to plot
    labels : list of str
        List of labels for each series
    colors : list of str
        List of colors for each series
    processed_dir : str
        Output directory for saving plots
    filename : str
        Base filename for the saved plot (without extension)
    config : dict
        Configuration dictionary
    light_cycle : str
        Light cycle for shading ('Constant', '8to24', '6to22')
    shading : bool, optional
        Whether to add day/night shading (default: True)
    smooth_sigma : float, optional
        Standard deviation for Gaussian smoothing
    x_axis_type : str, optional
        Type of x-axis: 'datetime' or 'scan_number' (default: 'datetime')
    """
    create_dir_if_needed(processed_dir)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for col, label, color in zip(columns, labels, colors):
        if col in df.columns:
            # Plot raw data
            # Use color parameter directly instead of format string to support named colors
            ax.plot(df.index, df[col], '.-', color=color, label=label, alpha=0.7, markersize=3)
            
            # Plot smoothed data if requested
            if smooth_sigma is not None:
                smoothed_values = gaussian_filter1d(df[col].values, sigma=smooth_sigma)
                ax.plot(df.index, smoothed_values, '--', color=color, label=f'Smoothed {label}', linewidth=2, alpha=0.8)
    
    # Only add shading for datetime-based plots (not meaningful for scan_number)
    if shading and x_axis_type == 'datetime':
        add_day_night_shading(ax, df.index.min(), df.index.max(), light_cycle=light_cycle)
    
    # Set x-axis label and formatting based on x_axis_type
    if x_axis_type == 'scan_number':
        ax.set_xlabel('Scan Number', fontsize=14)
    else:  # datetime
        ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
        ax.xaxis.set_major_locator(DayLocator())
        fig.autofmt_xdate()
    
    ax.set_ylabel('Value', fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    
    save_path = Path(processed_dir) / f'{filename.split(".")[0]}_{datetime.now().strftime("%Y-%m-%d")}.png'
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved successfully to {save_path}")
    plt.close()


def plot_time_series_wrapper(df, column, label, color, processed_dir, filename, config, light_cycle, shading=True, smooth_sigma=None, x_axis_type='datetime'):
    """
    Wrapper for plot_time_series that supports both datetime and scan_number x-axis.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with datetime or scan_number index
    column : str
        Column name to plot
    label : str
        Label for the plot
    color : str
        Color for the plot line
    processed_dir : str
        Output directory
    filename : str
        Base filename
    config : dict
        Configuration dictionary
    light_cycle : str
        Light cycle for shading
    shading : bool, optional
        Whether to add shading
    smooth_sigma : float, optional
        Standard deviation for Gaussian smoothing
    x_axis_type : str, optional
        Type of x-axis: 'datetime' or 'scan_number'
    """
    create_dir_if_needed(processed_dir)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot raw data
    ax.plot(df.index, df[column], '.-', color=color, label=label, alpha=0.7, markersize=3)
    
    # Plot smoothed data if requested
    if smooth_sigma is not None:
        smoothed_values = gaussian_filter1d(df[column].values, sigma=smooth_sigma)
        ax.plot(df.index, smoothed_values, '--', color=color, label=f'Smoothed {label}', linewidth=2, alpha=0.8)
    
    # Only add shading for datetime-based plots
    if shading and x_axis_type == 'datetime':
        add_day_night_shading(ax, df.index.min(), df.index.max(), light_cycle=light_cycle)
    
    # Set x-axis label and formatting based on x_axis_type
    if x_axis_type == 'scan_number':
        ax.set_xlabel('Scan Number', fontsize=14)
    else:  # datetime
        ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
        ax.xaxis.set_major_locator(DayLocator())
        fig.autofmt_xdate()
    
    ax.set_ylabel(label, fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    
    save_path = Path(processed_dir) / f'{filename.split(".")[0]}_{datetime.now().strftime("%Y-%m-%d")}.png'
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.close()


def create_time_series_plots(results_df, output_dir, config, light_cycle=None):
    """
    Create time series plots for metrics from batch processing.
    
    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame with datetime or scan_number index and metric columns
    output_dir : Path
        Output directory for saving plots
    config : dict
        Configuration dictionary
    light_cycle : str, optional
        Light cycle for shading ('Constant', '8to24', '6to22')
    """
    if results_df.empty:
        print("Warning: No data to plot in time series.")
        return
    
    # Get light cycle from config if not provided
    if light_cycle is None:
        light_cycle = config.get('light_cycle', 'Constant')
    
    # Get x-axis type from config (default: datetime)
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    if x_axis_type not in ['datetime', 'scan_number']:
        print(f"Warning: Invalid x_axis_type '{x_axis_type}', using 'datetime'")
        x_axis_type = 'datetime'
    
    output_dir_str = str(output_dir)
    
    # Create combined time series plot (similar to production pipeline)
    # Note: We need to map our column names to what plot_combined_time_series expects
    # For now, create custom plots with our column names
    print(f"\n=== Creating time series plots (x-axis: {x_axis_type}) ===")
    
    # ========================================================================
    # Raw (Original) Fluorescent Intensity Analysis
    # ========================================================================
    print("\n--- Raw (Original) Fluorescent Intensity Analysis ---")
    
    # Raw: Fluorescence comparison plot (Methods 1, 2, and raw AUC)
    if 'Raw_Fluorescence_Method1' in results_df.columns and 'Raw_Fluorescence_Method2' in results_df.columns and 'Raw_Fluorescence_RawAUC' in results_df.columns:
        plot_comparison_time_series(
            results_df,
            ['Raw_Fluorescence_Method1', 'Raw_Fluorescence_Method2', 'Raw_Fluorescence_RawAUC'],
            ['Raw Method 1 (Lieberfit baseline AUC)', 'Raw Method 2 (Raw - G-band - Background)', 'Raw AUC (no correction)'],
            ['r', 'b', 'orange'],
            output_dir_str,
            'raw_fluorescence_all_methods_timeseries',
            config,
            light_cycle,
            shading=True,
            x_axis_type=x_axis_type
        )
    
    # Raw: Fluorescence to G-band ratio comparison plot (both methods with G-band analysis)
    if 'Raw_Fluorescence_to_Gband_Ratio_Method1' in results_df.columns and 'Raw_Fluorescence_to_Gband_Ratio_Method2' in results_df.columns:
        plot_comparison_time_series(
            results_df,
            ['Raw_Fluorescence_to_Gband_Ratio_Method1', 'Raw_Fluorescence_to_Gband_Ratio_Method2'],
            ['Raw Method 1 / G-band', 'Raw Method 2 / G-band'],
            ['m', 'c'],
            output_dir_str,
            'raw_fluorescence_gband_ratio_comparison_timeseries',
            config,
            light_cycle,
            shading=True,
            smooth_sigma=config.get('sigma_gaussian', 25),
            x_axis_type=x_axis_type
        )
    
    # Raw: G-band area time series
    if 'Raw_Gband_area' in results_df.columns:
        plot_time_series_wrapper(
            results_df,
            'Raw_Gband_area',
            'Raw G-band Peak Area',
            'g',
            output_dir_str,
            'raw_gband_area_timeseries',
            config,
            light_cycle,
            shading=True,
            x_axis_type=x_axis_type
        )
    
    # ========================================================================
    # Normalized Fluorescent Intensity Analysis
    # ========================================================================
    print("\n--- Normalized Fluorescent Intensity Analysis ---")
    
    # Normalized: Fluorescence comparison plot (Methods 1, 2, and raw AUC)
    if 'Normalized_Fluorescence_Method1' in results_df.columns and 'Normalized_Fluorescence_Method2' in results_df.columns and 'Normalized_Fluorescence_RawAUC' in results_df.columns:
        plot_comparison_time_series(
            results_df,
            ['Normalized_Fluorescence_Method1', 'Normalized_Fluorescence_Method2', 'Normalized_Fluorescence_RawAUC'],
            ['Normalized Method 1 (Lieberfit baseline AUC)', 'Normalized Method 2 (Raw - G-band - Background)', 'Normalized Raw AUC (no correction)'],
            ['r', 'b', 'orange'],
            output_dir_str,
            'normalized_fluorescence_all_methods_timeseries',
            config,
            light_cycle,
            shading=True,
            x_axis_type=x_axis_type
        )
    
    # Normalized: Fluorescence to G-band ratio comparison plot (both methods with G-band analysis)
    if 'Normalized_Fluorescence_to_Gband_Ratio_Method1' in results_df.columns and 'Normalized_Fluorescence_to_Gband_Ratio_Method2' in results_df.columns:
        plot_comparison_time_series(
            results_df,
            ['Normalized_Fluorescence_to_Gband_Ratio_Method1', 'Normalized_Fluorescence_to_Gband_Ratio_Method2'],
            ['Normalized Method 1 / G-band', 'Normalized Method 2 / G-band'],
            ['m', 'c'],
            output_dir_str,
            'normalized_fluorescence_gband_ratio_comparison_timeseries',
            config,
            light_cycle,
            shading=True,
            smooth_sigma=config.get('sigma_gaussian', 25),
            x_axis_type=x_axis_type
        )
    
    # Normalized: G-band area time series
    if 'Normalized_Gband_area' in results_df.columns:
        plot_time_series_wrapper(
            results_df,
            'Normalized_Gband_area',
            'Normalized G-band Peak Area',
            'g',
            output_dir_str,
            'normalized_gband_area_timeseries',
            config,
            light_cycle,
            shading=True,
            x_axis_type=x_axis_type
        )
    
    # ========================================================================
    # Common metrics
    # ========================================================================
    print("\n--- Common Metrics ---")
    
    # Background time series plot (normalization factor diagnostic)
    if 'Background_Average_250_1250_cm-1' in results_df.columns:
        # Create enhanced diagnostic plot for normalization factor
        create_dir_if_needed(output_dir_str)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot normalization factor
        ax.plot(results_df.index, results_df['Background_Average_250_1250_cm-1'], 'o-', 
                color='orange', label='Normalization Factor', alpha=0.7, markersize=4)
        
        # Calculate statistics
        norm_factor = results_df['Background_Average_250_1250_cm-1'].dropna()
        if len(norm_factor) > 0:
            mean_norm = norm_factor.mean()
            std_norm = norm_factor.std()
            cv_norm = (std_norm / mean_norm * 100) if mean_norm > 0 else np.nan
            
            # Add mean line
            ax.axhline(y=mean_norm, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_norm:.2f}')
            
            # Add ±1 std bands
            x_min = results_df.index.min()
            x_max = results_df.index.max()
            ax.axhspan(mean_norm - std_norm, mean_norm + std_norm,
                      alpha=0.2, color='red', label=f'±1 SD: {std_norm:.2f}')
            
            # Add statistics text box
            stats_text = f'Mean: {mean_norm:.2f}\nStd: {std_norm:.2f}\nCV: {cv_norm:.2f}%'
            if cv_norm < 5:
                stability = 'Very Stable'
                color = 'green'
            elif cv_norm < 10:
                stability = 'Stable'
                color = 'yellow'
            elif cv_norm < 20:
                stability = 'Moderate Variation'
                color = 'orange'
            else:
                stability = 'High Variation'
                color = 'red'
            
            stats_text += f'\nStability: {stability}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
        
        # Only add shading for datetime-based plots
        if x_axis_type == 'datetime':
            add_day_night_shading(ax, results_df.index.min(), results_df.index.max(), light_cycle=light_cycle)
            ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
            fig.autofmt_xdate()
        else:  # scan_number
            ax.set_xlabel('Scan Number', fontsize=14)
        
        ax.set_ylabel('Normalization Factor\n(Average Intensity 250-1250 cm^-1)', fontsize=14)
        ax.set_title('Normalization Factor Diagnostic Plot\n(Stability Check)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=12)
        
        save_path = Path(output_dir_str) / f'normalization_factor_diagnostic_{datetime.now().strftime("%Y-%m-%d")}.png'
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Normalization factor diagnostic plot saved to: {save_path}")
        
        # Also create standard time series plot
        plot_time_series_wrapper(
            results_df,
            'Background_Average_250_1250_cm-1',
            'Background Average (250-1250 cm^-1)',
            'y',  # Yellow color
            output_dir_str,
            'background_timeseries',
            config,
            light_cycle,
            shading=True,
            x_axis_type=x_axis_type
        )
    
    # Raman peak area time series
    if 'Raman_peak_area' in results_df.columns:
        plot_time_series_wrapper(
            results_df,
            'Raman_peak_area',
            'Raman Peak Area',
            'b',
            output_dir_str,
            'raman_area_timeseries',
            config,
            light_cycle,
            shading=True,
            x_axis_type=x_axis_type
        )


def run_batch_with_optimization(
    scan_list,
    optimize,
    custom_order,
    custom_tot_iter,
    config,
    profile_name,
    raman_df,
    output_dir
):
    """
    Run batch processing with optimization on first scan, then apply to rest.
    
    Parameters
    ----------
    scan_list : list of int
        List of scan numbers to process
    optimize : bool
        Whether to optimize parameters on first scan
    custom_order : int, optional
        Override polynomial order
    custom_tot_iter : int, optional
        Override iterations
    config : dict
        Configuration dictionary
    profile_name : str
        Profile name
    raman_df : pandas.DataFrame
        Raman dataset
    output_dir : Path
        Output directory
    
    Returns
    -------
    list of dict
        List of summaries from processing
    """
    summaries = []
    optimized_order = custom_order
    optimized_iter = custom_tot_iter
    
    # Pre-compute wavenumber arrays once (cached for all scans)
    # This saves ~6-12ms per scan by avoiding repeated computation
    wavenumbers_full = np.array([float(col) for col in raman_df.columns if col not in ['Scan Number', 'Seconds']])
    wavenumber_filter = wavenumbers_full >= 250
    wavenumbers = wavenumbers_full[wavenumber_filter]
    
    # Process first scan with optimization if requested
    if scan_list:
        first_scan = scan_list[0]
        print(f"\n=== Processing first scan ({first_scan}) ===")
        
        if optimize:
            print("Running optimization on first scan...")
            summary = test_new_method(
                scan_number=first_scan,
                optimize_params=True,  # Enable optimization
                custom_order=custom_order,
                custom_tot_iter=custom_tot_iter,
                raman_df=raman_df,
                config=config,
                output_dir=output_dir,
                profile_name=profile_name,
                suppress_plot=False,  # Show plot for first scan
                cached_wavenumbers_full=wavenumbers_full,      # Pass cached arrays
                cached_wavenumber_filter=wavenumber_filter,
                cached_wavenumbers=wavenumbers,
            )
            
            if summary:
                # Extract optimized parameters from the summary
                optimized_order = summary.get('poly_order', custom_order)
                optimized_iter = summary.get('tot_iter', custom_tot_iter)
                print(f"\n[Batch] Using optimized parameters from first scan:")
                print(f"  Polynomial order: {optimized_order}")
                print(f"  Iterations: {optimized_iter}")
                summaries.append(summary)
        else:
            # Process first scan normally (with plot)
            summary = test_new_method(
                scan_number=first_scan,
                optimize_params=False,
                custom_order=custom_order,
                custom_tot_iter=custom_tot_iter,
                raman_df=raman_df,
                config=config,
                output_dir=output_dir,
                profile_name=profile_name,
                suppress_plot=False,  # Show plot for first scan
                cached_wavenumbers_full=wavenumbers_full,      # Pass cached arrays
                cached_wavenumber_filter=wavenumber_filter,
                cached_wavenumbers=wavenumbers,
            )
            
            if summary:
                # Still extract parameters used (may be from config)
                optimized_order = summary.get('poly_order', custom_order)
                optimized_iter = summary.get('tot_iter', custom_tot_iter)
                summaries.append(summary)
        
        # Process remaining scans without plots
        remaining_scans = scan_list[1:]
        if remaining_scans:
            print(f"\n=== Processing remaining {len(remaining_scans)} scans (no plots) ===")
            # Use tqdm for clean single-line progress bar
            # Configured for Windows PowerShell compatibility
            for scan in tqdm(remaining_scans, desc="Processing scans", unit="scan", 
                           ncols=80, mininterval=1.0, file=sys.stdout, 
                           dynamic_ncols=False, leave=False, ascii=True):
                summary = test_new_method(
                    scan_number=scan,
                    optimize_params=False,  # Don't optimize again
                    custom_order=optimized_order,
                    custom_tot_iter=optimized_iter,
                    raman_df=raman_df,
                    config=config,
                    output_dir=output_dir,
                    profile_name=profile_name,
                    suppress_plot=True,  # Suppress plots for remaining scans
                    verbose=False,  # Suppress verbose output for batch processing
                    skip_plot_creation=True,  # Skip figure creation entirely for speed
                    cached_wavenumbers_full=wavenumbers_full,      # Pass cached arrays
                    cached_wavenumber_filter=wavenumber_filter,
                    cached_wavenumbers=wavenumbers,
                )
                
                if summary:
                    summaries.append(summary)
    
    return summaries


def create_overlaid_spectra_plots(scan_list, raman_df, config, output_dir, wavenumbers_full=None, wavenumber_filter=None, wavenumbers=None):
    """
    Create overlaid plots of raw and normalized spectra for multiple selected scans.
    
    Parameters
    ----------
    scan_list : list of int
        List of scan numbers to overlay
    raman_df : pandas.DataFrame
        Raman dataset
    config : dict
        Configuration dictionary
    output_dir : Path
        Output directory
    wavenumbers_full : array, optional
        Pre-computed full wavenumber array
    wavenumber_filter : array, optional
        Pre-computed wavenumber filter mask
    wavenumbers : array, optional
        Pre-computed filtered wavenumber array
    """
    if len(scan_list) < 2:
        return  # Need at least 2 scans to overlay
    
    print(f"\n=== Creating overlaid spectra plots for {len(scan_list)} scans ===")
    
    # Get wavenumber arrays
    if wavenumbers_full is None or wavenumber_filter is None or wavenumbers is None:
        wavenumbers_full = np.array([float(col) for col in raman_df.columns if col not in ['Scan Number', 'Seconds']])
        wavenumber_filter = wavenumbers_full >= 250
        wavenumbers = wavenumbers_full[wavenumber_filter]
    
    # Get processing parameters
    window_size = config.get('window_size', 25)
    poly_order_sg = config.get('poly_order_sg', 2)
    poly_order = config.get('poly_order', 5)
    tot_iter = config.get('tot_iter', 100)
    
    # Collect spectra data
    raw_spectra = {}
    normalized_spectra = {}
    
    for scan_num in scan_list:
        scan_data = raman_df[raman_df['Scan Number'] == scan_num]
        if scan_data.empty:
            continue
        
        # Extract intensities
        intensities_raw_full = scan_data.iloc[0, 2:].values
        intensities_raw = intensities_raw_full[wavenumber_filter]
        
        # Smooth
        intensities_smooth = savgol_filter(intensities_raw, window_size, poly_order_sg)
        raw_spectra[scan_num] = intensities_smooth.copy()
        
        # Calculate normalization factor
        baseline_bg_mask = (wavenumbers >= 250) & (wavenumbers <= 1250)
        if np.any(baseline_bg_mask):
            bg_intensity_avg_250_1250 = np.mean(intensities_smooth[baseline_bg_mask])
        else:
            bg_intensity_avg_250_1250 = np.mean(intensities_smooth)
        
        # Normalize
        if bg_intensity_avg_250_1250 > 0:
            intensities_normalized = intensities_smooth / bg_intensity_avg_250_1250
            normalized_spectra[scan_num] = intensities_normalized.copy()
    
    if not raw_spectra:
        print("Warning: No valid scans found for overlay plot")
        return
    
    # Create raw intensity overlay plot
    fig_raw, ax_raw = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(raw_spectra)))
    
    for idx, (scan_num, spectrum) in enumerate(raw_spectra.items()):
        ax_raw.plot(wavenumbers, spectrum, '-', linewidth=1.5, alpha=0.7, 
                   color=colors[idx], label=f'Scan {scan_num}')
    
    ax_raw.set_xlabel('Wavenumber (cm^-1)', fontsize=14)
    ax_raw.set_ylabel('Intensity (Raw)', fontsize=14)
    ax_raw.set_title(f'Overlaid Raw Spectra - {len(raw_spectra)} Scans', fontsize=16, fontweight='bold')
    ax_raw.legend(fontsize=11, loc='best', ncol=2)
    ax_raw.grid(True, alpha=0.3)
    ax_raw.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    save_path_raw = Path(output_dir) / f'overlaid_raw_spectra_scans_{"_".join(map(str, scan_list))}.png'
    plt.savefig(str(save_path_raw), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overlaid raw spectra plot saved to: {save_path_raw}")
    
    # Create normalized intensity overlay plot
    if normalized_spectra:
        fig_norm, ax_norm = plt.subplots(figsize=(14, 8))
        
        for idx, (scan_num, spectrum) in enumerate(normalized_spectra.items()):
            ax_norm.plot(wavenumbers, spectrum, '-', linewidth=1.5, alpha=0.7, 
                        color=colors[idx], label=f'Scan {scan_num}')
        
        ax_norm.set_xlabel('Wavenumber (cm^-1)', fontsize=14)
        ax_norm.set_ylabel('Normalized Intensity', fontsize=14)
        ax_norm.set_title(f'Overlaid Normalized Spectra - {len(normalized_spectra)} Scans', fontsize=16, fontweight='bold')
        ax_norm.legend(fontsize=11, loc='best', ncol=2)
        ax_norm.grid(True, alpha=0.3)
        ax_norm.tick_params(axis='both', labelsize=12)
        
        plt.tight_layout()
        save_path_norm = Path(output_dir) / f'overlaid_normalized_spectra_scans_{"_".join(map(str, scan_list))}.png'
        plt.savefig(str(save_path_norm), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Overlaid normalized spectra plot saved to: {save_path_norm}")
    
    # Create difference plots (difference from first scan)
    if len(raw_spectra) >= 2:
        first_scan_num = scan_list[0]
        if first_scan_num in raw_spectra:
            reference_raw = raw_spectra[first_scan_num]
            reference_normalized = normalized_spectra.get(first_scan_num)
            
            # Raw difference plot
            fig_diff_raw, ax_diff_raw = plt.subplots(figsize=(14, 8))
            num_diff_scans = len(raw_spectra) - 1
            diff_colors = plt.cm.tab10(np.linspace(0, 1, num_diff_scans)) if num_diff_scans > 0 else []
            
            diff_idx = 0
            for scan_num, spectrum in raw_spectra.items():
                if scan_num == first_scan_num:
                    continue  # Skip reference scan
                difference = spectrum - reference_raw
                ax_diff_raw.plot(wavenumbers, difference, '-', linewidth=1.5, alpha=0.7, 
                               color=diff_colors[diff_idx] if diff_idx < len(diff_colors) else 'blue', 
                               label=f'Scan {scan_num} - Scan {first_scan_num}')
                diff_idx += 1
            
            ax_diff_raw.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax_diff_raw.set_xlabel('Wavenumber (cm^-1)', fontsize=14)
            ax_diff_raw.set_ylabel('Intensity Difference (Raw)', fontsize=14)
            ax_diff_raw.set_title(f'Raw Spectra Differences from Scan {first_scan_num}\n({len(raw_spectra)-1} scans)', 
                                 fontsize=16, fontweight='bold')
            ax_diff_raw.legend(fontsize=11, loc='best', ncol=2)
            ax_diff_raw.grid(True, alpha=0.3)
            ax_diff_raw.tick_params(axis='both', labelsize=12)
            
            plt.tight_layout()
            save_path_diff_raw = Path(output_dir) / f'overlaid_raw_difference_from_scan_{first_scan_num}_scans_{"_".join(map(str, scan_list))}.png'
            plt.savefig(str(save_path_diff_raw), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Raw difference plot saved to: {save_path_diff_raw}")
            
            # Normalized difference plot
            if reference_normalized is not None and normalized_spectra:
                fig_diff_norm, ax_diff_norm = plt.subplots(figsize=(14, 8))
                
                diff_idx = 0
                for scan_num, spectrum in normalized_spectra.items():
                    if scan_num == first_scan_num:
                        continue  # Skip reference scan
                    difference = spectrum - reference_normalized
                    ax_diff_norm.plot(wavenumbers, difference, '-', linewidth=1.5, alpha=0.7, 
                                     color=diff_colors[diff_idx] if diff_idx < len(diff_colors) else 'blue', 
                                     label=f'Scan {scan_num} - Scan {first_scan_num}')
                    diff_idx += 1
                
                ax_diff_norm.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                ax_diff_norm.set_xlabel('Wavenumber (cm^-1)', fontsize=14)
                ax_diff_norm.set_ylabel('Intensity Difference (Normalized)', fontsize=14)
                ax_diff_norm.set_title(f'Normalized Spectra Differences from Scan {first_scan_num}\n({len(normalized_spectra)-1} scans)', 
                                      fontsize=16, fontweight='bold')
                ax_diff_norm.legend(fontsize=11, loc='best', ncol=2)
                ax_diff_norm.grid(True, alpha=0.3)
                ax_diff_norm.tick_params(axis='both', labelsize=12)
                
                plt.tight_layout()
                save_path_diff_norm = Path(output_dir) / f'overlaid_normalized_difference_from_scan_{first_scan_num}_scans_{"_".join(map(str, scan_list))}.png'
                plt.savefig(str(save_path_diff_norm), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Normalized difference plot saved to: {save_path_diff_norm}")


def main():
    parser = build_parser()
    args = parser.parse_args()
    
    # Load configuration
    config = load_profile_config(args.config_file, args.profile)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_raman_dataset(config)
    raman_df = dataset.spectra
    
    print(f"Loaded {len(raman_df)} scans")
    print(f"Wavenumber range: {raman_df.columns[2]} to {raman_df.columns[-1]}")
    print(f"Raw data file: {dataset.source_path}")
    
    # Set output directory relative to raw data file location
    if args.output_dir is None:
        # Get the directory containing the raw data file (should be "Raw data" folder)
        raw_data_dir = Path(dataset.source_path).parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if args.batch:
            output_dir = raw_data_dir / f"test_outputs_batch_{timestamp}"
        else:
            output_dir = raw_data_dir / f"test_outputs_{timestamp}"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    if args.batch:
        # Batch processing mode
        print("\n=== Running batch processing ===")
        
        scan_numbers = None
        if args.scans:
            try:
                scan_numbers = [
                    int(val.strip())
                    for val in args.scans.split(",")
                    if val.strip()
                ]
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
        
        if args.max_scans:
            scan_list = scan_list[:args.max_scans]
        
        if not scan_list:
            print("[Batch] No scans left to process after filtering.")
            return
        
        print(f"[Batch] Processing {len(scan_list)} scans")
        print(f"[Batch] Scan list: {scan_list[:10]}{'...' if len(scan_list) > 10 else ''}")
        
        # Run batch processing with optimization on first scan if requested
        summaries = run_batch_with_optimization(
            scan_list=scan_list,
            optimize=args.optimize,
            custom_order=args.order,
            custom_tot_iter=args.iter,
            config=config,
            profile_name=args.profile,
            raman_df=raman_df,
            output_dir=output_dir,
        )
        
        if not summaries:
            print("[Batch] No summaries generated.")
            return
        
        print(f"\n[Batch] Completed processing {len(summaries)} scans")
        
        # Aggregate summaries into DataFrame
        print("\n=== Aggregating results ===")
        # Get x-axis type from config (default: datetime)
        x_axis_type = config.get('timeseries_x_axis', 'datetime')
        if x_axis_type not in ['datetime', 'scan_number']:
            print(f"Warning: Invalid x_axis_type '{x_axis_type}', using 'datetime'")
            x_axis_type = 'datetime'
        
        results_df = aggregate_summaries_to_dataframe(summaries, raman_df, x_axis_type=x_axis_type)
        
        if results_df.empty:
            print("[Batch] Warning: No data to aggregate.")
            return
        
        # Save aggregated results to CSV
        csv_path = output_dir / "batch_results_summary.csv"
        results_df.to_csv(csv_path)
        print(f"[Batch] Results saved to: {csv_path}")
        print(f"\n[Batch] Results summary:")
        print(results_df.head())
        print(f"\n[Batch] Total scans processed: {len(results_df)}")
        
        # Create time series plots
        light_cycle = config.get('light_cycle', 'Constant')
        create_time_series_plots(results_df, output_dir, config, light_cycle)
        
        # Create overlaid spectra plots if multiple scans were selected
        if scan_list and len(scan_list) >= 2:
            # Get wavenumber arrays for overlay plots
            wavenumbers_full = np.array([float(col) for col in raman_df.columns if col not in ['Scan Number', 'Seconds']])
            wavenumber_filter = wavenumbers_full >= 250
            wavenumbers = wavenumbers_full[wavenumber_filter]
            create_overlaid_spectra_plots(scan_list, raman_df, config, output_dir, 
                                         wavenumbers_full, wavenumber_filter, wavenumbers)
        
        print(f"\n[Batch] All outputs saved to: {output_dir}")
    else:
        # Single scan testing mode
        print("\n=== Running single scan test ===")
        test_new_method(
            scan_number=args.scan,
            optimize_params=args.optimize,
            custom_order=args.order,
            custom_tot_iter=args.iter,
            raman_df=raman_df,
            config=config,
            output_dir=output_dir,
            profile_name=args.profile,
        )


if __name__ == "__main__":
    main()

