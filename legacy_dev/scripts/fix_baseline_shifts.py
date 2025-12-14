"""
Standalone script to fix baseline shifts (step discontinuities) in normalized fluorescence data
using MAD-based jump detection and correction.

This script detects sudden jumps in the signal (caused by light cycle changes) and corrects
them by offsetting signal segments to remove discontinuities.

Usage:
    python fix_baseline_shifts.py <path_to_batch_results_summary.csv> [options]

Example:
    python fix_baseline_shifts.py "path/to/batch_results_summary.csv"
    python fix_baseline_shifts.py "path/to/batch_results_summary.csv" --threshold 5.0 --output "custom/output/dir"
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.signal import savgol_filter


def smooth_signal(signal, window_size=11, poly_order=2):
    """
    Smooth signal using Savitzky-Golay filter.
    
    Parameters
    ----------
    signal : np.array
        Array of signal values
    window_size : int
        Window size for Savitzky-Golay filter (must be odd, default: 11)
    poly_order : int
        Polynomial order for Savitzky-Golay filter (default: 2)
    
    Returns
    -------
    smoothed : np.array
        Smoothed signal
    """
    # Ensure window_size is odd and valid
    if window_size % 2 == 0:
        window_size += 1
    if window_size > len(signal):
        window_size = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
    if window_size < 3:
        return signal  # Too short to smooth
    
    # Ensure poly_order is less than window_size
    if poly_order >= window_size:
        poly_order = window_size - 1
    
    try:
        smoothed = savgol_filter(signal, window_size, poly_order)
        return smoothed
    except Exception as e:
        print(f"Warning: Smoothing failed ({e}), using original signal")
        return signal


def fix_step_discontinuities(signal, thresh_multiplier=5, smooth_first=True, 
                             smooth_window=11, smooth_poly_order=2):
    """
    Detect and correct step discontinuities (offset jumps) in a 1D signal.
    
    Uses Median Absolute Deviation (MAD) to detect jumps in the signal derivative,
    then corrects by offsetting segments after each detected jump.
    
    Parameters
    ----------
    signal : np.array
        Array of signal values
    thresh_multiplier : float
        Factor for MAD-based threshold (default 5 for moderate sensitivity).
        Higher values = less sensitive (fewer jumps detected)
        Lower values = more sensitive (more jumps detected)
    smooth_first : bool
        If True, smooth the signal before jump detection (default: True)
    smooth_window : int
        Window size for smoothing if smooth_first=True (default: 11)
    smooth_poly_order : int
        Polynomial order for smoothing if smooth_first=True (default: 2)
    
    Returns
    -------
    corrected : np.array
        Corrected signal with jumps removed
    jump_indices : np.array
        Indices where jumps were detected (0-indexed, jump occurs between index and index+1)
    jump_info : list
        List of dicts with jump information (index, jump_size, etc.)
    smoothed_signal : np.array
        Smoothed version of input signal (if smooth_first=True) or original signal
    """
    # Smooth signal first if requested
    if smooth_first:
        smoothed_signal = smooth_signal(signal, window_size=smooth_window, poly_order=smooth_poly_order)
    else:
        smoothed_signal = signal.copy()
    
    # Calculate first difference (derivative) on smoothed signal
    diff = np.diff(smoothed_signal)
    abs_diff = np.abs(diff)
    
    # Calculate MAD (Median Absolute Deviation)
    median_abs_diff = np.median(abs_diff)
    mad = np.median(np.abs(abs_diff - median_abs_diff))
    
    # Handle edge case where MAD is zero
    if mad == 0:
        mad = np.std(abs_diff) if np.std(abs_diff) > 0 else 1.0
    
    threshold = thresh_multiplier * mad
    jump_indices = np.where(abs_diff > threshold)[0]
    
    # Correct the signal
    corrected = signal.copy()
    jump_info = []
    
    for k in sorted(jump_indices):
        j = k + 1  # Index where correction starts (after the jump)
        if j >= len(corrected):
            continue
        
        jump_size = corrected[j] - corrected[j - 1]
        increment = -jump_size
        
        # Apply correction to all points after the jump
        corrected[j:] += increment
        
        jump_info.append({
            'index': k,
            'correction_start': j,
            'jump_size': jump_size,
            'correction_increment': increment,
            'before_value': corrected[j - 1] - increment,  # Original value before correction
            'after_value': corrected[j]  # Value after correction
        })
    
    return corrected, jump_indices, jump_info, smoothed_signal


def fix_baseline_shifts(csv_path, output_dir=None, thresh_multiplier=5.0, 
                       plot_comparison=True, columns_to_fix=None,
                       smooth_first=True, smooth_window=11, smooth_poly_order=2):
    """
    Fix baseline shifts in normalized fluorescence data from batch_results_summary.csv.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to batch_results_summary.csv file
    output_dir : str or Path, optional
        Output directory for corrected CSV and plots. 
        If None, defaults to Script/swnt_iaa_analysis_v2/scripts/test_outputs
    thresh_multiplier : float, optional
        MAD threshold multiplier for jump detection (default: 5.0)
    plot_comparison : bool, optional
        If True, create comparison plots showing original vs corrected signals (default: True)
    columns_to_fix : list, optional
        List of column names to fix. If None, fixes all normalized fluorescence columns.
    smooth_first : bool, optional
        If True, smooth signal before jump detection (default: True)
    smooth_window : int, optional
        Window size for Savitzky-Golay smoothing (must be odd, default: 11)
    smooth_poly_order : int, optional
        Polynomial order for Savitzky-Golay smoothing (default: 2)
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}")
        return
    
    # Load CSV
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("Error: CSV file is empty")
        return
    
    # Check for required columns
    if 'Scan Number' not in df.columns:
        print("Error: 'Scan Number' column not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Set output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "test_outputs"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Sort by scan number
    df = df.sort_values('Scan Number').reset_index(drop=True)
    scan_numbers = df['Scan Number'].values
    
    # Identify columns to fix
    if columns_to_fix is None:
        # Default: fix all normalized fluorescence columns
        normalized_columns = {
            'Normalized_Fluorescence_Method1': 'Normalized Method 1',
            'Normalized_Fluorescence_Method2': 'Normalized Method 2',
            'Normalized_Fluorescence_RawAUC': 'Normalized Raw AUC'
        }
        columns_to_fix = [col for col in normalized_columns.keys() if col in df.columns]
    
    if not columns_to_fix:
        print("Warning: No normalized fluorescence columns found to fix")
        print(f"Available columns: {list(df.columns)}")
        return
    
    print(f"\nFixing baseline shifts in {len(columns_to_fix)} column(s): {columns_to_fix}")
    print(f"Using MAD threshold multiplier: {thresh_multiplier}")
    if smooth_first:
        print(f"Smoothing parameters: window_size={smooth_window}, poly_order={smooth_poly_order}")
    else:
        print("Smoothing disabled")
    
    # Create corrected dataframe
    corrected_df = df.copy()
    all_jump_info = {}
    smoothed_signals = {}
    
    # Fix each column
    for col in columns_to_fix:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found, skipping...")
            continue
        
        signal = df[col].values
        
        # Remove NaN values for processing
        mask = ~np.isnan(signal)
        if np.sum(mask) < 3:
            print(f"Warning: Column '{col}' has insufficient data, skipping...")
            continue
        
        # Fix step discontinuities (with optional smoothing)
        corrected, jump_indices, jump_info, smoothed = fix_step_discontinuities(
            signal, thresh_multiplier=thresh_multiplier,
            smooth_first=smooth_first, smooth_window=smooth_window, 
            smooth_poly_order=smooth_poly_order
        )
        
        # Store smoothed signal for plotting
        smoothed_signals[col] = smoothed
        
        # Store corrected values
        corrected_col_name = f'{col}_Corrected'
        corrected_df[corrected_col_name] = corrected
        
        # Store jump information
        all_jump_info[col] = {
            'jump_indices': jump_indices,
            'jump_info': jump_info,
            'n_jumps': len(jump_indices)
        }
        
        print(f"\n{col}:")
        print(f"  Detected {len(jump_indices)} jump(s)")
        if jump_info:
            for info in jump_info:
                print(f"    Scan {scan_numbers[info['index']]:.0f}-{scan_numbers[info['correction_start']]:.0f}: "
                     f"jump_size={info['jump_size']:.4f}, correction={info['correction_increment']:.4f}")
    
    # Save corrected CSV
    corrected_csv_path = output_dir / 'baseline_corrected_results.csv'
    corrected_df.to_csv(corrected_csv_path, index=False)
    print(f"\nCorrected data saved to: {corrected_csv_path}")
    
    # Create comparison plots
    if plot_comparison:
        n_cols = len(columns_to_fix)
        if n_cols == 0:
            return
        
        fig, axes = plt.subplots(n_cols, 2, figsize=(16, 5 * n_cols))
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Baseline Shift Correction - Original vs Corrected', 
                     fontsize=16, fontweight='bold')
        
        for idx, col in enumerate(columns_to_fix):
            if col not in df.columns:
                continue
            
            signal = df[col].values
            corrected_col_name = f'{col}_Corrected'
            corrected = corrected_df[corrected_col_name].values
            smoothed = smoothed_signals.get(col, signal)
            
            # Plot original signal
            ax_orig = axes[idx, 0]
            ax_orig.plot(scan_numbers, signal, 'o-', linewidth=1.5, markersize=3, 
                         alpha=0.5, color='gray', label='Original', linestyle='--')
            if smooth_first:
                ax_orig.plot(scan_numbers, smoothed, '-', linewidth=2, 
                            alpha=0.8, color='blue', label='Smoothed')
            else:
                ax_orig.plot(scan_numbers, signal, 'o-', linewidth=2, markersize=4, 
                            alpha=0.7, color='blue', label='Original')
            
            # Mark jump points
            jump_info = all_jump_info[col]['jump_info']
            if jump_info:
                jump_scans = [scan_numbers[info['index']] for info in jump_info]
                if smooth_first:
                    jump_values = [smoothed[info['index']] for info in jump_info]
                else:
                    jump_values = [signal[info['index']] for info in jump_info]
                ax_orig.scatter(jump_scans, jump_values, color='red', s=100, 
                               zorder=5, marker='x', label=f'Jumps (n={len(jump_info)})')
            
            ax_orig.set_xlabel('Scan Number', fontsize=12)
            ax_orig.set_ylabel('Fluorescence', fontsize=12)
            ax_orig.set_title(f'{col} - Original', fontsize=13, fontweight='bold')
            ax_orig.grid(True, alpha=0.3)
            ax_orig.tick_params(axis='both', labelsize=11)
            ax_orig.legend()
            
            # Plot corrected signal
            ax_corr = axes[idx, 1]
            ax_corr.plot(scan_numbers, signal, 'o-', linewidth=1.5, markersize=3, 
                        alpha=0.4, color='gray', label='Original', linestyle='--')
            if smooth_first:
                ax_corr.plot(scan_numbers, smoothed, '-', linewidth=1.5, 
                            alpha=0.6, color='blue', label='Smoothed', linestyle=':')
            ax_corr.plot(scan_numbers, corrected, 'o-', linewidth=2, markersize=4, 
                        alpha=0.8, color='green', label='Corrected')
            
            # Mark jump points on corrected plot
            if jump_info:
                jump_scans = [scan_numbers[info['index']] for info in jump_info]
                jump_values_corr = [corrected[info['correction_start']] for info in jump_info]
                ax_corr.scatter(jump_scans, jump_values_corr, color='red', s=100, 
                               zorder=5, marker='x', label='Jump points')
            
            ax_corr.set_xlabel('Scan Number', fontsize=12)
            ax_corr.set_ylabel('Corrected Fluorescence', fontsize=12)
            ax_corr.set_title(f'{col} - Corrected', fontsize=13, fontweight='bold')
            ax_corr.grid(True, alpha=0.3)
            ax_corr.tick_params(axis='both', labelsize=11)
            ax_corr.legend()
        
        plt.tight_layout()
        plot_path = output_dir / 'baseline_correction_comparison.png'
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to: {plot_path}")
    
    print("\nDone!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fix baseline shifts (step discontinuities) in normalized fluorescence data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python fix_baseline_shifts.py "path/to/batch_results_summary.csv"
  
  # Custom threshold (higher = less sensitive, lower = more sensitive)
  python fix_baseline_shifts.py "path/to/batch_results_summary.csv" --threshold 3.0
  
  # Specify custom output directory
  python fix_baseline_shifts.py "path/to/batch_results_summary.csv" --output "custom/output/dir"
  
  # Fix specific columns only
  python fix_baseline_shifts.py "path/to/batch_results_summary.csv" --columns "Normalized_Fluorescence_Method1" "Normalized_Fluorescence_RawAUC"
  
  # No plots (just CSV output)
  python fix_baseline_shifts.py "path/to/batch_results_summary.csv" --no-plots
        """
    )
    
    parser.add_argument('csv_path', help='Path to batch_results_summary.csv file')
    parser.add_argument('--output', '-o', dest='output_dir', default=None,
                       help='Output directory for corrected CSV and plots (default: Script/swnt_iaa_analysis_v2/scripts/test_outputs)')
    parser.add_argument('--threshold', type=float, default=5.0,
                       help='MAD threshold multiplier for jump detection (default: 5.0, higher=less sensitive)')
    parser.add_argument('--columns', nargs='+', default=None,
                       help='Specific columns to fix (default: all normalized fluorescence columns)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip creating comparison plots')
    parser.add_argument('--no-smooth', action='store_true',
                       help='Disable smoothing before jump detection')
    parser.add_argument('--smooth-window', type=int, default=11,
                       help='Window size for Savitzky-Golay smoothing (must be odd, default: 11)')
    parser.add_argument('--smooth-poly', type=int, default=2,
                       help='Polynomial order for Savitzky-Golay smoothing (default: 2)')
    
    args = parser.parse_args()
    
    fix_baseline_shifts(
        args.csv_path,
        output_dir=args.output_dir,
        thresh_multiplier=args.threshold,
        plot_comparison=not args.no_plots,
        columns_to_fix=args.columns,
        smooth_first=not args.no_smooth,
        smooth_window=args.smooth_window,
        smooth_poly_order=args.smooth_poly
    )


if __name__ == "__main__":
    main()

