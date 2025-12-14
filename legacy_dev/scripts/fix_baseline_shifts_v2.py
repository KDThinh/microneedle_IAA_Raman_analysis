"""
Standalone script to fix baseline shifts (step discontinuities) in normalized fluorescence data
using MAD-based jump detection with median-based offset correction.

This script uses a refined algorithm that:
- Detects jumps using MAD on the derivative signal
- Filters consecutive jumps to avoid detecting ramps as multiple jumps
- Uses median (instead of mean) for robust offset calculation
- Applies cumulative corrections to stitch signal segments

Usage:
    python fix_baseline_shifts_v2.py <path_to_batch_results_summary.csv> [options]

Example:
    python fix_baseline_shifts_v2.py "path/to/batch_results_summary.csv"
    python fix_baseline_shifts_v2.py "path/to/batch_results_summary.csv" --threshold 10.0 --window 5
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


def correct_baseline_shifts(signal, threshold_multiplier=5, window_size=5,
                           smooth_first=True, smooth_window=11, smooth_poly_order=2,
                           correct_smoothed=False, shared_threshold=None,
                           detect_cumulative_jumps=True, cumulative_window=5):
    """
    Detect and correct baseline shifts using MAD-based detection with median offset calculation.
    
    This algorithm:
    1. Calculates derivative with prepend to maintain array length
    2. Uses MAD to detect outliers (individual jumps)
    3. Optionally detects cumulative changes over a window (gradual jumps over multiple points)
    4. Filters consecutive jumps to avoid detecting ramps
    5. Uses median (not mean) for robust offset calculation
    6. Applies cumulative corrections
    
    Parameters
    ----------
    signal : np.array
        Array of signal values
    threshold_multiplier : float
        How many times larger than the noise floor a jump must be to be corrected (default: 5)
    window_size : int
        Number of points to average before/after a jump to calculate precise offset (default: 5)
    smooth_first : bool
        If True, smooth the signal before jump detection (default: True)
    smooth_window : int
        Window size for smoothing if smooth_first=True (default: 11)
    smooth_poly_order : int
        Polynomial order for smoothing if smooth_first=True (default: 2)
    correct_smoothed : bool
        If True, apply correction to smoothed signal (offset calculated from smoothed).
        If False, apply correction to original signal (offset calculated from original) (default: False)
    shared_threshold : float, optional
        If provided, use this absolute threshold instead of calculating MAD-based threshold.
        This allows using a shared MAD from a reference signal (default: None)
    detect_cumulative_jumps : bool
        If True, also detect gradual jumps over multiple consecutive points (default: True)
    cumulative_window : int
        Window size for detecting cumulative changes (default: 5)
        Detects jumps where total change over N consecutive points exceeds threshold
    
    Returns
    -------
    corrected : np.array
        Corrected signal with jumps removed
    jump_indices : np.array
        Indices where jumps were detected
    jump_info : list
        List of dicts with jump information
    smoothed_signal : np.array
        Smoothed version of input signal (if smooth_first=True) or original signal
    """
    # Smooth signal first if requested
    if smooth_first:
        smoothed_signal = smooth_signal(signal, window_size=smooth_window, poly_order=smooth_poly_order)
    else:
        smoothed_signal = signal.copy()
    
    # Calculate the first difference (derivative) with prepend to maintain length
    diffs = np.diff(smoothed_signal, prepend=smoothed_signal[0])
    
    # Define threshold: use shared threshold if provided, otherwise calculate from MAD
    if shared_threshold is not None:
        threshold = shared_threshold
    else:
        # Estimate the "noise floor" using Median Absolute Deviation (MAD)
        # This is more robust than Standard Deviation for data with outliers/steps
        median_diff = np.median(diffs)
        mad = np.median(np.abs(diffs - median_diff))
        
        # Handle edge case where MAD is zero
        if mad == 0:
            mad = np.std(diffs) if np.std(diffs) > 0 else 1.0
        
        # Define a threshold for what constitutes a "Shift" vs just "Noise"
        threshold = mad * threshold_multiplier
    
    # Find indices where the jump exceeds the threshold (individual jumps)
    jump_indices_individual = np.where(np.abs(diffs) > threshold)[0]
    
    # Debug output: show detection statistics
    if shared_threshold is None:
        print(f"  Individual jump detection: MAD={mad:.6f}, threshold={threshold:.6f} (multiplier={threshold_multiplier}), detected {len(jump_indices_individual)} jump(s) before filtering")
    else:
        print(f"  Individual jump detection: shared_threshold={threshold:.6f}, detected {len(jump_indices_individual)} jump(s) before filtering")
    
    # Optionally detect cumulative changes over a window (gradual jumps)
    jump_indices_cumulative = np.array([], dtype=int)
    if detect_cumulative_jumps and len(smoothed_signal) > cumulative_window:
        cumulative_changes = []
        cumulative_indices = []
        
        for i in range(len(smoothed_signal) - cumulative_window):
            # Calculate total change over the window
            total_change = smoothed_signal[i + cumulative_window] - smoothed_signal[i]
            cumulative_changes.append(total_change)
            cumulative_indices.append(i + cumulative_window)  # Mark end of window
        
        # Calculate MAD for cumulative changes
        if len(cumulative_changes) > 0:
            median_cumulative = np.median(cumulative_changes)
            mad_cumulative = np.median(np.abs(np.array(cumulative_changes) - median_cumulative))
            
            if mad_cumulative == 0:
                mad_cumulative = np.std(cumulative_changes) if np.std(cumulative_changes) > 0 else 1.0
            
            cumulative_threshold = mad_cumulative * threshold_multiplier
            
            # Find where cumulative change exceeds threshold
            cumulative_array = np.array(cumulative_changes)
            cumulative_mask = np.abs(cumulative_array) > cumulative_threshold
            cumulative_detected = np.where(cumulative_mask)[0]
            
            print(f"  Cumulative jump detection: MAD_cumulative={mad_cumulative:.6f}, threshold={cumulative_threshold:.6f}, detected {len(cumulative_detected)} potential cumulative jump(s)")
            
            # Only consider cumulative jumps that don't overlap with individual jumps
            # (to avoid double-detection)
            for idx in cumulative_detected:
                jump_idx = cumulative_indices[idx]
                # Check if this index is far enough from individual jumps
                if len(jump_indices_individual) == 0 or np.min(np.abs(jump_indices_individual - jump_idx)) > window_size:
                    jump_indices_cumulative = np.append(jump_indices_cumulative, jump_idx)
            
            print(f"  Cumulative jumps after overlap filtering: {len(jump_indices_cumulative)} jump(s)")
    
    # Combine both types of jump detections
    jump_indices_combined = np.unique(np.concatenate([jump_indices_individual, jump_indices_cumulative]))
    jump_indices_combined = np.sort(jump_indices_combined)
    
    # Track which jumps are individual vs cumulative
    jump_type_map = {}
    for idx in jump_indices_individual:
        jump_type_map[idx] = 'individual'
    for idx in jump_indices_cumulative:
        jump_type_map[idx] = 'cumulative'
    
    print(f"  Combined jumps (individual + cumulative): {len(jump_indices_combined)} jump(s)")
    
    # Filter indices: ensure we don't pick up consecutive points (ramp) as multiple jumps
    # We only take the peak of the jump
    clean_indices = []
    clean_jump_types = []
    if len(jump_indices_combined) > 0:
        clean_indices.append(jump_indices_combined[0])
        clean_jump_types.append(jump_type_map.get(jump_indices_combined[0], 'unknown'))
        filtered_count = 0
        for i in range(1, len(jump_indices_combined)):
            if jump_indices_combined[i] - jump_indices_combined[i-1] > window_size:
                clean_indices.append(jump_indices_combined[i])
                clean_jump_types.append(jump_type_map.get(jump_indices_combined[i], 'unknown'))
            else:
                filtered_count += 1
                print(f"  Filtered out jump at index {jump_indices_combined[i]} (too close to previous jump at {jump_indices_combined[i-1]}, distance={jump_indices_combined[i] - jump_indices_combined[i-1]} <= window_size={window_size})")
        
        if filtered_count > 0:
            print(f"  Total jumps before filtering: {len(jump_indices_combined)}, after filtering: {len(clean_indices)}, filtered out: {filtered_count}")
    
    jump_indices = np.array(clean_indices, dtype=int)
    
    # Apply Correction
    # Choose which signal to correct based on correct_smoothed parameter
    if correct_smoothed:
        # Correct smoothed signal: use smoothed signal for both offset calculation and correction
        target_signal = smoothed_signal.copy().astype(float)
        y_corrected = smoothed_signal.copy().astype(float)
    else:
        # Correct original signal: use original signal for offset calculation and correction
        target_signal = signal.copy().astype(float)
        y_corrected = signal.copy().astype(float)
    
    cumulative_offset = 0.0
    jump_info = []
    
    # Iterate through the detected jumps and "stitch" the segments
    # Calculate all offsets from original signal first, then apply corrections cumulatively
    for idx in clean_indices:
        # Define a small window before and after the jump index
        start_idx = max(0, idx - window_size)
        end_idx = min(len(target_signal), idx + window_size)
        
        # Calculate the median level before and after the jump using ORIGINAL target signal
        # This ensures consistent offset calculation regardless of previous corrections
        # We exclude the jump point from both windows to avoid bias
        
        # Before jump: exclude jump point (idx is not included)
        val_before = np.median(target_signal[start_idx:idx])
        
        # After jump: exclude jump point (use idx+1 to start after the jump)
        # This prevents the jump point itself from biasing the median
        after_start = idx + 1
        if after_start < end_idx:
            val_after = np.median(target_signal[after_start:end_idx])
        else:
            # Edge case: if jump is very close to the end, use a smaller window
            # or use the value right after the jump if available
            if idx + 1 < len(target_signal):
                val_after = np.median(target_signal[idx+1:min(len(target_signal), idx+1+window_size)])
            else:
                # Last point: use the value before as fallback
                val_after = val_before
        
        # The jump magnitude (calculated from original target signal)
        step_change = val_after - val_before
        
        # We accumulate this offset
        cumulative_offset += step_change
        
        # Apply correction to the target signal
        # (We subtract the jump to bring the new baseline down/up to the old one)
        # This is applied cumulatively: each correction affects all subsequent points
        y_corrected[idx:] -= step_change
        
        jump_info.append({
            'index': idx,
            'val_before': val_before,
            'val_after': val_after,
            'step_change': step_change,
            'cumulative_offset': cumulative_offset,
            'jump_type': jump_type_map.get(idx, 'unknown')
        })
    
    return y_corrected, np.array(clean_indices), jump_info, smoothed_signal


def correct_baseline_shifts_with_jump_indices(signal, jump_indices, window_size=5,
                                             smooth_first=True, smooth_window=11, smooth_poly_order=2,
                                             correct_smoothed=False):
    """
    Apply baseline correction using pre-defined jump indices (from reference signal).
    
    This function is used when jump points are detected from a reference signal (e.g., fluorescence)
    and we want to apply corrections to related signals (e.g., G-band, Raman peak) at the same points.
    
    Parameters
    ----------
    signal : np.array
        Array of signal values to correct
    jump_indices : np.array
        Pre-defined jump indices (from reference signal)
    window_size : int
        Number of points to average before/after a jump to calculate precise offset (default: 5)
    smooth_first : bool
        If True, smooth the signal before offset calculation (default: True)
    smooth_window : int
        Window size for smoothing if smooth_first=True (default: 11)
    smooth_poly_order : int
        Polynomial order for smoothing if smooth_first=True (default: 2)
    correct_smoothed : bool
        If True, apply correction to smoothed signal (offset calculated from smoothed).
        If False, apply correction to original signal (offset calculated from original) (default: False)
    
    Returns
    -------
    corrected : np.array
        Corrected signal with jumps removed
    jump_info : list
        List of dicts with jump information
    smoothed_signal : np.array
        Smoothed version of input signal (if smooth_first=True) or original signal
    """
    # Smooth signal first if requested
    if smooth_first:
        smoothed_signal = smooth_signal(signal, window_size=smooth_window, poly_order=smooth_poly_order)
    else:
        smoothed_signal = signal.copy()
    
    # Choose which signal to correct based on correct_smoothed parameter
    if correct_smoothed:
        target_signal = smoothed_signal.copy().astype(float)
        y_corrected = smoothed_signal.copy().astype(float)
    else:
        target_signal = signal.copy().astype(float)
        y_corrected = signal.copy().astype(float)
    
    cumulative_offset = 0.0
    jump_info = []
    
    # Filter jump indices to ensure they're within valid range
    valid_jump_indices = [idx for idx in jump_indices if 0 <= idx < len(target_signal)]
    
    # Iterate through the pre-defined jump indices and "stitch" the segments
    for idx in valid_jump_indices:
        # Define a small window before and after the jump index
        start_idx = max(0, idx - window_size)
        end_idx = min(len(target_signal), idx + window_size)
        
        # Calculate the median level before and after the jump using ORIGINAL target signal
        # We exclude the jump point from both windows to avoid bias
        
        # Before jump: exclude jump point (idx is not included)
        val_before = np.median(target_signal[start_idx:idx])
        
        # After jump: exclude jump point (use idx+1 to start after the jump)
        after_start = idx + 1
        if after_start < end_idx:
            val_after = np.median(target_signal[after_start:end_idx])
        else:
            # Edge case: if jump is very close to the end
            if idx + 1 < len(target_signal):
                val_after = np.median(target_signal[idx+1:min(len(target_signal), idx+1+window_size)])
            else:
                val_after = val_before
        
        # The jump magnitude (calculated from original target signal)
        step_change = val_after - val_before
        
        # We accumulate this offset
        cumulative_offset += step_change
        
        # Apply correction to the target signal
        y_corrected[idx:] -= step_change
        
        jump_info.append({
            'index': idx,
            'val_before': val_before,
            'val_after': val_after,
            'step_change': step_change,
            'cumulative_offset': cumulative_offset,
            'jump_type': 'unknown'  # Using pre-defined jump indices, type not available
        })
    
    return y_corrected, jump_info, smoothed_signal


def fix_baseline_shifts_v2(csv_path, output_dir=None, threshold_multiplier=5.0, 
                           window_size=5, plot_comparison=True, columns_to_fix=None,
                           smooth_first=True, smooth_window=11, smooth_poly_order=2,
                           correct_smoothed=False):
    """
    Fix baseline shifts in normalized fluorescence data from batch_results_summary.csv.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to batch_results_summary.csv file
    output_dir : str or Path, optional
        Output directory for corrected CSV and plots. 
        If None, defaults to Script/swnt_iaa_analysis_v2/scripts/test_outputs
    threshold_multiplier : float, optional
        MAD threshold multiplier for jump detection (default: 5.0)
    window_size : int, optional
        Number of points to average before/after jump for offset calculation (default: 5)
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
    correct_smoothed : bool, optional
        If True, apply correction to smoothed signal (offset calculated from smoothed).
        If False, apply correction to original signal (offset calculated from original) (default: False)
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
    print(f"Using MAD threshold multiplier: {threshold_multiplier}")
    print(f"Window size for offset calculation: {window_size}")
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
        corrected, jump_indices, jump_info, smoothed = correct_baseline_shifts(
            signal, threshold_multiplier=threshold_multiplier, window_size=window_size,
            smooth_first=smooth_first, smooth_window=smooth_window, 
            smooth_poly_order=smooth_poly_order, correct_smoothed=correct_smoothed
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
                print(f"    Scan {scan_numbers[info['index']]:.0f}: "
                     f"step_change={info['step_change']:.4f}, "
                     f"cumulative_offset={info['cumulative_offset']:.4f}")
    
    # Save corrected CSV
    corrected_csv_path = output_dir / 'baseline_corrected_results_v2.csv'
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
        
        fig.suptitle('Baseline Shift Correction v2 - Original vs Corrected', 
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
                jump_values_corr = [corrected[info['index']] for info in jump_info]
                ax_corr.scatter(jump_scans, jump_values_corr, color='red', s=100, 
                               zorder=5, marker='x', label='Jump points')
            
            ax_corr.set_xlabel('Scan Number', fontsize=12)
            ax_corr.set_ylabel('Corrected Fluorescence', fontsize=12)
            ax_corr.set_title(f'{col} - Corrected', fontsize=13, fontweight='bold')
            ax_corr.grid(True, alpha=0.3)
            ax_corr.tick_params(axis='both', labelsize=11)
            ax_corr.legend()
        
        plt.tight_layout()
        plot_path = output_dir / 'baseline_correction_comparison_v2.png'
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to: {plot_path}")
    
    print("\nDone!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Fix baseline shifts v2 - using median-based offset correction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python fix_baseline_shifts_v2.py "path/to/batch_results_summary.csv"
  
  # Custom threshold and window size
  python fix_baseline_shifts_v2.py "path/to/batch_results_summary.csv" --threshold 10.0 --window 5
  
  # Specify custom output directory
  python fix_baseline_shifts_v2.py "path/to/batch_results_summary.csv" --output "custom/output/dir"
  
  # Fix specific columns only
  python fix_baseline_shifts_v2.py "path/to/batch_results_summary.csv" --columns "Normalized_Fluorescence_Method1" "Normalized_Fluorescence_RawAUC"
  
  # No plots (just CSV output)
  python fix_baseline_shifts_v2.py "path/to/batch_results_summary.csv" --no-plots
  
  # Disable smoothing
  python fix_baseline_shifts_v2.py "path/to/batch_results_summary.csv" --no-smooth
        """
    )
    
    parser.add_argument('csv_path', help='Path to batch_results_summary.csv file')
    parser.add_argument('--output', '-o', dest='output_dir', default=None,
                       help='Output directory for corrected CSV and plots (default: Script/swnt_iaa_analysis_v2/scripts/test_outputs)')
    parser.add_argument('--threshold', type=float, default=5.0,
                       help='MAD threshold multiplier for jump detection (default: 5.0, higher=less sensitive)')
    parser.add_argument('--window', type=int, default=5,
                       help='Window size for offset calculation (default: 5)')
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
    parser.add_argument('--correct-smoothed', action='store_true',
                       help='Apply correction to smoothed signal instead of original signal')
    
    args = parser.parse_args()
    
    fix_baseline_shifts_v2(
        args.csv_path,
        output_dir=args.output_dir,
        threshold_multiplier=args.threshold,
        window_size=args.window,
        plot_comparison=not args.no_plots,
        columns_to_fix=args.columns,
        smooth_first=not args.no_smooth,
        smooth_window=args.smooth_window,
        smooth_poly_order=args.smooth_poly,
        correct_smoothed=args.correct_smoothed
    )


if __name__ == "__main__":
    main()

