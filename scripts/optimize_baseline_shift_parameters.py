"""
Standalone script to optimize baseline shift correction parameters (threshold_multiplier and window_size).
Tests different parameter combinations and evaluates jump detection quality.

Usage:
    python optimize_baseline_shift_parameters.py "path/to/batch_summary_data_v3.csv" [options]

Example:
    python optimize_baseline_shift_parameters.py "test_outputs_v3_batch_20251123_231801/batch_summary_data_v3.csv"
    python optimize_baseline_shift_parameters.py "path/to/data.csv" --threshold-range 3 10 --window-range 5 20
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from datetime import datetime
import itertools

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Workspace root (parent of Script directory)
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent

from scripts.fix_baseline_shifts_v2 import correct_baseline_shifts
from pipeline.utils import add_day_night_shading
from matplotlib.dates import DateFormatter, DayLocator


def evaluate_baseline_correction(original_signal, corrected_signal, jump_info, sampling_interval_hours):
    """
    Evaluate baseline correction quality using multiple metrics.
    
    Parameters:
    -----------
    original_signal : array
        Original signal (after smoothing if applicable)
    corrected_signal : array
        Baseline-corrected signal
    jump_info : list
        List of dicts with jump information
    sampling_interval_hours : float
        Sampling interval in hours
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    # 1. Number of jumps detected
    metrics['n_jumps'] = len(jump_info)
    
    # 2. Residual sum of squares (RSS) - lower is better
    residuals = corrected_signal - np.mean(corrected_signal)
    metrics['rss'] = np.sum(residuals**2)
    
    # 3. Mean absolute deviation (MAD) - lower is better
    metrics['mad'] = np.median(np.abs(residuals - np.median(residuals)))
    
    # 4. Signal variance preservation (variance ratio) - closer to 1 is better
    original_var = np.var(original_signal)
    corrected_var = np.var(corrected_signal)
    metrics['variance_ratio'] = corrected_var / original_var if original_var > 0 else 0
    
    # 5. Signal smoothness (second derivative variance) - lower is better for smooth signals
    corrected_diff2 = np.diff(corrected_signal, n=2)
    metrics['smoothness'] = np.var(corrected_diff2)
    
    # 6. Jump magnitude statistics
    if jump_info:
        jump_magnitudes = [abs(info['step_change']) for info in jump_info]
        metrics['mean_jump_magnitude'] = np.mean(jump_magnitudes)
        metrics['max_jump_magnitude'] = np.max(jump_magnitudes)
        metrics['jump_magnitude_std'] = np.std(jump_magnitudes)
    else:
        metrics['mean_jump_magnitude'] = 0
        metrics['max_jump_magnitude'] = 0
        metrics['jump_magnitude_std'] = 0
    
    # 7. Signal stability (coefficient of variation) - lower is better
    metrics['cv'] = np.std(corrected_signal) / np.abs(np.mean(corrected_signal)) if np.mean(corrected_signal) != 0 else np.inf
    
    # 8. Periodic component preservation (FFT power at diurnal frequency ~0.04 cycles/hour)
    n = len(corrected_signal)
    signal_detrended = corrected_signal - np.mean(corrected_signal)
    fft_result = np.fft.fft(signal_detrended)
    frequencies = np.fft.fftfreq(n, d=sampling_interval_hours)
    
    # Focus on positive frequencies
    positive_mask = frequencies > 0
    positive_freqs = frequencies[positive_mask]
    magnitude = np.abs(fft_result[positive_mask])
    
    # Diurnal frequency range (0.03 to 0.05 cycles/hour, ~20-33 hour period)
    diurnal_mask = (positive_freqs >= 0.03) & (positive_freqs <= 0.05)
    if np.any(diurnal_mask):
        metrics['diurnal_power'] = np.sum(magnitude[diurnal_mask]**2)
        metrics['diurnal_peak'] = np.max(magnitude[diurnal_mask])
    else:
        metrics['diurnal_power'] = 0
        metrics['diurnal_peak'] = 0
    
    # Total power in periodic range (0.01 to 0.2 cycles/hour)
    periodic_mask = (positive_freqs >= 0.01) & (positive_freqs <= 0.2)
    if np.any(periodic_mask):
        metrics['periodic_power'] = np.sum(magnitude[periodic_mask]**2)
    else:
        metrics['periodic_power'] = 0
    
    # 9. Combined score (weighted combination of metrics)
    # Lower RSS, MAD, CV, and smoothness are better
    # Higher variance ratio and periodic power are better
    # Moderate number of jumps is better (not too few, not too many)
    score = (
        -metrics['rss'] / (metrics['rss'] + 1e-6) +  # Normalized RSS (inverted)
        -metrics['mad'] / (metrics['mad'] + 1e-6) +  # Normalized MAD (inverted)
        metrics['variance_ratio'] * 0.2 +  # Preserve signal variance
        -metrics['cv'] / (metrics['cv'] + 1e-6) * 0.2 +  # Lower CV is better
        -metrics['smoothness'] / (metrics['smoothness'] + 1e-6) * 0.1 +  # Lower smoothness is better
        metrics['diurnal_power'] / (metrics['diurnal_power'] + 1e-6) * 0.2 +  # Preserve diurnal patterns
        metrics['periodic_power'] / (metrics['periodic_power'] + 1e-6) * 0.1  # Preserve periodic patterns
    )
    
    # Penalize too many or too few jumps
    if metrics['n_jumps'] == 0:
        score -= 0.1  # Penalize no jumps detected
    elif metrics['n_jumps'] > 20:
        score -= 0.1  # Penalize too many jumps
    
    metrics['combined_score'] = score
    
    return metrics


def optimize_baseline_shift_parameters(csv_path, ratio_column='Normalized_Fluorescence_Intensity',
                                      threshold_range=(3.0, 10.0), window_range=(5, 20),
                                      n_threshold=8, n_window=8, smooth_first=True, smooth_window=11,
                                      smooth_poly_order=2, correct_smoothed=True,
                                      detect_cumulative_jumps=True, cumulative_window=5,
                                      output_dir=None, threshold_multiplier=None, window_size=None):
    """
    Optimize baseline shift correction parameters by testing different combinations.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to batch_summary_data_v3.csv
    ratio_column : str
        Column name to optimize for
    threshold_range : tuple
        (min, max) range for threshold_multiplier - ignored if threshold_multiplier is provided
    window_range : tuple
        (min, max) range for window_size - ignored if window_size is provided
    n_threshold : int
        Number of threshold values to test - ignored if threshold_multiplier is provided
    n_window : int
        Number of window values to test - ignored if window_size is provided
    smooth_first : bool
        Whether to smooth before jump detection
    smooth_window : int
        Window size for smoothing
    smooth_poly_order : int
        Polynomial order for smoothing
    correct_smoothed : bool
        Whether to correct smoothed signal
    detect_cumulative_jumps : bool
        Whether to detect cumulative jumps
    cumulative_window : int
        Window size for cumulative jump detection
    output_dir : Path, optional
        Output directory for results
    threshold_multiplier : float, optional
        Predefined threshold_multiplier. If provided with window_size, skips optimization.
    window_size : int, optional
        Predefined window_size. If provided with threshold_multiplier, skips optimization.
    
    Returns:
    --------
    pandas.DataFrame
        Results DataFrame with all tested combinations and metrics
    dict
        Best parameters found
    """
    csv_path_orig = Path(csv_path)
    
    # Resolve path: if not absolute, try relative to workspace root first, then current directory
    if csv_path_orig.is_absolute():
        csv_path = csv_path_orig.resolve()
    else:
        # Try workspace root first (most common case)
        workspace_path = WORKSPACE_ROOT / csv_path_orig
        if workspace_path.exists():
            csv_path = workspace_path.resolve()
        # Try relative to current directory
        elif (Path.cwd() / csv_path_orig).exists():
            csv_path = (Path.cwd() / csv_path_orig).resolve()
        # Try as-is (might be relative to script location)
        elif csv_path_orig.exists():
            csv_path = csv_path_orig.resolve()
        else:
            # Last attempt: try relative to workspace root
            csv_path = workspace_path
    
    if not csv_path.exists():
        tried_paths = [
            f"  - Workspace root: {WORKSPACE_ROOT / csv_path_orig}",
            f"  - Current directory: {Path.cwd() / csv_path_orig}",
            f"  - As provided: {csv_path_orig}"
        ]
        raise FileNotFoundError(
            f"CSV file not found: {csv_path_orig}\n"
            f"Workspace root: {WORKSPACE_ROOT}\n"
            f"Current directory: {Path.cwd()}\n"
            f"Tried paths:\n" + "\n".join(tried_paths)
        )
    
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if ratio_column not in df.columns:
        raise ValueError(f"Column '{ratio_column}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Ensure datetime index
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')
        df = df.set_index('Datetime')
    elif df.index.name == 'Datetime' or pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, errors='coerce')
    else:
        raise ValueError("No datetime index found. CSV must have 'Datetime' column or datetime index.")
    
    df = df.sort_index()
    
    # Extract signal (use original fluorescence intensity before baseline correction)
    # Try to find the original column (remove _BaselineCorrected suffix if present)
    original_column = ratio_column.replace('_BaselineCorrected', '')
    if original_column in df.columns:
        signal = df[original_column].values
        print(f"Using original signal column: {original_column}")
    elif ratio_column in df.columns:
        # Fallback to specified column
        signal = df[ratio_column].values
        print(f"Using specified signal column: {ratio_column}")
    else:
        # Try common fluorescence intensity column names
        possible_columns = [
            'Normalized_Fluorescence_Intensity',
            'Raw_Fluorescence_Intensity',
            'Fluorescence_Intensity',
            'Normalized_Fluorescence_Intensity_BaselineCorrected'
        ]
        found = False
        for col in possible_columns:
            if col in df.columns:
                signal = df[col].values
                print(f"Using fallback signal column: {col}")
                found = True
                break
        if not found:
            raise ValueError(f"Signal column '{ratio_column}' not found. Available columns: {list(df.columns)}")
    
    mask = ~np.isnan(signal)
    
    if np.sum(mask) < 10:
        raise ValueError(f"Insufficient data: only {np.sum(mask)} valid points")
    
    signal_valid = signal[mask]
    datetimes_valid = df.index[mask]
    
    # Calculate sampling interval (in hours)
    time_diffs = np.diff(datetimes_valid)
    # Convert timedelta64 to hours: convert to seconds first, then to hours
    sampling_interval_hours = np.mean([pd.Timedelta(td).total_seconds() / 3600.0 for td in time_diffs])
    print(f"Sampling interval: {sampling_interval_hours:.3f} hours")
    print(f"Data length: {len(signal_valid)} points")
    print(f"Time span: {datetimes_valid[0]} to {datetimes_valid[-1]}")
    
    # Check if predefined parameters are provided
    if threshold_multiplier is not None and window_size is not None:
        # Use predefined parameters (skip optimization)
        print(f"\n=== Using predefined parameters ===")
        print(f"threshold_multiplier: {threshold_multiplier:.2f}")
        print(f"window_size: {window_size}")
        
        try:
            # Apply baseline correction
            corrected, jump_indices, jump_info, smoothed = correct_baseline_shifts(
                signal_valid,
                threshold_multiplier=threshold_multiplier,
                window_size=window_size,
                smooth_first=smooth_first,
                smooth_window=smooth_window,
                smooth_poly_order=smooth_poly_order,
                correct_smoothed=correct_smoothed,
                detect_cumulative_jumps=detect_cumulative_jumps,
                cumulative_window=cumulative_window
            )
            
            # Evaluate
            metrics = evaluate_baseline_correction(
                smoothed if correct_smoothed else signal_valid,
                corrected,
                jump_info,
                sampling_interval_hours
            )
            
            results_df = pd.DataFrame([{
                'threshold_multiplier': threshold_multiplier,
                'window_size': window_size,
                'n_jumps': metrics['n_jumps'],
                'rss': metrics['rss'],
                'mad': metrics['mad'],
                'variance_ratio': metrics['variance_ratio'],
                'smoothness': metrics['smoothness'],
                'cv': metrics['cv'],
                'diurnal_power': metrics['diurnal_power'],
                'diurnal_peak': metrics['diurnal_peak'],
                'periodic_power': metrics['periodic_power'],
                'mean_jump_magnitude': metrics['mean_jump_magnitude'],
                'max_jump_magnitude': metrics['max_jump_magnitude'],
                'combined_score': metrics['combined_score']
            }])
            
            best_params = {
                'threshold_multiplier': threshold_multiplier,
                'window_size': window_size,
                'score': metrics['combined_score']
            }
            
        except Exception as e:
            raise ValueError(f"Failed to apply baseline correction with predefined parameters: {e}")
    
    else:
        # Run optimization grid search
        # Generate parameter grids
        threshold_values = np.linspace(threshold_range[0], threshold_range[1], n_threshold)
        window_values = np.arange(window_range[0], window_range[1] + 1, 
                                 max(1, (window_range[1] - window_range[0]) // (n_window - 1))).astype(int)
        
        print(f"\nTesting {len(threshold_values)} x {len(window_values)} = {len(threshold_values) * len(window_values)} parameter combinations...")
        print(f"threshold_multiplier range: {threshold_range[0]:.1f} to {threshold_range[1]:.1f}")
        print(f"window_size range: {window_range[0]} to {window_range[1]}")
        
        # Test all combinations
        results = []
        best_score = -np.inf
        best_params = None
        
        total_combinations = len(threshold_values) * len(window_values)
        for idx, (threshold, window) in enumerate(itertools.product(threshold_values, window_values)):
            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx + 1}/{total_combinations} combinations tested...")
            
            try:
                # Apply baseline correction
                corrected, jump_indices, jump_info, smoothed = correct_baseline_shifts(
                    signal_valid,
                    threshold_multiplier=threshold,
                    window_size=window,
                    smooth_first=smooth_first,
                    smooth_window=smooth_window,
                    smooth_poly_order=smooth_poly_order,
                    correct_smoothed=correct_smoothed,
                    detect_cumulative_jumps=detect_cumulative_jumps,
                    cumulative_window=cumulative_window
                )
                
                # Evaluate
                metrics = evaluate_baseline_correction(
                    smoothed if correct_smoothed else signal_valid,
                    corrected,
                    jump_info,
                    sampling_interval_hours
                )
                
                results.append({
                    'threshold_multiplier': threshold,
                    'window_size': window,
                    'n_jumps': metrics['n_jumps'],
                    'rss': metrics['rss'],
                    'mad': metrics['mad'],
                    'variance_ratio': metrics['variance_ratio'],
                    'smoothness': metrics['smoothness'],
                    'cv': metrics['cv'],
                    'diurnal_power': metrics['diurnal_power'],
                    'diurnal_peak': metrics['diurnal_peak'],
                    'periodic_power': metrics['periodic_power'],
                    'mean_jump_magnitude': metrics['mean_jump_magnitude'],
                    'max_jump_magnitude': metrics['max_jump_magnitude'],
                    'combined_score': metrics['combined_score']
                })
                
                # Track best
                if metrics['combined_score'] > best_score:
                    best_score = metrics['combined_score']
                    best_params = {
                        'threshold_multiplier': threshold,
                        'window_size': window,
                        'score': best_score
                    }
            
            except Exception as e:
                print(f"  Warning: Failed for threshold={threshold:.2f}, window={window}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        if results_df.empty:
            raise ValueError("No valid results generated!")
        
        # Sort by combined score
        results_df = results_df.sort_values('combined_score', ascending=False)
    
    if threshold_multiplier is not None and window_size is not None:
        print(f"\n=== Evaluation Results (Predefined Parameters) ===")
        print(f"Parameters used:")
        print(f"  threshold_multiplier: {best_params['threshold_multiplier']:.2f}")
        print(f"  window_size: {best_params['window_size']}")
        print(f"  Combined score: {best_params['score']:.4f}")
        print(f"\nMetrics:")
        print(results_df[['threshold_multiplier', 'window_size', 'combined_score', 'n_jumps', 'diurnal_power', 'variance_ratio', 'cv', 'rss', 'mad']].to_string(index=False))
    else:
        print(f"\n=== Optimization Results ===")
        print(f"Best parameters:")
        print(f"  threshold_multiplier: {best_params['threshold_multiplier']:.2f}")
        print(f"  window_size: {best_params['window_size']}")
        print(f"  Combined score: {best_params['score']:.4f}")
        
        print(f"\nTop 5 parameter combinations:")
        print(results_df[['threshold_multiplier', 'window_size', 'combined_score', 'n_jumps', 'diurnal_power', 'variance_ratio', 'cv']].head())
    
    # Create output directory
    if output_dir is None:
        output_dir = csv_path.parent / f"baseline_shift_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")
    
    # Save results CSV
    results_csv = output_dir / "baseline_shift_optimization_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to: {results_csv}")
    
    # Create visualization plots
    print("\nCreating visualization plots...")
    
    # Only create heatmaps if running optimization (not with predefined params)
    if threshold_multiplier is None or window_size is None:
        # Plot 1: Parameter space heatmap (combined score)
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_score = results_df.pivot(index='window_size', columns='threshold_multiplier', values='combined_score')
        im = ax.imshow(pivot_score.values, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xticks(range(len(pivot_score.columns)))
        ax.set_xticklabels([f'{x:.1f}' for x in pivot_score.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(pivot_score.index)))
        ax.set_yticklabels([f'{int(y)}' for y in pivot_score.index])
        ax.set_xlabel('Threshold Multiplier', fontsize=12)
        ax.set_ylabel('Window Size', fontsize=12)
        ax.set_title('Baseline Shift Parameter Optimization - Combined Score Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Combined Score')
        plt.tight_layout()
        plt.savefig(output_dir / 'baseline_shift_optimization_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Number of jumps heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_jumps = results_df.pivot(index='window_size', columns='threshold_multiplier', values='n_jumps')
        im = ax.imshow(pivot_jumps.values, aspect='auto', cmap='plasma', origin='lower')
        ax.set_xticks(range(len(pivot_jumps.columns)))
        ax.set_xticklabels([f'{x:.1f}' for x in pivot_jumps.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(pivot_jumps.index)))
        ax.set_yticklabels([f'{int(y)}' for y in pivot_jumps.index])
        ax.set_xlabel('Threshold Multiplier', fontsize=12)
        ax.set_ylabel('Window Size', fontsize=12)
        ax.set_title('Baseline Shift Parameter Optimization - Number of Jumps Detected', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Number of Jumps')
        plt.tight_layout()
        plt.savefig(output_dir / 'baseline_shift_optimization_jumps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2b: NEW - Threshold Multiplier vs Number of Jumps (line plot for different window sizes)
        fig, ax = plt.subplots(figsize=(12, 8))
        unique_windows = sorted(results_df['window_size'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_windows)))
        
        for window, color in zip(unique_windows, colors):
            window_data = results_df[results_df['window_size'] == window].sort_values('threshold_multiplier')
            ax.plot(window_data['threshold_multiplier'], window_data['n_jumps'], 
                   'o-', label=f'Window Size = {window}', color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel('Threshold Multiplier', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Jumps Detected', fontsize=14, fontweight='bold')
        ax.set_title('Effect of Threshold Multiplier on Jump Detection\n(Lower Multiplier = More Jumps)', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=12)
        
        # Add annotation explaining the relationship
        ax.text(0.02, 0.98, 
               'Lower threshold multiplier → Lower threshold → More jumps detected\n'
               'Higher threshold multiplier → Higher threshold → Fewer jumps detected',
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'threshold_multiplier_vs_jumps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2c: NEW - Threshold Multiplier Sensitivity Analysis (showing rate of change)
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for window, color in zip(unique_windows, colors):
            window_data = results_df[results_df['window_size'] == window].sort_values('threshold_multiplier')
            # Calculate rate of change (jumps per unit threshold)
            threshold_diff = np.diff(window_data['threshold_multiplier'])
            jumps_diff = np.diff(window_data['n_jumps'])
            rate_of_change = -jumps_diff / threshold_diff  # Negative because fewer jumps as threshold increases
            midpoints = window_data['threshold_multiplier'].values[:-1] + threshold_diff / 2
            
            ax.plot(midpoints, rate_of_change, 's-', label=f'Window Size = {window}', 
                   color=color, linewidth=2, markersize=5, alpha=0.7)
        
        ax.set_xlabel('Threshold Multiplier', fontsize=14, fontweight='bold')
        ax.set_ylabel('Rate of Change (Jumps per Unit Threshold)', fontsize=14, fontweight='bold')
        ax.set_title('Sensitivity Analysis: How Quickly Jump Count Changes with Threshold\n'
                    '(Higher values = More sensitive to threshold changes)', 
                    fontsize=16, fontweight='bold')
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=12)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'threshold_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Diurnal power heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_diurnal = results_df.pivot(index='window_size', columns='threshold_multiplier', values='diurnal_power')
        im = ax.imshow(pivot_diurnal.values, aspect='auto', cmap='plasma', origin='lower')
        ax.set_xticks(range(len(pivot_diurnal.columns)))
        ax.set_xticklabels([f'{x:.1f}' for x in pivot_diurnal.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(pivot_diurnal.index)))
        ax.set_yticklabels([f'{int(y)}' for y in pivot_diurnal.index])
        ax.set_xlabel('Threshold Multiplier', fontsize=12)
        ax.set_ylabel('Window Size', fontsize=12)
        ax.set_title('Baseline Shift Parameter Optimization - Diurnal Power Preservation', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Diurnal Power')
        plt.tight_layout()
        plt.savefig(output_dir / 'baseline_shift_optimization_diurnal_power.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 4: Parameter comparison
    threshold_best = best_params['threshold_multiplier']
    window_best = best_params['window_size']
    
    corrected_best, jump_indices_best, jump_info_best, smoothed_best = correct_baseline_shifts(
        signal_valid,
        threshold_multiplier=threshold_best,
        window_size=window_best,
        smooth_first=smooth_first,
        smooth_window=smooth_window,
        smooth_poly_order=smooth_poly_order,
        correct_smoothed=correct_smoothed,
        detect_cumulative_jumps=detect_cumulative_jumps,
        cumulative_window=cumulative_window
    )
    
    # Also plot with default parameters for comparison (unless using predefined params)
    if threshold_multiplier is not None and window_size is not None:
        # Only show predefined parameters
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Original signal
        axes[0].plot(datetimes_valid, signal_valid, 'm-', label='Original Signal', alpha=0.7, linewidth=1.5)
        axes[0].set_ylabel('Ratio', fontsize=12)
        axes[0].set_title('Original Signal', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Corrected signal
        axes[1].plot(datetimes_valid, signal_valid, 'm-', label='Original', alpha=0.5, linewidth=1)
        axes[1].plot(datetimes_valid, corrected_best, 'g-', label=f'Corrected (threshold={threshold_best:.2f}, window={window_best})', alpha=0.8, linewidth=1.5)
        
        # Mark jump points
        if jump_info_best:
            jump_indices_list = [info['index'] for info in jump_info_best]
            jump_x = datetimes_valid[jump_indices_list]
            jump_y = corrected_best[jump_indices_list]
            axes[1].scatter(jump_x, jump_y, color='red', s=150, zorder=5, marker='x', 
                          linewidths=2, label=f'Jump points (n={len(jump_info_best)})')
        
        axes[1].set_ylabel('Ratio', fontsize=12)
        axes[1].set_xlabel('Date time (MM-DD HH)', fontsize=12)
        axes[1].set_title(f'Baseline Correction (Score: {best_params["score"]:.4f})', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
        axes[1].xaxis.set_major_locator(DayLocator())
    else:
        # Show best vs default comparison
        threshold_default = 7.5
        window_default = 15
        corrected_default, jump_indices_default, jump_info_default, smoothed_default = correct_baseline_shifts(
            signal_valid,
            threshold_multiplier=threshold_default,
            window_size=window_default,
            smooth_first=smooth_first,
            smooth_window=smooth_window,
            smooth_poly_order=smooth_poly_order,
            correct_smoothed=correct_smoothed,
            detect_cumulative_jumps=detect_cumulative_jumps,
            cumulative_window=cumulative_window
        )
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Original signal
        axes[0].plot(datetimes_valid, signal_valid, 'm-', label='Original Signal', alpha=0.7, linewidth=1.5)
        axes[0].set_ylabel('Ratio', fontsize=12)
        axes[0].set_title('Original Signal', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Best parameters
        axes[1].plot(datetimes_valid, signal_valid, 'm-', label='Original', alpha=0.5, linewidth=1)
        axes[1].plot(datetimes_valid, corrected_best, 'g-', label=f'Corrected (Best: threshold={threshold_best:.2f}, window={window_best})', alpha=0.8, linewidth=1.5)
        
        # Mark jump points
        if jump_info_best:
            jump_indices_list = [info['index'] for info in jump_info_best]
            jump_x = datetimes_valid[jump_indices_list]
            jump_y = corrected_best[jump_indices_list]
            axes[1].scatter(jump_x, jump_y, color='red', s=150, zorder=5, marker='x', 
                          linewidths=2, label=f'Jump points (n={len(jump_info_best)})')
        
        axes[1].set_ylabel('Ratio', fontsize=12)
        axes[1].set_title(f'Best Parameters (Score: {best_params["score"]:.4f})', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Default parameters
        axes[2].plot(datetimes_valid, signal_valid, 'm-', label='Original', alpha=0.5, linewidth=1)
        axes[2].plot(datetimes_valid, corrected_default, 'b-', label=f'Corrected (Default: threshold={threshold_default:.2f}, window={window_default})', alpha=0.8, linewidth=1.5)
        
        # Mark jump points
        if jump_info_default:
            jump_indices_list = [info['index'] for info in jump_info_default]
            jump_x = datetimes_valid[jump_indices_list]
            jump_y = corrected_default[jump_indices_list]
            axes[2].scatter(jump_x, jump_y, color='red', s=150, zorder=5, marker='x', 
                          linewidths=2, label=f'Jump points (n={len(jump_info_default)})')
        
        axes[2].set_ylabel('Ratio', fontsize=12)
        axes[2].set_xlabel('Date time (MM-DD HH)', fontsize=12)
        axes[2].set_title('Default Parameters (for comparison)', fontsize=13, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
        axes[2].xaxis.set_major_locator(DayLocator())
    
    plt.tight_layout()
    plt.savefig(output_dir / 'baseline_shift_optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: FFT comparison
    if threshold_multiplier is not None and window_size is not None:
        # Single FFT plot for predefined parameters
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n = len(corrected_best)
        freqs_best = np.fft.fftfreq(n, d=sampling_interval_hours)
        fft_best = np.fft.fft(corrected_best - np.mean(corrected_best))
        magnitude_best = np.abs(fft_best)
        
        # Plot positive frequencies only
        positive_mask = freqs_best > 0
        freqs_plot = freqs_best[positive_mask]
        
        ax.plot(freqs_plot, magnitude_best[positive_mask], 'g-', 
               label=f'Parameters (threshold={threshold_best:.2f}, window={window_best})', linewidth=2)
        ax.axvspan(0.03, 0.05, alpha=0.2, color='yellow', label='Diurnal Range (0.03-0.05 cycles/hour)')
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.set_xlabel('Frequency (cycles/hour)', fontsize=12)
        ax.set_title(f'FFT - Predefined Parameters', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'baseline_shift_optimization_fft_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # FFT comparison for best vs default
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Calculate FFT for best and default
        n = len(corrected_best)
        freqs_best = np.fft.fftfreq(n, d=sampling_interval_hours)
        fft_best = np.fft.fft(corrected_best - np.mean(corrected_best))
        magnitude_best = np.abs(fft_best)
        
        freqs_default = np.fft.fftfreq(n, d=sampling_interval_hours)
        fft_default = np.fft.fft(corrected_default - np.mean(corrected_default))
        magnitude_default = np.abs(fft_default)
        
        # Plot positive frequencies only
        positive_mask = freqs_best > 0
        freqs_plot = freqs_best[positive_mask]
        
        axes[0].plot(freqs_plot, magnitude_best[positive_mask], 'g-', label='Best Parameters', linewidth=2)
        axes[0].axvspan(0.03, 0.05, alpha=0.2, color='yellow', label='Diurnal Range (0.03-0.05 cycles/hour)')
        axes[0].set_ylabel('Magnitude', fontsize=12)
        axes[0].set_title(f'FFT - Best Parameters (threshold={threshold_best:.2f}, window={window_best})', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 0.2)
        
        axes[1].plot(freqs_plot, magnitude_default[positive_mask], 'b-', label='Default Parameters', linewidth=2)
        axes[1].axvspan(0.03, 0.05, alpha=0.2, color='yellow', label='Diurnal Range (0.03-0.05 cycles/hour)')
        axes[1].set_ylabel('Magnitude', fontsize=12)
        axes[1].set_xlabel('Frequency (cycles/hour)', fontsize=12)
        axes[1].set_title(f'FFT - Default Parameters (threshold={threshold_default:.2f}, window={window_default})', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 0.2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'baseline_shift_optimization_fft_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nAll plots saved to: {output_dir}")
    print(f"\n=== Recommended Parameters ===")
    print(f"threshold_multiplier: {best_params['threshold_multiplier']:.2f}")
    print(f"window_size: {best_params['window_size']}")
    print(f"\nAdd to config/pipeline.yml:")
    print(f"  baseline_correction_threshold: {best_params['threshold_multiplier']:.2f}")
    print(f"  baseline_correction_window: {best_params['window_size']}")
    
    return results_df, best_params


def main():
    parser = argparse.ArgumentParser(
        description='Optimize baseline shift correction parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default ranges (optimization)
  python optimize_baseline_shift_parameters.py "test_outputs_v3_batch_20251123_231801/batch_summary_data_v3.csv"
  
  # Custom parameter ranges (optimization)
  python optimize_baseline_shift_parameters.py "path/to/data.csv" --threshold-range 3 10 --window-range 5 20
  
  # Use predefined parameters (skip optimization)
  python optimize_baseline_shift_parameters.py "path/to/data.csv" --threshold-multiplier 5 --window-size 15
  
  # Different ratio column
  python optimize_baseline_shift_parameters.py "path/to/data.csv" --ratio-column "Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected"
  
  # Disable cumulative jump detection
  python optimize_baseline_shift_parameters.py "path/to/data.csv" --no-cumulative-jumps
        """
    )
    
    parser.add_argument('csv_path', nargs='?', default=None,
                       help='Path to batch_summary_data_v3.csv. If not provided, uses default sample file.')
    parser.add_argument('--ratio-column', default='Normalized_Fluorescence_Intensity',
                       help='Column name to optimize for (default: Normalized_Fluorescence_Intensity). Can also use ratio columns like Fluorescence_to_Gband_Ratio_BaselineCorrected.')
    parser.add_argument('--threshold-range', nargs=2, type=float, default=[3.0, 10.0],
                       metavar=('MIN', 'MAX'),
                       help='Range for threshold_multiplier (default: 3.0 10.0)')
    parser.add_argument('--window-range', nargs=2, type=int, default=[5, 20],
                       metavar=('MIN', 'MAX'),
                       help='Range for window_size (default: 5 20)')
    parser.add_argument('--n-threshold', type=int, default=8,
                       help='Number of threshold values to test (default: 8)')
    parser.add_argument('--n-window', type=int, default=8,
                       help='Number of window values to test (default: 8)')
    parser.add_argument('--smooth-first', action='store_true', default=True,
                       help='Smooth signal before jump detection (default: True)')
    parser.add_argument('--no-smooth-first', dest='smooth_first', action='store_false',
                       help='Disable smoothing before jump detection')
    parser.add_argument('--smooth-window', type=int, default=11,
                       help='Window size for smoothing (default: 11)')
    parser.add_argument('--smooth-poly-order', type=int, default=2,
                       help='Polynomial order for smoothing (default: 2)')
    parser.add_argument('--correct-smoothed', action='store_true', default=True,
                       help='Apply correction to smoothed signal (default: True)')
    parser.add_argument('--no-correct-smoothed', dest='correct_smoothed', action='store_false',
                       help='Apply correction to original signal')
    parser.add_argument('--detect-cumulative-jumps', action='store_true', default=True,
                       help='Detect cumulative jumps (default: True)')
    parser.add_argument('--no-cumulative-jumps', dest='detect_cumulative_jumps', action='store_false',
                       help='Disable cumulative jump detection')
    parser.add_argument('--cumulative-window', type=int, default=5,
                       help='Window size for cumulative jump detection (default: 5)')
    parser.add_argument('--threshold-multiplier', type=float, default=None,
                       help='Predefined threshold_multiplier. If provided with --window-size, skips optimization.')
    parser.add_argument('--window-size', type=int, default=None,
                       help='Predefined window_size. If provided with --threshold-multiplier, skips optimization.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: creates timestamped directory)')
    
    args = parser.parse_args()
    
    # Use default sample file if not provided
    if args.csv_path is None:
        default_csv = WORKSPACE_ROOT / "IAA Nanosensor Experiment" / "In planta" / "Nb" / "Treatment_Shade" / "Light_8to24" / "Temp_Hum_Constant" / "Run 1" / "Raw data" / "test_outputs_v3_batch_20251124_235109" / "batch_summary_v3_final.csv"
        if default_csv.exists():
            args.csv_path = str(default_csv)
            print(f"Using default sample file: {args.csv_path}")
        else:
            parser.error("No CSV path provided and default file not found. Please provide --csv-path.")
    
    # Validate predefined parameters
    if (args.threshold_multiplier is not None) != (args.window_size is not None):
        parser.error("--threshold-multiplier and --window-size must be provided together to use predefined parameters")
    
    try:
        results_df, best_params = optimize_baseline_shift_parameters(
            csv_path=args.csv_path,
            ratio_column=args.ratio_column,
            threshold_range=tuple(args.threshold_range),
            window_range=tuple(args.window_range),
            n_threshold=args.n_threshold,
            n_window=args.n_window,
            smooth_first=args.smooth_first,
            smooth_window=args.smooth_window,
            smooth_poly_order=args.smooth_poly_order,
            correct_smoothed=args.correct_smoothed,
            detect_cumulative_jumps=args.detect_cumulative_jumps,
            cumulative_window=args.cumulative_window,
            output_dir=args.output_dir,
            threshold_multiplier=args.threshold_multiplier,
            window_size=args.window_size
        )
        
        print("\nOptimization complete!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

