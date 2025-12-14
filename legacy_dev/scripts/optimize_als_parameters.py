"""
Standalone script to optimize ALS baseline correction parameters (lam_als and p_als).
Applies Gaussian smoothing BEFORE ALS correction, then tests different parameter combinations.

Usage:
    python optimize_als_parameters.py "path/to/batch_summary_data_v3.csv" [options]

Example:
    python optimize_als_parameters.py "test_outputs_v3_batch_20251123_231801/batch_summary_data_v3.csv"
    python optimize_als_parameters.py "test_outputs_v3_batch_20251123_231801/batch_summary_data_v3.csv" --lam-range 1e4 1e8 --p-range 0.0001 0.01
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft, fftfreq
from datetime import datetime
import itertools

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

# Workspace root (parent of Script directory)
WORKSPACE_ROOT = PROJECT_ROOT.parent.parent

from pipeline.utils import add_day_night_shading
from matplotlib.dates import DateFormatter, DayLocator


def baseline_als(y, lam=1e7, p=0.001, niter=20):
    """
    ALS baseline correction function.
    
    Parameters:
    -----------
    y : array
        Input signal
    lam : float
        Smoothness parameter (higher = smoother baseline)
    p : float
        Asymmetry parameter (lower = baseline stays below data)
    niter : int
        Number of iterations
    
    Returns:
    --------
    array
        Baseline estimate
    """
    L = len(y)
    D = diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), format='csc')
    w = np.ones(L)
    for i in range(niter):
        W = diags(w, 0, shape=(L, L), format='csc')
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def evaluate_als_correction(original_signal, corrected_signal, baseline, sampling_interval_hours):
    """
    Evaluate ALS correction quality using multiple metrics.
    
    Parameters:
    -----------
    original_signal : array
        Original signal (after Gaussian smoothing)
    corrected_signal : array
        ALS-corrected signal
    baseline : array
        Estimated baseline
    sampling_interval_hours : float
        Sampling interval in hours
    
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    metrics = {}
    
    # 1. Residual sum of squares (RSS) - lower is better
    residuals = corrected_signal - np.mean(corrected_signal)
    metrics['rss'] = np.sum(residuals**2)
    
    # 2. Mean absolute deviation (MAD) - lower is better
    metrics['mad'] = np.median(np.abs(residuals - np.median(residuals)))
    
    # 3. Baseline smoothness (second derivative variance) - lower is better
    baseline_diff2 = np.diff(baseline, n=2)
    metrics['baseline_smoothness'] = np.var(baseline_diff2)
    
    # 4. Signal preservation (variance ratio) - closer to 1 is better
    original_var = np.var(original_signal)
    corrected_var = np.var(corrected_signal)
    metrics['variance_ratio'] = corrected_var / original_var if original_var > 0 else 0
    
    # 5. Negative fraction (corrected signal should be mostly positive) - lower is better
    metrics['negative_fraction'] = np.sum(corrected_signal < 0) / len(corrected_signal)
    
    # 6. Periodic component preservation (FFT power at diurnal frequency ~0.04 cycles/hour)
    # Calculate FFT
    n = len(corrected_signal)
    fft_result = fft(corrected_signal - np.mean(corrected_signal))
    frequencies = fftfreq(n, d=sampling_interval_hours)
    
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
    
    # 7. Combined score (weighted combination of metrics)
    # Lower RSS and MAD are better, higher variance ratio and periodic power are better
    # Normalize metrics for scoring
    score = (
        -metrics['rss'] / (metrics['rss'] + 1e-6) +  # Normalized RSS (inverted)
        -metrics['mad'] / (metrics['mad'] + 1e-6) +  # Normalized MAD (inverted)
        metrics['variance_ratio'] * 0.3 +  # Preserve signal variance
        metrics['diurnal_power'] / (metrics['diurnal_power'] + 1e-6) * 0.3 +  # Preserve diurnal patterns
        -metrics['negative_fraction'] * 0.2  # Minimize negative values
    )
    metrics['combined_score'] = score
    
    return metrics


def optimize_als_parameters(csv_path, ratio_column='Fluorescence_to_Gband_Ratio_BaselineCorrected',
                           sigma_gaussian=25, lam_range=(1e4, 1e8), p_range=(0.0001, 0.01),
                           n_lam=10, n_p=8, niter_als=20, output_dir=None,
                           lam_als=None, p_als=None):
    """
    Optimize ALS parameters by testing different combinations, or apply predefined parameters.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to batch_summary_data_v3.csv
    ratio_column : str
        Column name to optimize for
    sigma_gaussian : float
        Gaussian smoothing sigma (applied BEFORE ALS)
    lam_range : tuple
        (min, max) range for lam_als (will use log scale) - ignored if lam_als is provided
    p_range : tuple
        (min, max) range for p_als (will use log scale) - ignored if p_als is provided
    n_lam : int
        Number of lam values to test - ignored if lam_als is provided
    n_p : int
        Number of p values to test - ignored if p_als is provided
    niter_als : int
        Number of ALS iterations
    output_dir : Path, optional
        Output directory for results
    lam_als : float, optional
        Predefined lam_als parameter. If provided, skips optimization and uses this value.
    p_als : float, optional
        Predefined p_als parameter. If provided, skips optimization and uses this value.
    
    Returns:
    --------
    pandas.DataFrame
        Results DataFrame with all tested combinations and metrics (or single result if predefined params)
    dict
        Best parameters found (or predefined parameters)
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
    
    # Extract signal
    signal = df[ratio_column].values
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
    
    # Apply Gaussian smoothing FIRST (as requested)
    print(f"\nApplying Gaussian smoothing (sigma={sigma_gaussian}) BEFORE ALS correction...")
    signal_smoothed = gaussian_filter1d(signal_valid, sigma=sigma_gaussian)
    
    # Check if predefined parameters are provided
    if lam_als is not None and p_als is not None:
        # Use predefined parameters (skip optimization)
        print(f"\n=== Using predefined parameters ===")
        print(f"lam_als: {lam_als:.2e}")
        print(f"p_als: {p_als:.6f}")
        
        try:
            # Apply ALS baseline correction
            baseline = baseline_als(signal_smoothed, lam=lam_als, p=p_als, niter=niter_als)
            corrected = signal_smoothed - baseline
            
            # Evaluate
            metrics = evaluate_als_correction(
                signal_smoothed, corrected, baseline, sampling_interval_hours
            )
            
            results_df = pd.DataFrame([{
                'lam_als': lam_als,
                'p_als': p_als,
                'rss': metrics['rss'],
                'mad': metrics['mad'],
                'baseline_smoothness': metrics['baseline_smoothness'],
                'variance_ratio': metrics['variance_ratio'],
                'negative_fraction': metrics['negative_fraction'],
                'diurnal_power': metrics['diurnal_power'],
                'diurnal_peak': metrics['diurnal_peak'],
                'periodic_power': metrics['periodic_power'],
                'combined_score': metrics['combined_score']
            }])
            
            best_params = {'lam_als': lam_als, 'p_als': p_als, 'score': metrics['combined_score']}
            
        except Exception as e:
            raise ValueError(f"Failed to apply ALS with predefined parameters: {e}")
    
    else:
        # Run optimization grid search
        # Generate parameter grids (log scale for lam, log scale for p)
        lam_values = np.logspace(np.log10(lam_range[0]), np.log10(lam_range[1]), n_lam)
        p_values = np.logspace(np.log10(p_range[0]), np.log10(p_range[1]), n_p)
        
        print(f"\nTesting {len(lam_values)} x {len(p_values)} = {len(lam_values) * len(p_values)} parameter combinations...")
        print(f"lam_als range: {lam_range[0]:.1e} to {lam_range[1]:.1e}")
        print(f"p_als range: {p_range[0]:.6f} to {p_range[1]:.6f}")
        
        # Test all combinations
        results = []
        best_score = -np.inf
        best_params = None
        
        total_combinations = len(lam_values) * len(p_values)
        for idx, (lam, p) in enumerate(itertools.product(lam_values, p_values)):
            if (idx + 1) % 10 == 0:
                print(f"  Progress: {idx + 1}/{total_combinations} combinations tested...")
            
            try:
                # Apply ALS baseline correction
                baseline = baseline_als(signal_smoothed, lam=lam, p=p, niter=niter_als)
                corrected = signal_smoothed - baseline
                
                # Evaluate
                metrics = evaluate_als_correction(
                    signal_smoothed, corrected, baseline, sampling_interval_hours
                )
                
                results.append({
                    'lam_als': lam,
                    'p_als': p,
                    'rss': metrics['rss'],
                    'mad': metrics['mad'],
                    'baseline_smoothness': metrics['baseline_smoothness'],
                    'variance_ratio': metrics['variance_ratio'],
                    'negative_fraction': metrics['negative_fraction'],
                    'diurnal_power': metrics['diurnal_power'],
                    'diurnal_peak': metrics['diurnal_peak'],
                    'periodic_power': metrics['periodic_power'],
                    'combined_score': metrics['combined_score']
                })
                
                # Track best
                if metrics['combined_score'] > best_score:
                    best_score = metrics['combined_score']
                    best_params = {'lam_als': lam, 'p_als': p, 'score': best_score}
            
            except Exception as e:
                print(f"  Warning: Failed for lam={lam:.2e}, p={p:.6f}: {e}")
                continue
        
        results_df = pd.DataFrame(results)
        
        if results_df.empty:
            raise ValueError("No valid results generated!")
        
        # Sort by combined score
        results_df = results_df.sort_values('combined_score', ascending=False)
    
    if lam_als is not None and p_als is not None:
        print(f"\n=== Evaluation Results (Predefined Parameters) ===")
        print(f"Parameters used:")
        print(f"  lam_als: {best_params['lam_als']:.2e}")
        print(f"  p_als: {best_params['p_als']:.6f}")
        print(f"  Combined score: {best_params['score']:.4f}")
        print(f"\nMetrics:")
        print(results_df[['lam_als', 'p_als', 'combined_score', 'diurnal_power', 'variance_ratio', 'negative_fraction', 'rss', 'mad']].to_string(index=False))
    else:
        print(f"\n=== Optimization Results ===")
        print(f"Best parameters:")
        print(f"  lam_als: {best_params['lam_als']:.2e}")
        print(f"  p_als: {best_params['p_als']:.6f}")
        print(f"  Combined score: {best_params['score']:.4f}")
        
        print(f"\nTop 5 parameter combinations:")
        print(results_df[['lam_als', 'p_als', 'combined_score', 'diurnal_power', 'variance_ratio', 'negative_fraction']].head())
    
    # Create output directory
    if output_dir is None:
        output_dir = csv_path.parent / f"als_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {output_dir}")
    
    # Save results CSV
    results_csv = output_dir / "als_optimization_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"Results saved to: {results_csv}")
    
    # Create visualization plots
    print("\nCreating visualization plots...")
    
    # Only create heatmaps if running optimization (not with predefined params)
    if lam_als is None or p_als is None:
        # Plot 1: Parameter space heatmap (combined score)
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_score = results_df.pivot(index='p_als', columns='lam_als', values='combined_score')
        im = ax.imshow(pivot_score.values, aspect='auto', cmap='viridis', origin='lower')
        ax.set_xticks(range(len(pivot_score.columns)))
        ax.set_xticklabels([f'{x:.1e}' for x in pivot_score.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(pivot_score.index)))
        ax.set_yticklabels([f'{y:.6f}' for y in pivot_score.index])
        ax.set_xlabel('lam_als (smoothness parameter)', fontsize=12)
        ax.set_ylabel('p_als (asymmetry parameter)', fontsize=12)
        ax.set_title('ALS Parameter Optimization - Combined Score Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Combined Score')
        plt.tight_layout()
        plt.savefig(output_dir / 'als_optimization_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Diurnal power heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_diurnal = results_df.pivot(index='p_als', columns='lam_als', values='diurnal_power')
        im = ax.imshow(pivot_diurnal.values, aspect='auto', cmap='plasma', origin='lower')
        ax.set_xticks(range(len(pivot_diurnal.columns)))
        ax.set_xticklabels([f'{x:.1e}' for x in pivot_diurnal.columns], rotation=45, ha='right')
        ax.set_yticks(range(len(pivot_diurnal.index)))
        ax.set_yticklabels([f'{y:.6f}' for y in pivot_diurnal.index])
        ax.set_xlabel('lam_als (smoothness parameter)', fontsize=12)
        ax.set_ylabel('p_als (asymmetry parameter)', fontsize=12)
        ax.set_title('ALS Parameter Optimization - Diurnal Power Preservation', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Diurnal Power')
        plt.tight_layout()
        plt.savefig(output_dir / 'als_optimization_diurnal_power.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 3: Parameter comparison
    lam_best = best_params['lam_als']
    p_best = best_params['p_als']
    
    baseline_best = baseline_als(signal_smoothed, lam=lam_best, p=p_best, niter=niter_als)
    corrected_best = signal_smoothed - baseline_best
    
    # Also plot with default parameters for comparison (unless using predefined params)
    if lam_als is not None and p_als is not None:
        # Only show predefined parameters
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Original smoothed signal
        axes[0].plot(datetimes_valid, signal_smoothed, 'm-', label='Gaussian Smoothed Signal', alpha=0.7, linewidth=1.5)
        axes[0].set_ylabel('Ratio', fontsize=12)
        axes[0].set_title('Original Signal (Gaussian Smoothed)', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Predefined parameters
        axes[1].plot(datetimes_valid, signal_smoothed, 'm-', label='Original', alpha=0.5, linewidth=1)
        axes[1].plot(datetimes_valid, baseline_best, 'r--', label=f'ALS Baseline (lam={lam_best:.2e}, p={p_best:.6f})', linewidth=2)
        axes[1].plot(datetimes_valid, corrected_best, 'g-', label='Corrected', alpha=0.8, linewidth=1.5)
        axes[1].set_ylabel('Ratio', fontsize=12)
        axes[1].set_xlabel('Date time (MM-DD HH)', fontsize=12)
        axes[1].set_title(f'ALS Correction (Score: {best_params["score"]:.4f})', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
        axes[1].xaxis.set_major_locator(DayLocator())
    else:
        # Show best vs default comparison
        lam_default = 1e7
        p_default = 0.0001
        baseline_default = baseline_als(signal_smoothed, lam=lam_default, p=p_default, niter=niter_als)
        corrected_default = signal_smoothed - baseline_default
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
        
        # Original smoothed signal
        axes[0].plot(datetimes_valid, signal_smoothed, 'm-', label='Gaussian Smoothed Signal', alpha=0.7, linewidth=1.5)
        axes[0].set_ylabel('Ratio', fontsize=12)
        axes[0].set_title('Original Signal (Gaussian Smoothed)', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Best parameters
        axes[1].plot(datetimes_valid, signal_smoothed, 'm-', label='Original', alpha=0.5, linewidth=1)
        axes[1].plot(datetimes_valid, baseline_best, 'r--', label=f'ALS Baseline (lam={lam_best:.2e}, p={p_best:.6f})', linewidth=2)
        axes[1].plot(datetimes_valid, corrected_best, 'g-', label='Corrected (Best)', alpha=0.8, linewidth=1.5)
        axes[1].set_ylabel('Ratio', fontsize=12)
        axes[1].set_title(f'Best Parameters (Score: {best_params["score"]:.4f})', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Default parameters
        axes[2].plot(datetimes_valid, signal_smoothed, 'm-', label='Original', alpha=0.5, linewidth=1)
        axes[2].plot(datetimes_valid, baseline_default, 'r--', label=f'ALS Baseline (lam={lam_default:.2e}, p={p_default:.6f})', linewidth=2)
        axes[2].plot(datetimes_valid, corrected_default, 'g-', label='Corrected (Default)', alpha=0.8, linewidth=1.5)
        axes[2].set_ylabel('Ratio', fontsize=12)
        axes[2].set_xlabel('Date time (MM-DD HH)', fontsize=12)
        axes[2].set_title('Default Parameters (for comparison)', fontsize=13, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
        axes[2].xaxis.set_major_locator(DayLocator())
    
    plt.tight_layout()
    plt.savefig(output_dir / 'als_optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: FFT comparison
    if lam_als is not None and p_als is not None:
        # Single FFT plot for predefined parameters
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n = len(corrected_best)
        freqs_best = fftfreq(n, d=sampling_interval_hours)
        fft_best = fft(corrected_best - np.mean(corrected_best))
        magnitude_best = np.abs(fft_best)
        
        # Plot positive frequencies only
        positive_mask = freqs_best > 0
        freqs_plot = freqs_best[positive_mask]
        
        ax.plot(freqs_plot, magnitude_best[positive_mask], 'g-', label=f'Parameters (lam={lam_best:.2e}, p={p_best:.6f})', linewidth=2)
        ax.axvspan(0.03, 0.05, alpha=0.2, color='yellow', label='Diurnal Range (0.03-0.05 cycles/hour)')
        ax.set_ylabel('Magnitude', fontsize=12)
        ax.set_xlabel('Frequency (cycles/hour)', fontsize=12)
        ax.set_title(f'FFT - Predefined Parameters', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 0.2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'als_optimization_fft_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # FFT comparison for best vs default
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Calculate FFT for best and default
        n = len(corrected_best)
        freqs_best = fftfreq(n, d=sampling_interval_hours)
        fft_best = fft(corrected_best - np.mean(corrected_best))
        magnitude_best = np.abs(fft_best)
        
        freqs_default = fftfreq(n, d=sampling_interval_hours)
        fft_default = fft(corrected_default - np.mean(corrected_default))
        magnitude_default = np.abs(fft_default)
        
        # Plot positive frequencies only
        positive_mask = freqs_best > 0
        freqs_plot = freqs_best[positive_mask]
        
        axes[0].plot(freqs_plot, magnitude_best[positive_mask], 'g-', label='Best Parameters', linewidth=2)
        axes[0].axvspan(0.03, 0.05, alpha=0.2, color='yellow', label='Diurnal Range (0.03-0.05 cycles/hour)')
        axes[0].set_ylabel('Magnitude', fontsize=12)
        axes[0].set_title(f'FFT - Best Parameters (lam={lam_best:.2e}, p={p_best:.6f})', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 0.2)
        
        axes[1].plot(freqs_plot, magnitude_default[positive_mask], 'b-', label='Default Parameters', linewidth=2)
        axes[1].axvspan(0.03, 0.05, alpha=0.2, color='yellow', label='Diurnal Range (0.03-0.05 cycles/hour)')
        axes[1].set_ylabel('Magnitude', fontsize=12)
        axes[1].set_xlabel('Frequency (cycles/hour)', fontsize=12)
        axes[1].set_title(f'FFT - Default Parameters (lam={lam_default:.2e}, p={p_default:.6f})', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 0.2)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'als_optimization_fft_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"\nAll plots saved to: {output_dir}")
    print(f"\n=== Recommended Parameters ===")
    print(f"lam_als: {best_params['lam_als']:.2e}")
    print(f"p_als: {best_params['p_als']:.6f}")
    print(f"\nAdd to config/pipeline.yml:")
    print(f"  lam_als: {int(best_params['lam_als'])}")
    print(f"  p_als: {best_params['p_als']:.6f}")
    
    return results_df, best_params


def main():
    parser = argparse.ArgumentParser(
        description='Optimize ALS baseline correction parameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default ranges (optimization)
  python optimize_als_parameters.py "test_outputs_v3_batch_20251123_231801/batch_summary_data_v3.csv"
  
  # Custom parameter ranges (optimization)
  python optimize_als_parameters.py "path/to/data.csv" --lam-range 1e5 1e7 --p-range 0.0001 0.001
  
  # Use predefined parameters (skip optimization)
  python optimize_als_parameters.py "path/to/data.csv" --lam-als 1e6 --p-als 0.001
  
  # Different ratio column
  python optimize_als_parameters.py "path/to/data.csv" --ratio-column "Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected"
  
  # Custom Gaussian smoothing
  python optimize_als_parameters.py "path/to/data.csv" --sigma-gaussian 30
        """
    )
    
    parser.add_argument('csv_path', help='Path to batch_summary_data_v3.csv')
    parser.add_argument('--ratio-column', default='Fluorescence_to_Gband_Ratio_BaselineCorrected',
                       help='Column name to optimize for (default: Fluorescence_to_Gband_Ratio_BaselineCorrected)')
    parser.add_argument('--sigma-gaussian', type=float, default=25,
                       help='Gaussian smoothing sigma (applied BEFORE ALS, default: 25)')
    parser.add_argument('--lam-range', nargs=2, type=float, default=[1e4, 1e8],
                       metavar=('MIN', 'MAX'),
                       help='Range for lam_als (default: 1e4 1e8)')
    parser.add_argument('--p-range', nargs=2, type=float, default=[0.0001, 0.01],
                       metavar=('MIN', 'MAX'),
                       help='Range for p_als (default: 0.0001 0.01)')
    parser.add_argument('--n-lam', type=int, default=10,
                       help='Number of lam values to test (default: 10)')
    parser.add_argument('--n-p', type=int, default=8,
                       help='Number of p values to test (default: 8)')
    parser.add_argument('--niter-als', type=int, default=20,
                       help='Number of ALS iterations (default: 20)')
    parser.add_argument('--lam-als', type=float, default=None,
                       help='Predefined lam_als parameter. If provided with --p-als, skips optimization and uses these values.')
    parser.add_argument('--p-als', type=float, default=None,
                       help='Predefined p_als parameter. If provided with --lam-als, skips optimization and uses these values.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: creates timestamped directory)')
    
    args = parser.parse_args()
    
    # Validate predefined parameters
    if (args.lam_als is not None) != (args.p_als is not None):
        parser.error("--lam-als and --p-als must be provided together to use predefined parameters")
    
    try:
        results_df, best_params = optimize_als_parameters(
            csv_path=args.csv_path,
            ratio_column=args.ratio_column,
            sigma_gaussian=args.sigma_gaussian,
            lam_range=tuple(args.lam_range),
            p_range=tuple(args.p_range),
            n_lam=args.n_lam,
            n_p=args.n_p,
            niter_als=args.niter_als,
            output_dir=args.output_dir,
            lam_als=args.lam_als,
            p_als=args.p_als
        )
        
        print("\nOptimization complete!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

