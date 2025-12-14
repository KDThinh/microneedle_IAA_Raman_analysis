#!/usr/bin/env python3
"""
Calibration script for IAA sensor response.

This script:
1. Parses scan_info.txt to extract IAA addition points
2. Calculates cumulative IAA concentrations
3. Reads sensor response data from batch_summary CSV
4. Calculates normalized sensor responses
5. Plots calibration curves
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from datetime import datetime
import re
from scipy.optimize import curve_fit


def parse_scan_info(scan_info_path):
    """
    Parse scan_info.txt file to extract IAA addition information.
    
    Parameters:
    -----------
    scan_info_path : str or Path
        Path to scan_info.txt file
        
    Returns:
    --------
    dict : Dictionary containing:
        - initial_volume_mL : float
        - stock_concentration_mM : float
        - additions : list of dicts with 'scan' and 'volume_uL' keys
    """
    scan_info_path = Path(scan_info_path)
    
    if not scan_info_path.exists():
        raise FileNotFoundError(f"Scan info file not found: {scan_info_path}")
    
    with open(scan_info_path, 'r') as f:
        lines = f.readlines()
    
    info = {
        'initial_volume_mL': None,
        'stock_concentration_mM': None,
        'additions': []
    }
    
    # Parse header information
    for line in lines:
        line = line.strip()
        if 'Initial volumne' in line or 'Initial volume' in line:
            # Extract volume (handle typo "volumne")
            match = re.search(r'(\d+\.?\d*)\s*mL', line, re.IGNORECASE)
            if match:
                info['initial_volume_mL'] = float(match.group(1))
        elif 'IAA in DMSO' in line:
            # Extract concentration
            match = re.search(r'(\d+\.?\d*)\s*mM', line, re.IGNORECASE)
            if match:
                info['stock_concentration_mM'] = float(match.group(1))
        elif line.startswith('scan') and 'IAA' in line.lower():
            # Header line for additions table
            continue
        elif line and not line.startswith('Date') and not line.startswith('Time') and \
             not line.startswith('No.') and not line.startswith('Time Lapsed') and \
             not line.startswith('Int.') and not line.startswith('Averaging') and \
             not line.startswith('Smoothing') and line != '':
            # Parse addition data (tab or space separated)
            parts = line.split()
            if len(parts) >= 2:
                try:
                    scan = int(parts[0])
                    volume_uL = float(parts[1])
                    info['additions'].append({'scan': scan, 'volume_uL': volume_uL})
                except ValueError:
                    continue
    
    # Validate
    if info['initial_volume_mL'] is None:
        raise ValueError("Could not find initial volume in scan_info file")
    if info['stock_concentration_mM'] is None:
        raise ValueError("Could not find stock concentration in scan_info file")
    if len(info['additions']) == 0:
        raise ValueError("No IAA additions found in scan_info file")
    
    # Sort additions by scan number
    info['additions'].sort(key=lambda x: x['scan'])
    
    return info


def calculate_cumulative_concentrations(scan_info):
    """
    Calculate cumulative IAA concentrations after each addition.
    
    Parameters:
    -----------
    scan_info : dict
        Parsed scan info dictionary
        
    Returns:
    --------
    pd.DataFrame : DataFrame with columns:
        - scan : scan number
        - volume_added_uL : volume added at this scan
        - cumulative_volume_uL : cumulative volume added
        - total_volume_mL : total volume after addition
        - concentration_uM : cumulative IAA concentration in uM
    """
    initial_volume_mL = scan_info['initial_volume_mL']
    stock_concentration_mM = scan_info['stock_concentration_mM']
    additions = scan_info['additions']
    
    # Convert initial volume to uL
    total_volume_uL = initial_volume_mL * 1000
    cumulative_moles = 0.0  # in moles
    
    results = []
    
    for addition in additions:
        scan = addition['scan']
        volume_added_uL = addition['volume_uL']
        
        # Calculate moles added (stock is in mM, volume in uL)
        # 1 mM = 1e-3 M, 1 uL = 1e-6 L
        # moles = (volume_uL * 1e-6 L) * (stock_mM * 1e-3 M) = volume_uL * stock_mM * 1e-9
        moles_added = volume_added_uL * stock_concentration_mM * 1e-9
        
        cumulative_moles += moles_added
        total_volume_uL += volume_added_uL
        
        # Calculate concentration in uM
        # concentration_M = cumulative_moles / (total_volume_uL * 1e-6)
        # concentration_uM = concentration_M * 1e6 = cumulative_moles * 1e6 / (total_volume_uL * 1e-6)
        # = cumulative_moles * 1e12 / total_volume_uL
        concentration_uM = (cumulative_moles * 1e12) / total_volume_uL if total_volume_uL > 0 else 0
        
        results.append({
            'scan': scan,
            'volume_added_uL': volume_added_uL,
            'cumulative_volume_uL': total_volume_uL - initial_volume_mL * 1000,
            'total_volume_mL': total_volume_uL / 1000,
            'concentration_uM': concentration_uM
        })
    
    return pd.DataFrame(results)


def load_sensor_data(csv_path):
    """
    Load sensor response data from batch_summary CSV.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to batch_summary CSV file
        
    Returns:
    --------
    pd.DataFrame : DataFrame with sensor data
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Check for required columns
    required_cols = ['Normalized_Fluorescence_Intensity', 'Normalized_Gband_Area', 'Fluorescence_to_Gband_Ratio']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")
    
    # Check if 'Scan Number' column exists, if not try to infer from index
    if 'Scan Number' not in df.columns:
        # Try to infer from Seconds column or index
        if 'Seconds' in df.columns:
            # Assume scans are sequential starting from 0
            # This is a fallback - ideally Scan Number should be in the CSV
            print("Warning: 'Scan Number' column not found. Attempting to infer from row index.")
            df['Scan Number'] = df.index
        else:
            raise ValueError("'Scan Number' column not found and cannot be inferred")
    
    return df


def calculate_sensor_responses(df, concentration_df, n_average=20):
    """
    Calculate average sensor responses for each concentration point.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Sensor data DataFrame
    concentration_df : pd.DataFrame
        Concentration data with scan numbers
    n_average : int
        Number of scans to average before next addition (default: 20)
        
    Returns:
    --------
    pd.DataFrame : DataFrame with columns:
        - concentration_uM : IAA concentration
        - normalized_fluorescence_ratio : normalized fluorescence intensity ratio
        - normalized_fluorescence_ratio_err : error (std) for normalized fluorescence ratio
        - normalized_gband_area_ratio : normalized G-band area ratio
        - normalized_gband_area_ratio_err : error (std) for normalized G-band area ratio
        - fluorescence_to_gband_ratio : fluorescence to G-band ratio (as-is)
        - fluorescence_to_gband_ratio_err : error (std) for fluorescence to G-band ratio
    """
    # Get first addition scan (skip scan 0 if it's the baseline)
    additions = concentration_df['scan'].values
    first_addition_scan = additions[0] if additions[0] > 0 else additions[1] if len(additions) > 1 else additions[0]
    
    # Calculate baseline averages (before first addition)
    baseline_mask = df['Scan Number'] < first_addition_scan
    if baseline_mask.sum() == 0:
        # If no baseline data, use first few scans
        baseline_mask = df['Scan Number'] <= min(10, len(df) - 1)
    
    baseline_fluo_avg = df.loc[baseline_mask, 'Normalized_Fluorescence_Intensity'].mean()
    baseline_gband_avg = df.loc[baseline_mask, 'Normalized_Gband_Area'].mean()
    
    print(f"\nBaseline averages (scans < {first_addition_scan}):")
    print(f"  Normalized Fluorescence Intensity: {baseline_fluo_avg:.4f}")
    print(f"  Normalized G-band Area: {baseline_gband_avg:.4f}")
    
    results = []
    
    for idx, row in concentration_df.iterrows():
        scan = row['scan']
        concentration_uM = row['concentration_uM']
        
        # Skip baseline (scan 0 with 0 concentration)
        if concentration_uM == 0 and scan == 0:
            continue
        
        # Determine scan range for averaging
        # Use last n_average scans before next addition
        if idx < len(concentration_df) - 1:
            next_scan = concentration_df.iloc[idx + 1]['scan']
            end_scan = next_scan - 1
            start_scan = max(first_addition_scan, end_scan - n_average + 1)
        else:
            # Last addition: use last n_average scans available
            end_scan = df['Scan Number'].max()
            start_scan = max(first_addition_scan, end_scan - n_average + 1)
        
        # Filter data for this range
        mask = (df['Scan Number'] >= start_scan) & (df['Scan Number'] <= end_scan)
        subset = df[mask]
        
        if len(subset) == 0:
            print(f"Warning: No data found for scan range {start_scan}-{end_scan} (concentration {concentration_uM:.2f} uM)")
            continue
        
        # Calculate averages and standard deviations
        avg_fluo = subset['Normalized_Fluorescence_Intensity'].mean()
        std_fluo = subset['Normalized_Fluorescence_Intensity'].std()
        avg_gband_area = subset['Normalized_Gband_Area'].mean()
        std_gband_area = subset['Normalized_Gband_Area'].std()
        avg_ratio = subset['Fluorescence_to_Gband_Ratio'].mean()
        std_ratio = subset['Fluorescence_to_Gband_Ratio'].std()
        
        # Calculate normalized ratios (for fluorescence and G-band area)
        normalized_fluo_ratio = avg_fluo / baseline_fluo_avg if baseline_fluo_avg > 0 else np.nan
        normalized_gband_ratio = avg_gband_area / baseline_gband_avg if baseline_gband_avg > 0 else np.nan
        
        # Calculate error propagation for normalized ratios
        # For ratio = value / baseline, error = std_value / baseline
        normalized_fluo_ratio_err = std_fluo / baseline_fluo_avg if baseline_fluo_avg > 0 else np.nan
        normalized_gband_ratio_err = std_gband_area / baseline_gband_avg if baseline_gband_avg > 0 else np.nan
        
        results.append({
            'concentration_uM': concentration_uM,
            'scan_range': f"{start_scan}-{end_scan}",
            'n_scans': len(subset),
            'normalized_fluorescence_ratio': normalized_fluo_ratio,
            'normalized_fluorescence_ratio_err': normalized_fluo_ratio_err,
            'normalized_gband_area_ratio': normalized_gband_ratio,
            'normalized_gband_area_ratio_err': normalized_gband_ratio_err,
            'fluorescence_to_gband_ratio': avg_ratio,
            'fluorescence_to_gband_ratio_err': std_ratio
        })
    
    return pd.DataFrame(results)


def hill_model(concentration_uM, beta, n, Kd):
    """
    Hill model equation for sensor response.
    
    Parameters:
    -----------
    concentration_uM : array-like
        IAA concentration in uM
    beta : float
        Amplitude parameter
    n : float
        Hill coefficient
    Kd : float
        Dissociation constant in uM
        
    Returns:
    --------
    array-like : Normalized fluorescence intensity ratio
    """
    # Handle zero concentrations
    concentration_uM = np.array(concentration_uM)
    result = np.ones_like(concentration_uM, dtype=float)
    
    # Apply Hill equation for non-zero concentrations
    mask = concentration_uM > 0
    if np.any(mask):
        C_n = concentration_uM[mask] ** n
        result[mask] = -beta * C_n / (C_n + Kd ** n) + 1
    
    return result


def fit_hill_model(concentrations, responses, errors=None):
    """
    Fit Hill model to calibration data.
    
    Parameters:
    -----------
    concentrations : array-like
        IAA concentrations in uM
    responses : array-like
        Normalized fluorescence intensity ratios
    errors : array-like, optional
        Error bars for weighting the fit
        
    Returns:
    --------
    tuple : (fitted_params, param_errors, r_squared)
        fitted_params: [beta, n, Kd]
        param_errors: standard errors for parameters
        r_squared: R-squared value for the fit
    """
    # Filter out NaN and invalid values
    mask = ~(np.isnan(concentrations) | np.isnan(responses))
    concentrations = np.array(concentrations)[mask]
    responses = np.array(responses)[mask]
    
    if len(concentrations) < 3:
        raise ValueError("Need at least 3 data points for fitting")
    
    # Filter out zero concentrations for fitting (they're baseline)
    mask = concentrations > 0
    concentrations_fit = concentrations[mask]
    responses_fit = responses[mask]
    
    if len(concentrations_fit) < 3:
        raise ValueError("Need at least 3 non-zero concentration points for fitting")
    
    # Initial parameter estimates
    # beta: roughly the max response change from baseline (1.0)
    max_response = np.max(responses_fit)
    min_response = np.min(responses_fit)
    beta_init = max(max_response - 1.0, 0.1)
    
    # n: Hill coefficient, start with 1 (Michaelis-Menten)
    n_init = 1.0
    
    # Kd: roughly the concentration at half-maximal response
    # Use median concentration as initial guess
    Kd_init = np.median(concentrations_fit)
    
    p0 = [beta_init, n_init, Kd_init]
    
    # Parameter bounds: beta > 0, n > 0, Kd > 0
    bounds = ([0, 0.1, 0], [np.inf, 10, np.inf])
    
    # Weight by inverse of errors if provided
    if errors is not None:
        errors_fit = np.array(errors)[mask]
        sigma = errors_fit
        # Avoid division by zero
        sigma = np.where(sigma > 0, sigma, np.mean(sigma[sigma > 0]) if np.any(sigma > 0) else 1.0)
    else:
        sigma = None
    
    try:
        # Fit the model
        popt, pcov = curve_fit(
            hill_model,
            concentrations_fit,
            responses_fit,
            p0=p0,
            bounds=bounds,
            sigma=sigma,
            maxfev=10000
        )
        
        # Calculate parameter errors
        param_errors = np.sqrt(np.diag(pcov))
        
        # Calculate R-squared
        y_pred = hill_model(concentrations_fit, *popt)
        ss_res = np.sum((responses_fit - y_pred) ** 2)
        ss_tot = np.sum((responses_fit - np.mean(responses_fit)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return popt, param_errors, r_squared
    
    except Exception as e:
        print(f"Warning: Hill model fitting failed: {e}")
        return None, None, None


def plot_calibration_curves(calibration_df, output_dir):
    """
    Plot calibration curves for sensor responses vs IAA concentration.
    
    Parameters:
    -----------
    calibration_df : pd.DataFrame
        Calibration data DataFrame
    output_dir : Path
        Output directory for saving plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    concentrations = calibration_df['concentration_uM'].values
    
    # Plot 1: Normalized Fluorescence Intensity Ratio with Hill model fit
    ax1 = axes[0]
    ax1.errorbar(concentrations, calibration_df['normalized_fluorescence_ratio'],
                 yerr=calibration_df['normalized_fluorescence_ratio_err'],
                 fmt='o', color='blue', markersize=8, capsize=5, capthick=2, elinewidth=2,
                 label='Data', zorder=2)
    
    # Fit Hill model
    try:
        popt, param_errors, r_squared = fit_hill_model(
            concentrations,
            calibration_df['normalized_fluorescence_ratio'].values,
            errors=calibration_df['normalized_fluorescence_ratio_err'].values
        )
        
        if popt is not None:
            beta, n, Kd = popt
            beta_err, n_err, Kd_err = param_errors
            
            # Generate smooth curve for plotting
            conc_min = np.min(concentrations[concentrations > 0])
            conc_max = np.max(concentrations)
            conc_fit = np.logspace(np.log10(conc_min), np.log10(conc_max), 200)
            response_fit = hill_model(conc_fit, beta, n, Kd)
            
            # Plot fitted curve
            ax1.plot(conc_fit, response_fit, 'r-', linewidth=2, label='Fit', zorder=1)
            
            # Add fit parameters as text
            fit_text = (f'Fit:\n'
                       f'β = {beta:.3f} ± {beta_err:.3f}\n'
                       f'n = {n:.3f} ± {n_err:.3f}\n'
                       f'Kd = {Kd:.2f} ± {Kd_err:.2f} µM\n'
                       f'R² = {r_squared:.4f}')
            ax1.text(0.05, 0.05, fit_text, transform=ax1.transAxes,
                    fontsize=9, verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            print(f"\nHill Model Fit Results:")
            print(f"  β = {beta:.4f} ± {beta_err:.4f}")
            print(f"  n = {n:.4f} ± {n_err:.4f}")
            print(f"  Kd = {Kd:.4f} ± {Kd_err:.4f} µM")
            print(f"  R² = {r_squared:.4f}")
    except Exception as e:
        print(f"Warning: Could not fit Hill model: {e}")
    
    # Determine concentration range for tick labels
    conc_min = np.min(concentrations[concentrations > 0])
    conc_max = np.max(concentrations)
    log_min = np.floor(np.log10(conc_min))
    log_max = np.ceil(np.log10(conc_max))
    
    # Generate tick locations: include major ticks (powers of 10) and key intermediate ticks
    major_ticks = []
    minor_ticks = []
    
    for exp in np.arange(log_min, log_max + 1):
        # Major ticks: powers of 10
        major_val = 10 ** exp
        if conc_min <= major_val <= conc_max:
            major_ticks.append(major_val)
        
        # Intermediate ticks with labels: 2, 5 times powers of 10
        for mult in [2, 5]:
            tick_val = mult * (10 ** exp)
            if conc_min <= tick_val <= conc_max:
                major_ticks.append(tick_val)
        
        # Other intermediate ticks as minor ticks (for grid lines)
        for mult in [3, 4, 6, 7, 8, 9]:
            tick_val = mult * (10 ** exp)
            if conc_min <= tick_val <= conc_max:
                minor_ticks.append(tick_val)
    
    # Sort ticks
    major_ticks = sorted(major_ticks)
    minor_ticks = sorted(minor_ticks)
    
    ax1.set_xlabel('IAA Concentration (uM)', fontsize=12)
    ax1.set_ylabel('Normalized Fluorescence Intensity Ratio', fontsize=12)
    ax1.set_title('Fluorescence Intensity Response', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    # Set more tick labels on log scale
    ax1.set_xticks(major_ticks)
    ax1.set_xticks(minor_ticks, minor=True)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(loc='best', fontsize=9)
    
    # Plot 2: Normalized G-band Area Ratio
    ax2 = axes[1]
    ax2.errorbar(concentrations, calibration_df['normalized_gband_area_ratio'],
                 yerr=calibration_df['normalized_gband_area_ratio_err'],
                 fmt='o', color='green', markersize=8, capsize=5, capthick=2, elinewidth=2)
    ax2.set_xlabel('IAA Concentration (uM)', fontsize=12)
    ax2.set_ylabel('Normalized G-band Area Ratio', fontsize=12)
    ax2.set_title('G-band Area Response', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    # Set more tick labels on log scale
    ax2.set_xticks(major_ticks)
    ax2.set_xticks(minor_ticks, minor=True)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Fluorescence to G-band Ratio
    ax3 = axes[2]
    ax3.errorbar(concentrations, calibration_df['fluorescence_to_gband_ratio'],
                 yerr=calibration_df['fluorescence_to_gband_ratio_err'],
                 fmt='o', color='red', markersize=8, capsize=5, capthick=2, elinewidth=2)
    ax3.set_xlabel('IAA Concentration (uM)', fontsize=12)
    ax3.set_ylabel('Fluorescence to G-band Ratio', fontsize=12)
    ax3.set_title('Fluorescence/G-band Ratio Response', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    # Set more tick labels on log scale
    ax3.set_xticks(major_ticks)
    ax3.set_xticks(minor_ticks, minor=True)
    ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = output_dir / f'iaa_calibration_curves_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nCalibration plot saved to: {plot_path}")
    
    plt.close()
    
    # Also save calibration data as CSV
    csv_path = output_dir / f'calibration_data_{timestamp}.csv'
    calibration_df.to_csv(csv_path, index=False)
    print(f"Calibration data saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate IAA sensor response from batch summary data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python calibrate_iaa_sensor.py \\
    --csv "path/to/batch_summary_v4_final.csv" \\
    --scan-info "path/to/run 1_scan info.txt" \\
    --n-average 20
        """
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to batch_summary_v4_final.csv file'
    )
    
    parser.add_argument(
        '--scan-info',
        type=str,
        required=True,
        help='Path to scan_info.txt file'
    )
    
    parser.add_argument(
        '--n-average',
        type=int,
        default=20,
        help='Number of scans to average before next addition (default: 20)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for plots (default: calibration folder in CSV directory)'
    )
    
    args = parser.parse_args()
    
    # Parse scan info
    print("Parsing scan info file...")
    scan_info = parse_scan_info(args.scan_info)
    print(f"Initial volume: {scan_info['initial_volume_mL']} mL")
    print(f"Stock concentration: {scan_info['stock_concentration_mM']} mM")
    print(f"Number of additions: {len(scan_info['additions'])}")
    for add in scan_info['additions']:
        print(f"  Scan {add['scan']}: {add['volume_uL']} uL")
    
    # Calculate concentrations
    print("\nCalculating cumulative concentrations...")
    concentration_df = calculate_cumulative_concentrations(scan_info)
    print("\nConcentration summary:")
    print(concentration_df.to_string(index=False))
    
    # Load sensor data
    print(f"\nLoading sensor data from {args.csv}...")
    sensor_df = load_sensor_data(args.csv)
    print(f"Loaded {len(sensor_df)} data points")
    
    # Calculate sensor responses
    print(f"\nCalculating sensor responses (averaging last {args.n_average} scans per addition)...")
    calibration_df = calculate_sensor_responses(sensor_df, concentration_df, n_average=args.n_average)
    print("\nCalibration data:")
    print(calibration_df.to_string(index=False))
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        csv_path = Path(args.csv)
        output_dir = csv_path.parent / 'calibration'
    
    # Plot calibration curves
    print("\nGenerating calibration plots...")
    plot_calibration_curves(calibration_df, output_dir)
    
    print("\nCalibration analysis complete!")


if __name__ == '__main__':
    main()

