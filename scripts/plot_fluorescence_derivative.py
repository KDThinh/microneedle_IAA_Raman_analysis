"""
Standalone script to plot raw and normalized fluorescence data vs scan number
with first derivatives from batch_results_summary.csv

Usage:
    python plot_fluorescence_derivative.py <path_to_batch_results_summary.csv> [output_dir]
    
Example:
    python plot_fluorescence_derivative.py "path/to/batch_results_summary.csv"
    python plot_fluorescence_derivative.py "path/to/batch_results_summary.csv" "custom/output/dir"
"""
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def calculate_derivative(x, y):
    """
    Calculate first derivative using numpy gradient.
    
    Parameters
    ----------
    x : array-like
        Independent variable (scan numbers)
    y : array-like
        Dependent variable (fluorescence values)
    
    Returns
    -------
    dy_dx : array
        First derivative
    """
    # Remove NaN values for derivative calculation
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 2:
        return np.full_like(y, np.nan)
    
    x_clean = np.array(x[mask])
    y_clean = np.array(y[mask])
    
    # Calculate derivative
    dy_dx = np.gradient(y_clean, x_clean)
    
    # Create full array with NaN where original data was NaN
    result = np.full_like(y, np.nan)
    result[mask] = dy_dx
    
    return result


def plot_fluorescence_with_derivative(csv_path, output_dir=None):
    """
    Plot raw and normalized fluorescence data vs scan number with first derivatives.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to batch_results_summary.csv file
    output_dir : str or Path, optional
        Output directory for plots. If None, defaults to Script/swnt_iaa_analysis_v2/scripts/test_outputs
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
    required_cols = ['Scan Number']
    if 'Scan Number' not in df.columns:
        print("Error: 'Scan Number' column not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Set output directory
    if output_dir is None:
        # Default to Script/swnt_iaa_analysis_v2/scripts/test_outputs
        script_dir = Path(__file__).parent
        output_dir = script_dir / "test_outputs"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Sort by scan number
    df = df.sort_values('Scan Number').reset_index(drop=True)
    scan_numbers = df['Scan Number'].values
    
    # ========================================================================
    # Raw Fluorescence Plots
    # ========================================================================
    raw_columns = {
        'Raw_Fluorescence_Method1': 'Raw Method 1 (Lieberfit baseline AUC)',
        'Raw_Fluorescence_Method2': 'Raw Method 2 (Raw - G-band - Background)',
        'Raw_Fluorescence_RawAUC': 'Raw Raw AUC (no correction)'
    }
    
    # Find available raw columns
    available_raw_cols = {col: label for col, label in raw_columns.items() if col in df.columns}
    
    if available_raw_cols:
        fig_raw, axes_raw = plt.subplots(len(available_raw_cols), 2, figsize=(16, 5 * len(available_raw_cols)))
        if len(available_raw_cols) == 1:
            axes_raw = axes_raw.reshape(1, -1)
        
        fig_raw.suptitle('Raw Fluorescence vs Scan Number with First Derivatives', 
                        fontsize=16, fontweight='bold')
        
        for idx, (col, label) in enumerate(available_raw_cols.items()):
            values = df[col].values
            
            # Plot fluorescence
            ax_data = axes_raw[idx, 0]
            ax_data.plot(scan_numbers, values, 'o-', linewidth=2, markersize=4, alpha=0.7, color='blue')
            ax_data.set_xlabel('Scan Number', fontsize=12)
            ax_data.set_ylabel('Fluorescence', fontsize=12)
            ax_data.set_title(f'{label}', fontsize=13, fontweight='bold')
            ax_data.grid(True, alpha=0.3)
            ax_data.tick_params(axis='both', labelsize=11)
            
            # Plot first derivative
            ax_deriv = axes_raw[idx, 1]
            derivative = calculate_derivative(scan_numbers, values)
            ax_deriv.plot(scan_numbers, derivative, 's-', linewidth=2, markersize=4, alpha=0.7, color='red')
            ax_deriv.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax_deriv.set_xlabel('Scan Number', fontsize=12)
            ax_deriv.set_ylabel('First Derivative (dF/dScan)', fontsize=12)
            ax_deriv.set_title(f'{label} - First Derivative', fontsize=13, fontweight='bold')
            ax_deriv.grid(True, alpha=0.3)
            ax_deriv.tick_params(axis='both', labelsize=11)
        
        plt.tight_layout()
        save_path_raw = output_dir / 'raw_fluorescence_with_derivative.png'
        plt.savefig(str(save_path_raw), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Raw fluorescence plot saved to: {save_path_raw}")
    
    # ========================================================================
    # Normalized Fluorescence Plots
    # ========================================================================
    normalized_columns = {
        'Normalized_Fluorescence_Method1': 'Normalized Method 1 (Lieberfit baseline AUC)',
        'Normalized_Fluorescence_Method2': 'Normalized Method 2 (Raw - G-band - Background)',
        'Normalized_Fluorescence_RawAUC': 'Normalized Raw AUC (no correction)'
    }
    
    # Find available normalized columns
    available_norm_cols = {col: label for col, label in normalized_columns.items() if col in df.columns}
    
    if available_norm_cols:
        fig_norm, axes_norm = plt.subplots(len(available_norm_cols), 2, figsize=(16, 5 * len(available_norm_cols)))
        if len(available_norm_cols) == 1:
            axes_norm = axes_norm.reshape(1, -1)
        
        fig_norm.suptitle('Normalized Fluorescence vs Scan Number with First Derivatives', 
                         fontsize=16, fontweight='bold')
        
        for idx, (col, label) in enumerate(available_norm_cols.items()):
            values = df[col].values
            
            # Plot fluorescence
            ax_data = axes_norm[idx, 0]
            ax_data.plot(scan_numbers, values, 'o-', linewidth=2, markersize=4, alpha=0.7, color='green')
            ax_data.set_xlabel('Scan Number', fontsize=12)
            ax_data.set_ylabel('Normalized Fluorescence', fontsize=12)
            ax_data.set_title(f'{label}', fontsize=13, fontweight='bold')
            ax_data.grid(True, alpha=0.3)
            ax_data.tick_params(axis='both', labelsize=11)
            
            # Plot first derivative
            ax_deriv = axes_norm[idx, 1]
            derivative = calculate_derivative(scan_numbers, values)
            ax_deriv.plot(scan_numbers, derivative, 's-', linewidth=2, markersize=4, alpha=0.7, color='orange')
            ax_deriv.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax_deriv.set_xlabel('Scan Number', fontsize=12)
            ax_deriv.set_ylabel('First Derivative (dF/dScan)', fontsize=12)
            ax_deriv.set_title(f'{label} - First Derivative', fontsize=13, fontweight='bold')
            ax_deriv.grid(True, alpha=0.3)
            ax_deriv.tick_params(axis='both', labelsize=11)
        
        plt.tight_layout()
        save_path_norm = output_dir / 'normalized_fluorescence_with_derivative.png'
        plt.savefig(str(save_path_norm), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Normalized fluorescence plot saved to: {save_path_norm}")
    
    if not available_raw_cols and not available_norm_cols:
        print("Warning: No fluorescence columns found in CSV")
        print(f"Available columns: {list(df.columns)}")
        print("\nExpected columns:")
        print("  Raw: Raw_Fluorescence_Method1, Raw_Fluorescence_Method2, Raw_Fluorescence_RawAUC")
        print("  Normalized: Normalized_Fluorescence_Method1, Normalized_Fluorescence_Method2, Normalized_Fluorescence_RawAUC")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python plot_fluorescence_derivative.py <path_to_batch_results_summary.csv> [output_dir]")
        print("\nExample:")
        print('  python plot_fluorescence_derivative.py "path/to/batch_results_summary.csv"')
        print('  python plot_fluorescence_derivative.py "path/to/batch_results_summary.csv" "output/dir"')
        sys.exit(1)
    
    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    plot_fluorescence_with_derivative(csv_path, output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()

