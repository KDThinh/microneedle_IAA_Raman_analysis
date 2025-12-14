"""
Utility script to plot overlaid normalized Raman spectroscopy spectra.

This script reads representative Raman data from CSV and creates publication-quality
overlay plots showing normalized spectra for petiole, IAA nanosensor, and IAA patch,
with annotations for G-band peak and Raman peak at 850 cm^-1.

Usage:
    python plot_representative_raman_spectra.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from pathlib import Path

# Note: This script is standalone and doesn't require imports from src.pipeline


def normalize_spectrum_to_max(intensities):
    """
    Normalize spectrum to maximum value (0-1 range).
    
    Parameters:
    -----------
    intensities : array-like
        Intensity values
    
    Returns:
    --------
    normalized : array
        Normalized intensities (divided by maximum)
    """
    intensities = np.array(intensities)
    max_val = np.max(intensities)
    
    if max_val > 0:
        return intensities / max_val
    else:
        return intensities


def find_peak_position(wavenumbers, intensities, target_wavenumber, window=20):
    """
    Find the peak position near a target wavenumber.
    
    Parameters:
    -----------
    wavenumbers : array
        Wavenumber values
    intensities : array
        Intensity values
    target_wavenumber : float
        Target wavenumber to search around
    window : float
        Search window around target (cm^-1)
    
    Returns:
    --------
    peak_pos : float or None
        Peak position if found, None otherwise
    peak_intensity : float or None
        Peak intensity if found, None otherwise
    """
    # Find indices within window
    mask = (wavenumbers >= target_wavenumber - window) & (wavenumbers <= target_wavenumber + window)
    
    if not np.any(mask):
        return None, None
    
    # Find local maximum in this region
    local_wavenumbers = wavenumbers[mask]
    local_intensities = intensities[mask]
    
    # Find peaks in this region
    peaks, properties = find_peaks(local_intensities, prominence=np.max(local_intensities) * 0.1)
    
    if len(peaks) > 0:
        # Find peak closest to target
        peak_indices = peaks
        peak_positions = local_wavenumbers[peak_indices]
        distances = np.abs(peak_positions - target_wavenumber)
        closest_idx = np.argmin(distances)
        
        peak_pos = peak_positions[closest_idx]
        peak_idx = peak_indices[closest_idx]
        peak_intensity = local_intensities[peak_idx]
        
        return peak_pos, peak_intensity
    else:
        # If no peak found, return maximum in region
        max_idx = np.argmax(local_intensities)
        return local_wavenumbers[max_idx], local_intensities[max_idx]


def plot_overlaid_raman_spectra(csv_path, output_dir=None, excitation_nm=830):
    """
    Plot overlaid normalized Raman spectra with peak annotations.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to CSV file with Raman data
    output_dir : str or Path, optional
        Output directory for saving plot. If None, uses 'Data for publication' folder.
    excitation_nm : float
        Excitation wavelength in nm (default: 830)
    """
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extract data columns
    # Expected columns: wavenumber_petiole, petiole, wavenumber_IAA_sensor, IAA_nanosensor,
    #                  wavenumber_IAA_patch, IAA_patch
    wavenumber_petiole = df['wavenumber_petiole'].values
    petiole = df['petiole'].values
    wavenumber_IAA_sensor = df['wavenumber_IAA_sensor'].values
    IAA_nanosensor = df['IAA_nanosensor'].values
    wavenumber_IAA_patch = df['wavenumber_IAA_patch'].values
    IAA_patch = df['IAA_patch'].values
    
    # Remove NaN values
    valid_petiole = ~np.isnan(petiole)
    valid_sensor = ~np.isnan(IAA_nanosensor)
    valid_patch = ~np.isnan(IAA_patch)
    
    wavenumber_petiole = wavenumber_petiole[valid_petiole]
    petiole = petiole[valid_petiole]
    wavenumber_IAA_sensor = wavenumber_IAA_sensor[valid_sensor]
    IAA_nanosensor = IAA_nanosensor[valid_sensor]
    wavenumber_IAA_patch = wavenumber_IAA_patch[valid_patch]
    IAA_patch = IAA_patch[valid_patch]
    
    # Filter wavenumbers below 250 cm^-1
    print("Filtering wavenumbers below 250 cm^-1...")
    min_wavenumber = 250
    
    mask_petiole = wavenumber_petiole >= min_wavenumber
    mask_sensor = wavenumber_IAA_sensor >= min_wavenumber
    mask_patch = wavenumber_IAA_patch >= min_wavenumber
    
    wavenumber_petiole = wavenumber_petiole[mask_petiole]
    petiole = petiole[mask_petiole]
    wavenumber_IAA_sensor = wavenumber_IAA_sensor[mask_sensor]
    IAA_nanosensor = IAA_nanosensor[mask_sensor]
    wavenumber_IAA_patch = wavenumber_IAA_patch[mask_patch]
    IAA_patch = IAA_patch[mask_patch]
    
    # Apply Savitzky-Golay smoothing
    print("Applying Savitzky-Golay smoothing...")
    window_size = 25  # Window size (must be odd)
    poly_order_sg = 2  # Polynomial order
    
    # Ensure window_size is odd and less than data length
    def apply_savgol(data, window, poly_order):
        if len(data) < window:
            window = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window < 3:
            return data  # Too few points, return original
        return savgol_filter(data, window, poly_order)
    
    petiole_smooth = apply_savgol(petiole, window_size, poly_order_sg)
    IAA_nanosensor_smooth = apply_savgol(IAA_nanosensor, window_size, poly_order_sg)
    IAA_patch_smooth = apply_savgol(IAA_patch, window_size, poly_order_sg)
    
    # Normalize spectra to maximum value first
    print("Normalizing spectra to maximum value...")
    petiole_norm_temp = normalize_spectrum_to_max(petiole_smooth)
    IAA_nanosensor_norm_temp = normalize_spectrum_to_max(IAA_nanosensor_smooth)
    IAA_patch_norm_temp = normalize_spectrum_to_max(IAA_patch_smooth)
    
    # Calculate baseline average in 250-1000 cm^-1 range for normalized spectra
    print("Calculating baseline averages (250-1000 cm^-1) on normalized spectra...")
    baseline_range_mask_petiole = (wavenumber_petiole >= 250) & (wavenumber_petiole <= 1000)
    baseline_range_mask_sensor = (wavenumber_IAA_sensor >= 250) & (wavenumber_IAA_sensor <= 1000)
    baseline_range_mask_patch = (wavenumber_IAA_patch >= 250) & (wavenumber_IAA_patch <= 1000)
    
    baseline_avg_petiole = np.mean(petiole_norm_temp[baseline_range_mask_petiole])
    baseline_avg_sensor = np.mean(IAA_nanosensor_norm_temp[baseline_range_mask_sensor])
    baseline_avg_patch = np.mean(IAA_patch_norm_temp[baseline_range_mask_patch])
    
    print(f"  Petiole baseline avg: {baseline_avg_petiole:.4f}")
    print(f"  IAA Nanosensor baseline avg: {baseline_avg_sensor:.4f}")
    print(f"  IAA Patch baseline avg: {baseline_avg_patch:.4f} (reference)")
    
    # Shift normalized spectra to align baselines using IAA Patch as reference
    print("Shifting normalized spectra to align baselines...")
    petiole_norm = petiole_norm_temp - baseline_avg_petiole + baseline_avg_patch
    IAA_nanosensor_norm = IAA_nanosensor_norm_temp - baseline_avg_sensor + baseline_avg_patch
    IAA_patch_norm = IAA_patch_norm_temp  # Reference, no shift needed
    
    # Find G-band peak (around 1590 cm^-1) and peak at 850 cm^-1
    print("Finding peaks...")
    
    # G-band peak (around 1590 cm^-1) - typical G-band for carbon nanotubes
    gband_target = 1590
    gband_window = 100  # Wider window to find G-band
    gband_petiole_pos, gband_petiole_int = find_peak_position(wavenumber_petiole, petiole_norm, gband_target, window=gband_window)
    gband_sensor_pos, gband_sensor_int = find_peak_position(wavenumber_IAA_sensor, IAA_nanosensor_norm, gband_target, window=gband_window)
    gband_patch_pos, gband_patch_int = find_peak_position(wavenumber_IAA_patch, IAA_patch_norm, gband_target, window=gband_window)
    
    # Print found G-band peaks for debugging
    print("G-band peaks found:")
    if gband_petiole_pos:
        print(f"  Petiole: {gband_petiole_pos:.1f} cm^-1 (intensity: {gband_petiole_int:.3f})")
    else:
        print("  Petiole: Not found")
    if gband_sensor_pos:
        print(f"  IAA Nanosensor: {gband_sensor_pos:.1f} cm^-1 (intensity: {gband_sensor_int:.3f})")
    else:
        print("  IAA Nanosensor: Not found")
    if gband_patch_pos:
        print(f"  IAA Patch: {gband_patch_pos:.1f} cm^-1 (intensity: {gband_patch_int:.3f})")
    else:
        print("  IAA Patch: Not found")
    
    # Find peak at 850 cm^-1 for IAA patch
    peak850_target = 850
    peak850_window = 50
    peak850_patch_pos, peak850_patch_int = find_peak_position(wavenumber_IAA_patch, IAA_patch_norm, peak850_target, window=peak850_window)
    
    if peak850_patch_pos:
        print(f"IAA Patch 850 cm^-1 peak: {peak850_patch_pos:.1f} cm^-1 (intensity: {peak850_patch_int:.3f})")
    else:
        print("IAA Patch 850 cm^-1 peak: Not found")
    
    # Create publication-quality plot
    print("Creating plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot normalized spectra with reduced line thickness
    ax.plot(wavenumber_petiole, petiole_norm, 'b-', label='Silk MN on petiole', linewidth=1.5, alpha=0.8)
    ax.plot(wavenumber_IAA_sensor, IAA_nanosensor_norm, 'r-', label='IAA Nanosensor', linewidth=1.5, alpha=0.8)
    ax.plot(wavenumber_IAA_patch, IAA_patch_norm, 'g-', label='IAA Sensor Patch', linewidth=1.5, alpha=0.8)
    
    # Add vertical dashed lines and annotations for G-band peak on IAA patch
    if gband_patch_pos is not None:
        ax.axvline(x=gband_patch_pos, color='green', linestyle='--', linewidth=1.5, alpha=0.7, zorder=4)
        ax.annotate(f'G-band\n{gband_patch_pos:.0f} cm⁻¹', 
                   xy=(gband_patch_pos, gband_patch_int),
                   xytext=(10, 20), textcoords='offset points',
                   fontsize=10, ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Add vertical dashed lines and annotations for 850 cm^-1 peak on IAA patch
    if peak850_patch_pos is not None:
        ax.axvline(x=peak850_patch_pos, color='green', linestyle='--', linewidth=1.5, alpha=0.7, zorder=4)
        ax.annotate(f'{peak850_patch_pos:.0f} cm⁻¹', 
                   xy=(peak850_patch_pos, peak850_patch_int),
                   xytext=(10, -30), textcoords='offset points',
                   fontsize=9, ha='left', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='black'))
    
    # Formatting for publication quality
    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Intensity', fontsize=14, fontweight='bold')
    ax.set_title('Normalized Raman Spectra Comparison\nPetiole, IAA Nanosensor, and IAA Patch', 
                fontsize=16, fontweight='bold', pad=15)
    
    ax.legend(loc='best', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(left=250)  # Start from 250 cm^-1 (filtered data)
    ax.set_ylim(bottom=0)
    
    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    
    # Tight layout
    plt.tight_layout()
    
    # Determine output directory
    if output_dir is None:
        # Use same directory as input CSV file
        output_dir = Path(csv_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    output_path = output_dir / 'Raman_p830_representative_overlay.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # Also save as PDF for publication
    output_path_pdf = output_dir / 'Raman_p830_representative_overlay.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path_pdf}")
    
    plt.close()
    
    print("Plotting complete!")


def main():
    """Main function."""
    # Determine paths - go up from scripts/ to swnt_iaa_analysis_v2/ to Script/ to workspace root
    script_dir = Path(__file__).parent  # scripts/
    project_dir = script_dir.parent  # swnt_iaa_analysis_v2/
    script_parent = project_dir.parent  # Script/
    workspace_root = script_parent.parent  # workspace root
    
    csv_path = workspace_root / 'Data for publication' / 'Raman spectroscopy of IAA-MN' / 'Raman_p830_representative.csv'
    
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print("Please ensure the CSV file exists at the expected location.")
        return
    
    # Create plot
    plot_overlaid_raman_spectra(csv_path)


if __name__ == '__main__':
    main()

