"""
Test script to compare Raman spectra before and after correction.
Tests single scan processing from Bok Choy Control sample.
"""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add src to path (go up two levels from scripts/ to project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline.ingestion import load_raman_dataset
from pipeline.processing import process_raman_data
from pipeline.config_loader import load_profile_config
from pipeline.utils import create_dir_if_needed, emission_nm_to_raman_wavenumber

def test_single_scan():
    """Test single scan processing and visualize before/after correction."""
    
    # Load configuration
    config_path = PROJECT_ROOT / "config" / "pipeline.yml"
    config = load_profile_config(str(config_path), "bok_choy_control_6to22_run1")
    
    # Create output directory for test results
    output_dir = PROJECT_ROOT / "scripts" / "test_outputs"
    create_dir_if_needed(str(output_dir))
    
    print("=== Single Scan Processing Test ===\n")
    print(f"Config loaded: bok_choy_control_6to22_run1")
    print(f"Excitation wavelength: {config.get('excitation_nm', 830)} nm")
    print(f"Window size: {config.get('window_size', 25)}")
    print(f"Poly order: {config.get('poly_order', 5)}")
    print(f"Lieberfit iterations: {config.get('tot_iter', 100)}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_raman_dataset(config)
    raman_df = dataset.spectra
    
    print(f"Loaded {len(raman_df)} scans")
    print(f"Wavenumber range: {raman_df.columns[2]} to {raman_df.columns[-1]}")
    
    # Select a scan number (try scan 100, or first available)
    available_scans = sorted(raman_df['Scan Number'].unique())
    scan_number = 100 if 100 in available_scans else available_scans[len(available_scans) // 2]
    
    print(f"\nProcessing Scan Number: {scan_number}")
    
    # Process the scan
    print("Processing scan...")
    result = process_raman_data(raman_df, scan_number, config)
    
    if result is None:
        print(f"ERROR: Scan {scan_number} not found!")
        return
    
    wavenumbers, emission_nm, intensities_raw, intensities_smooth, bg_intensity, intensities_bg_corrected, baseline_gband, corrected_spectrum_gband, datetime_val, seconds = result
    
    print("Processing complete!\n")
    print(f"Background intensity: {bg_intensity:.2f}\n")
    
    # Create comparison plots - expanded to show background correction
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Scan {scan_number} - Processing Steps Comparison\n'
                 f'DateTime: {datetime_val}, Seconds: {seconds:.1f}', 
                 fontsize=14, fontweight='bold')
    
    # Get background and G-band masks
    bg_emission_min = config.get('bg_emission_min', 850)
    bg_emission_max = config.get('bg_emission_max', 925)
    bg_wavenumber_min = emission_nm_to_raman_wavenumber(bg_emission_min, config.get('excitation_nm', 830))
    bg_wavenumber_max = emission_nm_to_raman_wavenumber(bg_emission_max, config.get('excitation_nm', 830))
    bg_wavenumber_range = (wavenumbers >= bg_wavenumber_min) & (wavenumbers <= bg_wavenumber_max)
    gband_mask = (wavenumbers >= 1500) & (wavenumbers <= 2100)
    gband_emission_mask = (emission_nm >= 952) & (emission_nm <= 958)
    
    # Plot 1: Wavenumber view - Raw and Smoothed
    ax1 = axes[0, 0]
    ax1.plot(wavenumbers, intensities_raw, 'b-', label='Raw Spectrum', alpha=0.6, linewidth=1)
    ax1.plot(wavenumbers, intensities_smooth, 'r-', label='Smoothed (Savitzky-Golay)', alpha=0.8, linewidth=1.5)
    ax1.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title('Step 1: Raw vs Smoothed', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(250, 2100)
    
    # Plot 2: Wavenumber view - Background Correction
    ax2 = axes[0, 1]
    ax2.plot(wavenumbers, intensities_smooth, 'gray', label='Smoothed Spectrum', alpha=0.6, linewidth=1)
    ax2.axhline(y=bg_intensity, color='orange', linestyle='--', linewidth=2, label=f'Background ({bg_intensity:.1f})')
    ax2.plot(wavenumbers, intensities_bg_corrected, 'purple', label='Background-Corrected', linewidth=1.5)
    ax2.fill_between(wavenumbers[bg_wavenumber_range], intensities_smooth[bg_wavenumber_range],
                     alpha=0.2, color='orange', label='Background Region')
    ax2.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.set_title('Step 2: Background Correction', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(250, 2100)
    
    # Plot 3: Wavenumber view - G-band Baseline Correction
    ax3 = axes[0, 2]
    ax3.plot(wavenumbers, intensities_bg_corrected, 'gray', label='Background-Corrected', alpha=0.6, linewidth=1)
    ax3.plot(wavenumbers, baseline_gband, 'r--', label='G-band Baseline', linewidth=2)
    ax3.plot(wavenumbers, corrected_spectrum_gband, 'g-', label='Final Corrected', linewidth=1.5)
    ax3.fill_between(wavenumbers[gband_mask], intensities_bg_corrected[gband_mask], 
                     baseline_gband[gband_mask], alpha=0.2, color='red', label='G-band Region')
    ax3.set_xlabel('Wavenumber (cm^-1)', fontsize=12)
    ax3.set_ylabel('Intensity', fontsize=12)
    ax3.set_title('Step 3: G-band Baseline Correction', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(250, 2100)
    
    # Plot 4: Emission wavelength view - Raw and Smoothed
    ax4 = axes[1, 0]
    ax4.plot(emission_nm, intensities_raw, 'b-', label='Raw Spectrum', alpha=0.6, linewidth=1)
    ax4.plot(emission_nm, intensities_smooth, 'r-', label='Smoothed (Savitzky-Golay)', alpha=0.8, linewidth=1.5)
    ax4.set_xlabel('Emission Wavelength (nm)', fontsize=12)
    ax4.set_ylabel('Intensity', fontsize=12)
    ax4.set_title('Step 1: Raw vs Smoothed', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Emission wavelength view - Background Correction
    ax5 = axes[1, 1]
    ax5.plot(emission_nm, intensities_smooth, 'gray', label='Smoothed Spectrum', alpha=0.6, linewidth=1)
    ax5.axhline(y=bg_intensity, color='orange', linestyle='--', linewidth=2, label=f'Background ({bg_intensity:.1f})')
    ax5.plot(emission_nm, intensities_bg_corrected, 'purple', label='Background-Corrected', linewidth=1.5)
    bg_emission_range = (emission_nm >= bg_emission_min) & (emission_nm <= bg_emission_max)
    ax5.fill_between(emission_nm[bg_emission_range], intensities_smooth[bg_emission_range],
                     alpha=0.2, color='orange', label='Background Region')
    ax5.set_xlabel('Emission Wavelength (nm)', fontsize=12)
    ax5.set_ylabel('Intensity', fontsize=12)
    ax5.set_title('Step 2: Background Correction', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Emission wavelength view - G-band Baseline Correction
    ax6 = axes[1, 2]
    ax6.plot(emission_nm, intensities_bg_corrected, 'gray', label='Background-Corrected', alpha=0.6, linewidth=1)
    ax6.plot(emission_nm, baseline_gband, 'r--', label='G-band Baseline', linewidth=2)
    ax6.plot(emission_nm, corrected_spectrum_gband, 'g-', label='Final Corrected', linewidth=1.5)
    ax6.fill_between(emission_nm[gband_emission_mask], intensities_bg_corrected[gband_emission_mask],
                     baseline_gband[gband_emission_mask], alpha=0.2, color='red', label='G-band Region')
    ax6.set_xlabel('Emission Wavelength (nm)', fontsize=12)
    ax6.set_ylabel('Intensity', fontsize=12)
    ax6.set_title('Step 3: G-band Baseline Correction', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot to organized output directory
    output_path = output_dir / f"test_scan_{scan_number}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to: {output_path}")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Raw spectrum range: [{intensities_raw.min():.1f}, {intensities_raw.max():.1f}]")
    print(f"Smoothed spectrum range: [{intensities_smooth.min():.1f}, {intensities_smooth.max():.1f}]")
    print(f"Background intensity: {bg_intensity:.1f}")
    print(f"Background-corrected range: [{intensities_bg_corrected.min():.1f}, {intensities_bg_corrected.max():.1f}]")
    print(f"G-band baseline range: [{baseline_gband[gband_mask].min():.1f}, {baseline_gband[gband_mask].max():.1f}]")
    print(f"Final corrected range: [{corrected_spectrum_gband[gband_mask].min():.1f}, {corrected_spectrum_gband[gband_mask].max():.1f}]")
    
    # G-band statistics
    gband_raw_max = intensities_raw[gband_mask].max()
    gband_smooth_max = intensities_smooth[gband_mask].max()
    gband_bg_corrected_max = intensities_bg_corrected[gband_mask].max()
    gband_final_corrected_max = corrected_spectrum_gband[gband_mask].max()
    
    print(f"\nG-band Peak Heights (1500-2100 cm^-1):")
    print(f"  Raw: {gband_raw_max:.1f}")
    print(f"  Smoothed: {gband_smooth_max:.1f}")
    print(f"  Background-corrected: {gband_bg_corrected_max:.1f}")
    print(f"  Final (G-band baseline-corrected): {gband_final_corrected_max:.1f}")
    
    # Show plot
    plt.show()
    
    print("\n[OK] Test completed successfully!")

if __name__ == "__main__":
    test_single_scan()

