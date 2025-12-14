import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
import os

# Read both CSV files
batch_csv = r"IAA Nanosensor Experiment\Bok Choy\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 1\Raw data\test_outputs_batch_20251122_162854\batch_results_summary.csv"
main_csv = r"IAA Nanosensor Experiment\Bok Choy\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 1\Raw data\processed_results\raman_analysis_results_emission_nm_2025-11-21.csv"

print("Loading CSV files...")
batch_df = pd.read_csv(batch_csv, parse_dates=['Datetime'], index_col='Datetime')
main_df = pd.read_csv(main_csv, parse_dates=['Datetime'], index_col='Datetime')

# Sort by datetime
batch_df = batch_df.sort_index()
main_df = main_df.sort_index()

# Create output directory
output_dir = r"IAA Nanosensor Experiment\Bok Choy\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 1\Raw data\test_outputs_batch_20251122_162854"
os.makedirs(output_dir, exist_ok=True)

def normalize_series(series):
    """Normalize a series to 0-1 range for overlay plotting"""
    if series.isna().all() or series.std() == 0:
        return series
    return (series - series.min()) / (series.max() - series.min())

def create_overlay_plot(df, title, output_path, method_name):
    """Create an overlay plot with multiple metrics"""
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    # Get available columns
    has_raman = 'Raman_peak_area' in df.columns
    has_gband = 'G-band_peak_area' in df.columns or 'G-band Area' in df.columns
    
    # Plot background (always available)
    if 'Background Average (250-1250 cm^-1)' in df.columns:
        bg_col = 'Background Average (250-1250 cm^-1)'
    elif 'Average Background Intensity' in df.columns:
        bg_col = 'Average Background Intensity'
    else:
        bg_col = None
    
    if bg_col:
        bg_normalized = normalize_series(df[bg_col])
        ax1.plot(df.index, bg_normalized, 'o-', color='orange', label='Background (normalized)', 
                alpha=0.7, markersize=2, linewidth=1)
    
    # Plot fluorescence
    if 'Fluorescence (Method 1)' in df.columns:
        fluo_col = 'Fluorescence (Method 1)'
        fluo_normalized = normalize_series(df[fluo_col])
        ax1.plot(df.index, fluo_normalized, 'r-', label='Fluorescence Method 1 (normalized)', 
                alpha=0.8, linewidth=1.5)
    elif 'Fluorescence (Method 2)' in df.columns:
        fluo_col = 'Fluorescence (Method 2)'
        fluo_normalized = normalize_series(df[fluo_col])
        ax1.plot(df.index, fluo_normalized, 'b-', label='Fluorescence Method 2 (normalized)', 
                alpha=0.8, linewidth=1.5)
    elif 'Final SWNT Fluorescence' in df.columns:
        fluo_col = 'Final SWNT Fluorescence'
        fluo_normalized = normalize_series(df[fluo_col])
        ax1.plot(df.index, fluo_normalized, 'g-', label='Final SWNT Fluorescence (normalized)', 
                alpha=0.8, linewidth=1.5)
    
    # Plot Raman peak area if available
    if has_raman:
        raman_col = 'Raman_peak_area'
        if raman_col in df.columns and not df[raman_col].isna().all():
            raman_normalized = normalize_series(df[raman_col])
            ax1.plot(df.index, raman_normalized, 'm-', label='Raman Peak Area (normalized)', 
                    alpha=0.7, linewidth=1.5)
    
    # Plot G-band area if available
    if has_gband:
        if 'G-band_peak_area' in df.columns:
            gband_col = 'G-band_peak_area'
        elif 'G-band Area' in df.columns:
            gband_col = 'G-band Area'
        else:
            gband_col = None
        
        if gband_col and gband_col in df.columns and not df[gband_col].isna().all():
            gband_normalized = normalize_series(df[gband_col])
            ax1.plot(df.index, gband_normalized, 'c-', label='G-band Area (normalized)', 
                    alpha=0.7, linewidth=1.5)
    
    ax1.set_xlabel('Datetime', fontsize=12)
    ax1.set_ylabel('Normalized Value (0-1)', fontsize=12)
    ax1.set_title(f'{title}\n{method_name}', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(DayLocator(interval=1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

# Create plots for each method
print("\nCreating overlay plots...")

# Method 1 (from batch results)
if 'Fluorescence (Method 1)' in batch_df.columns:
    create_overlay_plot(
        batch_df,
        'Method 1: Lieberfit Baseline AUC',
        os.path.join(output_dir, 'overlay_method1_all_metrics.png'),
        'Background, Fluorescence Method 1, Raman Peak Area, G-band Area'
    )

# Method 2 (from batch results)
if 'Fluorescence (Method 2)' in batch_df.columns:
    create_overlay_plot(
        batch_df,
        'Method 2: Raw - G-band - Background',
        os.path.join(output_dir, 'overlay_method2_all_metrics.png'),
        'Background, Fluorescence Method 2, Raman Peak Area, G-band Area'
    )

# Main.py method
if 'Final SWNT Fluorescence' in main_df.columns:
    create_overlay_plot(
        main_df,
        'Main.py Method',
        os.path.join(output_dir, 'overlay_main_method_all_metrics.png'),
        'Background, Final SWNT Fluorescence, G-band Area'
    )

# Also create a combined comparison plot with all three fluorescence methods
print("\nCreating combined comparison plot...")
fig, ax = plt.subplots(figsize=(18, 10))

# Normalize and plot all fluorescence methods
if 'Fluorescence (Method 1)' in batch_df.columns:
    fluo_m1_norm = normalize_series(batch_df['Fluorescence (Method 1)'])
    ax.plot(batch_df.index, fluo_m1_norm, 'r-', label='Method 1: Lieberfit Baseline AUC', 
            alpha=0.8, linewidth=2)

if 'Fluorescence (Method 2)' in batch_df.columns:
    fluo_m2_norm = normalize_series(batch_df['Fluorescence (Method 2)'])
    ax.plot(batch_df.index, fluo_m2_norm, 'b-', label='Method 2: Raw - G-band - Background', 
            alpha=0.8, linewidth=2)

if 'Final SWNT Fluorescence' in main_df.columns:
    fluo_main_norm = normalize_series(main_df['Final SWNT Fluorescence'])
    ax.plot(main_df.index, fluo_main_norm, 'g-', label='Main.py: Final SWNT Fluorescence', 
            alpha=0.8, linewidth=2)

# Plot background (use batch_df background)
if 'Background Average (250-1250 cm^-1)' in batch_df.columns:
    bg_norm = normalize_series(batch_df['Background Average (250-1250 cm^-1)'])
    ax.plot(batch_df.index, bg_norm, 'o-', color='orange', label='Background (normalized)', 
            alpha=0.6, markersize=2, linewidth=1)

# Plot G-band if available
if 'G-band_peak_area' in batch_df.columns:
    gband_norm = normalize_series(batch_df['G-band_peak_area'])
    ax.plot(batch_df.index, gband_norm, 'c--', label='G-band Area (normalized)', 
            alpha=0.6, linewidth=1.5)

# Plot Raman peak if available
if 'Raman_peak_area' in batch_df.columns:
    raman_norm = normalize_series(batch_df['Raman_peak_area'])
    ax.plot(batch_df.index, raman_norm, 'm--', label='Raman Peak Area (normalized)', 
            alpha=0.6, linewidth=1.5)

ax.set_xlabel('Datetime', fontsize=14)
ax.set_ylabel('Normalized Value (0-1)', fontsize=14)
ax.set_title('Comparison: All Methods Overlay\nBackground, Fluorescence Methods, Raman Peak, G-band', 
             fontsize=16, fontweight='bold')
ax.legend(loc='best', fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))
ax.xaxis.set_major_locator(DayLocator(interval=1))
plt.xticks(rotation=45)
plt.tight_layout()

combined_path = os.path.join(output_dir, 'overlay_all_methods_comparison.png')
plt.savefig(combined_path, dpi=300, bbox_inches='tight')
print(f"Saved: {combined_path}")
plt.close()

print("\nAll overlay plots created successfully!")

