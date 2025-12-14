import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
from matplotlib.ticker import FuncFormatter, MaxNLocator
from datetime import datetime
from .utils import add_day_night_shading, create_dir_if_needed, parse_light_transition_config, parse_shade_transition_config, parse_treatment_events_config
from .processing import lorentzian

def plot_raman_spectrum(wavenumbers, emission_nm, intensities_raw, intensities_smooth, baseline_gband,
                       corrected_spectrum_gband, scan_number, datetime_val, seconds, processed_dir, config):
    """
    Plot Raman spectrum in wavenumber and emission wavelength domains for a single scan.
    Includes plots showing: Raw vs Smoothed, and Raw vs Final Corrected spectrum.
    """
    gband_emission_min = config.get('gband_emission_min', 952)
    gband_emission_max = config.get('gband_emission_max', 965)
    gband_emission_window = (emission_nm >= gband_emission_min) & (emission_nm <= gband_emission_max)
    gband_mask = (wavenumbers >= 1500) & (wavenumbers <= 2100)
    create_dir_if_needed(processed_dir)
    
    # Plot 1: Wavenumber view - Raw and Smoothed
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavenumbers, intensities_raw, 'b-', label=f'Raw Scan {scan_number}', alpha=0.5)
    ax.plot(wavenumbers, intensities_smooth, 'r-', label=f'Smoothed Scan {scan_number}', alpha=0.7)
    ax.set_xlabel('Wavenumber (cm^-1)', fontsize=14)
    ax.set_ylabel('Intensity', fontsize=14)
    ax.set_title(f'Scan {scan_number} - Raw vs Smoothed', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.set_xlim(250, 2000)
    ax.tick_params(axis='both', labelsize=12)
    save_path = os.path.join(processed_dir, f'Raman_Scan_{scan_number}_Wavenumber_Lieberfit_Gband_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show()
    
    # Plot 2: Emission wavelength view - Raw and Smoothed with G-band area
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(emission_nm, intensities_raw, 'b-', label=f'Raw Scan {scan_number}', alpha=0.5)
    ax.plot(emission_nm, intensities_smooth, 'r-', label=f'Smoothed Scan {scan_number}', alpha=0.7)
    ax.fill_between(emission_nm[gband_emission_window], corrected_spectrum_gband[gband_emission_window],
                    color='green', alpha=0.3, label='G-band Area (Corrected)')
    ax.set_xlabel('Emission Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Intensity', fontsize=14)
    ax.set_title(f'Scan {scan_number} - Raw vs Smoothed (G-band highlighted)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=12)
    save_path = os.path.join(processed_dir, f'Raman_Scan_{scan_number}_Emission_nm_Lieberfit_Gband_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show()
    
    # Plot 3: Raw vs Final Corrected - Wavenumber view
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wavenumbers, intensities_raw, 'b-', label='Raw Spectrum', alpha=0.6, linewidth=1.5)
    ax.plot(wavenumbers, corrected_spectrum_gband, 'g-', label='Final Corrected Spectrum', alpha=0.8, linewidth=1.5)
    
    # Highlight G-band region
    ax.fill_between(wavenumbers[gband_mask], intensities_raw[gband_mask], 
                     corrected_spectrum_gband[gband_mask], alpha=0.2, color='orange', 
                     label='G-band Region (1500-2100 cm^-1)')
    
    ax.set_xlabel('Wavenumber (cm^-1)', fontsize=14)
    ax.set_ylabel('Intensity', fontsize=14)
    ax.set_title(f'Scan {scan_number} - Raw vs Final Corrected (Wavenumber)\nDateTime: {datetime_val}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(250, 2100)
    ax.tick_params(axis='both', labelsize=12)
    save_path = os.path.join(processed_dir, f'Raman_Scan_{scan_number}_Raw_vs_FinalCorrected_Wavenumber_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show()
    
    # Plot 4: Raw vs Final Corrected - Emission wavelength view
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(emission_nm, intensities_raw, 'b-', label='Raw Spectrum', alpha=0.6, linewidth=1.5)
    ax.plot(emission_nm, corrected_spectrum_gband, 'g-', label='Final Corrected Spectrum', alpha=0.8, linewidth=1.5)
    
    # Highlight G-band region in emission wavelength
    ax.fill_between(emission_nm[gband_emission_window], intensities_raw[gband_emission_window],
                     corrected_spectrum_gband[gband_emission_window], alpha=0.2, color='orange',
                     label='G-band Region (~952-958 nm)')
    
    ax.set_xlabel('Emission Wavelength (nm)', fontsize=14)
    ax.set_ylabel('Intensity', fontsize=14)
    ax.set_title(f'Scan {scan_number} - Raw vs Final Corrected (Emission Wavelength)\nDateTime: {datetime_val}', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    save_path = os.path.join(processed_dir, f'Raman_Scan_{scan_number}_Raw_vs_FinalCorrected_Emission_nm_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show()

def plot_time_series(df, column, label, color, processed_dir, filename, config, light_cycle, shading=True, smooth_sigma=None):
    """
    Plot a time-series metric with optional shading and Gaussian smoothing.
    
    Parameters:
    df : pandas.DataFrame
        DataFrame with time index and metric columns.
    column : str
        Column name to plot.
    label : str
        Label for the plot line.
    color : str
        Color for the main plot line (e.g., 'b' for blue).
    processed_dir : str
        Output directory for saving the plot.
    filename : str
        Base filename for the saved plot (without extension).
    config : dict
        Configuration dictionary (e.g., for sigma_gaussian).
    light_cycle : str
        Light cycle for shading ('Constant', '8to24', '6to22').
    shading : bool, optional
        Whether to add day/night shading (default: True).
    smooth_sigma : float, optional
        Standard deviation for Gaussian smoothing.
    """
    create_dir_if_needed(processed_dir)
    fig, ax = plt.subplots(figsize=(8, 12))
    # Use color parameter directly to support both single-character codes and named colors
    ax.plot(df.index, df[column], '.-', color=color, label=label)
    if smooth_sigma is not None:
        smoothed_values = gaussian_filter1d(df[column].values, sigma=smooth_sigma)
        ax.plot(df.index, smoothed_values, 'k--', label=f'Smoothed {label}')
    if shading:
        add_day_night_shading(ax, df.index.min(), df.index.max(), light_cycle=light_cycle)
    ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
    ax.set_ylabel(label, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=12)
    ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
    ax.xaxis.set_major_locator(DayLocator())
    fig.autofmt_xdate()
    save_path = os.path.join(processed_dir, f'{filename.split(".")[0]}_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show()

def plot_stacked_metrics(plot_df, plot_df_temp, processed_dir, config, light_cycle):
    """
    Create a stacked plot of Fluorescence to G-band Ratio, Temperature, and Relative Humidity.
    
    Parameters:
    plot_df : pandas.DataFrame
        DataFrame with Raman metrics.
    plot_df_temp : pandas.DataFrame
        DataFrame with temperature and humidity data.
    processed_dir : str
        Output directory for saving the plot.
    config : dict
        Configuration dictionary (e.g., 'sigma_gaussian_stacked').
    light_cycle : str
        Light cycle for shading ('Constant', '8to24', '6to22').
    """
    create_dir_if_needed(processed_dir)
    sigma = config.get('sigma_gaussian_stacked', 20)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12), sharex=True, gridspec_kw={'hspace': 0})
    ax1.plot(plot_df.index, plot_df['Fluorescence to G-band Ratio'], 'm.-', label='Fluorescence to G-band Ratio (Raw)')
    fluo_to_gband_smooth = gaussian_filter1d(plot_df['Fluorescence to G-band Ratio'].values, sigma=sigma)
    ax1.plot(plot_df.index, fluo_to_gband_smooth, 'k--', label='Smoothed')
    add_day_night_shading(ax1, plot_df.index.min(), plot_df.index.max(), light_cycle=light_cycle)
    ax1.set_ylabel('Fluorescence to G-band Ratio', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True)
    ax1.tick_params(axis='both', labelsize=12)
    ax2.plot(plot_df_temp.index, plot_df_temp['Temperature'], 'b.-', label='Temperature (°C)')
    add_day_night_shading(ax2, plot_df.index.min(), plot_df.index.max(), light_cycle=light_cycle)
    ax2.set_ylabel('Temperature (°C)', fontsize=14)
    ax2.legend(fontsize=12)
    ax2.grid(True)
    ax2.tick_params(axis='both', labelsize=12)
    ax3.plot(plot_df_temp.index, plot_df_temp['Relative_Humidity'], 'r.-', label='Relative Humidity (%)')
    add_day_night_shading(ax3, plot_df.index.min(), plot_df.index.max(), light_cycle=light_cycle)
    ax3.set_xlabel('Date time (MM-DD HH)', fontsize=14)
    ax3.set_ylabel('Relative Humidity (%)', fontsize=14)
    ax3.legend(fontsize=12)
    ax3.grid(True)
    ax3.tick_params(axis='both', labelsize=12)
    ax3.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
    ax3.xaxis.set_major_locator(DayLocator())
    ax3.xaxis.set_minor_locator(HourLocator(interval=6))
    fig.autofmt_xdate()
    save_path = os.path.join(processed_dir, f'Stacked_Metrics_over_time_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show()

def plot_extrema(results_df, processed_dir, config):
    """
    Plot the fluorescence-to-G-band ratio with Gaussian smoothing and local extrema marked.
    """
    from scipy.signal import find_peaks
    create_dir_if_needed(processed_dir)
    sigma = config.get('sigma_gaussian_extrema', 100.0)
    peak_distance = config.get('peak_distance', 100)
    ratio = results_df['Fluorescence to G-band Ratio'].values
    datetimes = results_df.index
    ratio_smoothed = gaussian_filter1d(ratio, sigma=sigma)
    peaks, _ = find_peaks(ratio_smoothed, distance=peak_distance)
    valleys, _ = find_peaks(-ratio_smoothed, distance=peak_distance)
    extrema_data = []
    for idx in peaks:
        extrema_data.append({
            'Type': 'Maxima',
            'Datetime': datetimes[idx],
            'Hour': datetimes[idx].strftime('%H:%M'),
            'Fluorescence to G-band Ratio': ratio_smoothed[idx]
        })
    for idx in valleys:
        extrema_data.append({
            'Type': 'Minima',
            'Datetime': datetimes[idx],
            'Hour': datetimes[idx].strftime('%H:%M'),
            'Fluorescence to G-band Ratio': ratio_smoothed[idx]
        })
    extrema_df = pd.DataFrame(extrema_data)
    extrema_df.sort_values('Datetime', inplace=True)
    output_file = os.path.join(processed_dir, f'fluo_gband_ratio_extrema_{datetime.now().strftime("%Y-%m-%d")}.csv')
    extrema_df.to_csv(output_file, index=False)
    print(f"Extrema results saved to {output_file}")
    print("\nFirst 5 rows of extrema:")
    print(extrema_df.head())
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(datetimes, ratio, 'm', label='Original PL/G')
    ax.plot(datetimes, ratio_smoothed, 'b-')
    ax.plot(datetimes[peaks], ratio_smoothed[peaks], 'ro', label='Maxima', markersize=8)
    ax.plot(datetimes[valleys], ratio_smoothed[valleys], 'go', label='Minima', markersize=8)
    ax.set_xlabel('Date Time (MM-DD HH)', fontsize=14)
    ax.set_ylabel('Fluorescence to G-band Ratio', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=12)
    ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
    ax.xaxis.set_major_locator(HourLocator(interval=24))
    fig.autofmt_xdate()
    save_path = os.path.join(processed_dir, f'gaussian_smoothed_fluo_gband_ratio_extrema_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show()

def plot_combined_time_series(df, processed_dir, config, light_cycle):
    """
    Generate a single stacked figure with time-series plots for key metrics.
    
    Parameters:
    df : pandas.DataFrame
        Filtered DataFrame with time index and metric columns.
    processed_dir : str
        Output directory for saving the plot.
    config : dict
        Configuration dictionary (e.g., 'sigma_gaussian').
    light_cycle : str
        Light cycle for shading ('Constant', '8to24', '6to22').
    """
    create_dir_if_needed(processed_dir)
    fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True, gridspec_kw={'hspace': 0.1})
    def plot_on_axis(ax, df, column, ylabel, legend_label, color, shading=True, smooth_sigma=None):
        # Use color parameter directly to support both single-character codes and named colors
        ax.plot(df.index, df[column], '.-', color=color, label=legend_label)
        if smooth_sigma is not None:
            smoothed_values = gaussian_filter1d(df[column].values, sigma=smooth_sigma)
            ax.plot(df.index, smoothed_values, 'k--', label=f'Smoothed {legend_label}')
        if shading:
            add_day_night_shading(ax, df.index.min(), df.index.max(), light_cycle=light_cycle)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=12)
    plot_on_axis(axs[0], df, 'Average Background Intensity', 'Intensity (a.u.)', 'Average Background Intensity', 'b', shading=True)
    plot_on_axis(axs[1], df, 'Final SWNT Fluorescence', 'Intensity (a.u.)', 'Background-corrected SWNT Fluorescence (PL)', 'r', shading=True)
    plot_on_axis(axs[2], df, 'G-band Height', 'Intensity (a.u.)', 'Background-corrected G-band Height (G)', 'g', shading=True)
    plot_on_axis(axs[3], df, 'Fluorescence to G-band Ratio', 'Ratio', 'PL/G ratio', 'm', shading=True, smooth_sigma=config.get('sigma_gaussian', 25))
    axs[3].set_xlabel('Date time (MM-DD HH)', fontsize=14)
    for ax in axs:
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
        ax.xaxis.set_major_locator(DayLocator())
    fig.autofmt_xdate()
    save_path = os.path.join(processed_dir, f'Combined_Time_Series_{datetime.now().strftime("%Y-%m-%d")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {save_path}")
    plt.show()
def plot_normalized_baseline_correction_comparison(df, output_dir, config, light_cycle=None, jump_info_dict=None):
    """
    Create combined plots showing normalized fluorescence intensity, G-band, and Raman peak 
    before and after baseline correction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with normalized data and corrected columns
    output_dir : Path
        Output directory
    config : dict
        Configuration dictionary
    light_cycle : str, optional
        Light cycle for shading
    jump_info_dict : dict, optional
        Dictionary mapping column names to jump information
    """
    if light_cycle is None:
        light_cycle = config.get('light_cycle', 'Constant')
    
    # Parse transition and event configs if available
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    output_dir_str = str(output_dir)
    create_dir_if_needed(output_dir_str)
    
    # Normalized data plots
    fig_norm, axes_norm = plt.subplots(3, 1, figsize=(14, 12))
    
    # Fluorescence
    if 'Normalized_Fluorescence_Intensity' in df.columns:
        # Determine what to plot as "Before Correction"
        before_col = 'Normalized_Fluorescence_Intensity_Smoothed' if ('Normalized_Fluorescence_Intensity_Smoothed' in df.columns and 
                                                               jump_info_dict and 
                                                               'Normalized_Fluorescence_Intensity' in jump_info_dict and
                                                               jump_info_dict['Normalized_Fluorescence_Intensity'].get('correct_smoothed', False)) else 'Normalized_Fluorescence_Intensity'
        
        if before_col in df.columns:
            before_label = 'Before Correction (Smoothed)' if before_col == 'Normalized_Fluorescence_Intensity_Smoothed' else 'Before Correction'
            axes_norm[0].plot(df.index, df[before_col], 'o-', 
                           label=before_label, alpha=0.7, markersize=3)
        
        if 'Normalized_Fluorescence_Intensity_BaselineCorrected' in df.columns:
            after_label = 'After Correction (Corrected Smoothed)' if (jump_info_dict and 
                                                                      'Normalized_Fluorescence_Intensity' in jump_info_dict and
                                                                      jump_info_dict['Normalized_Fluorescence_Intensity'].get('correct_smoothed', False)) else 'After Correction'
            axes_norm[0].plot(df.index, df['Normalized_Fluorescence_Intensity_BaselineCorrected'], 's-', 
                           label=after_label, alpha=0.7, markersize=3)
        
        # Mark jump points if available
        if jump_info_dict and 'Normalized_Fluorescence_Intensity' in jump_info_dict:
            jump_info = jump_info_dict['Normalized_Fluorescence_Intensity']['jump_info']
            if jump_info:
                jump_indices = [info['index'] for info in jump_info]
                jump_x = df.index[jump_indices]
                jump_y_before = df[before_col].iloc[jump_indices].values if before_col in df.columns else df['Normalized_Fluorescence_Intensity'].iloc[jump_indices].values
                jump_y_after = df['Normalized_Fluorescence_Intensity_BaselineCorrected'].iloc[jump_indices].values if 'Normalized_Fluorescence_Intensity_BaselineCorrected' in df.columns else jump_y_before
                axes_norm[0].scatter(jump_x, jump_y_before, color='red', s=150, 
                                  zorder=5, marker='x', linewidths=2, label=f'Jump points (n={len(jump_info)})')
                axes_norm[0].scatter(jump_x, jump_y_after, color='darkred', s=150, 
                                  zorder=5, marker='+', linewidths=2, alpha=0.7)
        
        axes_norm[0].set_ylabel('Normalized Fluorescence Intensity', fontsize=12)
        axes_norm[0].set_title('Normalized - Fluorescence Intensity', fontsize=13, fontweight='bold')
        axes_norm[0].legend()
        axes_norm[0].grid(True, alpha=0.3)
    
    # G-band (using Area)
    if 'Normalized_Gband_Area' in df.columns:
        # Determine what to plot as "Before Correction" (same logic as fluorescence)
        before_col = 'Normalized_Gband_Area_Smoothed' if ('Normalized_Gband_Area_Smoothed' in df.columns and 
                                                          jump_info_dict and 
                                                          'Normalized_Gband_Area' in jump_info_dict and
                                                          jump_info_dict['Normalized_Gband_Area'].get('correct_smoothed', False)) else 'Normalized_Gband_Area'
        
        if before_col in df.columns:
            before_label = 'Before Correction (Smoothed)' if before_col == 'Normalized_Gband_Area_Smoothed' else 'Before Correction'
            axes_norm[1].plot(df.index, df[before_col], 'o-', 
                           label=before_label, alpha=0.7, markersize=3)
        
        if 'Normalized_Gband_Area_BaselineCorrected' in df.columns:
            after_label = 'After Correction (Corrected Smoothed)' if (jump_info_dict and 
                                                                        'Normalized_Gband_Area' in jump_info_dict and
                                                                        jump_info_dict['Normalized_Gband_Area'].get('correct_smoothed', False)) else 'After Correction'
            axes_norm[1].plot(df.index, df['Normalized_Gband_Area_BaselineCorrected'], 's-', 
                           label=after_label, alpha=0.7, markersize=3)
        
        # Mark jump points if available
        if jump_info_dict and 'Normalized_Gband_Area' in jump_info_dict:
            jump_info = jump_info_dict['Normalized_Gband_Area']['jump_info']
            if jump_info:
                jump_indices = [info['index'] for info in jump_info]
                jump_x = df.index[jump_indices]
                jump_y_before = df[before_col].iloc[jump_indices].values if before_col in df.columns else df['Normalized_Gband_Area'].iloc[jump_indices].values
                jump_y_after = df['Normalized_Gband_Area_BaselineCorrected'].iloc[jump_indices].values if 'Normalized_Gband_Area_BaselineCorrected' in df.columns else jump_y_before
                axes_norm[1].scatter(jump_x, jump_y_before, color='red', s=150, 
                                  zorder=5, marker='x', linewidths=2, label=f'Jump points (n={len(jump_info)})')
                axes_norm[1].scatter(jump_x, jump_y_after, color='darkred', s=150, 
                                  zorder=5, marker='+', linewidths=2, alpha=0.7)
        
        axes_norm[1].set_ylabel('G-band Area (Lorentzian)', fontsize=12)
        axes_norm[1].set_title('Normalized - G-band Area', fontsize=13, fontweight='bold')
        axes_norm[1].legend()
        axes_norm[1].grid(True, alpha=0.3)
    
    # Raman peak 850 (using Area)
    if 'Normalized_Raman_Peak_850_Area' in df.columns:
        before_col = 'Normalized_Raman_Peak_850_Area_Smoothed' if ('Normalized_Raman_Peak_850_Area_Smoothed' in df.columns and 
                                                                    jump_info_dict and 
                                                                    'Normalized_Raman_Peak_850_Area' in jump_info_dict and
                                                                    jump_info_dict['Normalized_Raman_Peak_850_Area'].get('correct_smoothed', False)) else 'Normalized_Raman_Peak_850_Area'
        
        if before_col in df.columns:
            before_label = 'Before Correction (Smoothed)' if before_col == 'Normalized_Raman_Peak_850_Area_Smoothed' else 'Before Correction'
            axes_norm[2].plot(df.index, df[before_col], 'o-', 
                           label=before_label, alpha=0.7, markersize=3)
        
        if 'Normalized_Raman_Peak_850_Area_BaselineCorrected' in df.columns:
            after_label = 'After Correction (Corrected Smoothed)' if (jump_info_dict and 
                                                                      'Normalized_Raman_Peak_850_Area' in jump_info_dict and
                                                                      jump_info_dict['Normalized_Raman_Peak_850_Area'].get('correct_smoothed', False)) else 'After Correction'
            axes_norm[2].plot(df.index, df['Normalized_Raman_Peak_850_Area_BaselineCorrected'], 's-', 
                           label=after_label, alpha=0.7, markersize=3)
        
        # Mark jump points if available
        if jump_info_dict and 'Normalized_Raman_Peak_850_Area' in jump_info_dict:
            jump_info = jump_info_dict['Normalized_Raman_Peak_850_Area']['jump_info']
            if jump_info:
                jump_indices = [info['index'] for info in jump_info]
                jump_x = df.index[jump_indices]
                jump_y_before = df[before_col].iloc[jump_indices].values if before_col in df.columns else df['Normalized_Raman_Peak_850_Area'].iloc[jump_indices].values
                jump_y_after = df['Normalized_Raman_Peak_850_Area_BaselineCorrected'].iloc[jump_indices].values if 'Normalized_Raman_Peak_850_Area_BaselineCorrected' in df.columns else jump_y_before
                axes_norm[2].scatter(jump_x, jump_y_before, color='red', s=150, 
                                  zorder=5, marker='x', linewidths=2, label=f'Jump points (n={len(jump_info)})')
                axes_norm[2].scatter(jump_x, jump_y_after, color='darkred', s=150, 
                                  zorder=5, marker='+', linewidths=2, alpha=0.7)
        
        axes_norm[2].set_ylabel('Raman Peak 850 Area (Lorentzian)', fontsize=12)
        axes_norm[2].set_title('Normalized - Raman Peak 850 Area', fontsize=13, fontweight='bold')
        axes_norm[2].legend()
        axes_norm[2].grid(True, alpha=0.3)
    
    # Set x-axis formatting
    if x_axis_type == 'datetime':
        for ax in axes_norm:
            add_day_night_shading(ax, df.index.min(), df.index.max(), light_cycle=light_cycle, light_transition=light_transition)
            # Add vertical line to mark light transition if configured
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    ax.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
            # Add vertical line to mark shade transition if configured
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    label_text = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
            # Add markers for treatment events if configured
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if df.index.min() <= event_time <= df.index.max():
                        ax.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                 linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            ax.set_xlabel('Date time (MM-DD HH)', fontsize=12)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
        fig_norm.autofmt_xdate()
    else:  # scan_number
        for ax in axes_norm:
            ax.set_xlabel('Scan Number', fontsize=12)
            # Remove date formatters and use numeric formatting for scan numbers
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    
    plt.tight_layout()
    save_path_norm = Path(output_dir_str) / f'norm_baseline_comp_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(str(save_path_norm), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Normalized baseline correction comparison plot saved to: {save_path_norm}")


def plot_normalized_smoothing_comparison(df, output_dir, config, light_cycle=None, jump_info_dict=None):
    """
    Create combined plots showing normalized fluorescence intensity, G-band, and Raman peak 
    before and after smoothing (same layout as norm_baseline_comp).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with normalized data and optionally smoothed columns
    output_dir : Path
        Output directory
    config : dict
        Configuration dictionary
    light_cycle : str, optional
        Light cycle for shading
    jump_info_dict : dict, optional
        Dictionary mapping column names to jump information (used to check if smoothing was applied)
    """
    if light_cycle is None:
        light_cycle = config.get('light_cycle', 'Constant')
    
    # Parse transition and event configs if available
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    output_dir_str = str(output_dir)
    create_dir_if_needed(output_dir_str)
    
    # Get smoothing parameters (same as used in baseline correction)
    smooth_window = config.get('baseline_correction_smooth_window', 11)
    smooth_poly_order = config.get('baseline_correction_smooth_poly_order', 2)
    
    # Normalized data plots (same layout as norm_baseline_comp)
    fig_norm, axes_norm = plt.subplots(3, 1, figsize=(14, 12))
    
    # Helper function to get smoothed data (either from stored column or compute on the fly)
    def get_smoothed_signal(original_col, smoothed_col, signal_values):
        """Get smoothed signal, either from stored column or compute on the fly."""
        if smoothed_col in df.columns:
            return df[smoothed_col].values
        else:
            # Apply smoothing on the fly using Savitzky-Golay
            mask = ~np.isnan(signal_values)
            if np.sum(mask) >= smooth_window:
                smoothed = signal_values.copy()
                # Ensure window_size is odd and valid
                window_size = smooth_window if smooth_window % 2 == 1 else smooth_window - 1
                if window_size < 3:
                    window_size = 3
                if window_size >= len(signal_values[mask]):
                    window_size = len(signal_values[mask]) - 1 if len(signal_values[mask]) % 2 == 0 else len(signal_values[mask]) - 2
                if window_size < 3:
                    return signal_values  # Can't smooth, return original
                smoothed[mask] = savgol_filter(signal_values[mask], window_size, smooth_poly_order)
                return smoothed
            else:
                return signal_values
    
    # Fluorescence Intensity
    if 'Normalized_Fluorescence_Intensity' in df.columns:
        original_signal = df['Normalized_Fluorescence_Intensity'].values
        smoothed_signal = get_smoothed_signal(
            'Normalized_Fluorescence_Intensity',
            'Normalized_Fluorescence_Intensity_Smoothed',
            original_signal
        )
        
        axes_norm[0].plot(df.index, original_signal, 'o-', 
                       label='Before Smoothing (Original)', alpha=0.7, markersize=3, color='blue')
        axes_norm[0].plot(df.index, smoothed_signal, 's-', 
                       label='After Smoothing', alpha=0.7, markersize=3, color='green')
        
        axes_norm[0].set_ylabel('Normalized Fluorescence Intensity', fontsize=12)
        axes_norm[0].set_title('Normalized - Fluorescence Intensity (Smoothing Comparison)', fontsize=13, fontweight='bold')
        axes_norm[0].legend()
        axes_norm[0].grid(True, alpha=0.3)
    
    # G-band Area
    if 'Normalized_Gband_Area' in df.columns:
        original_signal = df['Normalized_Gband_Area'].values
        smoothed_signal = get_smoothed_signal(
            'Normalized_Gband_Area',
            'Normalized_Gband_Area_Smoothed',
            original_signal
        )
        
        axes_norm[1].plot(df.index, original_signal, 'o-', 
                       label='Before Smoothing (Original)', alpha=0.7, markersize=3, color='blue')
        axes_norm[1].plot(df.index, smoothed_signal, 's-', 
                       label='After Smoothing', alpha=0.7, markersize=3, color='green')
        
        axes_norm[1].set_ylabel('G-band Area (Lorentzian)', fontsize=12)
        axes_norm[1].set_title('Normalized - G-band Area (Smoothing Comparison)', fontsize=13, fontweight='bold')
        axes_norm[1].legend()
        axes_norm[1].grid(True, alpha=0.3)
    
    # Raman peak 850 Area
    if 'Normalized_Raman_Peak_850_Area' in df.columns:
        original_signal = df['Normalized_Raman_Peak_850_Area'].values
        smoothed_signal = get_smoothed_signal(
            'Normalized_Raman_Peak_850_Area',
            'Normalized_Raman_Peak_850_Area_Smoothed',
            original_signal
        )
        
        axes_norm[2].plot(df.index, original_signal, 'o-', 
                       label='Before Smoothing (Original)', alpha=0.7, markersize=3, color='blue')
        axes_norm[2].plot(df.index, smoothed_signal, 's-', 
                       label='After Smoothing', alpha=0.7, markersize=3, color='green')
        
        axes_norm[2].set_ylabel('Raman Peak 850 Area (Lorentzian)', fontsize=12)
        axes_norm[2].set_title('Normalized - Raman Peak 850 Area (Smoothing Comparison)', fontsize=13, fontweight='bold')
        axes_norm[2].legend()
        axes_norm[2].grid(True, alpha=0.3)
    
    # Set x-axis formatting (same as norm_baseline_comp)
    if x_axis_type == 'datetime':
        for ax in axes_norm:
            add_day_night_shading(ax, df.index.min(), df.index.max(), light_cycle=light_cycle, light_transition=light_transition)
            # Add vertical line to mark light transition if configured
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    ax.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
            # Add vertical line to mark shade transition if configured
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    label_text = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=label_text)
            # Add markers for treatment events if configured
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if df.index.min() <= event_time <= df.index.max():
                        ax.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                 linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            ax.set_xlabel('Date time (MM-DD HH)', fontsize=12)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
        fig_norm.autofmt_xdate()
    else:  # scan_number
        for ax in axes_norm:
            ax.set_xlabel('Scan Number', fontsize=12)
            # Remove date formatters and use numeric formatting for scan numbers
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    
    plt.tight_layout()
    save_path_norm = Path(output_dir_str) / f'norm_smoothing_comp_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(str(save_path_norm), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Normalized smoothing comparison plot saved to: {save_path_norm}")


def plot_representative_raman_spectrum(summary, output_dir, config):
    """
    Plot representative Raman spectrum from first processed scan showing:
    - Raw normalized spectrum
    - Lieberfit baseline
    - Lorentzian fitting for G-band and Raman peak at 850 cm-1
    
    Parameters:
    -----------
    summary : dict
        Summary dictionary from process_scan_v2() containing spectrum data
    output_dir : Path
        Output directory for saving plot
    config : dict
        Configuration dictionary
    """
    if summary is None:
        return
    
    output_dir_str = str(output_dir)
    create_dir_if_needed(output_dir_str)
    
    # Extract spectrum data
    wavenumbers = summary.get('wavenumbers')
    intensities_normalized_smooth = summary.get('intensities_normalized_smooth')
    baseline_normalized = summary.get('baseline_normalized')
    corrected_normalized = summary.get('corrected_normalized')
    raman_peak_850 = summary.get('raman_peak_850_normalized')
    gband_peak_1600 = summary.get('gband_peak_1600_normalized')
    scan_number = summary.get('scan_number')
    
    if wavenumbers is None or intensities_normalized_smooth is None:
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot normalized smoothed spectrum
    ax.plot(wavenumbers, intensities_normalized_smooth, '-', 
            color='black', label='Normalized Smoothed Spectrum', alpha=0.7, linewidth=1.5)
    
    # Plot Lieberfit baseline
    if baseline_normalized is not None:
        ax.plot(wavenumbers, baseline_normalized, '--', 
                color='red', label='Lieberfit Baseline', alpha=0.8, linewidth=2)
    
    # Plot corrected spectrum (after Lieberfit)
    if corrected_normalized is not None:
        ax.plot(wavenumbers, corrected_normalized, '-', 
                color='blue', label='Lieberfit-Corrected Spectrum', alpha=0.6, linewidth=1)
    
    # Plot Lorentzian fits for G-band
    if gband_peak_1600 and gband_peak_1600.get('fit_params') is not None:
        fit_params = gband_peak_1600['fit_params']
        center = gband_peak_1600.get('wavenumber', fit_params[1])
        width = gband_peak_1600.get('width', fit_params[2])
        
        # Create fine grid around peak for smooth Lorentzian curve
        peak_range_mask = (wavenumbers >= center - 3*width) & (wavenumbers <= center + 3*width)
        if np.any(peak_range_mask):
            wavenumbers_peak = wavenumbers[peak_range_mask]
            lorentzian_fit = lorentzian(wavenumbers_peak, fit_params[0], fit_params[1], fit_params[2], fit_params[3])
            ax.plot(wavenumbers_peak, lorentzian_fit, '-', 
                    color='green', label=f"G-band Lorentzian Fit (Area={gband_peak_1600.get('area', 0):.2f})", 
                    linewidth=2.5, alpha=0.9)
            # Mark peak center
            ax.axvline(center, color='green', linestyle=':', alpha=0.5, linewidth=1.5)
    
    # Plot Lorentzian fits for Raman peak 850
    if raman_peak_850 and raman_peak_850.get('fit_params') is not None:
        fit_params = raman_peak_850['fit_params']
        center = raman_peak_850.get('wavenumber', fit_params[1])
        width = raman_peak_850.get('width', fit_params[2])
        
        # Create fine grid around peak for smooth Lorentzian curve
        peak_range_mask = (wavenumbers >= center - 3*width) & (wavenumbers <= center + 3*width)
        if np.any(peak_range_mask):
            wavenumbers_peak = wavenumbers[peak_range_mask]
            lorentzian_fit = lorentzian(wavenumbers_peak, fit_params[0], fit_params[1], fit_params[2], fit_params[3])
            ax.plot(wavenumbers_peak, lorentzian_fit, '-', 
                    color='orange', label=f"Raman 850 cm⁻¹ Lorentzian Fit (Area={raman_peak_850.get('area', 0):.2f})", 
                    linewidth=2.5, alpha=0.9)
            # Mark peak center
            ax.axvline(center, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=14)
    ax.set_ylabel('Normalized Intensity', fontsize=14)
    ax.set_title(f'Representative Raman Spectrum - Scan {scan_number}\n(Normalized, Lieberfit Baseline, and Lorentzian Fits)', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(250, max(wavenumbers))
    
    save_path = Path(output_dir_str) / f'rep_raman_s{scan_number}_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Representative Raman spectrum plot saved to: {save_path}")


def plot_overlaid_normalized_spectra(summaries, output_dir, config):
    """
    Plot overlaid normalized Raman spectra for multiple scans.
    Shows normalized smoothed spectra starting from 250 cm^-1.
    
    Parameters:
    -----------
    summaries : list of dict
        List of summary dictionaries from process_scan_v4() containing spectrum data
    output_dir : Path
        Output directory for saving plot
    config : dict
        Configuration dictionary
    """
    if not summaries or len(summaries) == 0:
        return
    
    output_dir_str = str(output_dir)
    create_dir_if_needed(output_dir_str)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use a colormap to distinguish different scans
    colors = cm.tab10(np.linspace(0, 1, len(summaries)))
    
    for idx, summary in enumerate(summaries):
        # Extract spectrum data
        wavenumbers = summary.get('wavenumbers')
        intensities_normalized_smooth = summary.get('intensities_normalized_smooth')
        scan_number = summary.get('scan_number')
        
        if wavenumbers is None or intensities_normalized_smooth is None:
            continue
        
        # Filter to start from 250 cm^-1
        mask = wavenumbers >= 250
        wavenumbers_filtered = wavenumbers[mask]
        intensities_filtered = intensities_normalized_smooth[mask]
        
        # Plot normalized smoothed spectrum
        ax.plot(wavenumbers_filtered, intensities_filtered, '-', 
                color=colors[idx], label=f'Scan {scan_number}', 
                alpha=0.7, linewidth=1.5)
    
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=14)
    ax.set_ylabel('Normalized Intensity', fontsize=14)
    ax.legend(fontsize=11, loc='best', ncol=2 if len(summaries) > 10 else 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(250, None)  # Start from 250 cm^-1
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scan_numbers_str = '_'.join([str(s.get('scan_number', 'unknown')) for s in summaries[:5]])
    if len(summaries) > 5:
        scan_numbers_str += f'_and_{len(summaries)-5}_more'
    save_path = Path(output_dir_str) / f'overlaid_normalized_spectra_{scan_numbers_str}_{timestamp}.png'
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Overlaid normalized Raman spectra plot saved to: {save_path}")


def plot_ratio_timeseries(df, output_dir, config, light_cycle=None, use_baseline_corrected=True):
    """
    Plot timeseries for fluorescence ratios.
    Shows pre-smoothed data as markers with shade, and Gaussian smoothed as bold dashed line.
    Two plots stacked in one figure.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with ratio columns
    output_dir : Path
        Output directory for saving plots
    config : dict
        Configuration dictionary
    light_cycle : str, optional
        Light cycle for shading ('Constant', '8to24', '6to22')
    use_baseline_corrected : bool, optional
        If True, use baseline corrected ratios. If False, use non-baseline-corrected ratios.
        Default: True
    """
    if df.empty:
        print("Warning: No data to plot in time series.")
        return
    
    # Get light cycle from config if not provided
    if light_cycle is None:
        light_cycle = config.get('light_cycle', 'Constant')
    
    # Parse transition and event configs if available
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    # Get x-axis type from config (default: datetime)
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    if x_axis_type not in ['datetime', 'scan_number']:
        print(f"Warning: Invalid x_axis_type '{x_axis_type}', using 'datetime'")
        x_axis_type = 'datetime'
    
    # Parse transition and event configs if available
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    output_dir_str = str(output_dir)
    create_dir_if_needed(output_dir_str)
    
    # Get smoothing sigma from config (same as main.py - uses sigma_gaussian)
    # Default: 25 (matches config), fallback to 5 if not in config
    ratio_smooth_sigma = config.get('sigma_gaussian', 25)
    
    # Create stacked figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'hspace': 0.15})
    
    # Plot 1: Fluorescence to G-band ratio (top subplot)
    # Determine which column to use based on use_baseline_corrected flag
    if use_baseline_corrected and 'Fluorescence_to_Gband_Ratio_BaselineCorrected' in df.columns:
        ratio_col = 'Fluorescence_to_Gband_Ratio_BaselineCorrected'
        label_text = 'Baseline Corrected Ratio'
    elif 'Fluorescence_to_Gband_Ratio' in df.columns:
        ratio_col = 'Fluorescence_to_Gband_Ratio'
        label_text = 'Ratio'
    else:
        ratio_col = None
    
    if ratio_col:
        ratio_data = df[ratio_col].values
        mask = ~np.isnan(ratio_data)
        
        if np.sum(mask) > 0:
            # Plot pre-smoothed data as markers with transparency (no border)
            ax1.plot(df.index, ratio_data, 'o', 
                    color='m', label=label_text, alpha=0.5, markersize=4,
                    markeredgewidth=0)
            
            # Calculate and plot Gaussian smoothed as bold dashed line
            ratio_smoothed = ratio_data.copy()
            if np.sum(mask) > 1:
                ratio_smoothed[mask] = gaussian_filter1d(ratio_data[mask], sigma=ratio_smooth_sigma)
            ax1.plot(df.index, ratio_smoothed, '--', 
                    color='black', label='Gaussian Smoothed', alpha=1.0, linewidth=3)
        
        # Only add shading for datetime-based plots
        if x_axis_type == 'datetime':
            add_day_night_shading(ax1, df.index.min(), df.index.max(), light_cycle=light_cycle, light_transition=light_transition)
            # Add vertical line to mark light transition if configured
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    ax1.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
            # Add vertical line to mark shade transition if configured
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    shade_label = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        shade_label += f" (PPFD: {shade_transition['ppfd']})"
                    ax1.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=shade_label)
            # Add markers for treatment events if configured
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if df.index.min() <= event_time <= df.index.max():
                        ax1.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                  linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            ax1.set_xlabel('Date time (MM-DD HH)', fontsize=14)
            ax1.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax1.xaxis.set_major_locator(DayLocator())
            fig.autofmt_xdate()
        else:  # scan_number
            ax1.set_xlabel('Scan Number', fontsize=14)
            # Remove date formatters and use numeric formatting for scan numbers
            ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
            ax1.xaxis.set_major_locator(MaxNLocator(nbins=10))
        
        ax1.set_ylabel('Fluorescence / G-band Ratio', fontsize=14)
        ax1.set_title('Fluorescence to G-band Ratio Timeseries', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=12)
    
    # Plot 2: Fluorescence to Raman peak 850 ratio (bottom subplot)
    # Determine which column to use based on use_baseline_corrected flag
    if use_baseline_corrected and 'Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected' in df.columns:
        ratio_col_raman = 'Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected'
        label_text_raman = 'Baseline Corrected Ratio'
    elif 'Fluorescence_to_Raman_Peak_850_Ratio' in df.columns:
        ratio_col_raman = 'Fluorescence_to_Raman_Peak_850_Ratio'
        label_text_raman = 'Ratio'
    else:
        ratio_col_raman = None
    
    if ratio_col_raman:
        ratio_data = df[ratio_col_raman].values
        mask = ~np.isnan(ratio_data)
        
        if np.sum(mask) > 0:
            # Plot pre-smoothed data as markers with transparency (no border)
            ax2.plot(df.index, ratio_data, 'o', 
                    color='green', label=label_text_raman, alpha=0.5, markersize=4,
                    markeredgewidth=0)
            
            # Calculate and plot Gaussian smoothed as bold dashed line
            ratio_smoothed = ratio_data.copy()
            if np.sum(mask) > 1:
                ratio_smoothed[mask] = gaussian_filter1d(ratio_data[mask], sigma=ratio_smooth_sigma)
            ax2.plot(df.index, ratio_smoothed, '--', 
                    color='black', label='Gaussian Smoothed', alpha=1.0, linewidth=3)
        
        # Only add shading for datetime-based plots
        if x_axis_type == 'datetime':
            add_day_night_shading(ax2, df.index.min(), df.index.max(), light_cycle=light_cycle, light_transition=light_transition)
            # Add vertical line to mark light transition if configured
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    ax2.axvline(transition_time, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Light transition')
            # Add vertical line to mark shade transition if configured
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if df.index.min() <= transition_time <= df.index.max():
                    shade_label = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        shade_label += f" (PPFD: {shade_transition['ppfd']})"
                    ax2.axvline(transition_time, color='purple', linestyle='--', linewidth=2, alpha=0.7, label=shade_label)
            # Add markers for treatment events if configured
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if df.index.min() <= event_time <= df.index.max():
                        ax2.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                                  linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
            ax2.set_xlabel('Date time (MM-DD HH)', fontsize=14)
            ax2.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax2.xaxis.set_major_locator(DayLocator())
        else:  # scan_number
            ax2.set_xlabel('Scan Number', fontsize=14)
            # Remove date formatters and use numeric formatting for scan numbers
            ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
            ax2.xaxis.set_major_locator(MaxNLocator(nbins=10))
        
        ax2.set_ylabel('Fluorescence / Raman Peak 850 Ratio', fontsize=14)
        ax2.set_title('Fluorescence to Raman Peak 850 Ratio Timeseries', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    save_path = Path(output_dir_str) / f'fluo_ratios_ts_{datetime.now().strftime("%Y%m%d")}.png'
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Stacked fluorescence ratios plot saved to: {save_path}")


