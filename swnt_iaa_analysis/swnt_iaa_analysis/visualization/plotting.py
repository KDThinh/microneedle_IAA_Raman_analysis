"""Plotting functions for Raman spectra and timeseries."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 13
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.titlesize'] = 14
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.ticker import FuncFormatter, MaxNLocator
from datetime import datetime

from ..core.utils import (
    create_dir_if_needed,
    parse_light_transition_config,
    parse_shade_transition_config,
    parse_treatment_events_config,
    add_day_night_shading,
    smooth_signal,
)
from ..core.baseline import apply_gaussian_smoothing, apply_als_baseline


def _add_shading_for_cycle(ax, start_time, end_time, light_cycle, is_first=False):
    """Helper function to add shading for a specific light cycle period."""
    from datetime import timedelta
    
    current_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)
    
    while current_time < end_time:
        if light_cycle == 'Constant':
            day_start = current_time.replace(hour=0, minute=0)
            day_end = (current_time + timedelta(days=1)).replace(hour=0, minute=0)
            if day_start < end_time and day_end > start_time:
                ax.axvspan(max(day_start, start_time), min(day_end, end_time),
                           facecolor='yellow', alpha=0.1)
        else:
            day_start_hour = 8 if light_cycle == '8to24' else 6
            day_end_hour = 0 if light_cycle == '8to24' else 22
            day_end_day_offset = 1 if light_cycle == '8to24' else 0
            
            day_start = current_time.replace(hour=day_start_hour, minute=0)
            day_end = (current_time + timedelta(days=day_end_day_offset)).replace(hour=day_end_hour, minute=0)
            if day_start < end_time and day_end > start_time:
                ax.axvspan(max(day_start, start_time), min(day_end, end_time),
                           facecolor='yellow', alpha=0.1)
            
            night_start1 = current_time.replace(hour=0, minute=0)
            night_end1 = current_time.replace(hour=day_start_hour, minute=0)
            if night_start1 < end_time and night_end1 > start_time:
                ax.axvspan(max(night_start1, start_time), min(night_end1, end_time),
                           facecolor='blue', alpha=0.1)
            
            if day_end_hour != 0:
                night_start2 = current_time.replace(hour=day_end_hour, minute=0)
                night_end2 = (current_time + timedelta(days=1)).replace(hour=0, minute=0)
                if night_start2 < end_time and night_end2 > start_time:
                    ax.axvspan(max(night_start2, start_time), min(night_end2, end_time),
                               facecolor='blue', alpha=0.1)
        
        current_time += timedelta(days=1)


def plot_ratio_timeseries(df, column_name, output_dir, config, title=None):
    """
    Plot timeseries of ratio data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with datetime index and ratio column
    column_name : str
        Name of column to plot
    output_dir : Path
        Output directory for saving plot
    config : dict
        Configuration dictionary
    title : str, optional
        Plot title
    """
    create_dir_if_needed(str(output_dir))
    
    # Create SVG subfolder
    svg_dir = Path(output_dir) / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    
    if column_name not in df.columns:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x_values = df.index
    y_values = df[column_name].values
    
    ax.plot(x_values, y_values, label=column_name, linewidth=1.5)
    
    # Add day/night shading if datetime index
    if pd.api.types.is_datetime64_any_dtype(df.index):
        light_cycle = config.get('metadata', {}).get('light_cycle', 'Constant')
        light_transition = parse_light_transition_config(config)
        shade_transition = parse_shade_transition_config(config)
        treatment_events = parse_treatment_events_config(config)
        
        add_day_night_shading(ax, x_values.min(), x_values.max(), 
                             light_cycle=light_cycle, light_transition=light_transition)
        
        if light_transition:
            transition_time = light_transition['transition_datetime']
            if x_values.min() <= transition_time <= x_values.max():
                ax.axvline(transition_time, color='red', linestyle='--', 
                          linewidth=2, alpha=0.7, label='Light transition')
        
        if shade_transition:
            transition_time = shade_transition['transition_datetime']
            if x_values.min() <= transition_time <= x_values.max():
                ax.axvline(transition_time, color='purple', linestyle='--', 
                          linewidth=2, alpha=0.7, label='Shade transition')
        
        if treatment_events:
            for event in treatment_events:
                event_time = event['datetime']
                if x_values.min() <= event_time <= x_values.max():
                    ax.axvline(event_time, color=event['marker_color'], 
                             linestyle=event['marker_style'], linewidth=1.5, 
                             alpha=0.6, label=event.get('description', event['event_type']))
        
        ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
        ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
        ax.xaxis.set_major_locator(DayLocator())
        fig.autofmt_xdate()
    else:
        ax.set_xlabel('Scan Number', fontsize=14)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    
    # Clean up column name for title
    title_name = column_name.replace('_', ' ').replace('BaselineCorrected', '').strip()
    if title_name.endswith('Ratio'):
        title_name = title_name.replace('Fluorescence to', 'Fluorescence/')
    
    ax.set_ylabel(title_name, fontsize=12, fontweight='bold')
    ax.set_title(title or title_name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plot_filename = f'timeseries_{column_name}.png'
    plot_path = Path(output_dir) / plot_filename
    svg_path = svg_dir / f'timeseries_{column_name}.svg'
    
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(str(svg_path), format='svg', bbox_inches='tight', facecolor='white')
    plt.close()


def plot_signal_correction_comparison(df_original, df_corrected, output_dir, config, jump_info_dict=None):
    """
    Plot overlaid time-series showing before and after signal corrections.
    
    Creates separate plots for each corrected column showing original vs corrected data.
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        DataFrame with original (uncorrected) data
    df_corrected : pandas.DataFrame
        DataFrame with corrected data
    output_dir : Path
        Output directory for saving plots
    config : dict
        Configuration dictionary
    jump_info_dict : dict, optional
        Dictionary mapping column names to jump information dicts with 'jump_indices' key
    """
    create_dir_if_needed(str(output_dir))
    
    # Create SVG subfolder
    svg_dir = Path(output_dir) / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    
    # Columns to plot
    columns_to_plot = [
        'Normalized_Fluorescence_Intensity',
        'Normalized_Gband_Area',
        'Normalized_Raman_Peak_850_Area'
    ]
    
    # Get light cycle and transitions from config
    light_cycle = config.get('metadata', {}).get('light_cycle', 'Constant')
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    # Determine x-axis type
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    if x_axis_type not in ['datetime', 'scan_number']:
        x_axis_type = 'datetime'
    
    for col in columns_to_plot:
        if col not in df_original.columns:
            continue
        
        # Get the corrected column name (with _BaselineCorrected suffix)
        corrected_col_name = col + '_BaselineCorrected'
        if corrected_col_name not in df_corrected.columns:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x_values = df_original.index
        y_original = df_original[col].values
        y_corrected = df_corrected[corrected_col_name].values  # FIX: Use corrected column, not original
        
        # Plot original data (gray - conventional for raw/original data)
        ax.plot(x_values, y_original, 'o-', color='gray', alpha=0.7, 
                linewidth=1.5, markersize=4, label=f'{col} (Original)', zorder=2)
        
        # Plot corrected data (blue - conventional for processed/corrected data)
        ax.plot(x_values, y_corrected, 's-', color='#1f77b4', alpha=0.8, 
                linewidth=2, markersize=3, label=f'{col} (Corrected)', zorder=3)
        
        # Mark jump points on original data if available
        if jump_info_dict and col in jump_info_dict:
            jump_data = jump_info_dict[col]
            jump_indices = jump_data.get('jump_indices', [])
            if len(jump_indices) > 0:
                # Convert to numpy array if needed
                if not isinstance(jump_indices, np.ndarray):
                    jump_indices = np.array(jump_indices)
                
                # Filter indices to valid range
                valid_jump_indices = jump_indices[(jump_indices >= 0) & (jump_indices < len(x_values))]
                
                if len(valid_jump_indices) > 0:
                    # Convert jump indices to x-axis values
                    if isinstance(x_values, pd.DatetimeIndex):
                        jump_x_values = x_values[valid_jump_indices]
                    else:
                        jump_x_values = x_values[valid_jump_indices]
                    
                    # Get y-values at jump points (use original values, not corrected)
                    jump_y_values = y_original[valid_jump_indices]
                    
                    ax.scatter(jump_x_values, jump_y_values, color='red', marker='x', 
                             s=100, linewidths=3, zorder=4, label=f'Jump Points (n={len(valid_jump_indices)})')
        
        # Add day/night shading if datetime index
        if pd.api.types.is_datetime64_any_dtype(df_original.index):
            add_day_night_shading(ax, x_values.min(), x_values.max(), 
                                light_cycle=light_cycle, light_transition=light_transition)
            
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if x_values.min() <= transition_time <= x_values.max():
                    ax.axvline(transition_time, color='red', linestyle='--', 
                              linewidth=2, alpha=0.7, label='Light transition')
            
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if x_values.min() <= transition_time <= x_values.max():
                    label_text = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax.axvline(transition_time, color='purple', linestyle='--', 
                              linewidth=2, alpha=0.7, label=label_text)
            
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if x_values.min() <= event_time <= x_values.max():
                        ax.axvline(event_time, color=event['marker_color'], 
                                 linestyle=event['marker_style'], linewidth=1.5, 
                                 alpha=0.6, label=event.get('description', event['event_type']))
            
            ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
            fig.autofmt_xdate()
        else:
            ax.set_xlabel('Scan Number', fontsize=14)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        
        # Clean up column name for labels
        col_display = col.replace('_', ' ').replace('Normalized ', '')
        ax.set_ylabel(col_display, fontsize=12, fontweight='bold')
        ax.set_title(col_display, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plot_filename = f'signal_correction_comparison_{col}.png'
        plot_path = Path(output_dir) / plot_filename
        svg_path = svg_dir / f'signal_correction_comparison_{col}.svg'
        
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(str(svg_path), format='svg', bbox_inches='tight', facecolor='white')
        plt.close()


def plot_representative_raman_spectrum(spectrum_data, output_dir, config):
    """
    Plot representative Raman spectrum showing:
    - Normalized smoothed spectrum
    - Lieberfit baseline
    - Lorentzian fits for G-band and Raman peak at 850 cm⁻¹
    
    Parameters:
    -----------
    spectrum_data : dict
        Dictionary containing:
        - wavenumbers: array of wavenumbers
        - intensities_normalized: normalized smoothed intensities
        - baseline_normalized: Lieberfit baseline
        - corrected_normalized: Lieberfit-corrected spectrum
        - raman_peak_850: dict with peak info and fit_params
        - gband_peak_1600: dict with peak info and fit_params
        - scan_number: scan number
    output_dir : Path
        Output directory for saving plot
    config : dict
        Configuration dictionary
    """
    if spectrum_data is None:
        return
    
    create_dir_if_needed(str(output_dir))
    
    # Create SVG subfolder
    svg_dir = Path(output_dir) / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract spectrum data
    wavenumbers = spectrum_data.get('wavenumbers')
    intensities_normalized = spectrum_data.get('intensities_normalized')
    baseline_normalized = spectrum_data.get('baseline_normalized')
    corrected_normalized = spectrum_data.get('corrected_normalized')
    raman_peak_850 = spectrum_data.get('raman_peak_850')
    gband_peak_1600 = spectrum_data.get('gband_peak_1600')
    scan_number = spectrum_data.get('scan_number', 'unknown')
    
    if wavenumbers is None or intensities_normalized is None:
        return
    
    # Import lorentzian function
    from ..analysis.peaks import lorentzian
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot normalized smoothed spectrum
    ax.plot(wavenumbers, intensities_normalized, '-', 
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
        center = gband_peak_1600.get('wavenumber', fit_params[1] if len(fit_params) > 1 else np.nan)
        width = gband_peak_1600.get('width', fit_params[2] if len(fit_params) > 2 else np.nan)
        
        if not np.isnan(center) and not np.isnan(width) and width > 0:
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
        center = raman_peak_850.get('wavenumber', fit_params[1] if len(fit_params) > 1 else np.nan)
        width = raman_peak_850.get('width', fit_params[2] if len(fit_params) > 2 else np.nan)
        
        if not np.isnan(center) and not np.isnan(width) and width > 0:
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
    
    ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
    ax.set_title(f'Raman Spectrum (Scan {scan_number})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim(250, max(wavenumbers))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plot_filename = f'representative_raman_spectrum_scan_{scan_number}.png'
    plot_path = Path(output_dir) / plot_filename
    svg_path = svg_dir / f'representative_raman_spectrum_scan_{scan_number}.svg'
    
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(str(svg_path), format='svg', bbox_inches='tight', facecolor='white')
    plt.close()


def plot_spike_removal_comparison(df_original, df_cleaned, output_dir, config, spike_info_dict=None):
    """
    Plot overlaid time-series showing before and after spike removal.
    
    Creates separate plots for each column showing original vs spike-cleaned data.
    
    Parameters:
    -----------
    df_original : pandas.DataFrame
        DataFrame with original (with spikes) data
    df_cleaned : pandas.DataFrame
        DataFrame with spikes removed
    output_dir : Path
        Output directory for saving plots
    config : dict
        Configuration dictionary
    spike_info_dict : dict, optional
        Dictionary mapping column names to spike information dicts with 'spike_indices' key
    """
    create_dir_if_needed(str(output_dir))
    
    # Create SVG subfolder
    svg_dir = Path(output_dir) / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    
    # Columns to plot
    columns_to_plot = [
        'Normalized_Fluorescence_Intensity',
        'Normalized_Gband_Area',
        'Normalized_Raman_Peak_850_Area'
    ]
    
    # Get light cycle and transitions from config
    light_cycle = config.get('metadata', {}).get('light_cycle', 'Constant')
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    # Determine x-axis type
    x_axis_type = config.get('timeseries_x_axis', 'datetime')
    if x_axis_type not in ['datetime', 'scan_number']:
        x_axis_type = 'datetime'
    
    for col in columns_to_plot:
        if col not in df_original.columns:
            continue
        
        # For spike removal comparison, we compare original vs cleaned (before baseline correction)
        # The cleaned signal should be in the same column name (spikes are replaced in-place)
        # But we need to track which points were spikes
        fig, ax = plt.subplots(figsize=(10, 5))
        
        x_values = df_original.index
        y_original = df_original[col].values
        y_cleaned = df_cleaned[col].values
        
        # Plot original data (gray - with spikes)
        ax.plot(x_values, y_original, 'o-', color='gray', alpha=0.7, 
                linewidth=1.5, markersize=4, label=f'{col} (Original)', zorder=2)
        
        # Plot cleaned data (blue - spikes removed)
        ax.plot(x_values, y_cleaned, 's-', color='#1f77b4', alpha=0.8, 
                linewidth=2, markersize=3, label=f'{col} (Spikes Removed)', zorder=3)
        
        # Mark spike points on original data if available
        if spike_info_dict and col in spike_info_dict:
            spike_data = spike_info_dict[col]
            spike_indices = spike_data.get('spike_indices', [])
            if len(spike_indices) > 0:
                # Convert to numpy array if needed
                if not isinstance(spike_indices, np.ndarray):
                    spike_indices = np.array(spike_indices)
                
                # Filter indices to valid range
                valid_spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < len(x_values))]
                
                if len(valid_spike_indices) > 0:
                    # Convert spike indices to x-axis values
                    if isinstance(x_values, pd.DatetimeIndex):
                        spike_x_values = x_values[valid_spike_indices]
                    else:
                        spike_x_values = x_values[valid_spike_indices]
                    
                    # Get y-values at spike points (use original values)
                    spike_y_values = y_original[valid_spike_indices]
                    
                    ax.scatter(spike_x_values, spike_y_values, color='red', marker='x', 
                             s=100, linewidths=3, zorder=4, label=f'Spikes Removed (n={len(valid_spike_indices)})')
        
        # Add day/night shading if datetime index
        if pd.api.types.is_datetime64_any_dtype(df_original.index):
            add_day_night_shading(ax, x_values.min(), x_values.max(), 
                                light_cycle=light_cycle, light_transition=light_transition)
            
            if light_transition:
                transition_time = light_transition['transition_datetime']
                if x_values.min() <= transition_time <= x_values.max():
                    ax.axvline(transition_time, color='red', linestyle='--', 
                              linewidth=2, alpha=0.7, label='Light transition')
            
            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if x_values.min() <= transition_time <= x_values.max():
                    label_text = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax.axvline(transition_time, color='purple', linestyle='--', 
                              linewidth=2, alpha=0.7, label=label_text)
            
            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if x_values.min() <= event_time <= x_values.max():
                        ax.axvline(event_time, color=event['marker_color'], 
                                 linestyle=event['marker_style'], linewidth=1.5, 
                                 alpha=0.6, label=event.get('description', event['event_type']))
            
            ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
            fig.autofmt_xdate()
        else:
            ax.set_xlabel('Scan Number', fontsize=14)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        
        # Clean up column name for labels
        col_display = col.replace('_', ' ').replace('Normalized ', '')
        ax.set_ylabel(col_display, fontsize=12, fontweight='bold')
        ax.set_title(col_display, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plot_filename = f'spike_removal_comparison_{col}.png'
        plot_path = Path(output_dir) / plot_filename
        svg_path = svg_dir / f'spike_removal_comparison_{col}.svg'
        
        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(str(svg_path), format='svg', bbox_inches='tight', facecolor='white')
        plt.close()


def plot_fft_analysis(results_df, fft_results, output_dir, config, ratio_name='Fluorescence to G-band Ratio'):
    """
    Create publication-quality FFT plots with timeseries and spectrum.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with datetime index and ratio column
    fft_results : dict
        Dictionary containing FFT results from compute_fourier_transform
    output_dir : Path
        Output directory for saving plot
    config : dict
        Configuration dictionary
    ratio_name : str
        Name of the ratio being analyzed (for plot labels)
    """
    create_dir_if_needed(str(output_dir))
    
    # Create SVG subfolder
    svg_dir = Path(output_dir) / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    
    if not fft_results or 'peaks' not in fft_results or fft_results['peaks'] is None:
        return
    
    # Get ratio column name from results
    ratio_col = 'Fluorescence_to_Gband_Ratio_BaselineCorrected'
    if ratio_col not in results_df.columns:
        ratio_col = 'Fluorescence_to_Gband_Ratio'
    
    if ratio_col not in results_df.columns:
        return
    
    # Get signal and time data
    signal = results_df[ratio_col].values
    mask = ~np.isnan(signal)
    
    if np.sum(mask) < 3:
        return
    
    signal_valid = signal[mask]
    
    # Get FFT data
    peaks_df = fft_results['peaks']
    complete_df = fft_results.get('complete', None)
    
    # Extract positive frequencies and magnitude from complete data
    if complete_df is not None:
        positive_mask = complete_df['Frequency (cycles/hour)'] > 0
        positive_freqs = complete_df['Frequency (cycles/hour)'][positive_mask].values
        magnitude = complete_df['Magnitude'][positive_mask].values
    else:
        # Fallback: reconstruct from peaks (less ideal)
        peak_freqs = peaks_df['Frequency (cycles/hour)'].values
        peak_mags = peaks_df['Magnitude'].values
        # Create simple frequency range
        max_freq = min(0.5, peak_freqs.max() * 1.2) if len(peak_freqs) > 0 else 0.5
        positive_freqs = np.linspace(0.001, max_freq, 1000)
        # Interpolate magnitude (simplified)
        magnitude = np.interp(positive_freqs, peak_freqs, peak_mags)
    
    # Peak data (still computed for export, but not drawn in plot)
    top_peaks = config.get('fft_top_peaks', 10)
    if len(peaks_df) > 0:
        peak_freqs = peaks_df['Frequency (cycles/hour)'].values[:top_peaks]
        peak_magnitudes = peaks_df['Magnitude'].values[:top_peaks]
    else:
        peak_freqs = np.array([])
        peak_magnitudes = np.array([])
    
    # Get configuration for plot annotations
    light_cycle = config.get('metadata', {}).get('light_cycle', 'Constant')
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)
    
    # Compute Gaussian-smoothed ratio and ALS baseline for overlay (match FFT preprocessing)
    processing_cfg = config.get('processing', {}) or config.get('sections', {}).get('processing', {})
    sigma_gaussian = processing_cfg.get('sigma_gaussian', config.get('sigma_gaussian', 50))
    lam_als = processing_cfg.get('lam_als', config.get('lam_als', 100000000))
    p_als = processing_cfg.get('p_als', config.get('p_als', 0.000100))
    niter_als = processing_cfg.get('niter_als', config.get('niter_als', 20))
    
    signal_gaussian = apply_gaussian_smoothing(signal_valid, sigma=sigma_gaussian)
    als_baseline = apply_als_baseline(signal_gaussian, lam=lam_als, p=p_als, niter=niter_als)
    
    # Create combined plot: timeseries on top, FFT below
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=False, gridspec_kw={'hspace': 0.35})
    
    # Top subplot: Timeseries of ratio
    if pd.api.types.is_datetime64_any_dtype(results_df.index):
        datetimes_valid = results_df.index[mask]
        # Raw ratio
        ax1.plot(datetimes_valid, signal_valid, color='#2E86AB', alpha=0.8, linewidth=2, label=ratio_name)
        # Gaussian-smoothed ratio (used for FFT)
        ax1.plot(datetimes_valid, signal_gaussian, color='black', linestyle='--',
                 linewidth=1.5, alpha=0.7, label='Gaussian')
        # ALS baseline
        ax1.plot(datetimes_valid, als_baseline, color='red', linestyle=':',
                 linewidth=1.5, alpha=0.7, label='')
        
        # Add day/night shading
        add_day_night_shading(ax1, datetimes_valid.min(), datetimes_valid.max(), 
                             light_cycle=light_cycle, light_transition=light_transition)
        
        # Add vertical lines for transitions
        if light_transition:
            transition_time = light_transition['transition_datetime']
            if datetimes_valid.min() <= transition_time <= datetimes_valid.max():
                ax1.axvline(transition_time, color='red', linestyle='--', linewidth=2, 
                          alpha=0.7, label='Light transition')
        
        if shade_transition:
            transition_time = shade_transition['transition_datetime']
            if datetimes_valid.min() <= transition_time <= datetimes_valid.max():
                label_text = 'Shade transition'
                if 'ppfd' in shade_transition:
                    label_text += f" (PPFD: {shade_transition['ppfd']})"
                ax1.axvline(transition_time, color='purple', linestyle='--', linewidth=2, 
                          alpha=0.7, label=label_text)
        
        if treatment_events:
            for event in treatment_events:
                event_time = event['datetime']
                if datetimes_valid.min() <= event_time <= datetimes_valid.max():
                    ax1.axvline(event_time, color=event['marker_color'], linestyle=event['marker_style'], 
                              linewidth=1.5, alpha=0.6, label=event.get('description', event['event_type']))
        
        ax1.set_xlabel('Date time (MM-DD HH)', fontsize=13, fontweight='bold')
        ax1.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
        ax1.xaxis.set_major_locator(DayLocator())
        fig.autofmt_xdate()
    else:
        # Fallback for non-datetime index
        time_hours_valid = np.arange(len(signal_valid))
        ax1.plot(time_hours_valid, signal_valid, color='#2E86AB', alpha=0.8, linewidth=2, label=ratio_name)
        ax1.plot(time_hours_valid, signal_gaussian, color='black', linestyle='--',
                 linewidth=1.5, alpha=0.7, label='Gaussian')
        ax1.plot(time_hours_valid, als_baseline, color='red', linestyle=':',
                 linewidth=1.5, alpha=0.7, label='')
        ax1.set_xlabel('Time (hours from start)', fontsize=13, fontweight='bold')
    
    ax1.set_ylabel(ratio_name, fontsize=13, fontweight='bold')
    ax1.set_title(f'Timeseries - {ratio_name}', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=11, loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(axis='both', labelsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Bottom subplot: FFT Spectrum
    ax2.plot(positive_freqs, magnitude, color='#1E88E5', linewidth=2, label='FFT Magnitude', zorder=1)
    
    # Highlight diurnal range (0.03-0.05 cycles/hour, ~20-33 hour period)
    ax2.axvspan(0.03, 0.05, alpha=0.2, color='gold', label='Diurnal Range (20-33h)', zorder=0)
    
    # Add vertical line at 24h period (0.0417 cycles/hour)
    ax2.axvline(1/24, color='red', linestyle=':', linewidth=2, alpha=0.6, 
               label='24h Period', zorder=2)
    
    ax2.set_xlabel('Frequency (cycles/hour)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Magnitude', fontsize=13, fontweight='bold')
    ax2.set_title(f'FFT Spectrum - {ratio_name}', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlim(0, 0.5)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11, loc='best', framealpha=0.9)
    ax2.tick_params(axis='both', labelsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Clean up ratio name for filename
    ratio_name_clean = ratio_name.lower().replace(" ", "_").replace("/", "_")
    
    # Save plot
    plot_filename = f'fft_analysis_{ratio_name_clean}.png'
    plot_path = Path(output_dir) / plot_filename
    svg_path = svg_dir / f'fft_analysis_{ratio_name_clean}.svg'
    
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(str(svg_path), format='svg', bbox_inches='tight', facecolor='white')
    plt.close()


def plot_processing_stages(df_original, df_after_spikes, df_corrected, output_dir, config, jump_info_dict=None):
    """
    Plot overlaid time-series showing processing stages:
    - Raw data
    - Savitzky-Golay smoothed data
    - Spike-removed data (Hampel)
    - Baseline-corrected data
    """
    create_dir_if_needed(str(output_dir))

    # Create SVG subfolder
    svg_dir = Path(output_dir) / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)

    # Columns to plot
    columns_to_plot = [
        'Normalized_Fluorescence_Intensity',
        'Normalized_Gband_Area',
        'Normalized_Raman_Peak_850_Area'
    ]

    # Get smoothing parameters for time-series smoothing
    processing_cfg = config.get('processing', {}) or config.get('sections', {}).get('processing', {})
    ts_sg_window = processing_cfg.get('timeseries_sg_window', config.get('timeseries_sg_window', 5))
    ts_sg_poly_order = processing_cfg.get('timeseries_sg_poly_order', config.get('timeseries_sg_poly_order', 2))
    ts_method = processing_cfg.get('timeseries_smoothing_method', config.get('timeseries_smoothing_method', 'savitzky')).lower()
    ts_gaussian_sigma = processing_cfg.get('timeseries_gaussian_sigma', config.get('timeseries_gaussian_sigma', 10))

    # Get light cycle and transitions from config
    light_cycle = config.get('metadata', {}).get('light_cycle', 'Constant')
    light_transition = parse_light_transition_config(config)
    shade_transition = parse_shade_transition_config(config)
    treatment_events = parse_treatment_events_config(config)

    for col in columns_to_plot:
        if col not in df_original.columns:
            continue

        corrected_col_name = col + '_BaselineCorrected'
        if corrected_col_name not in df_corrected.columns:
            continue

        fig, ax = plt.subplots(figsize=(10, 5))

        x_values = df_original.index
        y_raw = df_original[col].values

        # Time-series smoothed data (Savitzky-Golay or Gaussian)
        if ts_method == 'gaussian':
            y_smooth = apply_gaussian_smoothing(y_raw.copy(), sigma=ts_gaussian_sigma)
        else:
            y_smooth = smooth_signal(y_raw.copy(), window_size=ts_sg_window, poly_order=ts_sg_poly_order)

        # Spike-removed data (after Hampel, before baseline correction)
        if col in df_after_spikes.columns:
            y_spikes_removed = df_after_spikes[col].values
        else:
            y_spikes_removed = y_smooth

        # Baseline-corrected data
        y_baseline_corrected = df_corrected[corrected_col_name].values

        # Plot raw (no markers, transparent)
        ax.plot(x_values, y_raw, color='gray', linewidth=1.0, alpha=0.4, label='Raw')

        # Plot smoothed series
        label_smooth = 'Gaussian' if ts_method == 'gaussian' else 'Savitzky-Golay'
        ax.plot(x_values, y_smooth, color='#1f77b4', linewidth=1.5, alpha=0.7, label=label_smooth)

        # Plot baseline-corrected
        ax.plot(x_values, y_baseline_corrected, color='red', linewidth=1.8, alpha=0.9, label='Baseline Corrected')

        # Mark jump points on Gaussian smoothed curve if available
        if jump_info_dict and col in jump_info_dict:
            jump_data = jump_info_dict[col]
            jump_indices = jump_data.get('jump_indices', [])
            if len(jump_indices) > 0:
                # Convert to numpy array if needed
                if not isinstance(jump_indices, np.ndarray):
                    jump_indices = np.array(jump_indices)

                # Filter indices to valid range
                valid_jump_indices = jump_indices[(jump_indices >= 0) & (jump_indices < len(x_values))]

                if len(valid_jump_indices) > 0:
                    # Map indices to x-axis values
                    if isinstance(x_values, pd.DatetimeIndex):
                        jump_x_values = x_values[valid_jump_indices]
                    else:
                        jump_x_values = x_values[valid_jump_indices]

                    # Get y-values from Gaussian smoothed curve
                    jump_y_values = y_smooth[valid_jump_indices]

                    ax.scatter(
                        jump_x_values,
                        jump_y_values,
                        edgecolors='orange',
                        facecolors='none',
                        marker='o',
                        s=100,
                        linewidths=2,
                        zorder=5,
                        label=''
                    )

        # Add day/night shading if datetime index
        if pd.api.types.is_datetime64_any_dtype(df_original.index):
            add_day_night_shading(ax, x_values.min(), x_values.max(),
                                  light_cycle=light_cycle, light_transition=light_transition)

            if light_transition:
                transition_time = light_transition['transition_datetime']
                if x_values.min() <= transition_time <= x_values.max():
                    ax.axvline(transition_time, color='red', linestyle='--',
                               linewidth=2, alpha=0.7, label='Light transition')

            if shade_transition:
                transition_time = shade_transition['transition_datetime']
                if x_values.min() <= transition_time <= x_values.max():
                    label_text = 'Shade transition'
                    if 'ppfd' in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax.axvline(transition_time, color='purple', linestyle='--',
                               linewidth=2, alpha=0.7, label=label_text)

            if treatment_events:
                for event in treatment_events:
                    event_time = event['datetime']
                    if x_values.min() <= event_time <= x_values.max():
                        ax.axvline(event_time, color=event['marker_color'],
                                   linestyle=event['marker_style'], linewidth=1.5,
                                   alpha=0.6, label=event.get('description', event['event_type']))

            ax.set_xlabel('Date time (MM-DD HH)', fontsize=14)
            ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H'))
            ax.xaxis.set_major_locator(DayLocator())
            fig.autofmt_xdate()
        else:
            ax.set_xlabel('Scan Number', fontsize=14)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10))

        # Clean up column name for labels
        col_display = col.replace('_', ' ').replace('Normalized ', '')
        ax.set_ylabel(col_display, fontsize=12, fontweight='bold')
        ax.set_title(f'{col_display} - Processing Stages', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='both', labelsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plot_filename = f'processing_stages_{col}.png'
        plot_path = Path(output_dir) / plot_filename
        svg_path = svg_dir / f'processing_stages_{col}.svg'

        plt.savefig(str(plot_path), dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(str(svg_path), format='svg', bbox_inches='tight', facecolor='white')
        plt.close()