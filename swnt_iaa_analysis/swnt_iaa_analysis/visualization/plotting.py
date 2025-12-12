"""Plotting functions for Raman spectra and timeseries."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.ticker import FuncFormatter, MaxNLocator
from datetime import datetime

from ..core.utils import (
    create_dir_if_needed,
    parse_light_transition_config,
    parse_shade_transition_config,
    parse_treatment_events_config,
    add_day_night_shading
)


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
    
    if column_name not in df.columns:
        print(f"Warning: Column '{column_name}' not found in DataFrame")
        return
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
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
    
    ax.set_ylabel(column_name, fontsize=14)
    ax.set_title(title or f'Timeseries - {column_name}', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=12)
    
    timestamp = datetime.now().strftime("%Y%m%d")
    plot_filename = f'timeseries_{column_name}_{timestamp}.png'
    plot_path = Path(output_dir) / plot_filename
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")

