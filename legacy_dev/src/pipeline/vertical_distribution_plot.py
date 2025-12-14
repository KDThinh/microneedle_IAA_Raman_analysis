"""

Key Features:
- Vertical boxplot and violin plot options
- Statistics table (mean, median, std, quartiles)
- Day/night grouping (optional)
- Export to high-res PNG and CSV
- Compatible with main_wo_append.py workflow
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
from datetime import datetime
from .utils import create_dir_if_needed, add_day_night_shading

def plot_vertical_distribution(results_df, processed_dir, column='Corrected Ratio (ALS)', 
                              light_cycle='Constant', group_by_day_night=False, 
                              figsize=(6, 10), save=True):
    """
    Create comprehensive vertical distribution plot(s) for corrected ALS data.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with time-indexed corrected ALS data
    processed_dir : str
        Output directory for saving plots and stats
    column : str, default='Corrected Ratio (ALS)'
        Column name for distribution analysis
    light_cycle : str, default='Constant'
        Light cycle for day/night grouping: 'Constant', '8to24', '6to22'
    group_by_day_night : bool, default=False
        Create separate distributions for day vs night periods
    figsize : tuple, default=(6, 10)
        Figure size (width, height) - tall for vertical emphasis
    save : bool, default=True
        Save plot and statistics to files
    
    Returns:
    --------
    stats_df : pandas.DataFrame
        Summary statistics
    """
    create_dir_if_needed(processed_dir)
    
    # Validate data
    if column not in results_df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame. Available: {list(results_df.columns)}")
    
    data = results_df[column].dropna()
    if len(data) == 0:
        print(f"No valid data found for '{column}'. Skipping plot.")
        return None
    
    # Compute statistics
    stats_dict = {
        'N': len(data),
        'Mean': np.mean(data),
        'Median': np.median(data),
        'Std': np.std(data),
        'Min': np.min(data),
        'Q1': np.percentile(data, 25),
        'Q3': np.percentile(data, 75),
        'Max': np.max(data),
        'IQR': np.percentile(data, 75) - np.percentile(data, 25),
        'Skewness': stats.skew(data)
    }
    stats_df = pd.DataFrame([stats_dict])
    
    # Create figure(s)
    if group_by_day_night:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1]*1.2), 
                                     gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3})
        axes = [ax1, ax2]
        
        # Split data by day/night
        datetimes = results_df.index
        day_mask = _get_day_mask(datetimes, light_cycle)
        day_data = data[day_mask]
        night_data = data[~day_mask]
        
        # Plot day distribution
        _plot_single_vertical(ax1, day_data, f'{column} - Daytime', stats_df)
        ax1.text(0.02, 0.98, f'N={len(day_data)}', transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot night distribution  
        _plot_single_vertical(ax2, night_data, f'{column} - Nighttime', stats_df)
        ax2.text(0.02, 0.98, f'N={len(night_data)}', transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.suptitle('Vertical Distribution: Day vs Night', fontsize=16, weight='bold')
        
        # Update stats for groups
        group_stats = pd.DataFrame({
            'Period': ['Daytime', 'Nighttime'],
            'N': [len(day_data), len(night_data)],
            'Mean': [np.mean(day_data), np.mean(night_data)],
            'Median': [np.median(day_data), np.median(night_data)]
        })
        
    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        axes = [ax]
        _plot_single_vertical(ax, data, column, stats_df)
        ax.text(0.02, 0.98, f'N={len(data)}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        plt.title(f'Vertical Distribution: {column}', fontsize=16, weight='bold')
    
    # Add statistics table
    if save:
        stats_table_path = os.path.join(processed_dir, f'vertical_distribution_stats_{column.replace(" ", "_")}_{datetime.now().strftime("%Y-%m-%d")}.csv')
        stats_df.to_csv(stats_table_path, index=False)
        print(f"Statistics saved to: {stats_table_path}")
        
        if group_by_day_night:
            group_stats_path = os.path.join(processed_dir, f'vertical_distribution_group_stats_{datetime.now().strftime("%Y-%m-%d")}.csv')
            group_stats.to_csv(group_stats_path, index=False)
            print(f"Group statistics saved to: {group_stats_path}")
    
    # Save plot
    if save:
        plot_filename = f'vertical_distribution{"_day_night" if group_by_day_night else ""}_{column.replace(" ", "_")}_{datetime.now().strftime("%Y-%m-%d")}.png'
        save_path = os.path.join(processed_dir, plot_filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Vertical distribution plot saved to: {save_path}")
    
    plt.tight_layout()
    plt.show(block=False)
    
    return stats_df

def _plot_single_vertical(ax, data, title, stats_df):
    """Helper function to plot single vertical distribution"""
    # Vertical boxplot
    box_plot = ax.boxplot(data, vert=True, patch_artist=True, 
                         boxprops=dict(facecolor='lightgreen', color='darkgreen', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(color='darkgreen', linewidth=1.5),
                         capprops=dict(color='darkgreen'),
                         flierprops=dict(marker='o', color='red', alpha=0.6, markersize=4))
    
    # Add quartiles as horizontal lines
    q1, q3 = np.percentile(data, [25, 75])
    ax.axhline(q1, color='orange', linestyle='--', alpha=0.7, label=f'Q1={q1:.3f}')
    ax.axhline(q3, color='purple', linestyle='--', alpha=0.7, label=f'Q3={q3:.3f}')
    
    # Formatting
    ax.set_ylabel(title, fontsize=14, weight='bold')
    ax.set_xticks([])
    ax.set_title(f'Mean: {stats_df["Mean"].iloc[0]:.3f} | Median: {stats_df["Median"].iloc[0]:.3f}', 
                fontsize=12, pad=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=12)
    
    # Set y-limits with padding
    y_min, y_max = ax.get_ylim()
    padding = (y_max - y_min) * 0.05
    ax.set_ylim(y_min - padding, y_max + padding)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)

def _get_day_mask(datetimes, light_cycle):
    """Helper to get boolean mask for daytime periods"""
    day_mask = np.zeros(len(datetimes), dtype=bool)
    
    for dt in datetimes:
        hour = dt.hour
        
        if light_cycle == 'Constant':
            day_mask[True] = True  # All daytime
        elif light_cycle == '8to24':
            day_mask[hour >= 8 or hour < 0] = True
        elif light_cycle == '6to22':
            day_mask[6 <= hour <= 22] = True
    
    return day_mask

def plot_vertical_histogram(results_df, processed_dir, column='Corrected Ratio (ALS)', 
                           bins=50, figsize=(8, 10), save=True):
    """
    Alternative: Vertical histogram (rotated density plot)
    """
    create_dir_if_needed(processed_dir)
    
    data = results_df[column].dropna()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create vertical histogram
    counts, bins_edges, _ = ax.hist(data, bins=bins, orientation='horizontal', 
                                   color='skyblue', alpha=0.7, density=True, 
                                   edgecolor='navy', linewidth=0.5)
    
    # Add KDE line
    from scipy import stats
    kde = stats.gaussian_kde(data)
    x_kde = np.linspace(data.min(), data.max(), 200)
    ax.plot(kde(x_kde), x_kde, 'r-', linewidth=2, label='Kernel Density')
    
    ax.set_xlabel('Density', fontsize=14)
    ax.set_ylabel(column, fontsize=14)
    ax.set_title(f'Vertical Density Distribution\nN={len(data)}', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save:
        filename = f'vertical_histogram_{column.replace(" ", "_")}_{datetime.now().strftime("%Y-%m-%d")}.png'
        save_path = os.path.join(processed_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Vertical histogram saved to: {save_path}")
    
    plt.tight_layout()
    plt.show(block=False)

def plot_multi_distribution(df, processed_dir, columns=None, light_cycle='Constant'):
    """
    Plot several vertical distributions side-by-side (box + violin overlay).
    """
    if columns is None:
        columns = ['Corrected Ratio (ALS)', 'Smoothed Corrected (ALS)', 'Fluorescence to G-band Ratio']
        columns = [c for c in columns if c in df.columns]

    if not columns:
        print("No valid columns for multi-distribution plot.")
        return

    fig, axes = plt.subplots(1, len(columns), figsize=(4*len(columns), 8))
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        data = df[col].dropna()
        if data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(col)
            continue

        # Boxplot
        bp = ax.boxplot(data, vert=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='black'), capprops=dict(color='black'),
                        flierprops=dict(marker='o', color='orange', markersize=4, alpha=0.6))

        # Violin overlay
        vp = ax.violinplot(data, widths=0.6, showmeans=True, showmedians=True)
        for pc in vp['bodies']:
            pc.set_facecolor('green')
            pc.set_alpha(0.4)
            pc.set_edgecolor('darkgreen')

        ax.set_ylabel(col, fontsize=12)
        ax.set_xticks([])
        ax.set_title(col, fontsize=13, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('Vertical Distributions – Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    timestamp = datetime.now().strftime("%Y-%m-%d")
    save_path = os.path.join(processed_dir,
                             f"multi_vertical_distribution_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Multi-distribution plot saved: {save_path}")
    plt.show(block=False)

# Example usage and test function
def demo_usage():
    """Demo function - call this to test the module"""
    print("Demo: Creating vertical distribution plots...")
    # This would be called from main_wo_append.py with actual data
    print("Module loaded successfully!")

if __name__ == "__main__":
    demo_usage()
