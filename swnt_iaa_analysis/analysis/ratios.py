"""Ratio calculation functions."""

import numpy as np
import pandas as pd


def calculate_ratios(df):
    """
    Calculate fluorescence ratios from normalized data (before and after baseline correction).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with normalized data and optionally baseline-corrected data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with ratio columns added
    """
    df = df.copy()
    
    # Calculate ratios from original normalized data
    if 'Normalized_Fluorescence_Intensity' in df.columns and 'Normalized_Gband_Area' in df.columns:
        df['Fluorescence_to_Gband_Ratio'] = df['Normalized_Fluorescence_Intensity'] / df['Normalized_Gband_Area']
        df['Fluorescence_to_Gband_Ratio'] = df['Fluorescence_to_Gband_Ratio'].replace([np.inf, -np.inf], np.nan)
    
    if 'Normalized_Fluorescence_Intensity' in df.columns and 'Normalized_Raman_Peak_850_Area' in df.columns:
        df['Fluorescence_to_Raman_Peak_850_Ratio'] = df['Normalized_Fluorescence_Intensity'] / df['Normalized_Raman_Peak_850_Area']
        df['Fluorescence_to_Raman_Peak_850_Ratio'] = df['Fluorescence_to_Raman_Peak_850_Ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate ratios from baseline-corrected data if available
    if 'Normalized_Fluorescence_Intensity_BaselineCorrected' in df.columns and 'Normalized_Gband_Area_BaselineCorrected' in df.columns:
        df['Fluorescence_to_Gband_Ratio_BaselineCorrected'] = df['Normalized_Fluorescence_Intensity_BaselineCorrected'] / df['Normalized_Gband_Area_BaselineCorrected']
        df['Fluorescence_to_Gband_Ratio_BaselineCorrected'] = df['Fluorescence_to_Gband_Ratio_BaselineCorrected'].replace([np.inf, -np.inf], np.nan)
    
    if 'Normalized_Fluorescence_Intensity_BaselineCorrected' in df.columns and 'Normalized_Raman_Peak_850_Area_BaselineCorrected' in df.columns:
        df['Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected'] = df['Normalized_Fluorescence_Intensity_BaselineCorrected'] / df['Normalized_Raman_Peak_850_Area_BaselineCorrected']
        df['Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected'] = df['Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected'].replace([np.inf, -np.inf], np.nan)
    
    return df

