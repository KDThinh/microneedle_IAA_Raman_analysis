"""Ratio calculation functions."""

import numpy as np
import pandas as pd

BACKGROUND_COLUMN = 'Average_Background_Intensity_250_1250_cm-1'

# (normalized column -> raw column) pairs recovered by undoing the per-scan normalization.
_RAW_COLUMN_MAP = [
    ('Normalized_Fluorescence_Intensity', 'Raw_Fluorescence_Intensity'),
    ('Normalized_Gband_Area', 'Raw_Gband_Area'),
    ('Normalized_Gband_Intensity', 'Raw_Gband_Intensity'),
    ('Normalized_Raman_Peak_850_Area', 'Raw_Raman_Peak_850_Area'),
]

# Corrected-channel triples, preferred order: raw space first, then normalized space.
_CORRECTED_TRIPLES = [
    ('Raw_Fluorescence_Intensity_BaselineCorrected',
     'Raw_Gband_Area_BaselineCorrected',
     'Raw_Raman_Peak_850_Area_BaselineCorrected'),
    ('Normalized_Fluorescence_Intensity_BaselineCorrected',
     'Normalized_Gband_Area_BaselineCorrected',
     'Normalized_Raman_Peak_850_Area_BaselineCorrected'),
]


def add_raw_columns(df):
    """
    Recover the raw (un-normalized) per-scan quantities.

    Each spectrum is normalized per scan by its 250-1250 background, and that background
    carries its own day/night pattern. Multiplying back by it recovers the physical
    magnitude -- e.g. the G-band area, which is nearly flat day/night once the
    normalization is undone. Idempotent, so it is safe to call more than once.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with the normalized columns and the background column.

    Returns
    -------
    pandas.DataFrame
        DataFrame with ``Raw_*`` columns added (unchanged if the background is absent).
    """
    if BACKGROUND_COLUMN not in df.columns:
        return df
    for norm_col, raw_col in _RAW_COLUMN_MAP:
        if norm_col in df.columns:
            df[raw_col] = df[norm_col] * df[BACKGROUND_COLUMN]
    return df


def _safe_ratio(numerator, denominator):
    """Element-wise ratio with infinities mapped to NaN."""
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def calculate_ratios(df):
    """
    Calculate fluorescence ratios (before and after baseline correction) and add the
    raw (un-normalized) peak columns.

    The uncorrected ratio is identical in either space, because the per-scan background
    cancels: Normalized_fluor / Normalized_gband == Raw_fluor / Raw_gband. The
    baseline-corrected ratio is computed from whichever corrected channels the baseline
    method produced (raw space is preferred; see the ``stitch_space`` setting).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with normalized data and optionally baseline-corrected data

    Returns
    -------
    pandas.DataFrame
        DataFrame with ratio and raw columns added
    """
    df = df.copy()

    # Raw (un-normalized) per-scan quantities.
    df = add_raw_columns(df)

    # Uncorrected ratios (background cancels, so the space does not matter).
    if 'Normalized_Fluorescence_Intensity' in df.columns and 'Normalized_Gband_Area' in df.columns:
        df['Fluorescence_to_Gband_Ratio'] = _safe_ratio(
            df['Normalized_Fluorescence_Intensity'], df['Normalized_Gband_Area'])

    if 'Normalized_Fluorescence_Intensity' in df.columns and 'Normalized_Raman_Peak_850_Area' in df.columns:
        df['Fluorescence_to_Raman_Peak_850_Ratio'] = _safe_ratio(
            df['Normalized_Fluorescence_Intensity'], df['Normalized_Raman_Peak_850_Area'])

    # Baseline-corrected ratios, from whichever corrected space is present.
    for fluor_col, gband_col, peak850_col in _CORRECTED_TRIPLES:
        if fluor_col not in df.columns:
            continue
        if gband_col in df.columns:
            df['Fluorescence_to_Gband_Ratio_BaselineCorrected'] = _safe_ratio(
                df[fluor_col], df[gband_col])
        if peak850_col in df.columns:
            df['Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected'] = _safe_ratio(
                df[fluor_col], df[peak850_col])
        break

    return df
