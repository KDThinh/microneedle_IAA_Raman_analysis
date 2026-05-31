"""Data loading functions for Raman datasets."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .utils import find_google_drive, correct_wavenumber_for_excitation


class IngestionError(Exception):
    """Base class for ingestion-related errors."""


class FileMissingError(IngestionError):
    """Raised when the Raman file cannot be located."""


class SchemaMismatchError(IngestionError):
    """Raised when the Raman TSV does not match the expected schema."""


class DatetimeParseError(IngestionError):
    """Raised when date/time columns cannot be parsed."""


class ConfigError(IngestionError):
    """Raised when the ingestion configuration is invalid."""


@dataclass
class RamanDataset:
    """Container for loaded Raman dataset."""
    spectra: pd.DataFrame
    wavenumbers: np.ndarray
    metadata: Dict[str, Any]
    source_path: str
    warnings: List[str] = field(default_factory=list)


def find_project_root() -> Optional[Path]:
    """
    Search for the specific project root directory on common drives.
    This finds the folder that CONTAINS 'IAA Nanosensor Experiment'.
    
    Returns:
    --------
    Path or None
        Path to project root if found, None otherwise
    """
    # The suffix that connects your Drive root to the Experiment folder
    project_path_suffix = "DiSTAP/Research/Auxin IAA/IAA-MN longitudinal"
    
    candidates = [
        f"H:/My Drive/Work/{project_path_suffix}",
        f"G:/My Drive/Work/{project_path_suffix}",
        "G:/My Drive/Work",
        "H:/My Drive/Work",
        os.path.expanduser("~/Google Drive/Work"),
    ]
    
    # Also try to find Google Drive and construct path
    try:
        drive_root = find_google_drive()
        if drive_root:
            candidates.insert(0, os.path.join(drive_root, "My Drive", "Work", project_path_suffix))
            candidates.insert(1, os.path.join(drive_root, "My Drive", "Work"))
    except Exception:
        pass
    
    for path in candidates:
        path_obj = Path(path)
        if path_obj.exists():
            return path_obj
    return None


def resolve_path(relative_path_str: Optional[str]) -> Optional[Path]:
    """
    Smartly resolve a path that might be:
    1. Absolute (use as is)
    2. Relative to current folder (use as is)
    3. Relative to Google Drive Project Root (find root and join)
    
    Parameters:
    -----------
    relative_path_str : str or None
        Path string to resolve
    
    Returns:
    --------
    Path or None
        Resolved path if found, None if input was None
    """
    if not relative_path_str:
        return None
        
    path_obj = Path(relative_path_str)
    
    # 1. Check if it exists as-is (absolute or relative to current dir)
    if path_obj.exists():
        return path_obj.resolve()
    
    # 2. Try finding it via Project Root on Google Drive
    project_root = find_project_root()
    if project_root:
        # Strip leading slash to ensure clean join
        clean_rel = str(relative_path_str).lstrip("/\\")
        # Try joining with project root
        potential_path = project_root / clean_rel
        if potential_path.exists():
            return potential_path.resolve()
        
        # Try with "IAA Nanosensor Experiment" prefix if not already present
        if "IAA Nanosensor Experiment" not in clean_rel:
            potential_path = project_root / "IAA Nanosensor Experiment" / clean_rel
            if potential_path.exists():
                return potential_path.resolve()
    
    # 3. Try traditional Google Drive resolution (backward compatibility)
    try:
        drive_root = find_google_drive()
        if drive_root:
            base_path = Path(drive_root) / "My Drive" / "Work"
            potential_path = base_path / relative_path_str
            if potential_path.exists():
                return potential_path.resolve()
    except Exception:
        pass
    
    # Return original path (will throw error later if still not found)
    return path_obj


def _resolve_path(relative_path: str, requires_google_drive: bool) -> str:
    """
    Legacy path resolver for backward compatibility.
    Use resolve_path() for new code.
    """
    resolved = resolve_path(relative_path)
    if resolved and resolved.exists():
        return str(resolved)
    
    # Fall back to old behavior
    if os.path.isabs(relative_path):
        return relative_path
    base_path = ""
    if requires_google_drive:
        base_path = find_google_drive()
        if not base_path:
            raise FileMissingError(
                "Google Drive path could not be located. "
                "Ensure Drive is mounted or set requires_google_drive=false."
            )
        base_path = os.path.join(base_path, "My Drive", "Work")
    return os.path.abspath(os.path.join(base_path, relative_path))


def _validate_savgol(window_size: int, spectrum_length: int) -> int:
    """Validate and adjust Savitzky-Golay window size."""
    adjusted_window = window_size
    if adjusted_window % 2 == 0:
        adjusted_window += 1
    if adjusted_window > spectrum_length:
        adjusted_window = spectrum_length - 1 if spectrum_length % 2 == 0 else spectrum_length
    if adjusted_window < 3:
        raise SchemaMismatchError(
            f"Savitzky-Golay window size {window_size} is incompatible with "
            f"spectrum length {spectrum_length}."
        )
    return adjusted_window


def _parse_datetime_columns(dates: pd.Series, times: pd.Series) -> pd.DatetimeIndex:
    """Parse datetime columns with multiple format attempts."""
    formats: Sequence[str] = (
        "%m-%d-%Y %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    )
    combined = dates.astype(str).str.replace("/", "-", regex=False) + " " + times.astype(str)
    for fmt in formats:
        try:
            parsed = pd.to_datetime(combined, format=fmt, errors="raise")
            return pd.DatetimeIndex(parsed)
        except (ValueError, TypeError):
            continue
    raise DatetimeParseError(
        "Unable to parse datetime columns. Please verify the date/time format."
    )


def load_raman_dataset(
    config: Dict[str, Any],
    metadata_overrides: Optional[Dict[str, Any]] = None,
) -> RamanDataset:
    """
    Load Raman TSV data, validate schema, and return a standardized dataset.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing data_source and processing parameters
    metadata_overrides : dict, optional
        Metadata overrides to apply
    
    Returns:
    --------
    RamanDataset
        Loaded and validated dataset
    """
    data_source = config.get("data_source", {})
    raman_rel_path = data_source.get("raman_relative_path")
    if not raman_rel_path:
        raise ConfigError("Configuration is missing 'raman_relative_path'.")

    # Use new generic path resolver
    file_path = resolve_path(raman_rel_path)
    if not file_path or not file_path.exists():
        raise FileMissingError(f"Raman data file not found: {raman_rel_path} (resolved to: {file_path})")
    
    file_path_str = str(file_path)

    df = pd.read_csv(file_path_str, sep="\t", header=None)
    if df.shape[1] < 4:
        raise SchemaMismatchError(
            "Expected at least four columns (date, time, seconds, spectra...)."
        )

    wavenumbers = df.iloc[0, 3:].astype(float).values
    
    # Check if wavenumber correction is needed
    recorded_excitation_nm = config.get("recorded_excitation_nm")
    if recorded_excitation_nm is None:
        processing_cfg = config.get("processing", {})
        if not processing_cfg and "sections" in config:
            processing_cfg = config.get("sections", {}).get("processing", {})
        recorded_excitation_nm = processing_cfg.get("recorded_excitation_nm")
    
    actual_excitation_nm = config.get("excitation_nm")
    if actual_excitation_nm is None:
        processing_cfg = config.get("processing", {})
        if not processing_cfg and "sections" in config:
            processing_cfg = config.get("sections", {}).get("processing", {})
        actual_excitation_nm = processing_cfg.get("excitation_nm")
    
    # Apply wavenumber correction if recorded_excitation_nm is specified and differs from actual
    if recorded_excitation_nm is not None and actual_excitation_nm is not None:
        if recorded_excitation_nm != actual_excitation_nm:
            wavenumbers = correct_wavenumber_for_excitation(
                wavenumbers, 
                recorded_excitation_nm=recorded_excitation_nm, 
                actual_excitation_nm=actual_excitation_nm
            )
    
    wavenumbers_diff = np.diff(wavenumbers)
    if np.any(wavenumbers_diff <= 0):
        raise SchemaMismatchError("Wavenumber axis must be strictly increasing.")

    dates = df.iloc[1:, 0]
    times = df.iloc[1:, 1]
    seconds = pd.to_numeric(df.iloc[1:, 2], errors="coerce").fillna(0.0)
    datetimes = _parse_datetime_columns(dates, times)

    spectra = pd.DataFrame(np.array(df.iloc[1:, 3:], dtype=float), columns=wavenumbers)
    spectra.insert(0, "Seconds", seconds.values)
    spectra.insert(0, "Scan Number", range(1, len(spectra) + 1))
    spectra["Datetime"] = datetimes.values
    spectra.set_index("Datetime", inplace=True)

    # Incomplete acquisitions often leave a truncated final row (fewer tab-separated
    # fields than the header). Pandas pads missing cells with NaN, which later breaks
    # polyfit / curve_fit. Drop any row with non-finite spectral values.
    spectral_cols = [c for c in spectra.columns if c not in ("Scan Number", "Seconds")]
    spec_mat = spectra[spectral_cols].to_numpy(dtype=float, copy=False)
    row_ok = np.isfinite(spec_mat).all(axis=1)
    n_bad = int((~row_ok).sum())
    if n_bad:
        log = logging.getLogger(__name__)
        log.warning(
            "Dropped %d Raman scan row(s) with NaN or Inf in spectral channels "
            "(often a truncated final line while the spectrometer is still writing). "
            "Source: %s",
            n_bad,
            file_path_str,
        )
        spectra = spectra.loc[row_ok]
    if spectra.empty:
        raise SchemaMismatchError(
            "All Raman spectral rows contained non-finite values after filtering; "
            "check raw file integrity."
        )

    processing_cfg = config.get("sections", {}).get("processing", {})
    if "spectral_sg_window" in processing_cfg:
        adjusted = _validate_savgol(processing_cfg["spectral_sg_window"], len(wavenumbers))
        if adjusted != processing_cfg["spectral_sg_window"]:
            processing_cfg["spectral_sg_window"] = adjusted
            config["spectral_sg_window"] = adjusted

    metadata = config.get("metadata", {}).copy()
    overrides = metadata_overrides or {}
    for key, value in overrides.items():
        if value is not None:
            metadata[key] = value

    warnings: List[str] = []
    if n_bad:
        warnings.append(
            f"Dropped {n_bad} incomplete spectral row(s) containing NaN or Inf."
        )
    if len(spectra.index.unique()) != len(spectra.index):
        warnings.append("Duplicate datetime entries detected in Raman data.")

    return RamanDataset(
        spectra=spectra,
        wavenumbers=wavenumbers,
        metadata=metadata,
        source_path=file_path_str,
        warnings=warnings,
    )


def load_temperature_data(config: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Load Temperature/Humidity dataset from configuration.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing data_source with temp_relative_path
    
    Returns:
    --------
    pandas.DataFrame or None
        Temperature data with datetime index, or None if not configured/not found
    """
    data_source = config.get("data_source", {})
    temp_rel_path = data_source.get("temp_relative_path")
    
    # Temperature data is often optional
    if not temp_rel_path:
        logger = logging.getLogger(__name__)
        logger.info("No temperature file configured.")
        return None

    # Use generic path resolver
    file_path = resolve_path(temp_rel_path)

    if not file_path or not file_path.exists():
        logger = logging.getLogger(__name__)
        logger.warning(f"Temperature file configured but not found: {temp_rel_path} (resolved to: {file_path})")
        return None

    logger = logging.getLogger(__name__)
    logger.info(f"Loading Temperature data from: {file_path}")
    
    # Load temperature CSV (assuming standard CSV structure)
    try:
        # Try common date column names
        date_columns = ['Timestamp', 'DateTime', 'Date', 'Time', 'datetime']
        temp_df = pd.read_csv(file_path)
        
        # Find datetime column
        datetime_col = None
        for col in date_columns:
            if col in temp_df.columns:
                datetime_col = col
                break
        
        if datetime_col:
            temp_df[datetime_col] = pd.to_datetime(temp_df[datetime_col], errors='coerce')
            temp_df.set_index(datetime_col, inplace=True)
        else:
            # Try parsing first column as datetime
            try:
                temp_df.iloc[:, 0] = pd.to_datetime(temp_df.iloc[:, 0], errors='coerce')
                temp_df.set_index(temp_df.columns[0], inplace=True)
            except Exception:
                logger.warning("Could not parse datetime column in temperature data")
        
        # Remove any rows with NaT in index
        temp_df = temp_df[temp_df.index.notna()]
        
        logger.info(f"Loaded {len(temp_df)} temperature records")
        return temp_df
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading temperature data: {e}")
        return None

