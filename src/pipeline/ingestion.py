from __future__ import annotations

import os
from dataclasses import dataclass, field
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


@dataclass
class RamanDataset:
    spectra: pd.DataFrame
    wavenumbers: np.ndarray
    metadata: Dict[str, Any]
    source_path: str
    warnings: List[str] = field(default_factory=list)


def _resolve_path(relative_path: str, requires_google_drive: bool) -> str:
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
    """
    data_source = config.get("data_source", {})
    requires_drive = data_source.get("requires_google_drive", True)
    raman_rel_path = data_source.get("raman_relative_path")
    if not raman_rel_path:
        raise ConfigError("Configuration is missing 'raman_relative_path'.")

    file_path = _resolve_path(raman_rel_path, requires_drive)
    if not os.path.exists(file_path):
        raise FileMissingError(f"Raman data file not found: {file_path}")

    df = pd.read_csv(file_path, sep="\t", header=None)
    if df.shape[1] < 4:
        raise SchemaMismatchError(
            "Expected at least four columns (date, time, seconds, spectra...)."
        )

    wavenumbers = df.iloc[0, 3:].astype(float).values
    
    # Check if wavenumber correction is needed
    # Config flattens processing parameters to top level, but also preserves nested structure
    # Check both top level and nested sections for compatibility
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

    processing_cfg = config.get("sections", {}).get("processing", {})
    if "window_size" in processing_cfg:
        adjusted = _validate_savgol(processing_cfg["window_size"], len(wavenumbers))
        if adjusted != processing_cfg["window_size"]:
            processing_cfg["window_size"] = adjusted
            config["window_size"] = adjusted

    metadata = config.get("metadata", {}).copy()
    overrides = metadata_overrides or {}
    for key, value in overrides.items():
        if value is not None:
            metadata[key] = value

    warnings: List[str] = []
    if len(datetimes.unique()) != len(datetimes):
        warnings.append("Duplicate datetime entries detected in Raman data.")

    return RamanDataset(
        spectra=spectra,
        wavenumbers=wavenumbers,
        metadata=metadata,
        source_path=file_path,
        warnings=warnings,
    )


class ConfigError(IngestionError):
    """Raised when the ingestion configuration is invalid."""

