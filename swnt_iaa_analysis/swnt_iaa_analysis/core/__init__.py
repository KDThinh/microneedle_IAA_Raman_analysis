"""Core processing modules for data loading, preprocessing, and baseline correction."""

from .loader import (
    load_raman_dataset,
    load_temperature_data,
    resolve_path,
    find_project_root,
    RamanDataset,
)

__all__ = [
    'load_raman_dataset',
    'load_temperature_data',
    'resolve_path',
    'find_project_root',
    'RamanDataset',
]

