"""Configuration management for SWNT IAA analysis."""

import copy
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError as exc:
    raise ImportError(
        "PyYAML is required to load pipeline configuration. "
        "Install it with 'pip install pyyaml' and rerun the script."
    ) from exc


class ConfigError(Exception):
    """Raised when configuration loading fails."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _flatten_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten profile structure for easier access."""
    flat_config: Dict[str, Any] = {}
    # Flatten processing and outputs sections to top level
    for section in ("processing", "outputs"):
        if section in profile:
            flat_config.update(profile[section])
    # Include other top-level keys
    for key, value in profile.items():
        if key not in ("processing", "outputs", "data_source", "metadata", "inherits", "description"):
            flat_config[key] = value
    # Preserve nested sections
    flat_config["data_source"] = profile.get("data_source", {}).copy()
    flat_config["metadata"] = profile.get("metadata", {}).copy()
    flat_config["sections"] = copy.deepcopy(profile)
    return flat_config


def _resolve_profile(profiles: Dict[str, Any], profile_name: str, visited: set) -> Dict[str, Any]:
    """Recursively resolve a profile with all its inheritance chain."""
    if profile_name not in profiles:
        raise ConfigError(f"Profile '{profile_name}' not found in configuration file.")
    
    if profile_name in visited:
        raise ConfigError(f"Circular inheritance detected involving profile '{profile_name}'.")
    
    profile = copy.deepcopy(profiles[profile_name])
    inherits = profile.get("inherits")
    
    if inherits:
        visited.add(profile_name)
        parent_profile = _resolve_profile(profiles, inherits, visited)
        visited.remove(profile_name)
        merged_profile = _deep_merge(parent_profile, profile)
    else:
        merged_profile = profile
    
    return merged_profile


def load_profile_config(
    config_path: str,
    profile_name: str,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load and resolve a profile from the YAML configuration file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
    profile_name : str
        Name of profile to load
    overrides : list, optional
        List of override strings in format "key=value"
    
    Returns:
    --------
    dict
        Flattened configuration dictionary
    """
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise ConfigError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as stream:
        config_data = yaml.safe_load(stream) or {}

    profiles = config_data.get("profiles")
    if not profiles:
        raise ConfigError("No profiles defined in configuration file.")
    if profile_name not in profiles:
        raise ConfigError(f"Profile '{profile_name}' not found in configuration file.")

    # Recursively resolve inheritance chain
    merged_profile = _resolve_profile(profiles, profile_name, set())
    
    # Apply overrides if provided
    if overrides:
        for override in overrides:
            if "=" not in override:
                continue
            dotted_key, raw_value = override.split("=", 1)
            keys = dotted_key.split(".")
            target = merged_profile
            for key in keys[:-1]:
                if key not in target or not isinstance(target[key], dict):
                    target[key] = {}
                target = target[key]
            leaf = keys[-1]
            # Try to coerce value
            if raw_value.lower() in {"true", "false"}:
                coerced_value = raw_value.lower() == "true"
            else:
                try:
                    coerced_value = int(raw_value)
                except ValueError:
                    try:
                        coerced_value = float(raw_value)
                    except ValueError:
                        coerced_value = raw_value
            target[leaf] = coerced_value
    
    return _flatten_profile(merged_profile)


def list_profiles(
    config_path: str,
    experiment_type: Optional[str] = None,
    exclude_base_profiles: bool = True,
) -> List[str]:
    """
    List profile names from the YAML configuration file.
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
    experiment_type : str, optional
        Filter by experiment type
    exclude_base_profiles : bool
        Exclude base profiles like 'processing_default'
    
    Returns:
    --------
    list
        Sorted list of profile names
    """
    if not os.path.exists(config_path):
        raise ConfigError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as stream:
        config_data = yaml.safe_load(stream) or {}

    profiles = config_data.get("profiles", {})
    
    matching_profiles = []
    for profile_name in profiles.keys():
        if exclude_base_profiles and profile_name == 'processing_default':
            continue
        
        if experiment_type is not None:
            profile = profiles[profile_name]
            metadata = profile.get('metadata', {})
            profile_experiment = metadata.get('experiment', '')
            if profile_experiment != experiment_type:
                continue
        
        matching_profiles.append(profile_name)
    
    return sorted(matching_profiles)


def find_latest_results_folder(config_path: str, profile_name: str) -> Optional[Path]:
    """
    Find the latest results_v4_* folder for a given profile.
    
    Strategy:
    1. Load profile config to get data file path
    2. Look in the data file's parent directory for results_v4_* folders
    3. Return the most recent one (by timestamp in folder name)
    
    Parameters:
    -----------
    config_path : str
        Path to YAML config file
    profile_name : str
        Name of profile to find results for
    
    Returns:
    --------
    Path or None
        Path to latest results folder, or None if not found
    """
    # Load config to get data file location
    try:
        config = load_profile_config(config_path, profile_name)
    except ConfigError:
        return None
    
    # Get data file path from config
    data_source = config.get('data_source', {})
    raman_path_str = data_source.get('raman_relative_path')
    
    if not raman_path_str:
        return None
    
    # Resolve path (using same logic as loader)
    from ..core.loader import resolve_path
    raman_path = resolve_path(raman_path_str)
    
    if not raman_path or not raman_path.exists():
        return None
    
    # Look in the parent directory for results_v4_* folders
    data_dir = raman_path.parent
    
    # Find all results_v4_* folders
    pattern = re.compile(r'results_v4_\d{8}_\d{6}$')
    results_folders = []
    
    for item in data_dir.iterdir():
        if item.is_dir() and pattern.match(item.name):
            results_folders.append(item)
    
    if not results_folders:
        return None
    
    # Sort by modification time (newest first) as fallback
    # But prefer sorting by timestamp in name for consistency
    def get_timestamp(folder_path: Path) -> tuple:
        """Extract timestamp from folder name for sorting."""
        match = re.search(r'(\d{8})_(\d{6})', folder_path.name)
        if match:
            date_str, time_str = match.groups()
            return (date_str, time_str)
        # Fallback to modification time if no timestamp found
        return (0, 0)
    
    results_folders.sort(key=get_timestamp, reverse=True)
    
    return results_folders[0]

