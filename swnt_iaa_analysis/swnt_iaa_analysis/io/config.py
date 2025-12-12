"""Configuration management for SWNT IAA analysis."""

import copy
import os
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

