import copy
import os
from typing import Any, Dict, List

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to load pipeline configuration. "
        "Install it with 'pip install pyyaml' and rerun the script."
    ) from exc


class ConfigError(Exception):
    """Raised when configuration loading fails."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _apply_overrides(config: Dict[str, Any], overrides: List[str]) -> None:
    for override in overrides or []:
        if "=" not in override:
            raise ConfigError(
                f"Invalid override '{override}'. Use dotted paths like section.key=value."
            )
        dotted_key, raw_value = override.split("=", 1)
        keys = dotted_key.split(".")
        target = config
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]
        leaf = keys[-1]
        coerced_value: Any = raw_value
        if raw_value.lower() in {"true", "false"}:
            coerced_value = raw_value.lower() == "true"
        else:
            try:
                coerced_value = int(raw_value)
            except ValueError:
                try:
                    coerced_value = float(raw_value)
                except ValueError:
                    pass
        target[leaf] = coerced_value


def _flatten_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    flat_config: Dict[str, Any] = {}
    # Flatten processing and outputs sections to top level
    for section in ("processing", "outputs"):
        if section in profile:
            flat_config.update(profile[section])
    # Include other top-level keys (like prompt_datetime)
    for key, value in profile.items():
        if key not in ("processing", "outputs", "data_source", "metadata", "inherits", "description"):
            flat_config[key] = value
    # Preserve nested sections for richer access
    flat_config["data_source"] = profile.get("data_source", {}).copy()
    flat_config["metadata"] = profile.get("metadata", {}).copy()
    flat_config["sections"] = copy.deepcopy(profile)

    # Backward-compatible path shortcuts
    data_source = profile.get("data_source", {})
    if "raman_relative_path" in data_source:
        flat_config["raman_relative_path"] = data_source["raman_relative_path"]
    if "temp_relative_path" in data_source:
        flat_config["temp_relative_path"] = data_source["temp_relative_path"]
    for meta_key, meta_value in profile.get("metadata", {}).items():
        flat_config[meta_key] = meta_value
    return flat_config


def _resolve_profile(profiles: Dict[str, Any], profile_name: str, visited: set) -> Dict[str, Any]:
    """
    Recursively resolve a profile with all its inheritance chain.
    
    Args:
        profiles: Dictionary of all profiles
        profile_name: Name of the profile to resolve
        visited: Set of already visited profiles (to detect circular dependencies)
    
    Returns:
        Fully resolved profile with all inherited values merged
    """
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


def list_profiles(
    config_path: str,
    experiment_type: str | None = None,
    exclude_base_profiles: bool = True,
) -> List[str]:
    """
    List profile names from the YAML configuration file, optionally filtered by experiment type.
    
    Parameters:
    -----------
    config_path : str
        Path to pipeline.yml config file
    experiment_type : str, optional
        Filter profiles by experiment type (e.g., 'in planta', 'in vitro').
        If None, returns all profiles (except base profiles if exclude_base_profiles=True).
    exclude_base_profiles : bool, optional
        If True, excludes base profiles like 'processing_default' (default: True)
    
    Returns:
    --------
    List[str]
        Sorted list of profile names matching the criteria
    """
    resolved_path = config_path or os.environ.get("IAA_PIPELINE_CONFIG")
    if not resolved_path:
        resolved_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "config",
            "pipeline.yml",
        )
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.abspath(resolved_path)
    if not os.path.exists(resolved_path):
        raise ConfigError(f"Configuration file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as stream:
        config_data = yaml.safe_load(stream) or {}

    profiles = config_data.get("profiles", {})
    
    # Filter profiles
    matching_profiles = []
    for profile_name in profiles.keys():
        # Skip base profiles if requested
        if exclude_base_profiles and profile_name == 'processing_default':
            continue
        
        # Filter by experiment type if specified
        if experiment_type is not None:
            profile = profiles[profile_name]
            metadata = profile.get('metadata', {})
            profile_experiment = metadata.get('experiment', '')
            if profile_experiment != experiment_type:
                continue
        
        matching_profiles.append(profile_name)
    
    return sorted(matching_profiles)


def get_profiles_by_experiment(
    config_path: str,
    experiment_type: str,
) -> List[str]:
    """
    Get all profile names filtered by experiment type.
    
    This is a convenience wrapper around list_profiles() for backward compatibility.
    
    Parameters:
    -----------
    config_path : str
        Path to pipeline.yml config file
    experiment_type : str
        Experiment type to filter by (e.g., 'in planta', 'in vitro')
    
    Returns:
    --------
    List[str]
        Sorted list of profile names with the specified experiment type
    """
    return list_profiles(config_path, experiment_type=experiment_type, exclude_base_profiles=True)


def load_profile_config(
    config_path: str,
    profile_name: str,
    overrides: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Load and resolve a profile from the YAML configuration file.
    """
    resolved_path = config_path or os.environ.get("IAA_PIPELINE_CONFIG")
    if not resolved_path:
        resolved_path = os.path.join(
            os.path.dirname(__file__),
            os.pardir,
            os.pardir,
            "config",
            "pipeline.yml",
        )
    if not os.path.isabs(resolved_path):
        resolved_path = os.path.abspath(resolved_path)
    if not os.path.exists(resolved_path):
        raise ConfigError(f"Configuration file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as stream:
        config_data = yaml.safe_load(stream) or {}

    profiles = config_data.get("profiles")
    if not profiles:
        raise ConfigError("No profiles defined in configuration file.")
    if profile_name not in profiles:
        raise ConfigError(f"Profile '{profile_name}' not found in configuration file.")

    # Recursively resolve inheritance chain
    merged_profile = _resolve_profile(profiles, profile_name, set())

    _apply_overrides(merged_profile, overrides)
    return _flatten_profile(merged_profile)

