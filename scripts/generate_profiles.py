"""
Auto-generate pipeline profiles for each Raw data file under the IAA Nanosensor Experiment tree.

Usage:
    python scripts/generate_profiles.py \
        --base-profile in_planta_default \
        --profile-prefix auto \
        --overwrite
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in os.sys.path:
    os.sys.path.append(str(SRC_DIR))

from pipeline.utils import find_google_drive  # type: ignore  # noqa

DEFAULT_DATA_ROOT = Path(
    "DiSTAP/Research/Auxin IAA/IAA-MN longitudinal/IAA Nanosensor Experiment"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pipeline profiles from raw data folders.")
    parser.add_argument(
        "--config-path",
        default=PROJECT_ROOT / "config" / "pipeline.yml",
        type=Path,
        help="Path to pipeline configuration YAML.",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Root directory corresponding to GoogleDrive\\My Drive\\Work. "
        "Defaults to auto-detect via find_google_drive().",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path from work-root to the 'IAA Nanosensor Experiment' directory.",
    )
    parser.add_argument(
        "--base-profile",
        default="in_planta_default",
        help="Profile whose settings will be copied when creating new entries.",
    )
    parser.add_argument(
        "--profile-prefix",
        default="auto",
        help="Prefix to add to generated profile names (e.g., 'auto').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing profiles with the same name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without modifying pipeline.yml.",
    )
    return parser.parse_args()


def sanitize_token(token: str) -> str:
    return token.strip().replace(" ", "_").lower()


def infer_metadata(raw_file: Path, experiment_root: Path) -> Optional[Dict[str, str]]:
    """
    Infer metadata from a path structure like:
    .../IAA Nanosensor Experiment/<plant>/Treatment_<treatment>/Light_<cycle>/Temp_Hum_<state>/Run X/Raw data/file.txt
    """
    try:
        relative = raw_file.relative_to(experiment_root)
    except ValueError:
        return None

    parts = relative.parts
    if len(parts) < 6 or parts[-2] != "Raw data":
        return None

    plant = parts[0]
    treatment_segment = parts[1]
    light_segment = parts[2]
    temp_segment = parts[3]
    run_segment = parts[4] if parts[4].lower().startswith("run") else None

    if not (
        treatment_segment.startswith("Treatment_")
        and light_segment.startswith("Light_")
        and temp_segment.startswith("Temp_Hum_")
    ):
        return None

    treatment = treatment_segment.replace("Treatment_", "")
    light_cycle = light_segment.replace("Light_", "")
    temp_control_raw = temp_segment.replace("Temp_Hum_", "")
    temp_hum_control = "Yes" if temp_control_raw.lower() == "constant" else "No"

    replicate_number = 1
    if run_segment and " " in run_segment:
        try:
            replicate_number = int(run_segment.split(" ")[-1])
        except ValueError:
            pass

    profile_slug = "-".join(
        sanitize_token(token)
        for token in [
            plant,
            treatment,
            light_cycle,
            temp_control_raw,
            f"run{replicate_number}",
        ]
    )

    return {
        "plant_type": plant.replace("_", " "),
        "treatment": treatment,
        "light_cycle": light_cycle,
        "temp_hum_control": temp_hum_control,
        "replicate_number": replicate_number,
        "profile_slug": profile_slug,
        "relative_path": relative,
    }


def main() -> None:
    args = parse_args()

    if not args.config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config_path}")

    work_root = args.work_root
    if work_root is None:
        drive_root = Path(find_google_drive())
        work_root = drive_root / "My Drive" / "Work"
    work_root = work_root.resolve()
    experiment_root = (work_root / args.data_root).resolve()
    if not experiment_root.exists():
        raise FileNotFoundError(f"Experiment root not found: {experiment_root}")

    with open(args.config_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    profiles = config.setdefault("profiles", {})
    if args.base_profile not in profiles:
        raise KeyError(f"Base profile '{args.base_profile}' not found in config.")

    additions = {}
    for raw_file in experiment_root.rglob("Raw data/*.txt"):
        metadata = infer_metadata(raw_file, experiment_root)
        if not metadata:
            continue

        profile_name = f"{args.profile_prefix}-{metadata['profile_slug']}".strip("-")
        if profile_name in profiles and not args.overwrite:
            continue

        new_profile: Dict[str, Dict] = {
            "inherits": args.base_profile,
            "data_source": {},
            "metadata": {},
        }
        data_source = new_profile["data_source"]
        rel_path = raw_file.relative_to(work_root)
        data_source["raman_relative_path"] = str(rel_path).replace("\\", "/")

        metadata_block = new_profile["metadata"]
        metadata_block["plant_type"] = metadata["plant_type"]
        metadata_block["treatment"] = metadata["treatment"]
        metadata_block["light_cycle"] = metadata["light_cycle"]
        metadata_block["temp_hum_control"] = metadata["temp_hum_control"]
        metadata_block["replicate_number"] = metadata["replicate_number"]

        additions[profile_name] = new_profile

    if not additions:
        print("No new profiles generated.")
        return

    print(f"Prepared {len(additions)} profile(s). Overwrite existing: {args.overwrite}. Dry run: {args.dry_run}.")
    if args.dry_run:
        for name in sorted(additions):
            print(f"- {name}")
        return

    profiles.update(additions)
    sorted_profiles = dict(sorted(profiles.items()))
    config["profiles"] = sorted_profiles

    with open(args.config_path, "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    print(f"Updated {args.config_path} with {len(additions)} profile(s).")


if __name__ == "__main__":
    main()

