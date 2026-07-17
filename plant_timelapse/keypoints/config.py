"""Central configuration for the keypoint pipeline.

Imaging conventions:
  - SENSOR_MAX = 4095 (12-bit camera), absolute-range 8-bit scaling. This matches
    the segmentation pipeline so a pixel means the same thing day/night.
  - ROI = None -> the keypoint model uses the FULL sensor frame (1944x2592).

Why full frame (no crop)? A fixed crop clips the apical meristem once the plant
bolts and grows above the crop's top edge, and any single crop risks clipping
FUTURE datasets where the plant is taller, shifted, or leaning. The full frame is
the only truly future-proof framing for this rig. Stem height is a vertical pixel
distance (base_y - meristem_y), so removing the crop does not change the measured
height -- it only removes the clipping risk.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

# --- imaging conventions ---
SENSOR_MAX = 4095
# (x0, y0, x1, y1) crop applied as img[y0:y1, x0:x1], or None for the full frame.
ROI = None

# --- keypoints ---
KEYPOINTS = ["base", "meristem"]

# --- light schedule (from the capture folder name: Light_6to22) ---
LIGHT_ON_HOUR = 6
LIGHT_OFF_HOUR = 22

# --- datasets (same rig, same species) ---
# name -> folder of *.tif frames
DATASETS = OrderedDict(
    [
        (
            "run2",
            r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Drought\Light_6to22\Temp_Hum_Variable\Run 2\Raw data\Time-lapse images\timelapse_2026-02-09_16-44-08",
        ),
        (
            "run4",
            r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 4\DEV_1AB22C05B465\timelapse_2026-05-07_12-56-08",
        ),
        (
            "run5",
            r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 5\DEV_1AB22C05B465\timelapse_2026-05-16_16-16-30",
        ),
    ]
)

# --- DeepLabCut project layout (all generated, gitignored) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DLC_DIR = REPO_ROOT / "plant_timelapse" / "dlc"
VIDEO_DIR = DLC_DIR / "videos_prepared"     # prepared 8-bit mp4s DLC ingests (full frame)
EXPORT_DIR = DLC_DIR / "exported_frames"    # optional PNG exports for QC
POINTER_FILE = DLC_DIR / "PROJECT_CONFIG.txt"  # stores path to the active config.yaml

PROJECT_NAME = "plant_meristem"
EXPERIMENTER = "ryank"

# how many frames DLC's kmeans should pick per video for labeling
NUMFRAMES2PICK = 50

# default backbone for the PyTorch engine
DEFAULT_NET_TYPE = "resnet_50"


def dataset_items():
    """Yield (name, Path) for each configured dataset."""
    for name, path in DATASETS.items():
        yield name, Path(path)


def video_path(name: str) -> Path:
    """Path to the prepared 8-bit mp4 for a dataset."""
    return VIDEO_DIR / f"{name}.mp4"
