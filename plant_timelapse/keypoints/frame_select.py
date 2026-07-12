"""Frame selection + timestamp parsing helpers.

DeepLabCut's kmeans extractor handles most of the "pick diverse frames" job, but
these helpers let us reason about time/day-night coverage and optionally hand-pick
an evenly spaced subset (e.g. for PNG export or QC).

Filenames look like: tl_2026-05-07_12-56-08.tif
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import LIGHT_OFF_HOUR, LIGHT_ON_HOUR

_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})")


def parse_timestamp(name) -> Optional[datetime]:
    """Extract the capture time from a frame filename, or None if absent.

    Parses only the file's basename so a timestamped parent folder (e.g.
    ``timelapse_2026-05-07_12-56-08``) never shadows the per-frame timestamp.
    """
    m = _TS_RE.search(Path(name).name)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)}_{m.group(2)}", "%Y-%m-%d_%H-%M-%S")


def is_daytime(dt: datetime, on: int = LIGHT_ON_HOUR, off: int = LIGHT_OFF_HOUR) -> bool:
    """True if the timestamp falls within the light-on window [on, off)."""
    return on <= dt.hour < off


def select_even(frames: Sequence[Path], n: int) -> list[Path]:
    """Pick n frames evenly spaced across the (time-sorted) sequence."""
    frames = list(frames)
    if n >= len(frames):
        return frames
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in sorted(set(idx.tolist()))]


def summarize_day_night(frames: Sequence[Path]) -> dict:
    """Count day vs night vs untimed frames for a set of paths."""
    day = night = untimed = 0
    for f in frames:
        dt = parse_timestamp(f)
        if dt is None:
            untimed += 1
        elif is_daytime(dt):
            day += 1
        else:
            night += 1
    return {"day": day, "night": night, "untimed": untimed, "total": len(list(frames))}
