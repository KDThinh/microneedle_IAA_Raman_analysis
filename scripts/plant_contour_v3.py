#!/usr/bin/env python3
"""Run plant timelapse contour segmentation (delegates to plant_timelapse.plant_contour_v3)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from plant_timelapse.plant_contour_v3 import main

if __name__ == "__main__":
    main()
