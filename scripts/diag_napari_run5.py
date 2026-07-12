"""Diagnose where napari hangs when opening run5 labeled-data."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PROJ = REPO / "plant_timelapse" / "dlc" / "plant_meristem-ryank-2026-07-08"
CFG = PROJ / "config.yaml"
RUN5 = PROJ / "labeled-data" / "run5"


def tick(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    tick("import napari")
    import napari

    tick("create Viewer")
    viewer = napari.Viewer(title="diag-run5")

    tick("add napari-deeplabcut dock widget")
    viewer.window.add_plugin_dock_widget("napari-deeplabcut", "Keypoint controls")

    files = [str(RUN5), str(CFG)]
    tick(f"open files: {files}")

    t0 = time.time()
    viewer.open(files, plugin="napari-deeplabcut", stack=True)
    tick(f"viewer.open done in {time.time() - t0:.1f}s")

    tick("SUCCESS - close window manually or Ctrl+C")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
