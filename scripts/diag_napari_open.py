"""Stepwise napari open tests for run4 vs run5."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PROJ = REPO / "plant_timelapse" / "dlc" / "plant_meristem-ryank-2026-07-08"
CFG = PROJ / "config.yaml"


def tick(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def test_open(folder: str, plugin: str | None, stack: bool, limit: int | None) -> bool:
    import napari

    folder_path = PROJ / "labeled-data" / folder
    pngs = sorted(folder_path.glob("*.png"))
    if limit:
        pngs = pngs[:limit]

    tick(f"=== {folder} plugin={plugin!r} stack={stack} n={len(pngs)} ===")
    viewer = napari.Viewer(show=False)
    if plugin == "napari-deeplabcut":
        viewer.window.add_plugin_dock_widget("napari-deeplabcut", "Keypoint controls")

    t0 = time.time()
    if plugin == "napari-deeplabcut" and limit is None:
        viewer.open([str(folder_path), str(CFG)], plugin=plugin, stack=stack)
    elif plugin:
        viewer.open([str(p) for p in pngs], plugin=plugin, stack=stack)
    else:
        viewer.open([str(p) for p in pngs], stack=stack)

    tick(f"open OK in {time.time() - t0:.1f}s")
    viewer.close()
    return True


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("folder", choices=["run4", "run5"])
    p.add_argument("--plugin", default=None)
    p.add_argument("--stack", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    test_open(args.folder, args.plugin, args.stack, args.limit)
