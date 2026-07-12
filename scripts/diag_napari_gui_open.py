"""GUI open tests without napari.run (show=False)."""
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


def try_open(label: str, paths: list[str], plugin: str | None, stack: bool) -> None:
    import napari

    tick(f"START {label}")
    viewer = napari.Viewer(show=False)
    if plugin == "napari-deeplabcut":
        viewer.window.add_plugin_dock_widget("napari-deeplabcut", "Keypoint controls")
    t0 = time.time()
    viewer.open(paths, plugin=plugin, stack=stack)
    tick(f"DONE  {label} in {time.time()-t0:.1f}s  layers={[l.name for l in viewer.layers]}")
    viewer.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["run4", "run5"])
    args = p.parse_args()
    folder = str(PROJ / "labeled-data" / args.dataset)

    try_open(f"{args.dataset} folder-only dlc", [folder], "napari-deeplabcut", False)
    try_open(f"{args.dataset} folder+config dlc stack", [folder, str(CFG)], "napari-deeplabcut", True)
    try_open(f"{args.dataset} folder-only raw", [folder], None, True)
