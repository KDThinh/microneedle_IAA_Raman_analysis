"""Headless-ish tests of napari-deeplabcut readers (no napari.run)."""
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


def test_reader(name: str, fn) -> None:
    tick(f"START {name}")
    t0 = time.time()
    out = fn()
    n = len(out) if out is not None else 0
    tick(f"DONE  {name} in {time.time()-t0:.1f}s -> {n} layers")


def main() -> None:
    from napari_deeplabcut._reader import get_folder_parser, get_config_reader
    from napari_deeplabcut.core.io import read_images

    for ds in ("run4", "run5"):
        folder = PROJ / "labeled-data" / ds
        tick(f"=== {ds} ===")
        test_reader(f"{ds} read_images", lambda f=folder: read_images(f))
        parser = get_folder_parser(str(folder))
        tick(f"{ds} folder_parser={parser is not None}")
        if parser:
            test_reader(f"{ds} folder_parser()", lambda p=parser: p(None))

    cfg_reader = get_config_reader(str(CFG))
    tick(f"config_reader={cfg_reader is not None}")
    if cfg_reader:
        test_reader("read_config", lambda: cfg_reader(str(CFG)))


if __name__ == "__main__":
    main()
