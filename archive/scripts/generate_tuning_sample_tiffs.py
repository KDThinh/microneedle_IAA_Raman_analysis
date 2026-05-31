#!/usr/bin/env python3
"""
Generate a synthetic grayscale TIFF timelapse for tuning segment_nb_timelapse.py.

Designed to mimic a fixed lab camera with:
- upward plant growth (canopy moves up and expands; no crop needed in real life),
- alternating day / night exposure and noise,
- static dark “rig” strip and occasional edge speckle (residual false positives).

Filenames match ``tl_YYYY-MM-DD_HH-MM-SS`` so ``parse_timestamp_from_name`` works.

Run from repo root::

    python scripts/generate_tuning_sample_tiffs.py --output-dir plant_timelapse/tuning_sample
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image


def _is_night_frame(t: int, rng: np.random.Generator, night_prob: float) -> bool:
    """Mix day/night: periodic blocks + jitter so both appear without being perfectly regular."""
    block = (t // 7) % 2
    jitter = rng.random() < night_prob
    return bool(block ^ jitter) if night_prob > 0 else bool(block)


def synthesize_frame(
    t: int,
    n_frames: int,
    height: int,
    width: int,
    rng: np.random.Generator,
    *,
    night_prob: float,
) -> np.ndarray:
    """Return uint16 (H, W) grayscale image."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    u = t / max(n_frames - 1, 1)

    is_night = _is_night_frame(t, rng, night_prob)

    # Plant: main elliptical canopy + smaller lobe; center drifts upward over time.
    cy = height * (0.62 - 0.22 * u) + 8.0 * np.sin(t * 0.31)
    cx = width * (0.52 + 0.04 * np.sin(t * 0.19))
    sx = 55.0 + 55.0 * u
    sy = 48.0 + 62.0 * u
    plant = np.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
    plant += 0.42 * np.exp(
        -0.5
        * (
            ((xx - cx - 0.35 * sx) / (0.45 * sx)) ** 2
            + ((yy - cy + 0.4 * sy) / (0.55 * sy)) ** 2
        )
    )

    plant_peak = 3200.0 if is_night else 4200.0
    plant *= plant_peak / max(float(plant.max()), 1e-6)

    # Base background level (day vs night).
    bg_lvl = 920.0 if is_night else 2100.0
    bg_lvl *= 1.0 + 0.06 * np.sin(t * 0.07) + 0.03 * rng.standard_normal()
    img = np.full((height, width), bg_lvl, dtype=np.float32)

    # Soft vignette / chamber falloff.
    cxn, cyn = width / 2, height / 2
    rr = np.sqrt(((xx - cxn) / width) ** 2 + ((yy - cyn) / height) ** 2)
    img *= 0.82 + 0.18 * np.clip(1.0 - 0.9 * rr, 0.65, 1.0)

    # Static vertical rig (dark strip) — can create residual structure like real hardware.
    rig_x0 = int(0.14 * width)
    rig_x1 = int(0.26 * width)
    rig = img[:, rig_x0:rig_x1]
    rig[:] = rig * np.linspace(0.45, 0.78, rig_x1 - rig_x0, dtype=np.float32)[np.newaxis, :]
    img[:, rig_x0:rig_x1] = rig + (15.0 if is_night else 6.0) * rng.standard_normal((height, rig_x1 - rig_x0)).astype(np.float32)

    # Table / rim band at bottom with slight instability (false positives candidate).
    band = slice(int(0.88 * height), height)
    img[band, :] *= np.linspace(0.88, 1.06, band.stop - band.start, dtype=np.float32)[:, np.newaxis]
    img[band, :] += (25.0 if is_night else 10.0) * rng.standard_normal((band.stop - band.start, width)).astype(np.float32)

    img = img + plant.astype(np.float32)

    # Read noise + occasional hot pixels (stronger at night).
    noise_sigma = 18.0 if is_night else 7.0
    img += noise_sigma * rng.standard_normal((height, width)).astype(np.float32)
    n_hot = rng.integers(2, 9)
    for _ in range(int(n_hot)):
        hy, hx = rng.integers(0, height), rng.integers(0, width)
        img[hy, hx] += rng.uniform(400, 2800)

    # Global exposure drift frame-to-frame.
    img *= float(np.clip(1.0 + 0.04 * np.sin(0.11 * t) + 0.015 * rng.standard_normal(), 0.92, 1.08))

    img = np.clip(img, 0.0, 65535.0)
    return img.astype(np.uint16)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write TIFFs (created if missing). Default: plant_timelapse/tuning_sample under repo root.",
    )
    p.add_argument("--frames", type=int, default=72, help="Number of synthetic frames (default: 72).")
    p.add_argument("--width", type=int, default=720, help="Image width in pixels.")
    p.add_argument("--height", type=int, default=540, help="Image height in pixels.")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for repeatable stacks.")
    p.add_argument(
        "--night-prob",
        type=float,
        default=0.25,
        help="Extra random mixing of night-like exposure (0–1). Default: 0.25.",
    )
    p.add_argument(
        "--start-time",
        type=str,
        default="2026-06-01T06:00:00",
        help='ISO-ish start datetime for filenames (default: "2026-06-01T06:00:00").',
    )
    p.add_argument(
        "--minutes-step",
        type=int,
        default=30,
        help="Simulated capture interval in minutes (default: 30).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    out = args.output_dir
    if out is None:
        out = repo_root / "plant_timelapse" / "tuning_sample"
    out = out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    start = datetime.fromisoformat(args.start_time)
    step = timedelta(minutes=args.minutes_step)

    n = max(1, args.frames)
    for t in range(n):
        arr = synthesize_frame(t, n, args.height, args.width, rng, night_prob=args.night_prob)
        ts = start + t * step
        name = f"tl_{ts.strftime('%Y-%m-%d_%H-%M-%S')}.tif"
        path = out / name
        Image.fromarray(arr).save(path)

    print(f"Wrote {n} TIFFs to {out}")


if __name__ == "__main__":
    main()
