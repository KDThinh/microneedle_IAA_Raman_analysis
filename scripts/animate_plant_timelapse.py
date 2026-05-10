"""Create a timelapse animation (GIF/MP4) from timestamped TIFF image folders."""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

# Quick tuning guide:
# - Per-frame contrast: TIFFs use luminance percentile stretch (2–98% by default),
#   not a global fixed scale across the timelapse.
# - --frame-step: keep every Nth image (higher = smaller/faster output).
# - --max-width: resize frames to this width (preserves aspect ratio).
# - --fps: playback speed.
# - --max-frames: hard cap to prevent very large output jobs.
# Output guide:
# - --output supports .gif or .mp4.
# - MP4 is usually much smaller and faster than GIF for large timelapse sets.


DEFAULT_DIRS = [
    Path(
        r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Drought\Light_6to22\Temp_Hum_Variable\Run 2\Raw data\Time-lapse images\timelapse_2026-02-09_16-44-08"
    ),
    Path(
        r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Drought\Light_6to22\Temp_Hum_Variable\Run 2\Raw data\Time-lapse images\timelapse_2026-02-13_12-08-55"
    ),
]

TIFF_GLOBS = ("*.tif", "*.tiff", "*.TIF", "*.TIFF")


def parse_timestamp_from_name(path: Path) -> datetime | None:
    """Parse common timestamp patterns from filename."""
    name = path.stem
    patterns = [
        r"(\d{4}-\d{2}-\d{2})[ _-](\d{2})[-:](\d{2})[-:](\d{2})",
        r"(\d{8})[ _-]?(\d{6})",
    ]

    for pattern in patterns:
        match = re.search(pattern, name)
        if not match:
            continue

        groups = match.groups()
        try:
            if len(groups) == 4 and "-" in groups[0]:
                return datetime.strptime(f"{groups[0]} {groups[1]}:{groups[2]}:{groups[3]}", "%Y-%m-%d %H:%M:%S")
            if len(groups) == 2:
                return datetime.strptime(f"{groups[0]} {groups[1]}", "%Y%m%d %H%M%S")
        except ValueError:
            continue
    return None


def collect_tiff_files(input_dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in input_dirs:
        if not directory.exists():
            print(f"Warning: directory not found, skipping: {directory}")
            continue
        for pattern in TIFF_GLOBS:
            files.extend(directory.glob(pattern))

    if not files:
        raise FileNotFoundError("No TIFF files found in input directories.")

    def sort_key(path: Path):
        ts = parse_timestamp_from_name(path)
        return (ts is None, ts or datetime.min, path.name)

    return sorted(files, key=sort_key)


def resolve_image_view_dirs(dirs: Sequence[Path] | None) -> list[Path]:
    """
    Return directories to use for one camera view (side / top / etc.).

    Skips missing entries: ``None``, empty strings, non-existent paths, non-directories.
    Returns ``[]`` when there is nothing to use — callers should skip that view entirely.
    """
    if not dirs:
        return []
    resolved: list[Path] = []
    for raw in dirs:
        if raw is None:
            continue
        s = str(raw).strip()
        if not s:
            continue
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            print(f"Warning: image directory not found, skipping view path: {raw}")
            continue
        if not p.is_dir():
            print(f"Warning: not a directory, skipping: {p}")
            continue
        resolved.append(p)
    return resolved


def collect_tiff_timeline(input_dirs: list[Path]) -> list[tuple[datetime, Path]]:
    """Return TIFF files with parsed timestamps for timeline syncing."""
    files = collect_tiff_files(input_dirs)
    timeline: list[tuple[datetime, Path]] = []
    for file_path in files:
        ts = parse_timestamp_from_name(file_path)
        if ts is not None:
            timeline.append((ts, file_path))
    if not timeline:
        raise ValueError("No timestamp could be parsed from TIFF filenames.")
    return timeline


def to_uint8_rgb(
    image: Image.Image,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    """
    Convert TIFF (possibly 16-bit/grayscale) to uint8 RGB.

    Uses per-frame, luminance-aware percentile clipping (not a fixed global scale)
    before stretching to uint8 — reduces domination by hot/cold outliers so the plant
    is usually easier to see than with raw min/max.
    """
    arr_np = np.asarray(image)
    rgb = np.atleast_3d(arr_np.astype(np.float32))

    # Luminance stats for percentile bounds (same lo/hi applied to RGB → stable colors).
    if rgb.shape[2] >= 3:
        r, g_, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.299 * r + 0.587 * g_ + 0.114 * b
    else:
        gray = rgb[:, :, 0]

    valid = np.isfinite(gray)
    if not np.any(valid):
        h, w = gray.shape
        return np.zeros((h, w, 3), dtype=np.uint8)

    pv = gray[valid]
    lo_f, hi_f = float(np.percentile(pv, p_low)), float(np.percentile(pv, p_high))
    if hi_f <= lo_f:
        lo_f, hi_f = float(np.nanmin(pv)), float(np.nanmax(pv))
    if hi_f <= lo_f:
        h, w = gray.shape
        return np.zeros((h, w, 3), dtype=np.uint8)

    nch = min(3, rgb.shape[2])
    base = rgb[:, :, :nch].astype(np.float32)
    stretched = np.clip((base - lo_f) / (hi_f - lo_f), 0.0, 1.0) * 255.0

    out = stretched.astype(np.uint8)
    if out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    elif out.shape[2] >= 4:
        out = out[:, :, :3]
    return out


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    new_h = int(h * (max_width / w))
    pil_img = Image.fromarray(frame)
    pil_img = pil_img.resize((max_width, new_h), Image.Resampling.LANCZOS)
    return np.array(pil_img)


def load_processed_frame(image_path: Path, max_width: int) -> np.ndarray:
    """Load one TIFF file and apply normalization + resizing."""
    with Image.open(image_path) as img:
        frame = to_uint8_rgb(img)
    return resize_frame(frame, max_width=max_width)


def build_timelapse(
    input_dirs: list[Path],
    output_path: Path,
    fps: int,
    frame_step: int,
    max_width: int,
    max_frames: int | None,
) -> None:
    files = collect_tiff_files(input_dirs)
    files = files[::frame_step]
    if max_frames is not None:
        files = files[:max_frames]

    if not files:
        raise ValueError("No frames left after applying frame-step/max-frames.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()
    if ext not in (".gif", ".mp4"):
        raise ValueError("Output must end with .gif or .mp4")

    if ext == ".mp4":
        ffmpeg_available = shutil.which("ffmpeg") is not None
        if not ffmpeg_available:
            fallback_path = output_path.with_suffix(".gif")
            print(
                "Warning: FFmpeg not found on PATH. Falling back to GIF export at:\n"
                f"  {fallback_path}"
            )
            output_path = fallback_path
            ext = ".gif"

    print(f"Writing {len(files)} frames to: {output_path}")
    if ext == ".mp4":
        writer = imageio.get_writer(output_path, format="FFMPEG", fps=fps)
    else:
        # GIF writer uses frame duration (seconds) rather than FPS argument.
        writer = imageio.get_writer(output_path, mode="I", duration=1.0 / fps, loop=0)

    with writer:
        for i, file_path in enumerate(files, start=1):
            frame = load_processed_frame(file_path, max_width=max_width)
            writer.append_data(frame)
            if i % 50 == 0 or i == len(files):
                print(f"  Progress: {i}/{len(files)} frames")

    print("Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create timelapse animation from timestamped TIFF images."
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_DIRS,
        help="One or more directories containing TIFF timelapse images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("timelapse_plant.gif"),
        help="Output animation path (.gif or .mp4).",
    )
    parser.add_argument("--fps", type=int, default=12, help="Output frames per second (default: 12).")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=2,
        help="Keep every Nth frame (default: 2).",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Resize frames to this max width (default: 1280).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional hard cap on number of frames.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps < 1:
        raise ValueError("--fps must be >= 1")
    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")
    if args.max_width < 64:
        raise ValueError("--max-width must be >= 64")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be >= 1")

    build_timelapse(
        input_dirs=args.input_dirs,
        output_path=args.output,
        fps=args.fps,
        frame_step=args.frame_step,
        max_width=args.max_width,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
