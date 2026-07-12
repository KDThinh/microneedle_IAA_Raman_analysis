"""Frame preparation for the keypoint model.

Converts 12-bit grayscale TIFFs to 8-bit using the ABSOLUTE sensor range (0-4095)
so a pixel means the same thing day and night and across datasets. With
``config.ROI = None`` the full sensor frame is kept (no crop), which avoids
clipping the meristem when the plant bolts; an explicit ROI can still be passed
for cropping if ever needed.

The primary output is an 8-bit mp4 per dataset (what DeepLabCut ingests).
PNG export is also provided for QC / non-video labeling.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from .config import ROI, SENSOR_MAX


def list_tiffs(dataset_dir) -> list[Path]:
    """Sorted list of *.tif frames in a dataset folder."""
    return sorted(Path(dataset_dir).glob("*.tif"))


def read_frame_u16(path) -> np.ndarray:
    """Read a TIFF at native bit depth (keeps uint16)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def to_u8_absolute(img: np.ndarray, sensor_max: int = SENSOR_MAX) -> np.ndarray:
    """Scale to 8-bit using the absolute sensor range (NOT per-frame min/max)."""
    return np.clip(img.astype(np.float32) / float(sensor_max) * 255.0, 0, 255).astype(np.uint8)


def crop_roi(img: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = ROI) -> np.ndarray:
    if roi is None:
        return img
    x0, y0, x1, y1 = roi
    return img[y0:y1, x0:x1]


def prepare_frame(
    path,
    roi: Optional[Tuple[int, int, int, int]] = ROI,
    sensor_max: int = SENSOR_MAX,
) -> np.ndarray:
    """Read TIFF -> absolute 8-bit -> ROI crop -> 3-channel BGR (for DLC)."""
    u8 = to_u8_absolute(read_frame_u16(path), sensor_max)
    u8 = crop_roi(u8, roi)
    if u8.ndim == 2:
        u8 = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    return u8


def build_video(
    dataset_dir,
    out_path,
    roi: Optional[Tuple[int, int, int, int]] = ROI,
    sensor_max: int = SENSOR_MAX,
    fps: int = 30,
    step: int = 1,
    limit: Optional[int] = None,
    progress: bool = True,
) -> Tuple[Path, int, Tuple[int, int]]:
    """Encode a dataset's frames into a cropped 8-bit mp4 for DeepLabCut.

    Returns (out_path, n_frames_written, (width, height)).
    """
    frames = list_tiffs(dataset_dir)[::step]
    if limit is not None:
        frames = frames[:limit]
    if not frames:
        raise ValueError(f"No .tif frames found in {dataset_dir}")

    first = prepare_frame(frames[0], roi, sensor_max)
    h, w = first.shape[:2]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open for {out_path}")

    iterator = frames
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(frames, desc=f"{Path(dataset_dir).name} -> {out_path.name}", unit="frame")
        except ImportError:
            pass

    try:
        for f in iterator:
            fr = prepare_frame(f, roi, sensor_max)
            if fr.shape[:2] != (h, w):
                fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(fr)
    finally:
        writer.release()

    return out_path, len(frames), (w, h)


def export_frames_png(
    frame_paths: Sequence[Path],
    out_dir,
    roi: Optional[Tuple[int, int, int, int]] = ROI,
    sensor_max: int = SENSOR_MAX,
) -> list[Path]:
    """Export selected TIFF frames as cropped 8-bit PNGs (QC / manual labeling)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for f in frame_paths:
        fr = prepare_frame(f, roi, sensor_max)
        dst = out_dir / (Path(f).stem + ".png")
        cv2.imwrite(str(dst), fr)
        written.append(dst)
    return written


def _draw_keypoints(
    img: np.ndarray,
    base_xy: Tuple[float, float],
    meristem_xy: Tuple[float, float],
    *,
    draw_stem_line: bool = True,
    label: Optional[str] = None,
) -> np.ndarray:
    """Draw base (circle) and meristem (cross) on a BGR image."""
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(h, w) / 1200.0
    line_th = max(2, int(round(2 * scale)))
    base_r = max(10, int(round(10 * scale)))
    cross_sz = max(18, int(round(18 * scale)))
    font_scale = max(0.8, 0.9 * scale)
    thickness = max(2, int(round(2 * scale)))

    bx, by = int(round(base_xy[0])), int(round(base_xy[1]))
    mx, my = int(round(meristem_xy[0])), int(round(meristem_xy[1]))
    if draw_stem_line:
        cv2.line(out, (bx, by), (mx, my), (255, 200, 0), line_th, cv2.LINE_AA)
    cv2.circle(out, (bx, by), base_r, (0, 255, 0), line_th, cv2.LINE_AA)
    cv2.drawMarker(
        out, (mx, my), (0, 0, 255), cv2.MARKER_TILTED_CROSS, cross_sz, line_th, cv2.LINE_AA
    )
    if label:
        pad = 8
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x0, y0 = pad, pad + th + baseline
        cv2.rectangle(out, (pad, pad), (pad + tw + 8, y0 + baseline + 4), (0, 0, 0), -1)
        cv2.putText(
            out,
            label,
            (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return out


def export_keypoint_overlay_video(
    dataset_dir,
    height_table,
    out_path,
    roi: Optional[Tuple[int, int, int, int]] = ROI,
    sensor_max: int = SENSOR_MAX,
    fps: int = 10,
    step: int = 1,
    draw_stem_line: bool = True,
    use_interp_coords: bool = True,
    resize_factor: float = 1.0,
    progress: bool = True,
) -> Tuple[Path, int, int]:
    """Encode a QC mp4 with base + meristem overlaid on every frame.

    ``height_table`` must include columns: frame, base_x, base_y, meristem_x,
    meristem_y (optionally height_px for on-frame labels). Coordinates should
    match the full-frame pixel space used in postprocess CSVs.
    """
    tiffs = list_tiffs(dataset_dir)[::step]
    if not tiffs:
        raise ValueError(f"No .tif frames found in {dataset_dir}")

    table = height_table.set_index("frame")
    table.index = table.index.astype(int)
    x0, y0 = (roi[0], roi[1]) if roi is not None else (0, 0)

    first = prepare_frame(tiffs[0], roi, sensor_max)
    h, w = first.shape[:2]
    resize_factor = float(resize_factor)
    if resize_factor <= 0:
        raise ValueError("resize_factor must be > 0")

    out_h, out_w = h, w
    if resize_factor != 1.0:
        out_w = max(1, int(round(w * resize_factor)))
        out_h = max(1, int(round(h * resize_factor)))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter failed to open for {out_path}")

    iterator = enumerate(tiffs)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=len(tiffs), desc=f"overlay -> {out_path.name}", unit="frame")
        except ImportError:
            pass

    n_written = 0
    n_drawn = 0
    try:
        for frame_idx, tiff in iterator:
            img = prepare_frame(tiff, roi, sensor_max)
            # Ensure writer resolution (and ROI settings) match.
            if img.shape[:2] != (out_h, out_w):
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

            if frame_idx in table.index:
                row = table.loc[frame_idx]
                if hasattr(row, "columns"):
                    row = row.iloc[0]
                parts = [f"frame {frame_idx}"]
                if "height_px" in row.index:
                    h_px = float(row["height_px"])
                    if np.isfinite(h_px):
                        parts.append(f"h={h_px:.0f}px")
                if "timestamp" in row.index:
                    ts = row["timestamp"]
                    if ts is not None and str(ts) not in ("", "nan", "NaT"):
                        parts.append(str(ts)[:16])
                label = "  ".join(parts)
                base_x_col = "base_x_interp" if use_interp_coords and "base_x_interp" in row.index else "base_x"
                base_y_col = "base_y_refined" if use_interp_coords and "base_y_refined" in row.index else (
                    "base_y_interp" if use_interp_coords and "base_y_interp" in row.index else "base_y"
                )
                mer_x_col = "meristem_x_interp" if use_interp_coords and "meristem_x_interp" in row.index else "meristem_x"
                mer_y_col = "meristem_y_refined" if use_interp_coords and "meristem_y_refined" in row.index else (
                    "meristem_y_interp" if use_interp_coords and "meristem_y_interp" in row.index else "meristem_y"
                )

                # Scale coordinates into the resized frame space.
                bx = (row[base_x_col] - x0) * resize_factor
                by = (row[base_y_col] - y0) * resize_factor
                mx = (row[mer_x_col] - x0) * resize_factor
                my = (row[mer_y_col] - y0) * resize_factor

                img = _draw_keypoints(
                    img,
                    (bx, by),
                    (mx, my),
                    draw_stem_line=draw_stem_line,
                    label=label,
                )
                n_drawn += 1

            writer.write(img)
            n_written += 1
    finally:
        writer.release()

    return out_path, n_written, n_drawn
