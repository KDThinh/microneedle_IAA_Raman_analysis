"""Export synchronized separate GIFs/MP4s: plant timelapse + PLG trace."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.dates import DateFormatter, DayLocator
from PIL import Image, ImageDraw, ImageFont

from animate_PLG import DEFAULT_RATIO_COLUMN, load_plg_series
from animate_plant_timelapse import collect_tiff_timeline, load_processed_frame, resolve_image_view_dirs
from swnt_iaa_analysis.core.utils import (
    add_day_night_shading,
    parse_light_transition_config,
    parse_shade_transition_config,
    parse_treatment_events_config,
)
from swnt_iaa_analysis.io.config import load_profile_config


# User-editable defaults
PLG_PROCESS_PATH = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Drought\Light_6to22\Temp_Hum_Variable\Run 2\Raw data\run 9\results_v4_20260429_001643\results_v4_20260429_121906_reprocess\processed_data.csv"
)
# Side-view timelapse folder(s). Use [] when this experiment has no sideview dataset.
IMAGES_SIDEVIEW_DIR = [
    Path(
        r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Drought\Light_6to22\Temp_Hum_Variable\Run 2\Raw data\Time-lapse images\timelapse_2026-02-09_16-44-08"
    ),
    Path(
        r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Drought\Light_6to22\Temp_Hum_Variable\Run 2\Raw data\Time-lapse images\timelapse_2026-02-13_12-08-55"
    ),
]
# Top-view timelapse folder(s). Leave empty [] if not available — that view will be skipped.
IMAGES_TOPVIEW_DIR: list[Path] = []
PROFILE_NAME = "Nb_Drought_6to22_Temp_Hum_Variable_Run2"
CONFIG_PATH = Path("config.yaml")

PLANT_SIDE_OUTPUT = Path("plant_timelapse_small_side.mp4")
PLG_SIDE_OUTPUT = Path("plg_synced_side.mp4")
PLANT_TOP_OUTPUT = Path("plant_timelapse_small_top.mp4")
PLG_TOP_OUTPUT = Path("plg_synced_top.mp4")

FPS = 8
IMAGE_FRAME_STEP = 6
IMAGE_MAX_WIDTH = 420
MAX_FRAMES = None
MARKER_SIZE = 7
GAUSSIAN_SIGMA = None
OVERLAP_ONLY = True


def _build_plg_index_map(plg_times: pd.Series, frame_times: list[pd.Timestamp]) -> np.ndarray:
    plg_ns = plg_times.values.astype("datetime64[ns]").astype("int64")
    frame_ns = np.array(
        [pd.Timestamp(ts).to_datetime64().astype("datetime64[ns]").astype("int64") for ts in frame_times],
        dtype=np.int64,
    )
    idx = np.searchsorted(plg_ns, frame_ns, side="right") - 1
    return np.clip(idx, 0, len(plg_ns) - 1)


def _make_writer(output_path: Path, fps: int, label: str):
    ext = output_path.suffix.lower()
    if ext == ".mp4":
        return imageio.get_writer(output_path, format="FFMPEG", fps=fps)
    if ext == ".gif":
        return imageio.get_writer(output_path, mode="I", duration=1.0 / fps, loop=0)
    raise ValueError(f"{label} output must end with .gif or .mp4")


def export_plant_media(
    frame_paths: list[Path],
    frame_times: list[pd.Timestamp],
    output_path: Path,
    max_width: int,
    fps: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    day0 = pd.Timestamp(frame_times[0]).normalize()

    try:
        day_font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        day_font = ImageFont.load_default()

    with _make_writer(output_path, fps=fps, label="Plant") as writer:
        for i, (path, ts) in enumerate(zip(frame_paths, frame_times), start=1):
            frame = load_processed_frame(path, max_width=max_width)
            day_n = (pd.Timestamp(ts).normalize() - day0).days + 1

            img = Image.fromarray(frame)
            draw = ImageDraw.Draw(img)
            draw.text((12, 12), f"Day {day_n}", fill=(255, 255, 255), font=day_font)

            writer.append_data(np.asarray(img))
            if i % 50 == 0 or i == len(frame_paths):
                print(f"Plant progress: {i}/{len(frame_paths)}")
    print(f"Saved plant output: {output_path}")


def export_plg_media(
    plg_x: pd.Series,
    y_raw: pd.Series,
    y_smooth: pd.Series,
    frame_times: list[pd.Timestamp],
    plg_idx_map: np.ndarray,
    output_path: Path,
    fps: int,
    marker_size: int,
    ratio_column: str,
    sigma_used: float | None,
    plot_config: dict | None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    raw_line, = ax.plot([], [], color="gray", linewidth=1.4, alpha=0.6, label=ratio_column)
    smooth_label = f"{ratio_column} (Gaussian"
    if sigma_used is not None:
        smooth_label += f", sigma={sigma_used:g}"
    smooth_label += ")"
    smooth_line, = ax.plot([], [], color="#1f77b4", linewidth=2.0, label=smooth_label)
    marker, = ax.plot([], [], "o", color="crimson", markersize=marker_size, label="Current point")

    y_min = min(float(y_raw.min()), float(y_smooth.min()))
    y_max = max(float(y_raw.max()), float(y_smooth.max()))
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
    frame_start = min(frame_times)
    frame_end = max(frame_times)
    ax.set_xlim(frame_start, frame_end)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel("Datetime")
    ax.set_ylabel("PLG")
    ax.grid(alpha=0.3, linestyle="--")

    if plot_config:
        light_cycle = plot_config.get("metadata", {}).get("light_cycle", "Constant")
        light_transition = parse_light_transition_config(plot_config)
        shade_transition = parse_shade_transition_config(plot_config)
        treatment_events = parse_treatment_events_config(plot_config)
        add_day_night_shading(ax, frame_start, frame_end, light_cycle=light_cycle, light_transition=light_transition)
        if light_transition:
            tt = light_transition["transition_datetime"]
            if frame_start <= tt <= frame_end:
                ax.axvline(tt, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Light transition")
        if shade_transition:
            tt = shade_transition["transition_datetime"]
            if frame_start <= tt <= frame_end:
                ax.axvline(tt, color="purple", linestyle="--", linewidth=2, alpha=0.7, label="Shade transition")
        if treatment_events:
            for event in treatment_events:
                event_time = event["datetime"]
                if frame_start <= event_time <= frame_end:
                    ax.axvline(
                        event_time,
                        color=event["marker_color"],
                        linestyle=event["marker_style"],
                        linewidth=1.5,
                        alpha=0.6,
                        label=event.get("description", event["event_type"]),
                    )

    ax.xaxis.set_major_formatter(DateFormatter("%m-%d %H"))
    ax.xaxis.set_major_locator(DayLocator())
    fig.autofmt_xdate()

    with _make_writer(output_path, fps=fps, label="PLG") as writer:
        for i, idx in enumerate(plg_idx_map, start=1):
            x_slice = plg_x.iloc[: idx + 1]
            raw_slice = y_raw.iloc[: idx + 1]
            smooth_slice = y_smooth.iloc[: idx + 1]
            raw_line.set_data(x_slice, raw_slice)
            smooth_line.set_data(x_slice, smooth_slice)
            marker.set_data([x_slice.iloc[-1]], [smooth_slice.iloc[-1]])
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(frame)
            if i % 50 == 0 or i == len(plg_idx_map):
                print(f"PLG progress: {i}/{len(plg_idx_map)}")

    plt.close(fig)
    print(f"Saved PLG output: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export synchronized separate plant+PLG media (.gif/.mp4).")
    parser.add_argument("--csv-path", type=Path, default=PLG_PROCESS_PATH)
    parser.add_argument(
        "--sideview-dirs",
        nargs="*",
        type=Path,
        default=None,
        help=f"Sideview TIFF directories (default: script constants). Use empty list via CLI carefully.",
    )
    parser.add_argument(
        "--topview-dirs",
        nargs="*",
        type=Path,
        default=None,
        help="Topview TIFF directories (default: script constants). Omitted dirs skip top export.",
    )
    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    parser.add_argument("--profile-name", default=PROFILE_NAME)
    parser.add_argument("--plant-side-output", type=Path, default=PLANT_SIDE_OUTPUT)
    parser.add_argument("--plg-side-output", type=Path, default=PLG_SIDE_OUTPUT)
    parser.add_argument("--plant-top-output", type=Path, default=PLANT_TOP_OUTPUT)
    parser.add_argument("--plg-top-output", type=Path, default=PLG_TOP_OUTPUT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--image-frame-step", type=int, default=IMAGE_FRAME_STEP)
    parser.add_argument("--image-max-width", type=int, default=IMAGE_MAX_WIDTH)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--marker-size", type=int, default=MARKER_SIZE)
    parser.add_argument("--gaussian-sigma", type=float, default=GAUSSIAN_SIGMA)
    parser.add_argument("--ratio-column", default=DEFAULT_RATIO_COLUMN)
    parser.add_argument(
        "--overlap-only",
        action="store_true",
        default=OVERLAP_ONLY,
        help="Keep only frames within overlapping PLG/image datetime range.",
    )
    return parser.parse_args()


def _export_single_view(
    *,
    label: str,
    view_dirs: Sequence[Path] | None,
    plant_output: Path,
    plg_output: Path,
    plg_x: pd.Series,
    y_raw: pd.Series,
    y_smooth: pd.Series,
    plot_config: dict | None,
    args: argparse.Namespace,
    sigma_used: float | None,
) -> bool:
    """Return True if this view produced outputs."""
    resolved = resolve_image_view_dirs(view_dirs)
    if not resolved:
        print(f"Skipping {label} view: no valid image directories.")
        return False
    try:
        image_timeline = collect_tiff_timeline(resolved)
    except FileNotFoundError as exc:
        print(f"Skipping {label} view: {exc}")
        return False

    image_timeline = image_timeline[:: args.image_frame_step]
    if args.overlap_only:
        plg_start = pd.Timestamp(plg_x.min())
        plg_end = pd.Timestamp(plg_x.max())
        image_timeline = [
            (ts, path) for ts, path in image_timeline if plg_start <= pd.Timestamp(ts) <= plg_end
        ]
    if args.max_frames is not None:
        image_timeline = image_timeline[: args.max_frames]
    if not image_timeline:
        print(
            f"Skipping {label} view: no image frames after sampling/overlap filtering."
        )
        return False

    frame_times = [pd.Timestamp(ts) for ts, _ in image_timeline]
    frame_paths = [path for _, path in image_timeline]
    plg_idx_map = _build_plg_index_map(plg_x, frame_times)

    print(f"Exporting {label} view ({len(frame_paths)} frames)...")
    export_plant_media(frame_paths, frame_times, plant_output, args.image_max_width, args.fps)
    export_plg_media(
        plg_x=plg_x,
        y_raw=y_raw,
        y_smooth=y_smooth,
        frame_times=frame_times,
        plg_idx_map=plg_idx_map,
        output_path=plg_output,
        fps=args.fps,
        marker_size=args.marker_size,
        ratio_column=args.ratio_column,
        sigma_used=sigma_used,
        plot_config=plot_config,
    )
    return True


def main() -> None:
    args = parse_args()
    if args.fps < 1:
        raise ValueError("--fps must be >= 1")
    if args.image_frame_step < 1:
        raise ValueError("--image-frame-step must be >= 1")
    if args.image_max_width < 64:
        raise ValueError("--image-max-width must be >= 64")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be >= 1")

    plot_config = None
    if args.config_path and args.profile_name:
        plot_config = load_profile_config(str(args.config_path), args.profile_name)

    plg = load_plg_series(
        csv_path=args.csv_path,
        ratio_column=args.ratio_column,
        gaussian_smoothed=True,
        gaussian_sigma=args.gaussian_sigma,
        plot_config=plot_config,
    )
    if not plg["is_datetime"]:
        raise ValueError("Synchronized export requires Datetime x-axis in processed_data.csv.")

    plg_x = pd.to_datetime(plg["x"]).reset_index(drop=True)
    y_raw = plg["y_raw"].reset_index(drop=True)
    y_smooth = plg["y_plot"].reset_index(drop=True)

    sigma_used = plg["sigma_used"]
    side_dirs = (
        args.sideview_dirs if args.sideview_dirs is not None else IMAGES_SIDEVIEW_DIR
    )
    top_dirs = args.topview_dirs if args.topview_dirs is not None else IMAGES_TOPVIEW_DIR

    side_ok = _export_single_view(
        label="side",
        view_dirs=side_dirs,
        plant_output=args.plant_side_output,
        plg_output=args.plg_side_output,
        plg_x=plg_x,
        y_raw=y_raw,
        y_smooth=y_smooth,
        plot_config=plot_config,
        args=args,
        sigma_used=sigma_used,
    )
    top_ok = _export_single_view(
        label="top",
        view_dirs=top_dirs,
        plant_output=args.plant_top_output,
        plg_output=args.plg_top_output,
        plg_x=plg_x,
        y_raw=y_raw,
        y_smooth=y_smooth,
        plot_config=plot_config,
        args=args,
        sigma_used=sigma_used,
    )
    if not side_ok and not top_ok:
        raise ValueError(
            "No synchronized exports were written. Configure at least one view with "
            "valid TIFF directories (or lower --image-frame-step / overlap settings)."
        )


if __name__ == "__main__":
    main()
