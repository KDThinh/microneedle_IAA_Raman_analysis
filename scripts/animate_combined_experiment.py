"""Synchronized animation of plant timelapse images and PLG plot.

When both top- and side-view image directories resolve to valid timelines, builds
one video with panels left-to-right: top view | side view | PLG. Otherwise renders
single-view animations (image | PLG) to the respective output paths.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter, DayLocator
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from animate_PLG import DEFAULT_RATIO_COLUMN, load_plg_series
from animate_plant_timelapse import collect_tiff_timeline, load_processed_frame, resolve_image_view_dirs
from swnt_iaa_analysis.core.utils import (
    add_day_night_shading,
    parse_light_transition_config,
    parse_shade_transition_config,
    parse_treatment_events_config,
)
from swnt_iaa_analysis.io.config import load_profile_config

# -------------------------
# User-editable defaults
# -------------------------
PLG_PROCESS_PATH = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 3\Raw data\run 10\results_v4_20260507_005813\results_v4_20260507_011229_reprocess\processed_data.csv"
)
# Side-view timelapse directories. Use [] when this experiment has no sideview dataset.
IMAGES_SIDEVIEW_DIR: list[Path] = [
    Path(
        r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 3\DEV_1AB22C05B465\timelapse_2026-04-21_17-23-17"
    ),
    Path(
        r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 3_2\DEV_1AB22C05B465\timelapse_2026-05-03_11-45-21"
    ),
]
# Top-view timelapse directories. Leave [] if not available — that view is skipped.
IMAGES_TOPVIEW_DIR: list[Path] = [
    Path(
        r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 3\DEV_1AB22C0B9113\timelapse_2026-04-21_17-23-17"	
    ),
    Path(
        r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 3_2\DEV_1AB22C0B9113\timelapse_2026-05-03_11-45-21"	
    ),
]
DEFAULT_OUTPUT_SIDE = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 3\combined_timelapse_plg_side.mp4"
)
DEFAULT_OUTPUT_TOP = Path("combined_timelapse_plg_top.mp4")
# Written when both top- and side-view timelapses exist: top | side | PLG in one file.
DEFAULT_OUTPUT_COMBINED = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 3\combined_timelapse_plg_top_side.mp4"
)
DEFAULT_FPS = 10
DEFAULT_MARKER_SIZE = 11

# Plant timelapse sampling/render parameters
DEFAULT_IMAGE_FRAME_STEP = 6
DEFAULT_IMAGE_MAX_WIDTH = 100
DEFAULT_MAX_FRAMES = None
DEFAULT_PROFILE_NAME = "Nb_Control_6to22_Temp_Hum_Variable_Run3"
DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_OVERLAP_ONLY = True


def _resolve_output_path(output_path: Path) -> Path:
    ext = output_path.suffix.lower()
    if ext not in (".gif", ".mp4"):
        raise ValueError("Output must end with .gif or .mp4")
    if ext == ".mp4" and shutil.which("ffmpeg") is None:
        fallback = output_path.with_suffix(".gif")
        print(f"Warning: FFmpeg not found. Falling back to GIF: {fallback}")
        return fallback
    return output_path


def _build_plg_index_map(plg_times: pd.Series, frame_times: list[pd.Timestamp]) -> np.ndarray:
    plg_ns = plg_times.values.astype("datetime64[ns]").astype("int64")
    frame_ns = np.array(
        [pd.Timestamp(ts).to_datetime64().astype("datetime64[ns]").astype("int64") for ts in frame_times],
        dtype=np.int64,
    )
    idx = np.searchsorted(plg_ns, frame_ns, side="right") - 1
    idx = np.clip(idx, 0, len(plg_ns) - 1)
    return idx


def _align_top_to_side_timelines(
    side_times: list[pd.Timestamp],
    side_paths: list[Path],
    top_times: list[pd.Timestamp],
    top_paths: list[Path],
    *,
    tolerance: pd.Timedelta,
) -> tuple[list[pd.Timestamp], list[Path], list[Path]] | None:
    """Pair each side-frame time with the nearest top-frame time within ``tolerance``.

    Returns aligned (frame_times, top_paths, side_paths) for triptych rendering, or
    ``None`` if no pairs remain.
    """
    left = pd.DataFrame(
        {"t": pd.to_datetime(side_times), "path_side": [str(p) for p in side_paths]}
    ).sort_values("t")
    right = pd.DataFrame(
        {"t_top": pd.to_datetime(top_times), "path_top": [str(p) for p in top_paths]}
    ).sort_values("t_top")
    merged = pd.merge_asof(
        left,
        right,
        left_on="t",
        right_on="t_top",
        direction="nearest",
        tolerance=tolerance,
    )
    bad = merged["path_top"].isna()
    if bad.any():
        print(
            f"Warning: dropped {int(bad.sum())} side frames with no top-view match "
            f"within {tolerance}."
        )
    merged = merged.dropna(subset=["path_top"])
    if merged.empty:
        return None
    return (
        [pd.Timestamp(x) for x in merged["t"].tolist()],
        [Path(p) for p in merged["path_top"].tolist()],
        [Path(p) for p in merged["path_side"].tolist()],
    )


def _prepare_timeline_for_view(
    *,
    label: str,
    view_dirs: Sequence[Path] | None,
    plg_x: pd.Series,
    args: argparse.Namespace,
) -> tuple[list[pd.Timestamp], list[Path]] | None:
    resolved = resolve_image_view_dirs(view_dirs)
    if not resolved:
        print(f"Skipping {label} view: no valid image directories configured.")
        return None
    try:
        image_timeline = collect_tiff_timeline(resolved)
    except FileNotFoundError as exc:
        print(f"Skipping {label} view: {exc}")
        return None

    image_timeline = image_timeline[:: args.image_frame_step]
    if args.overlap_only:
        plg_start = pd.Timestamp(plg_x.min())
        plg_end = pd.Timestamp(plg_x.max())
        image_timeline = [
            (ts, path)
            for ts, path in image_timeline
            if plg_start <= pd.Timestamp(ts) <= plg_end
        ]
    if args.max_frames is not None:
        image_timeline = image_timeline[: args.max_frames]
    if not image_timeline:
        print(
            f"Skipping {label} view: no image frames after sampling/overlap filtering."
        )
        return None

    frame_times = [pd.Timestamp(ts) for ts, _ in image_timeline]
    frame_paths = [path for _, path in image_timeline]
    return frame_times, frame_paths


def _decorate_plg_axis(
    *,
    ax_plot,
    frame_times: list[pd.Timestamp],
    plg_x: pd.Series,
    y_raw: pd.Series,
    y_smooth: pd.Series,
    plg: dict,
    plot_config,
    args: argparse.Namespace,
) -> tuple:
    """Configure PLG subplot and return (raw_line, smooth_line, marker)."""
    raw_line, = ax_plot.plot([], [], color="gray", linewidth=2.0, alpha=0.6, label=args.ratio_column)
    smooth_label = f"{args.ratio_column} (Gaussian"
    if plg["sigma_used"] is not None:
        smooth_label += f", sigma={plg['sigma_used']:g}"
    smooth_label += ")"
    smooth_line, = ax_plot.plot([], [], color="#1f77b4", linewidth=2.8, label=smooth_label)
    marker, = ax_plot.plot([], [], "o", color="crimson", markersize=args.marker_size, label="Current point")

    y_min = min(float(y_raw.min()), float(y_smooth.min()))
    y_max = max(float(y_raw.max()), float(y_smooth.max()))
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1
    frame_start = min(frame_times)
    frame_end = max(frame_times)
    ax_plot.set_xlim(frame_start, frame_end)
    ax_plot.set_ylim(y_min - y_pad, y_max + y_pad)
    ax_plot.set_xlabel("")
    ax_plot.set_ylabel("PLG", fontsize=16)
    ax_plot.grid(alpha=0.3, linestyle="--")
    ax_plot.tick_params(axis="both", labelsize=12)

    if plot_config:
        light_cycle = plot_config.get("metadata", {}).get("light_cycle", "Constant")
        light_transition = parse_light_transition_config(plot_config)
        shade_transition = parse_shade_transition_config(plot_config)
        treatment_events = parse_treatment_events_config(plot_config)
        add_day_night_shading(
            ax_plot, frame_start, frame_end, light_cycle=light_cycle, light_transition=light_transition
        )
        if light_transition:
            transition_time = light_transition["transition_datetime"]
            if frame_start <= transition_time <= frame_end:
                ax_plot.axvline(
                    transition_time, color="red", linestyle="--", linewidth=2, alpha=0.7, label="Light transition"
                )
        if shade_transition:
            transition_time = shade_transition["transition_datetime"]
            if frame_start <= transition_time <= frame_end:
                ax_plot.axvline(
                    transition_time,
                    color="purple",
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label="Shade transition",
                )
        if treatment_events:
            for event in treatment_events:
                event_time = event["datetime"]
                if frame_start <= event_time <= frame_end:
                    ax_plot.axvline(
                        event_time,
                        color=event["marker_color"],
                        linestyle=event["marker_style"],
                        linewidth=1.5,
                        alpha=0.6,
                        label=event.get("description", event["event_type"]),
                    )

    ax_plot.xaxis.set_major_formatter(DateFormatter("%m-%d %H"))
    ax_plot.xaxis.set_major_locator(DayLocator())
    return raw_line, smooth_line, marker


def _render_combined_animation(
    *,
    label: str,
    output_path: Path,
    frame_paths: list[Path],
    frame_times: list[pd.Timestamp],
    plg_x: pd.Series,
    y_raw: pd.Series,
    y_smooth: pd.Series,
    plg_idx_map: np.ndarray,
    plg: dict,
    plot_config,
    args: argparse.Namespace,
) -> None:
    fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    first_frame = load_processed_frame(frame_paths[0], max_width=args.image_max_width)
    image_artist = ax_img.imshow(first_frame)
    ax_img.axis("off")
    time_text = ax_img.text(
        0.02,
        0.98,
        "",
        transform=ax_img.transAxes,
        color="white",
        fontsize=20,
        va="top",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.5},
    )

    raw_line, smooth_line, marker = _decorate_plg_axis(
        ax_plot=ax_plot,
        frame_times=frame_times,
        plg_x=plg_x,
        y_raw=y_raw,
        y_smooth=y_smooth,
        plg=plg,
        plot_config=plot_config,
        args=args,
    )
    fig.autofmt_xdate()

    def init():
        raw_line.set_data([], [])
        smooth_line.set_data([], [])
        marker.set_data([], [])
        time_text.set_text("")
        return image_artist, raw_line, smooth_line, marker, time_text

    day0 = pd.Timestamp(frame_times[0]).normalize()

    def update(frame: int):
        frame_time = frame_times[frame]
        frame_img = load_processed_frame(frame_paths[frame], max_width=args.image_max_width)
        image_artist.set_data(frame_img)
        day_n = (pd.Timestamp(frame_time).normalize() - day0).days + 1
        time_text.set_text(f"Day {day_n}")

        plg_idx = int(plg_idx_map[frame])
        x_slice = plg_x.iloc[: plg_idx + 1]
        raw_slice = y_raw.iloc[: plg_idx + 1]
        smooth_slice = y_smooth.iloc[: plg_idx + 1]
        raw_line.set_data(x_slice, raw_slice)
        smooth_line.set_data(x_slice, smooth_slice)
        marker.set_data([x_slice.iloc[-1]], [smooth_slice.iloc[-1]])
        return image_artist, raw_line, smooth_line, marker, time_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_paths),
        init_func=init,
        interval=1000 / args.fps,
        blit=False,
        repeat=False,
    )

    output_path = _resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gif":
        anim.save(output_path, writer="pillow", fps=args.fps)
    else:
        anim.save(output_path, writer="ffmpeg", fps=args.fps)
    print(f"Saved synchronized {label} animation: {output_path}")
    plt.close(fig)


def _render_triptych_animation(
    *,
    output_path: Path,
    top_paths: list[Path],
    side_paths: list[Path],
    frame_times: list[pd.Timestamp],
    plg_x: pd.Series,
    y_raw: pd.Series,
    y_smooth: pd.Series,
    plg_idx_map: np.ndarray,
    plg: dict,
    plot_config,
    args: argparse.Namespace,
) -> None:
    """Top view | side view | PLG (single synchronized animation)."""
    fig = plt.figure(figsize=(21, 6))
    # Nest two columns for images with small wspace so top sits tight against side;
    # keep a wider gap before the PLG panel.
    gs_main = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.72], wspace=0.16)
    gs_pair = GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=gs_main[0, 0],
        wspace=0.015,
        width_ratios=[1, 1],
    )
    ax_top = fig.add_subplot(gs_pair[0, 0])
    ax_side = fig.add_subplot(gs_pair[0, 1])
    ax_plot = fig.add_subplot(gs_main[0, 1])
    fig.subplots_adjust(left=0.035, right=0.992, top=0.96, bottom=0.11)

    ft0_top = load_processed_frame(top_paths[0], max_width=args.image_max_width)
    ft0_side = load_processed_frame(side_paths[0], max_width=args.image_max_width)
    img_top = ax_top.imshow(ft0_top)
    img_side = ax_side.imshow(ft0_side)
    ax_top.axis("off")
    ax_side.axis("off")

    time_text_top = ax_top.text(
        0.02,
        0.98,
        "",
        transform=ax_top.transAxes,
        color="white",
        fontsize=16,
        va="top",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "black", "alpha": 0.5},
    )
    raw_line, smooth_line, marker = _decorate_plg_axis(
        ax_plot=ax_plot,
        frame_times=frame_times,
        plg_x=plg_x,
        y_raw=y_raw,
        y_smooth=y_smooth,
        plg=plg,
        plot_config=plot_config,
        args=args,
    )
    fig.autofmt_xdate()

    def init():
        raw_line.set_data([], [])
        smooth_line.set_data([], [])
        marker.set_data([], [])
        time_text_top.set_text("")
        return img_top, img_side, raw_line, smooth_line, marker, time_text_top

    day0 = pd.Timestamp(frame_times[0]).normalize()

    def update(frame: int):
        ft = frame_times[frame]
        img_top.set_data(load_processed_frame(top_paths[frame], max_width=args.image_max_width))
        img_side.set_data(load_processed_frame(side_paths[frame], max_width=args.image_max_width))
        day_n = (pd.Timestamp(ft).normalize() - day0).days + 1
        time_text_top.set_text(f"Day {day_n}")

        plg_idx = int(plg_idx_map[frame])
        x_slice = plg_x.iloc[: plg_idx + 1]
        raw_slice = y_raw.iloc[: plg_idx + 1]
        smooth_slice = y_smooth.iloc[: plg_idx + 1]
        raw_line.set_data(x_slice, raw_slice)
        smooth_line.set_data(x_slice, smooth_slice)
        marker.set_data([x_slice.iloc[-1]], [smooth_slice.iloc[-1]])
        return img_top, img_side, raw_line, smooth_line, marker, time_text_top

    anim = FuncAnimation(
        fig,
        update,
        frames=len(side_paths),
        init_func=init,
        interval=1000 / args.fps,
        blit=False,
        repeat=False,
    )

    output_path = _resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".gif":
        anim.save(output_path, writer="pillow", fps=args.fps)
    else:
        anim.save(output_path, writer="ffmpeg", fps=args.fps)
    print(f"Saved top + side + PLG animation: {output_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create synchronized animation of plant timelapse and PLG plot."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=PLG_PROCESS_PATH,
        help=f"Path to processed_data.csv (default: {PLG_PROCESS_PATH}).",
    )
    parser.add_argument(
        "--sideview-dirs",
        nargs="*",
        type=Path,
        default=None,
        help="Sideview timelapse directories (default: IMAGES_SIDEVIEW_DIR constant). Empty / missing dirs skip.",
    )
    parser.add_argument(
        "--topview-dirs",
        nargs="*",
        type=Path,
        default=None,
        help="Topview timelapse directories (default: IMAGES_TOPVIEW_DIR constant).",
    )
    parser.add_argument(
        "--output-side",
        type=Path,
        default=DEFAULT_OUTPUT_SIDE,
        help="Output animation for sideview (.gif or .mp4).",
    )
    parser.add_argument(
        "--output-top",
        type=Path,
        default=DEFAULT_OUTPUT_TOP,
        help="Output animation for topview only (.gif or .mp4).",
    )
    parser.add_argument(
        "--output-combined",
        type=Path,
        default=DEFAULT_OUTPUT_COMBINED,
        help="When both top- and side-view dirs are valid: top | side | PLG in one file.",
    )
    parser.add_argument(
        "--image-pair-tolerance-minutes",
        type=float,
        default=30.0,
        help="When merging top+side timelines, max time gap for nearest-neighbor pairing (default: 30).",
    )
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Output frames per second.")
    parser.add_argument("--marker-size", type=int, default=DEFAULT_MARKER_SIZE, help="PLG marker size.")
    parser.add_argument(
        "--ratio-column",
        default=DEFAULT_RATIO_COLUMN,
        help=f"PLG column name (default: {DEFAULT_RATIO_COLUMN}).",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=None,
        help="Optional sigma override for PLG Gaussian smoothing.",
    )
    parser.add_argument(
        "--image-frame-step",
        type=int,
        default=DEFAULT_IMAGE_FRAME_STEP,
        help="Use every Nth image frame (default: 2).",
    )
    parser.add_argument(
        "--image-max-width",
        type=int,
        default=DEFAULT_IMAGE_MAX_WIDTH,
        help="Resize image panel max width.",
    )
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES, help="Optional cap on output frames.")
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml for shading.",
    )
    parser.add_argument("--profile-name", default=DEFAULT_PROFILE_NAME, help="Profile name in config file.")
    parser.add_argument(
        "--overlap-only",
        action="store_true",
        default=DEFAULT_OVERLAP_ONLY,
        help="Use only the overlapped time range between PLG and images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fps < 1:
        raise ValueError("--fps must be >= 1")
    if args.image_frame_step < 1:
        raise ValueError("--image-frame-step must be >= 1")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("--max-frames must be >= 1")
    if (args.config_path is None) != (args.profile_name is None):
        raise ValueError("Use --config-path and --profile-name together.")

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
        raise ValueError("Combined sync animation requires Datetime x-axis in processed_data.csv.")

    plg_x = pd.to_datetime(plg["x"]).reset_index(drop=True)
    y_raw = plg["y_raw"].reset_index(drop=True)
    y_smooth = plg["y_plot"].reset_index(drop=True)

    side_dirs = (
        args.sideview_dirs if args.sideview_dirs is not None else IMAGES_SIDEVIEW_DIR
    )
    top_dirs = args.topview_dirs if args.topview_dirs is not None else IMAGES_TOPVIEW_DIR

    side_prep = _prepare_timeline_for_view(label="side", view_dirs=side_dirs, plg_x=plg_x, args=args)
    top_prep = _prepare_timeline_for_view(label="top", view_dirs=top_dirs, plg_x=plg_x, args=args)

    any_rendered = False
    tolerance = pd.Timedelta(minutes=args.image_pair_tolerance_minutes)
    wrote_triptych = False

    if side_prep is not None and top_prep is not None:
        s_times, s_paths = side_prep
        t_times, t_paths = top_prep
        aligned = _align_top_to_side_timelines(
            s_times, s_paths, t_times, t_paths, tolerance=tolerance
        )
        if aligned is not None:
            frame_times, top_paths, side_paths = aligned
            plg_idx_map = _build_plg_index_map(plg_x, frame_times)
            _render_triptych_animation(
                output_path=args.output_combined,
                top_paths=top_paths,
                side_paths=side_paths,
                frame_times=frame_times,
                plg_x=plg_x,
                y_raw=y_raw,
                y_smooth=y_smooth,
                plg_idx_map=plg_idx_map,
                plg=plg,
                plot_config=plot_config,
                args=args,
            )
            any_rendered = True
            wrote_triptych = True
        else:
            print(
                "Could not align top and side timelines within tolerance; "
                "writing separate single-view animations instead."
            )

    if not wrote_triptych:
        if side_prep is not None:
            frame_times, frame_paths = side_prep
            plg_idx_map = _build_plg_index_map(plg_x, frame_times)
            _render_combined_animation(
                label="side",
                output_path=args.output_side,
                frame_paths=frame_paths,
                frame_times=frame_times,
                plg_x=plg_x,
                y_raw=y_raw,
                y_smooth=y_smooth,
                plg_idx_map=plg_idx_map,
                plg=plg,
                plot_config=plot_config,
                args=args,
            )
            any_rendered = True
        if top_prep is not None:
            frame_times, frame_paths = top_prep
            plg_idx_map = _build_plg_index_map(plg_x, frame_times)
            _render_combined_animation(
                label="top",
                output_path=args.output_top,
                frame_paths=frame_paths,
                frame_times=frame_times,
                plg_x=plg_x,
                y_raw=y_raw,
                y_smooth=y_smooth,
                plg_idx_map=plg_idx_map,
                plg=plg,
                plot_config=plot_config,
                args=args,
            )
            any_rendered = True

    if not any_rendered:
        raise ValueError(
            "No combined animations were written. Configure at least one view with "
            "valid TIFF directories (or relax --image-frame-step / overlap filtering)."
        )


if __name__ == "__main__":
    main()
