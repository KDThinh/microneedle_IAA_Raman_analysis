"""Animate baseline-corrected Fluorescence/G-band ratio from processed_data.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.dates import DateFormatter, DayLocator

from swnt_iaa_analysis.core.utils import (
    add_day_night_shading,
    parse_light_transition_config,
    parse_shade_transition_config,
    parse_treatment_events_config,
)
from swnt_iaa_analysis.core.baseline import apply_gaussian_smoothing
from swnt_iaa_analysis.io.config import load_profile_config


DEFAULT_RATIO_COLUMN = "Fluorescence_to_Gband_Ratio_BaselineCorrected"
X_CANDIDATES = ("Datetime", "Seconds", "Scan Number")

# Quick tuning guide:
# - --interval-ms: Playback speed (smaller = faster animation, larger = slower).
# - --marker-size: Size of the moving "head" marker.
# - --fps: Frame rate only for saved files (GIF/MP4), not live playback.
# - --ratio-column: Use a different column if needed.
# - --config-path and --profile-name: Use pipeline config for day/night shading.
# - --gaussian-smoothed: Animate Gaussian-smoothed ratio instead of raw ratio.
# - --gaussian-sigma: Override smoothing sigma (else use config/default).
# Output path guide:
# - --output sets both output directory and filename.
#   Example: --output "outputs/ratio_anim.gif"
#   -> creates folder "outputs" if needed and writes "ratio_anim.gif" there.
# - If --output is omitted, animation is shown interactively and not saved.


def _load_data(csv_path: Path, ratio_column: str) -> tuple[pd.Series, pd.Series, str]:
    df = pd.read_csv(csv_path)

    if ratio_column not in df.columns:
        raise ValueError(
            f"Column '{ratio_column}' not found.\n"
            f"Available columns: {', '.join(df.columns)}"
        )

    x_column = None
    for candidate in X_CANDIDATES:
        if candidate in df.columns:
            x_column = candidate
            break

    if x_column is None:
        x_data = pd.Series(range(len(df)), name="Frame")
        x_label = "Frame"
    else:
        x_data = df[x_column]
        x_label = x_column
        if x_column == "Datetime":
            x_data = pd.to_datetime(x_data, errors="coerce")
            if x_data.isna().all():
                raise ValueError("Datetime column exists but all values failed to parse.")
        else:
            x_data = pd.to_numeric(x_data, errors="coerce")

    y_data = pd.to_numeric(df[ratio_column], errors="coerce")
    valid_mask = ~(x_data.isna() | y_data.isna())
    x_data = x_data[valid_mask].reset_index(drop=True)
    y_data = y_data[valid_mask].reset_index(drop=True)

    if len(x_data) < 2:
        raise ValueError("Need at least 2 valid points for animation.")

    return x_data, y_data, x_label


def load_plg_series(
    csv_path: Path,
    ratio_column: str = DEFAULT_RATIO_COLUMN,
    gaussian_smoothed: bool = False,
    gaussian_sigma: float | None = None,
    plot_config: dict | None = None,
) -> dict:
    """Load PLG series as reusable data structure for external scripts."""
    x_data, y_data, x_label = _load_data(csv_path, ratio_column)
    y_raw = y_data.copy()
    y_plot = y_data.copy()
    sigma_used: float | None = None

    if gaussian_smoothed:
        sigma = gaussian_sigma
        if sigma is None:
            sigma = 5.0
            if plot_config:
                processing_cfg = plot_config.get("processing", {}) or plot_config.get("sections", {}).get("processing", {})
                sigma = float(
                    processing_cfg.get(
                        "timeseries_gaussian_sigma",
                        plot_config.get("timeseries_gaussian_sigma", 5),
                    )
                )
        sigma_used = float(sigma)
        y_plot = pd.Series(apply_gaussian_smoothing(y_raw.values, sigma=sigma_used))

    return {
        "x": x_data,
        "y_raw": y_raw,
        "y_plot": y_plot,
        "x_label": x_label,
        "ratio_column": ratio_column,
        "sigma_used": sigma_used,
        "is_datetime": bool(pd.api.types.is_datetime64_any_dtype(x_data)),
    }


def build_animation(
    csv_path: Path,
    ratio_column: str,
    interval_ms: int,
    marker_size: int,
    plot_config: dict | None = None,
    gaussian_smoothed: bool = False,
    gaussian_sigma: float | None = None,
) -> tuple[FuncAnimation, plt.Figure]:
    series_data = load_plg_series(
        csv_path=csv_path,
        ratio_column=ratio_column,
        gaussian_smoothed=gaussian_smoothed,
        gaussian_sigma=gaussian_sigma,
        plot_config=plot_config,
    )
    x_data = series_data["x"]
    y_raw = series_data["y_raw"]
    y_plot = series_data["y_plot"]
    x_label = series_data["x_label"]
    sigma_used = series_data["sigma_used"]
    y_label = "Fluorescence/G-band ratio (baseline corrected)"

    fig, ax = plt.subplots(figsize=(10, 5))
    if gaussian_smoothed:
        raw_line, = ax.plot(
            [],
            [],
            color="gray",
            linewidth=1.4,
            alpha=0.6,
            label=ratio_column,
        )
        smooth_line, = ax.plot(
            [],
            [],
            color="#1f77b4",
            linewidth=2.0,
            label=f"{ratio_column} (Gaussian, sigma={sigma_used:g})",
        )
    else:
        raw_line = None
        smooth_line, = ax.plot([], [], color="#1f77b4", linewidth=2.0, label=ratio_column)

    head, = ax.plot([], [], "o", color="crimson", markersize=marker_size, label="Current point")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Fluorescence/G-band Ratio Progression")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best")

    x_min, x_max = x_data.min(), x_data.max()
    y_min = min(float(y_raw.min()), float(y_plot.min()))
    y_max = max(float(y_raw.max()), float(y_plot.max()))
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else 0.1

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    if pd.api.types.is_datetime64_any_dtype(x_data):
        if plot_config:
            light_cycle = plot_config.get("metadata", {}).get("light_cycle", "Constant")
            light_transition = parse_light_transition_config(plot_config)
            shade_transition = parse_shade_transition_config(plot_config)
            treatment_events = parse_treatment_events_config(plot_config)

            add_day_night_shading(
                ax,
                x_data.min(),
                x_data.max(),
                light_cycle=light_cycle,
                light_transition=light_transition,
            )

            if light_transition:
                transition_time = light_transition["transition_datetime"]
                if x_data.min() <= transition_time <= x_data.max():
                    ax.axvline(
                        transition_time,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.7,
                        label="Light transition",
                    )

            if shade_transition:
                transition_time = shade_transition["transition_datetime"]
                if x_data.min() <= transition_time <= x_data.max():
                    label_text = "Shade transition"
                    if "ppfd" in shade_transition:
                        label_text += f" (PPFD: {shade_transition['ppfd']})"
                    ax.axvline(
                        transition_time,
                        color="purple",
                        linestyle="--",
                        linewidth=2,
                        alpha=0.7,
                        label=label_text,
                    )

            if treatment_events:
                for event in treatment_events:
                    event_time = event["datetime"]
                    if x_data.min() <= event_time <= x_data.max():
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

    def init():
        if raw_line is not None:
            raw_line.set_data([], [])
        smooth_line.set_data([], [])
        head.set_data([], [])
        if raw_line is None:
            return smooth_line, head
        return raw_line, smooth_line, head

    def update(frame: int):
        x_slice = x_data.iloc[: frame + 1]
        y_raw_slice = y_raw.iloc[: frame + 1]
        y_slice = y_plot.iloc[: frame + 1]
        if raw_line is not None:
            raw_line.set_data(x_slice, y_raw_slice)
        smooth_line.set_data(x_slice, y_slice)
        head.set_data([x_slice.iloc[-1]], [y_slice.iloc[-1]])
        if raw_line is None:
            return smooth_line, head
        return raw_line, smooth_line, head

    anim = FuncAnimation(
        fig,
        update,
        frames=len(x_data),
        init_func=init,
        interval=interval_ms,
        blit=True,
        repeat=False,
    )
    return anim, fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate baseline-corrected Fluorescence/G-band ratio from processed_data.csv."
    )
    parser.add_argument("csv_path", type=Path, help="Path to processed_data.csv")
    parser.add_argument(
        "--ratio-column",
        default=DEFAULT_RATIO_COLUMN,
        help=f"Column to animate (default: {DEFAULT_RATIO_COLUMN})",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=120,
        help="Milliseconds between frames (default: 120)",
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=7,
        help="Head marker size (default: 7)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output animation path (.gif or .mp4). If omitted, opens interactive window.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="FPS used when saving output (default: 12).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Optional path to config.yaml used for day/night shading.",
    )
    parser.add_argument(
        "--profile-name",
        default=None,
        help="Profile name in config file (required if --config-path is provided).",
    )
    parser.add_argument(
        "--gaussian-smoothed",
        action="store_true",
        help="Animate Gaussian-smoothed ratio (uses config sigma if available).",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=None,
        help="Sigma for Gaussian smoothing (overrides config when provided).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.config_path is None) != (args.profile_name is None):
        raise ValueError("Use --config-path and --profile-name together.")

    plot_config = None
    if args.config_path and args.profile_name:
        plot_config = load_profile_config(str(args.config_path), args.profile_name)

    anim, fig = build_animation(
        csv_path=args.csv_path,
        ratio_column=args.ratio_column,
        interval_ms=args.interval_ms,
        marker_size=args.marker_size,
        plot_config=plot_config,
        gaussian_smoothed=args.gaussian_smoothed,
        gaussian_sigma=args.gaussian_sigma,
    )

    if args.output:
        output_path = args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix.lower()
        if suffix == ".gif":
            anim.save(output_path, writer="pillow", fps=args.fps)
        elif suffix == ".mp4":
            anim.save(output_path, writer="ffmpeg", fps=args.fps)
        else:
            raise ValueError("Output must end with .gif or .mp4")
        print(f"Saved animation: {output_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
