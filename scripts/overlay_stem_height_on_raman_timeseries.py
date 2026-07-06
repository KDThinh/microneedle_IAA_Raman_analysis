"""
Overlay stem height and total plant height on the swnt_iaa_analysis F/G ratio
time series (datetime x-axis, 6→22 day/night shading).

Raman scans (~5 min) and timelapse frames (~30 min) share calendar time but not
sample times, so height is plotted at its own timestamps on a secondary y-axis.

Example (defaults match Run 4 / run 11 / DEV_1AB22C05B465):
  python scripts/overlay_stem_height_on_raman_timeseries.py
  python scripts/overlay_stem_height_on_raman_timeseries.py -o overlay.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from swnt_iaa_analysis.core.utils import add_day_night_shading

RUN4 = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1"
)
DEFAULT_RAMAN_CSV = (
    RUN4 / "Raw data" / "run 11" / "results_v4_20260513_210329" / "processed_data.csv"
)
DEFAULT_STEM_CSV = (
    RUN4 / "DEV_1AB22C05B465" / "stem_height_analysis_v2" / "stem_height_timeseries.csv"
)
DEFAULT_RATIO_COL = "Fluorescence_to_Gband_Ratio_BaselineCorrected"


def _load_raman_csv(path: Path, ratio_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Datetime" not in df.columns:
        raise ValueError(f"Expected 'Datetime' column in {path}")
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
    if ratio_col not in df.columns:
        raise ValueError(f"Column {ratio_col!r} not found in {path}")
    return df


def _load_stem_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "datetime_iso" not in df.columns:
        raise ValueError(f"Expected 'datetime_iso' column in {path}")
    df["Datetime"] = pd.to_datetime(df["datetime_iso"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
    for col in ("stem_h_px_filtered", "total_h_px_filtered"):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found in {path}")
    return df


def _ratio_title(column: str) -> str:
    name = column.replace("_", " ").replace("BaselineCorrected", "").strip()
    if name.endswith("Ratio"):
        name = name.replace("Fluorescence to", "Fluorescence/")
    return name


def plot_overlay(
    raman_df: pd.DataFrame,
    stem_df: pd.DataFrame,
    *,
    ratio_col: str,
    light_cycle: str,
    stem_col: str = "stem_h_px_filtered",
    total_col: str = "total_h_px_filtered",
) -> plt.Figure:
    ratio = pd.to_numeric(raman_df[ratio_col], errors="coerce")
    stem_h = pd.to_numeric(stem_df[stem_col], errors="coerce")
    total_h = pd.to_numeric(stem_df[total_col], errors="coerce")

    x_min = min(raman_df.index.min(), stem_df.index.min())
    x_max = max(raman_df.index.max(), stem_df.index.max())

    fig, ax_ratio = plt.subplots(figsize=(10, 5))
    add_day_night_shading(ax_ratio, x_min, x_max, light_cycle=light_cycle)

    ax_ratio.plot(
        raman_df.index,
        ratio,
        color="#1f77b4",
        linewidth=1.5,
        label=ratio_col,
        zorder=3,
    )
    ax_ratio.set_xlabel("Date time (MM-DD HH)", fontsize=14)
    ax_ratio.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    ax_ratio.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()

    ylabel_ratio = _ratio_title(ratio_col)
    ax_ratio.set_ylabel(ylabel_ratio, fontsize=12, fontweight="bold", color="#1f77b4")
    ax_ratio.tick_params(axis="y", labelcolor="#1f77b4", labelsize=11)
    ax_ratio.set_title(ylabel_ratio, fontsize=13, fontweight="bold")

    ax_height = ax_ratio.twinx()
    ax_height.plot(
        stem_df.index,
        stem_h,
        color="#2ca02c",
        linewidth=2.0,
        marker="o",
        markersize=4,
        label="stem height (filtered)",
        zorder=4,
    )
    ax_height.plot(
        stem_df.index,
        total_h,
        color="#9467bd",
        linewidth=1.6,
        marker="s",
        markersize=3,
        alpha=0.85,
        label="total plant height (filtered)",
        zorder=4,
    )
    ax_height.set_ylabel("height (px)", fontsize=12, fontweight="bold")
    ax_height.tick_params(axis="y", labelsize=11)

    ax_ratio.grid(True, alpha=0.3, linestyle="--", zorder=1)
    ax_ratio.spines["top"].set_visible(False)

    lines_r, labels_r = ax_ratio.get_legend_handles_labels()
    lines_h, labels_h = ax_height.get_legend_handles_labels()
    ax_ratio.legend(
        lines_r + lines_h,
        labels_r + labels_h,
        fontsize=9,
        loc="upper right",
        framealpha=0.92,
    )

    return fig


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Overlay stem/total height on swnt_iaa_analysis F/G ratio time series.",
    )
    ap.add_argument("--raman-csv", type=Path, default=DEFAULT_RAMAN_CSV)
    ap.add_argument("--stem-csv", type=Path, default=DEFAULT_STEM_CSV)
    ap.add_argument("--ratio-column", default=DEFAULT_RATIO_COL)
    ap.add_argument(
        "--light-cycle",
        default="6to22",
        help="Day/night shading cycle (default: 6to22).",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <raman_csv parent>/timeseries_<ratio>_with_stem_height.png).",
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    raman_csv = args.raman_csv.resolve()
    stem_csv = args.stem_csv.resolve()
    if not raman_csv.exists():
        print(f"Error: Raman CSV not found: {raman_csv}", file=sys.stderr)
        return 1
    if not stem_csv.exists():
        print(f"Error: stem height CSV not found: {stem_csv}", file=sys.stderr)
        return 1

    raman_df = _load_raman_csv(raman_csv, args.ratio_column)
    stem_df = _load_stem_csv(stem_csv)

    overlap_start = max(raman_df.index.min(), stem_df.index.min())
    overlap_end = min(raman_df.index.max(), stem_df.index.max())
    print(f"Raman:  {len(raman_df)} points  {raman_df.index.min()} -> {raman_df.index.max()}")
    print(f"Stem:   {len(stem_df)} points  {stem_df.index.min()} -> {stem_df.index.max()}")
    print(f"Overlap calendar range: {overlap_start} -> {overlap_end}")

    fig = plot_overlay(
        raman_df,
        stem_df,
        ratio_col=args.ratio_column,
        light_cycle=args.light_cycle,
    )

    if args.output is None:
        out = raman_csv.parent / f"timeseries_{args.ratio_column}_with_stem_height.png"
    else:
        out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out}")

    if not args.no_show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
