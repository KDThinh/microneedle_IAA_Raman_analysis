"""
Stacked Fluorescence/G-band ratio time series: Nb Control vs Drought (6→22, variable Temp/Hum).

Loads latest results_v4_* processed_data.csv per profile, maps datetime to elapsed days from
first valid observation (matches pipeline output after skip_scans_before), plots sequential scans as solid lines (no markers).

Optional shading: local clock intervals outside [lights-on, lights-off) (default hours 6-22).

Optional second figure: same layout with FFT-style preprocessing on the printed ratio —
Gaussian smoothing then ALS baseline removal — matching the pipeline FFT path.

Optional third figure: FFT magnitude spectra (frequency domain) from ``fft_complete_gband.csv``
in each profile's latest results folder, same stacked layout as the time-series figures.

Night shading uses the first plotted profile's t0 anchor on each subplot; when overlays differ
widely in calendar start, shading is illustrative only.

Extend CONTROL_PROFILES / DROUGHT_PROFILES below when additional runs exist.

Example (from repo root, with editable install):
  python scripts/plot_nb_control_drought_fg_ratio_timeseries.py
  python scripts/plot_nb_control_drought_fg_ratio_timeseries.py --mark-treatment -o fg_ratio_comparison.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo scripts may run without package on PYTHONPATH when launched from cwd
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from swnt_iaa_analysis.core.baseline import apply_als_baseline, apply_gaussian_smoothing
from swnt_iaa_analysis.io.config import find_latest_results_folder, load_profile_config


# --- Hardcoded profile lists (add more profiles here when available) ---
CONTROL_PROFILES: List[str] = [
    "Nb_Control_6to22_Temp_Hum_Variable_Run1",
    "Nb_Control_6to22_Temp_Hum_Variable_Run2",
]

DROUGHT_PROFILES: List[str] = [
    "Nb_Drought_6to22_Temp_Hum_Variable_Run1",
    "Nb_Drought_6to22_Temp_Hum_Variable_Run2",
]


RATIO_BC = "Fluorescence_to_Gband_Ratio_BaselineCorrected"
RATIO_RAW = "Fluorescence_to_Gband_Ratio"

DEFAULT_LIGHT_ON_HOUR = 6.0
DEFAULT_LIGHT_OFF_HOUR = 22.0

# Match swnt_iaa_analysis.core.utils._add_shading_for_cycle (6↔22 + 8↔24 non-Constant cycles)
DAY_SHADE_COLOR = "yellow"
NIGHT_SHADE_COLOR = "blue"
DAY_NIGHT_SHADE_ALPHA = 0.1

# Publication-style defaults (tuned for print / two-column figures)
PUBLICATION_DPI = 300
PUBLICATION_FIGSIZE_IN = (6.5, 5.2)

# Shared Figure.supylabel between stacked axes (PL = photoluminescence / fluorescence proxy)
YLABEL_PRIMARY_FIGURE = "PL/G"
YLABEL_ALS_FIGURE = "Baseline-corrected PL/G"

# Pipeline exports (see swnt_iaa_analysis.visualization.plotting.plot_fft_analysis)
FFT_COMPLETE_FILENAME = "fft_complete_gband.csv"
FFT_FREQ_COL = "Frequency (cycles/hour)"
FFT_MAG_COL = "Magnitude"
FFT_XLIM_RIGHT = 0.5

YLABEL_FFT_FIGURE = "FFT magnitude"


def _profile_legend_label(profile_name: str, resolved: Optional[Dict[str, Any]]) -> str:
    rep = resolved.get("metadata", {}).get("replicate_number") if resolved else None
    if rep is not None:
        return f"Rep {rep}"
    m = re.search(r"Run(\d+)", profile_name)
    return f"Rep {m.group(1)}" if m else profile_name


def _pick_ratio_column(df: pd.DataFrame, prefer_baseline_corrected: bool) -> Optional[str]:
    if prefer_baseline_corrected and RATIO_BC in df.columns:
        return RATIO_BC
    if RATIO_RAW in df.columns:
        return RATIO_RAW
    if RATIO_BC in df.columns:
        return RATIO_BC
    return None


def _load_processed_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    idx = pd.to_datetime(df.index, errors="coerce")
    if idx.isna().all():
        raise ValueError(f"Could not parse datetime index: {csv_path}")
    df.index = idx
    return df.sort_index()


def _t0_from_valid(df: pd.DataFrame, ratio_col: str) -> pd.Timestamp:
    """First time point with finite ratio (exported CSV already reflects skip_scans_before)."""
    y = pd.to_numeric(df[ratio_col], errors="coerce")
    mask = np.isfinite(y.values)
    if not np.any(mask):
        raise ValueError("No finite ratio values in series.")
    return df.index[mask][0]


def _elapsed_days(timestamps: pd.DatetimeIndex, t0: pd.Timestamp) -> np.ndarray:
    delta = timestamps - t0
    return delta.total_seconds().to_numpy(dtype=float) / 86400.0


def _fft_style_als_detrend(y: np.ndarray, resolved_cfg: Dict[str, Any]) -> np.ndarray:
    """Match pipeline FFT prep: Gaussian smooth, ALS baseline z, signal = smoothed - z."""
    if y.size < 3:
        return np.asarray(y, dtype=float)
    sig = np.asarray(y, dtype=float)
    sigma = resolved_cfg.get("fft_gaussian_sigma", 50)
    lam = resolved_cfg.get("fft_als_lambda", 100_000_000)
    p = resolved_cfg.get("fft_als_p", 0.0001)
    niter = resolved_cfg.get("fft_als_iterations", 20)
    smoothed = apply_gaussian_smoothing(sig, sigma=sigma)
    als_z = apply_als_baseline(smoothed, lam=lam, p=p, niter=niter)
    return smoothed - als_z


def _axvspan_contiguous(
    ax: plt.Axes,
    d_rel: np.ndarray,
    mask: np.ndarray,
    *,
    facecolor: str,
    alpha: float,
    zorder: float,
) -> None:
    """Fill contiguous True regions in ``mask`` along ``d_rel`` (same idea as pipeline axvspan loops)."""
    i = 0
    n = len(d_rel)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        left = float(d_rel[i])
        right = float(d_rel[j - 1])
        ax.axvspan(left, right, facecolor=facecolor, alpha=alpha, zorder=zorder, linewidth=0)
        i = j


def _apply_pipeline_style_diurnal_shading(
    ax: plt.Axes,
    t0_anchor: pd.Timestamp,
    *,
    light_on_hour: float,
    light_off_hour: float,
    zorder: float = 0,
) -> None:
    """
    Yellow daytime + blue nighttime at alpha 0.1, matching ``core.utils._add_shading_for_cycle``.
    Day = ``[light_on_hour, light_off_hour)`` in local clock hours (fractional).
    Anchored via ``t0_anchor`` + fractional days on the x-axis.
    """
    xmin, xmax = ax.get_xlim()
    n = max(4000, int((xmax - xmin) * 120))
    d_rel = np.linspace(xmin, xmax, n)
    ts = pd.DatetimeIndex(pd.to_datetime(t0_anchor) + pd.to_timedelta(d_rel, unit="d"))
    hr = ts.hour.astype(float)
    hr += ts.minute / 60.0 + ts.second / 3600.0 + ts.microsecond / 3.6e9
    is_day = (hr >= light_on_hour) & (hr < light_off_hour)
    is_night = ~is_day

    _axvspan_contiguous(ax, d_rel, is_day, facecolor=DAY_SHADE_COLOR, alpha=DAY_NIGHT_SHADE_ALPHA, zorder=zorder)
    _axvspan_contiguous(ax, d_rel, is_night, facecolor=NIGHT_SHADE_COLOR, alpha=DAY_NIGHT_SHADE_ALPHA, zorder=zorder)


def _load_raw_profile_yaml(config_path: Path, profile_name: str) -> Dict[str, Any]:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    profiles = data.get("profiles") or {}
    if profile_name not in profiles:
        raise KeyError(f"Profile '{profile_name}' not in {config_path}")
    return profiles[profile_name]


def _treatment_day_markers(config_path: Path, profile_name: str, t0: pd.Timestamp) -> List[Tuple[float, Dict[str, Any]]]:
    """Return list of (days_since_start, event_dict)."""
    raw = _load_raw_profile_yaml(config_path, profile_name)
    events = raw.get("treatment_events") or []
    out: List[Tuple[float, Dict[str, Any]]] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        ds = ev.get("datetime")
        if ds is None:
            continue
        tt = pd.to_datetime(ds)
        days = (tt - t0).total_seconds() / 86400.0
        out.append((float(days), ev))
    return sorted(out, key=lambda x: x[0])


def _plot_group(
    ax: plt.Axes,
    profiles: Sequence[str],
    config_path: Path,
    *,
    ratio_prefer_bc: bool,
    mark_treatment: bool,
    series_mode: str = "direct",
    night_shading: bool = False,
    light_on_hour: float = DEFAULT_LIGHT_ON_HOUR,
    light_off_hour: float = DEFAULT_LIGHT_OFF_HOUR,
) -> None:
    """series_mode ``direct``: plot CSV ratios. ``als``: FFT-style Gaussian + ALS detrend."""

    shading_anchor_t0: Optional[pd.Timestamp] = None

    for name in profiles:
        folder = find_latest_results_folder(str(config_path), name)
        if folder is None:
            print(f"Warning: no results_v4_* folder for profile '{name}' - skipping.")
            continue
        csv_path = folder / "processed_data.csv"
        if not csv_path.exists():
            print(f"Warning: missing {csv_path} - skipping.")
            continue
        df = _load_processed_csv(csv_path)

        if series_mode == "als":
            ratio_col = _pick_ratio_column(df, prefer_baseline_corrected=True)
            if ratio_col is None:
                ratio_col = _pick_ratio_column(df, prefer_baseline_corrected=False)
            if ratio_col == RATIO_RAW:
                print(f"Warning: ALS plot for '{name}' falls back to raw ratio (baseline-corrected column missing).")
        else:
            ratio_col = _pick_ratio_column(df, prefer_baseline_corrected=ratio_prefer_bc)

        if ratio_col is None:
            print(f"Warning: no F/G ratio column in {csv_path} - skipping.")
            continue
        t0 = _t0_from_valid(df, ratio_col)
        if shading_anchor_t0 is None:
            shading_anchor_t0 = t0

        y = pd.to_numeric(df[ratio_col], errors="coerce")
        mask = np.isfinite(y.values)
        t_days = _elapsed_days(df.index[mask], t0)
        yv = y.values[mask]

        resolved: Dict[str, Any] = {}
        try:
            resolved = load_profile_config(str(config_path), name)
        except Exception:
            resolved = {}

        if series_mode == "als":
            y_plot = _fft_style_als_detrend(yv, resolved)
        else:
            y_plot = yv

        label = _profile_legend_label(name, resolved if resolved else None)

        ax.plot(
            t_days,
            y_plot,
            linestyle="-",
            linewidth=1.0,
            label=label,
            alpha=0.9,
            zorder=2,
        )

        if mark_treatment:
            for day_offset, ev in _treatment_day_markers(config_path, name, t0):
                color = ev.get("marker_color") or "0.35"
                style = "-"
                if ev.get("marker_style") == "dotted":
                    style = ":"
                elif ev.get("marker_style") == "dashed":
                    style = "--"
                ax.axvline(
                    day_offset,
                    color=color,
                    linestyle=style,
                    linewidth=1.2,
                    alpha=0.85,
                    zorder=3,
                )

    ax.autoscale_view()
    if night_shading and shading_anchor_t0 is not None:
        _apply_pipeline_style_diurnal_shading(
            ax,
            shading_anchor_t0,
            light_on_hour=light_on_hour,
            light_off_hour=light_off_hour,
            zorder=0,
        )


def _decorate_fft_axis(ax: plt.Axes) -> None:
    """Match pipeline ``plot_fft_analysis`` frequency panel: diurnal band + 24 h line."""
    ax.axvspan(0.03, 0.05, alpha=0.2, color="gold", zorder=0, linewidth=0)
    ax.axvline(1.0 / 24.0, color="red", linestyle=":", linewidth=1.25, alpha=0.6, zorder=1)
    ax.set_xlim(0.0, FFT_XLIM_RIGHT)


def _plot_fft_group(ax: plt.Axes, profiles: Sequence[str], config_path: Path) -> None:
    """Positive-frequency FFT magnitude vs frequency (cycles/hour) per profile."""

    plotted = False
    for name in profiles:
        folder = find_latest_results_folder(str(config_path), name)
        if folder is None:
            print(f"Warning: no results folder for FFT profile '{name}' - skipping.")
            continue
        fft_path = folder / FFT_COMPLETE_FILENAME
        if not fft_path.exists():
            print(f"Warning: missing {fft_path} (run pipeline with FFT) - skipping.")
            continue
        df = pd.read_csv(fft_path)
        if FFT_FREQ_COL not in df.columns or FFT_MAG_COL not in df.columns:
            print(f"Warning: unexpected columns in {fft_path} - skipping.")
            continue

        fq = pd.to_numeric(df[FFT_FREQ_COL], errors="coerce").to_numpy(dtype=float)
        mag = pd.to_numeric(df[FFT_MAG_COL], errors="coerce").to_numpy(dtype=float)
        mask = (fq > 0.0) & np.isfinite(mag)
        fq = fq[mask]
        mag = mag[mask]
        if fq.size == 0:
            print(f"Warning: no positive-frequency FFT bins in {fft_path} - skipping.")
            continue

        order = np.argsort(fq)
        fq = fq[order]
        mag = mag[order]

        resolved: Dict[str, Any] = {}
        try:
            resolved = load_profile_config(str(config_path), name)
        except Exception:
            resolved = {}

        ax.plot(fq, mag, linestyle="-", linewidth=1.0, label=_profile_legend_label(name, resolved), alpha=0.9, zorder=2)
        plotted = True

    if plotted:
        _decorate_fft_axis(ax)


def _build_fft_comparison_figure(
    cfg_path: Path,
    *,
    shared_y_axis_label: str,
) -> plt.Figure:
    _apply_publication_style()
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=PUBLICATION_FIGSIZE_IN,
        layout="constrained",
        gridspec_kw={"hspace": 0.1},
    )
    fig.patch.set_facecolor("white")

    _plot_fft_group(ax_top, CONTROL_PROFILES, cfg_path)
    leg_top = ax_top.legend(loc="upper right", fontsize=8, frameon=True, fancybox=False, edgecolor="0.85")
    if leg_top is not None:
        leg_top.get_frame().set_linewidth(0.6)

    _plot_fft_group(ax_bot, DROUGHT_PROFILES, cfg_path)
    leg_bot = ax_bot.legend(loc="upper right", fontsize=8, frameon=True, fancybox=False, edgecolor="0.85")
    if leg_bot is not None:
        leg_bot.get_frame().set_linewidth(0.6)

    ax_bot.set_xlabel("Frequency (cycles/hour)")
    _configure_publication_axes(ax_top)
    _configure_publication_axes(ax_bot)
    fig.supylabel(shared_y_axis_label)
    return fig


def _configure_publication_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(0.8)


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": PUBLICATION_DPI,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.labelpad": 4,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "mathtext.fontset": "dejavusans",
        }
    )


def _build_comparison_figure(
    cfg_path: Path,
    *,
    ratio_prefer_bc: bool,
    mark_treatment: bool,
    series_mode: str,
    night_shading: bool,
    light_on_hour: float,
    light_off_hour: float,
    shared_y_axis_label: str,
) -> plt.Figure:
    _apply_publication_style()
    # Use constrained_layout so the supylabel (and tick labels) are spaced automatically.
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=PUBLICATION_FIGSIZE_IN,
        layout="constrained",
        gridspec_kw={"hspace": 0.1},
    )
    fig.patch.set_facecolor("white")

    _plot_group(
        ax_top,
        CONTROL_PROFILES,
        cfg_path,
        ratio_prefer_bc=ratio_prefer_bc,
        mark_treatment=mark_treatment,
        series_mode=series_mode,
        night_shading=night_shading,
        light_on_hour=light_on_hour,
        light_off_hour=light_off_hour,
    )
    leg_top = ax_top.legend(loc="upper right", fontsize=8, frameon=True, fancybox=False, edgecolor="0.85")
    if leg_top is not None:
        leg_top.get_frame().set_linewidth(0.6)

    _plot_group(
        ax_bot,
        DROUGHT_PROFILES,
        cfg_path,
        ratio_prefer_bc=ratio_prefer_bc,
        mark_treatment=mark_treatment,
        series_mode=series_mode,
        night_shading=night_shading,
        light_on_hour=light_on_hour,
        light_off_hour=light_off_hour,
    )
    leg_bot = ax_bot.legend(loc="upper right", fontsize=8, frameon=True, fancybox=False, edgecolor="0.85")
    if leg_bot is not None:
        leg_bot.get_frame().set_linewidth(0.6)

    ax_bot.set_xlabel("Days since start (run)")
    _configure_publication_axes(ax_top)
    _configure_publication_axes(ax_bot)

    # Constrained layout positions the shared y-axis label adjacent to the y-tick labels.
    fig.supylabel(shared_y_axis_label)

    return fig


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Stacked Control vs Drought F/G ratio vs days since run start.",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path.cwd() / "config.yaml",
        help="Path to config.yaml",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path.cwd() / "nb_control_vs_drought_fg_ratio_timeseries.png",
        help="Output path for the primary CSV ratio figure",
    )
    p.add_argument(
        "--use-raw-ratio",
        action="store_true",
        help=f"Prefer {RATIO_RAW} over baseline-corrected ratio when both exist (primary figure only).",
    )
    p.add_argument(
        "--mark-treatment",
        action="store_true",
        help="Plot vertical markers from YAML treatment_events (profiles without events unaffected).",
    )
    p.add_argument(
        "--no-night-shading",
        action="store_true",
        help="Disable 6→22 cycle day/night shading (see --light-hour options).",
    )
    p.add_argument(
        "--light-on-hour",
        type=float,
        default=DEFAULT_LIGHT_ON_HOUR,
        help="Lights-on fractional hour local time (default: 6 = 06:00).",
    )
    p.add_argument(
        "--light-off-hour",
        type=float,
        default=DEFAULT_LIGHT_OFF_HOUR,
        help="Lights-off fractional hour local time (default: 22 = 22:00).",
    )
    p.add_argument(
        "--skip-als-figure",
        action="store_true",
        help="Skip saving the second figure (Gaussian + ALS detrend preview).",
    )
    p.add_argument(
        "--als-output",
        type=Path,
        default=None,
        help=(
            "Path for ALS detrend PNG (default: <primary stem>_als<suffix> next to "
            "the primary output)."
        ),
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening an interactive window.",
    )
    p.add_argument(
        "--skip-fft-figure",
        action="store_true",
        help="Skip saving stacked FFT magnitude plot (needs fft_complete_gband.csv per profile).",
    )
    p.add_argument(
        "--fft-output",
        type=Path,
        default=None,
        help=(
            "Path for FFT comparison PNG (default: <primary stem>_fft<suffix> next to "
            "primary output)."
        ),
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    cfg = args.config.resolve()
    if not cfg.exists():
        print(f"Error: config not found: {cfg}", file=sys.stderr)
        return 1

    ratio_prefer_bc = not args.use_raw_ratio
    night_on = not args.no_night_shading

    fig1 = _build_comparison_figure(
        cfg,
        ratio_prefer_bc=ratio_prefer_bc,
        mark_treatment=args.mark_treatment,
        series_mode="direct",
        night_shading=night_on,
        light_on_hour=args.light_on_hour,
        light_off_hour=args.light_off_hour,
        shared_y_axis_label=YLABEL_PRIMARY_FIGURE,
    )

    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(outp, dpi=PUBLICATION_DPI, facecolor="white", edgecolor="none")
    print(f"Wrote {outp.resolve()}")

    if not args.skip_als_figure:
        fig2 = _build_comparison_figure(
            cfg,
            ratio_prefer_bc=ratio_prefer_bc,
            mark_treatment=args.mark_treatment,
            series_mode="als",
            night_shading=night_on,
            light_on_hour=args.light_on_hour,
            light_off_hour=args.light_off_hour,
            shared_y_axis_label=YLABEL_ALS_FIGURE,
        )
        als_out = Path(args.als_output) if args.als_output else outp.with_name(f"{outp.stem}_als{outp.suffix}")
        als_out.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(als_out, dpi=PUBLICATION_DPI, facecolor="white", edgecolor="none")
        print(f"Wrote {als_out.resolve()}")
        plt.close(fig2)

    if not args.skip_fft_figure:
        fig3 = _build_fft_comparison_figure(
            cfg,
            shared_y_axis_label=YLABEL_FFT_FIGURE,
        )
        fft_out = Path(args.fft_output) if args.fft_output else outp.with_name(f"{outp.stem}_fft{outp.suffix}")
        fft_out.parent.mkdir(parents=True, exist_ok=True)
        fig3.savefig(fft_out, dpi=PUBLICATION_DPI, facecolor="white", edgecolor="none")
        print(f"Wrote {fft_out.resolve()}")
        plt.close(fig3)

    if not args.no_show:
        plt.show()
    plt.close(fig1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
