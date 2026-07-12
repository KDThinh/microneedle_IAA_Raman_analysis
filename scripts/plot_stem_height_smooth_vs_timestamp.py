"""
Plot smoothed stem height versus timestamp with 6→22 day/night shading.

The day/night shading is inferred from the `swnt_iaa_analysis` codebase via
`swnt_iaa_analysis.core.utils.add_day_night_shading`.

Example:
  python scripts/plot_stem_height_smooth_vs_timestamp.py --csv "path/to/stem_height_run4.csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, hilbert, lombscargle

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from swnt_iaa_analysis.core.utils import add_day_night_shading
from swnt_iaa_analysis.core.baseline import apply_gaussian_smoothing


DEFAULT_CSV = Path(
    r"C:\Users\ryank\Code\microneedle_IAA_Raman_analysis\plant_timelapse\dlc\stem_height_run4.csv"
)

DEFAULT_RAMAN_CSV = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_6to22\Temp_Hum_Variable\Run 4\Raman data\run 11\results_v4_20260513_210329\processed_data.csv"
)
DEFAULT_RATIO_COL = "Fluorescence_to_Gband_Ratio_BaselineCorrected"

# Mirrors `scripts/plot_nb_control_drought_fg_ratio_timeseries.py` publication defaults.
PUBLICATION_DPI = 300
PUBLICATION_FIGSIZE_IN = (6.5, 4.6)


def _apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": PUBLICATION_DPI,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.labelpad": 4,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
            "mathtext.fontset": "dejavusans",
        }
    )


def _configure_publication_axes(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", length=4, width=0.8, direction="out")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_linewidth(0.8)


def load_stem_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Expected 'timestamp' column in {csv_path}")
    if "height_smooth" not in df.columns:
        raise ValueError(f"Expected 'height_smooth' column in {csv_path}")

    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    y = pd.to_numeric(df["height_smooth"], errors="coerce")

    out = pd.DataFrame({"timestamp": ts, "height_smooth": y}).dropna()
    out = out.sort_values("timestamp")
    if out.empty:
        raise ValueError(f"No valid rows after parsing timestamp/height_smooth in {csv_path}")
    # Also compute elapsed time in hours to enable a proper time-based derivative.
    dt = out["timestamp"].diff().dt.total_seconds() / 3600.0
    out["dt_hours"] = dt
    return out


def load_raman_csv(csv_path: Path, ratio_col: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Raman CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Datetime" not in df.columns:
        raise ValueError(f"Expected 'Datetime' column in {csv_path}")
    if ratio_col not in df.columns:
        raise ValueError(f"Expected ratio column {ratio_col!r} in {csv_path}")

    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"]).set_index("Datetime").sort_index()
    return df


def _compute_first_derivative(df: pd.DataFrame) -> pd.Series:
    """
    First derivative of height_smooth with respect to time (px/hour).

    Uses central differences where possible, falling back to forward/backward
    differences at the edges. dt is taken from actual time deltas between
    timestamps (in hours).
    """
    h = df["height_smooth"].to_numpy(dtype=float)
    dt = df["dt_hours"].to_numpy(dtype=float)
    n = h.size
    if n < 2:
        return pd.Series([float("nan")] * n, index=df["timestamp"])

    deriv = [float("nan")] * n

    # Forward difference for the first point where dt[1] is valid.
    if n >= 2 and dt[1] and not pd.isna(dt[1]):
        deriv[0] = (h[1] - h[0]) / dt[1]

    # Central differences for interior points when both sides have valid dt.
    for i in range(1, n - 1):
        if dt[i] and dt[i + 1] and not (pd.isna(dt[i]) or pd.isna(dt[i + 1])):
            # Effective dt is half of (dt_left + dt_right)
            dt_eff = 0.5 * (dt[i] + dt[i + 1])
            deriv[i] = (h[i + 1] - h[i - 1]) / (2.0 * dt_eff)

    # Backward difference for the last point where dt[n-1] is valid.
    if n >= 2 and dt[-1] and not pd.isna(dt[-1]):
        deriv[-1] = (h[-1] - h[-2]) / dt[-1]

    return pd.Series(deriv, index=df["timestamp"])


def plot_height_and_derivative(
    df: pd.DataFrame,
    *,
    light_cycle: str,
) -> plt.Figure:
    _apply_publication_style()

    x = df["timestamp"]
    y = df["height_smooth"]
    dydt = _compute_first_derivative(df)

    xmin = x.min()
    xmax = x.max()

    fig, (ax_h, ax_d) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(PUBLICATION_FIGSIZE_IN[0], PUBLICATION_FIGSIZE_IN[1] * 1.6),
        layout="constrained",
    )
    fig.patch.set_facecolor("white")

    # Day/night shading on both panels.
    add_day_night_shading(ax_h, xmin.to_pydatetime(), xmax.to_pydatetime(), light_cycle=light_cycle)
    add_day_night_shading(ax_d, xmin.to_pydatetime(), xmax.to_pydatetime(), light_cycle=light_cycle)

    # Height panel.
    ax_h.plot(
        x,
        y,
        color="#2ca02c",
        linewidth=2.2,
        marker=None,
        zorder=3,
    )
    ax_h.set_ylabel("Stem height (px, smoothed)", fontsize=11, fontweight="bold")
    ax_h.set_title("Height_smooth vs timestamp", fontsize=11, fontweight="bold")
    ax_h.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, zorder=1)
    _configure_publication_axes(ax_h)

    # Derivative (growth rate) panel.
    ax_d.plot(
        dydt.index,
        dydt.values,
        color="#d62728",
        linewidth=1.8,
        marker=None,
        zorder=3,
    )
    ax_d.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)
    ax_d.set_xlabel("Timestamp (MM-DD HH)", fontsize=11, fontweight="bold")
    ax_d.set_ylabel("Growth rate (px/hour)", fontsize=11, fontweight="bold")
    ax_d.set_title("d(Height_smooth)/dt vs timestamp", fontsize=11, fontweight="bold")
    ax_d.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, zorder=1)

    # Reasonable time ticks for publication figures (shared x-axis).
    ax_d.xaxis.set_major_locator(mdates.DayLocator())
    ax_d.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    fig.autofmt_xdate()

    _configure_publication_axes(ax_d)
    return fig


def plot_height_only(df: pd.DataFrame, *, light_cycle: str) -> plt.Figure:
    """Standalone plant height vs timestamp (with 6→22 day/night shading)."""
    _apply_publication_style()
    x = df["timestamp"]
    y = df["height_smooth"]

    xmin = x.min()
    xmax = x.max()

    fig, ax = plt.subplots(figsize=PUBLICATION_FIGSIZE_IN, layout="constrained")
    fig.patch.set_facecolor("white")

    add_day_night_shading(ax, xmin.to_pydatetime(), xmax.to_pydatetime(), light_cycle=light_cycle)

    ax.plot(x, y, color="#2ca02c", linewidth=2.2, marker=None, zorder=3)
    ax.set_xlabel("Timestamp (MM-DD HH)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Stem height (px, smoothed)", fontsize=11, fontweight="bold")
    ax.set_title("Plant height vs timestamp", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, zorder=1)

    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    fig.autofmt_xdate()

    _configure_publication_axes(ax)
    return fig


def plot_ratio_and_growth_rate_stack(
    raman_df: pd.DataFrame,
    stem_df: pd.DataFrame,
    *,
    ratio_col: str,
    light_cycle: str,
    ratio_gaussian_sigma: float,
    ratio_gaussian_linewidth: float,
) -> plt.Figure:
    """
    Stacked 2-row figure:
      1) F/G ratio (with thicker black dashed Gaussian smoothing overlay)
      2) Growth rate (first derivative, px/hour)
    sharing the same datetime x-axis and 6→22 day/night shading.
    """
    _apply_publication_style()

    ratio = pd.to_numeric(raman_df[ratio_col], errors="coerce")
    ratio_series = pd.Series(ratio.values, index=raman_df.index).sort_index()

    # Gaussian smooth curve overlay.
    # Fill missing points via time interpolation so gaussian_filter1d works.
    ratio_filled = ratio_series.interpolate(method="time").ffill().bfill()
    ratio_smoothed = (
        apply_gaussian_smoothing(ratio_filled.values, sigma=ratio_gaussian_sigma)
        if ratio_gaussian_sigma > 0
        else None
    )

    dydt = _compute_first_derivative(stem_df)

    xmin = min(raman_df.index.min(), stem_df["timestamp"].min())
    xmax = max(raman_df.index.max(), stem_df["timestamp"].max())

    fig, (ax_r, ax_g) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(PUBLICATION_FIGSIZE_IN[0], PUBLICATION_FIGSIZE_IN[1] * 1.9),
        layout="constrained",
    )
    fig.patch.set_facecolor("white")

    # Day/night shading.
    add_day_night_shading(ax_r, xmin.to_pydatetime(), xmax.to_pydatetime(), light_cycle=light_cycle)
    add_day_night_shading(ax_g, xmin.to_pydatetime(), xmax.to_pydatetime(), light_cycle=light_cycle)

    # Ratio panel (top).
    ax_r.plot(
        ratio_series.index,
        ratio_series.values,
        color="#1f77b4",
        linewidth=1.8,
        marker=None,
        zorder=3,
        label=ratio_col,
    )
    if ratio_smoothed is not None:
        ax_r.plot(
            ratio_series.index,
            ratio_smoothed,
            color="black",
            linewidth=ratio_gaussian_linewidth,
            linestyle="--",
            alpha=0.98,
            zorder=4,
            label=f"Gaussian smooth (sigma={ratio_gaussian_sigma:g})",
        )

    ax_r.set_ylabel("F/G ratio", fontsize=11, fontweight="bold")
    ax_r.set_title(f"{ratio_col} vs timestamp", fontsize=11, fontweight="bold")
    ax_r.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, zorder=1)
    _configure_publication_axes(ax_r)
    leg = ax_r.legend(
        loc="upper right",
        fontsize=8,
        frameon=True,
        fancybox=False,
        edgecolor="0.85",
    )
    if leg is not None:
        leg.get_frame().set_linewidth(0.6)

    # Growth-rate panel (bottom).
    ax_g.plot(
        dydt.index,
        dydt.values,
        color="#d62728",
        linewidth=1.8,
        marker=None,
        zorder=3,
    )
    ax_g.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7, zorder=2)
    ax_g.set_xlabel("Timestamp (MM-DD HH)", fontsize=11, fontweight="bold")
    ax_g.set_ylabel("Growth rate (px/hour)", fontsize=11, fontweight="bold")
    ax_g.set_title("d(Height_smooth)/dt vs timestamp", fontsize=11, fontweight="bold")
    ax_g.grid(True, alpha=0.25, linestyle="--", linewidth=0.6, zorder=1)

    ax_g.xaxis.set_major_locator(mdates.DayLocator())
    ax_g.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H"))
    fig.autofmt_xdate()
    _configure_publication_axes(ax_g)

    return fig


def _build_common_grid(
    ratio_series: pd.Series,
    growth_series: pd.Series,
    *,
    resample_minutes: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, float]:
    """
    Resample ratio (already smoothed) and growth-rate series to a common uniform grid.

    Returns (time_index, ratio_vals, growth_vals, dt_hours).
    """
    # Overlap in calendar time.
    t_start = max(ratio_series.index.min(), growth_series.index.min())
    t_end = min(ratio_series.index.max(), growth_series.index.max())
    if t_end <= t_start:
        raise ValueError("No temporal overlap between ratio and growth-rate series.")

    dt = f"{resample_minutes}min"
    t_grid = pd.date_range(t_start, t_end, freq=dt)
    dt_hours = resample_minutes / 60.0

    # Important: do NOT `reindex(t_grid)` before time interpolation.
    # Most samples won't land exactly on the resample grid, and `reindex` + `.interpolate(method="time")`
    # would drop all original points, yielding constant/NaN results.
    #
    # Instead, interpolate numerically using the original timestamps mapped to "hours since t_start".
    def _interp_to_grid(series: pd.Series) -> np.ndarray:
        # Ensure unique timestamps by averaging duplicates.
        s = series.groupby(level=0).mean()
        idx = s.index
        x = (idx - t_start).total_seconds().to_numpy(dtype=float) / 3600.0
        y = s.to_numpy(dtype=float)

        # Keep only finite samples.
        mask = np.isfinite(y) & np.isfinite(x)
        x = x[mask]
        y = y[mask]
        if x.size == 0:
            return np.full(t_grid.size, np.nan, dtype=float)
        if x.size == 1:
            return np.full(t_grid.size, float(y[0]), dtype=float)

        # np.interp requires x to be increasing.
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        t = (t_grid - t_start).total_seconds().to_numpy(dtype=float) / 3600.0
        return np.interp(t, x, y).astype(float)

    r_grid = _interp_to_grid(ratio_series)
    g_grid = _interp_to_grid(growth_series)
    return t_grid, r_grid, g_grid, dt_hours


def _estimate_dominant_period_hours(
    t_index: pd.DatetimeIndex,
    signal_vals: np.ndarray,
    *,
    period_min_hours: float,
    period_max_hours: float,
) -> float:
    """
    Estimate dominant period (hours) via Lomb-Scargle on irregular timestamps.
    """
    if len(t_index) < 10:
        raise ValueError("Not enough points to estimate dominant period.")

    # Use irregular timestamps in hours since start.
    t0 = t_index[0]
    t_hours = (t_index - t0).total_seconds().to_numpy(dtype=float) / 3600.0
    y = np.asarray(signal_vals, dtype=float)
    # Detrend by removing mean.
    y = y - np.nanmean(y)

    # Frequency grid in cycles/hour; Lomb-Scargle expects angular frequency (rad/hour).
    f_min = 1.0 / period_max_hours
    f_max = 1.0 / period_min_hours
    freqs = np.linspace(f_min, f_max, 500)
    ang_freqs = 2.0 * np.pi * freqs

    power = lombscargle(t_hours, y, ang_freqs)
    if not np.isfinite(power).any():
        raise ValueError("Lomb-Scargle power is not finite; cannot estimate period.")

    best_idx = int(np.nanargmax(power))
    best_freq = freqs[best_idx]
    if best_freq <= 0:
        raise ValueError("Estimated dominant frequency is non-positive.")
    return float(1.0 / best_freq)


def _normalized_cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    *,
    dt_hours: float,
    max_lag_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalized cross-correlation of two equal-length series.

    Returns (lags_hours, corr) where lags_hours > 0 means x leads y by that many hours.
    """
    if x.size != y.size:
        raise ValueError("x and y must have the same length for cross-correlation.")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x - np.nanmean(x)
    y = y - np.nanmean(y)

    sx = np.nanstd(x)
    sy = np.nanstd(y)
    if sx == 0 or sy == 0:
        raise ValueError("Cannot compute correlation: zero variance series.")

    # Replace any residual NaNs with zeros after centering.
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)

    corr_full = np.correlate(x / sx, y / sy, mode="full")
    lags = np.arange(-x.size + 1, x.size, dtype=int)
    lags_hours = lags * dt_hours

    mask = np.abs(lags_hours) <= max_lag_hours
    return lags_hours[mask], corr_full[mask]


def _bandpass_for_period(
    values: np.ndarray,
    *,
    dt_hours: float,
    period_hours: float,
    width_fraction: float = 0.2,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass around the dominant period.
    """
    fs = 1.0 / dt_hours  # samples per hour
    f0 = 1.0 / period_hours  # cycles/hour
    low = f0 * (1.0 - width_fraction)
    high = f0 * (1.0 + width_fraction)
    if low <= 0:
        low = f0 * 0.5
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = min(high / nyq, 0.99)
    if not (0 < low_n < high_n < 1):
        # Fallback: return detrended-only signal if band design fails.
        return values - np.nanmean(values)

    b, a = butter(3, [low_n, high_n], btype="bandpass")
    return filtfilt(b, a, values)


def _hilbert_phase_metrics(
    t_grid: pd.DatetimeIndex,
    r_vals: np.ndarray,
    g_vals: np.ndarray,
    *,
    dt_hours: float,
    dominant_period_hours: float,
) -> tuple[float, float]:
    """
    Compute mean phase offset (degrees) and PLV between ratio and growth-rate cycles.
    """
    # Bandpass both signals around dominant period.
    r_bp = _bandpass_for_period(r_vals, dt_hours=dt_hours, period_hours=dominant_period_hours)
    g_bp = _bandpass_for_period(g_vals, dt_hours=dt_hours, period_hours=dominant_period_hours)

    # Hilbert analytic signals.
    z_r = hilbert(r_bp)
    z_g = hilbert(g_bp)
    phi_r = np.angle(z_r)
    phi_g = np.angle(z_g)

    delta = phi_r - phi_g
    # Wrap to [-pi, pi].
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi

    vec = np.exp(1j * delta)
    mean_vec = np.mean(vec)
    plv = np.abs(mean_vec)
    mean_phase_deg = float(np.degrees(np.angle(mean_vec)))
    return mean_phase_deg, float(plv)


def _compute_light_phase_hours(
    t_grid: pd.DatetimeIndex,
    *,
    light_on_hour: float = 6.0,
) -> np.ndarray:
    """
    Map timestamps to local diurnal phase (hours since lights-on, modulo 24).
    """
    hours = t_grid.hour.astype(float)
    hours += t_grid.minute / 60.0
    hours += t_grid.second / 3600.0
    phase = (hours - light_on_hour) % 24.0
    return phase


def _binned_phase_curves(
    t_grid: pd.DatetimeIndex,
    r_vals: np.ndarray,
    g_vals: np.ndarray,
    *,
    phase_bin_hours: float,
    light_on_hour: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bin ratio and growth signals by local diurnal phase hour.

    Returns (phase_centers, r_binned, g_binned).
    """
    phase = _compute_light_phase_hours(t_grid, light_on_hour=light_on_hour)
    bins = np.arange(0.0, 24.0 + phase_bin_hours, phase_bin_hours)
    if bins.size < 3:
        raise ValueError("phase_bin_hours too large; need at least a few bins.")

    # Use pandas for grouping convenience.
    df_phase = pd.DataFrame(
        {
            "phase": phase,
            "ratio": r_vals,
            "growth": g_vals,
        }
    )
    df_phase["bin"] = pd.cut(df_phase["phase"], bins=bins, include_lowest=True, right=False)
    grouped = df_phase.groupby("bin", observed=True).mean(numeric_only=True)

    # Bin centers.
    phase_centers = []
    for interval in grouped.index:
        if interval is None:
            continue
        left = float(interval.left)
        right = float(interval.right)
        phase_centers.append(0.5 * (left + right))
    phase_centers = np.asarray(phase_centers, dtype=float)

    r_binned = grouped["ratio"].to_numpy(dtype=float)
    g_binned = grouped["growth"].to_numpy(dtype=float)
    return phase_centers, r_binned, g_binned


def _best_circular_shift(
    a: np.ndarray,
    b: np.ndarray,
    *,
    bin_width_hours: float,
) -> tuple[float, float]:
    """
    Find circular shift (in hours) aligning b to a, maximizing Pearson correlation.
    """
    if a.size != b.size:
        raise ValueError("Arrays must have same length for circular shift comparison.")
    n = a.size
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)

    best_r = -np.inf
    best_shift_bins = 0
    for k in range(n):
        b_shift = np.roll(b, k)
        num = np.nansum(a * b_shift)
        den = np.sqrt(np.nansum(a * a) * np.nansum(b_shift * b_shift))
        if den == 0:
            continue
        r = num / den
        if r > best_r:
            best_r = r
            best_shift_bins = k

    shift_hours = best_shift_bins * bin_width_hours
    return float(shift_hours), float(best_r)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Plot height_smooth vs timestamp and its first derivative (growth rate) with day/night shading."
    )
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input CSV file.")
    ap.add_argument(
        "--raman-csv",
        type=Path,
        default=None,
        help=(
            "Optional Raman processed_data.csv to overlay F/G ratio; "
            "if omitted, only height + derivative are plotted."
        ),
    )
    ap.add_argument(
        "--ratio-column",
        default=DEFAULT_RATIO_COL,
        help="F/G ratio column name in Raman CSV "
        f"(default: {DEFAULT_RATIO_COL}).",
    )
    ap.add_argument(
        "--ratio-gaussian-sigma",
        type=float,
        default=20.0,
        help="Gaussian sigma (in index points) for smoothing F/G ratio overlay.",
    )
    ap.add_argument(
        "--ratio-gaussian-linewidth",
        type=float,
        default=2.6,
        help="Line width for the Gaussian smooth overlay on F/G ratio.",
    )
    ap.add_argument("--light-cycle", default="6to22", help="Day/night cycle (default: 6to22).")
    # Cycle-synchronization analysis options.
    ap.add_argument(
        "--cycle-sync",
        action="store_true",
        help=(
            "Compute cycle-synchronization metrics between F/G ratio and growth rate "
            "(requires --raman-csv)."
        ),
    )
    ap.add_argument(
        "--resample-minutes",
        type=int,
        default=10,
        help="Resampling interval (minutes) for analysis grid (default: 10).",
    )
    ap.add_argument(
        "--max-lag-hours",
        type=float,
        default=48.0,
        help="Maximum lag (hours) to scan in cross-correlation (default: 48).",
    )
    ap.add_argument(
        "--period-search-hours-min",
        type=float,
        default=12.0,
        help="Minimum period (hours) for dominant-period search (default: 12).",
    )
    ap.add_argument(
        "--period-search-hours-max",
        type=float,
        default=36.0,
        help="Maximum period (hours) for dominant-period search (default: 36).",
    )
    ap.add_argument(
        "--phase-bin-hours",
        type=float,
        default=1.0,
        help="Bin width (hours) for diurnal-phase averaging (default: 1).",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Stack output PNG path when --raman-csv is provided. "
            "The standalone height plot will be written next to it with '_height' suffix. "
            "If --raman-csv is omitted, this remains the single figure output."
        ),
    )
    ap.add_argument("--dpi", type=int, default=PUBLICATION_DPI, help="Output DPI (default: 300).")
    ap.add_argument("--no-show", action="store_true", help="Do not open the plot window.")
    args = ap.parse_args(argv)

    stem_df = load_stem_csv(args.csv.resolve())

    # Decide which figure(s) to build.
    if args.raman_csv is not None:
        raman_path = args.raman_csv.resolve()
        raman_df = load_raman_csv(raman_path, args.ratio_column)

        # 1) Independent height plot.
        height_fig = plot_height_only(stem_df, light_cycle=args.light_cycle)

        # 2) Stacked ratio (top) + growth rate (bottom).
        stack_fig = plot_ratio_and_growth_rate_stack(
            raman_df,
            stem_df,
            ratio_col=args.ratio_column,
            light_cycle=args.light_cycle,
            ratio_gaussian_sigma=args.ratio_gaussian_sigma,
            ratio_gaussian_linewidth=args.ratio_gaussian_linewidth,
        )

        if args.output is None:
            stack_out = raman_path.parent / "ratio_to_growth_rate_vs_timestamp.png"
        else:
            stack_out = args.output.resolve()

        height_out = stack_out.with_name(stack_out.stem + "_height.png")

        height_out.parent.mkdir(parents=True, exist_ok=True)
        height_fig.savefig(height_out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"Wrote {height_out}")

        stack_out.parent.mkdir(parents=True, exist_ok=True)
        stack_fig.savefig(stack_out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"Wrote {stack_out}")

        if args.cycle_sync:
            # --- Cycle synchronization analysis ---
            # Use the same Gaussian-smoothed ratio as for the stacked plot.
            ratio_raw = pd.to_numeric(raman_df[args.ratio_column], errors="coerce")
            ratio_series = pd.Series(ratio_raw.values, index=raman_df.index).sort_index()
            ratio_filled = ratio_series.interpolate(method="time").ffill().bfill()
            ratio_smoothed = apply_gaussian_smoothing(
                ratio_filled.values, sigma=args.ratio_gaussian_sigma
            )
            ratio_smoothed_series = pd.Series(ratio_smoothed, index=ratio_series.index)

            # Growth rate series on original timestamps.
            growth_series = _compute_first_derivative(stem_df)

            # Common grid and period estimation.
            t_grid, r_grid, g_grid, dt_hours = _build_common_grid(
                ratio_smoothed_series, growth_series, resample_minutes=args.resample_minutes
            )

            # Dominant period from ratio signal.
            try:
                dominant_period_hours = _estimate_dominant_period_hours(
                    t_grid,
                    r_grid,
                    period_min_hours=args.period_search_hours_min,
                    period_max_hours=args.period_search_hours_max,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: dominant period estimation failed: {exc}")
                dominant_period_hours = 24.0

            # Cross-correlation (ratio vs growth).
            try:
                lags_hours, corr_vals = _normalized_cross_correlation(
                    r_grid, g_grid, dt_hours=dt_hours, max_lag_hours=args.max_lag_hours
                )
                best_corr_idx = int(np.nanargmax(corr_vals))
                best_lag_hours = float(lags_hours[best_corr_idx])
                best_corr = float(corr_vals[best_corr_idx])
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: cross-correlation failed: {exc}")
                lags_hours = np.array([])
                corr_vals = np.array([])
                best_lag_hours = np.nan
                best_corr = np.nan

            # Hilbert phase metrics.
            try:
                mean_phase_deg, plv = _hilbert_phase_metrics(
                    t_grid,
                    r_grid,
                    g_grid,
                    dt_hours=dt_hours,
                    dominant_period_hours=dominant_period_hours,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: Hilbert phase analysis failed: {exc}")
                mean_phase_deg = np.nan
                plv = np.nan

            # Light-time binning.
            try:
                phase_centers, r_binned, g_binned = _binned_phase_curves(
                    t_grid,
                    r_grid,
                    g_grid,
                    phase_bin_hours=args.phase_bin_hours,
                    light_on_hour=6.0,
                )
                best_shift_hours, best_phase_corr = _best_circular_shift(
                    r_binned, g_binned, bin_width_hours=args.phase_bin_hours
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Warning: light-time binning failed: {exc}")
                phase_centers = np.array([])
                r_binned = np.array([])
                g_binned = np.array([])
                best_shift_hours = np.nan
                best_phase_corr = np.nan

            # Metrics summary CSV.
            metrics = {
                "dominant_period_hours": [dominant_period_hours],
                "crosscorr_best_lag_hours_ratio_leads_growth": [best_lag_hours],
                "crosscorr_best_corr": [best_corr],
                "hilbert_mean_phase_deg_ratio_minus_growth": [mean_phase_deg],
                "hilbert_plv": [plv],
                "light_phase_best_shift_hours_ratio_vs_growth": [best_shift_hours],
                "light_phase_best_corr": [best_phase_corr],
            }
            metrics_df = pd.DataFrame(metrics)
            metrics_csv = stack_out.with_name("ratio_growth_cycle_sync_metrics.csv")
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"Wrote {metrics_csv}")

            # Diagnostic figure.
            diag_fig, (ax_cc, ax_phase, ax_bins) = plt.subplots(
                3,
                1,
                figsize=(PUBLICATION_FIGSIZE_IN[0], PUBLICATION_FIGSIZE_IN[1] * 2.3),
                layout="constrained",
            )
            diag_fig.patch.set_facecolor("white")

            # Cross-correlation panel.
            if lags_hours.size > 0:
                ax_cc.plot(lags_hours, corr_vals, color="#1f77b4", linewidth=1.5)
                ax_cc.axvline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
                if np.isfinite(best_lag_hours):
                    ax_cc.axvline(
                        best_lag_hours,
                        color="#d62728",
                        linestyle="--",
                        linewidth=1.0,
                        alpha=0.8,
                        label=f"best lag = {best_lag_hours:.2f} h",
                    )
                ax_cc.set_xlabel("Lag (hours, ratio leads > 0)")
                ax_cc.set_ylabel("Normalized cross-correlation")
                ax_cc.set_title("Cross-correlation: ratio vs growth rate")
                ax_cc.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
                if ax_cc.get_legend_handles_labels()[0]:
                    ax_cc.legend(fontsize=8, loc="best")

            # Hilbert phase panel: histogram of phase differences.
            if np.isfinite(mean_phase_deg) and np.isfinite(plv):
                # For simplicity, reuse delta from Hilbert computation if we recompute quickly.
                try:
                    r_bp = _bandpass_for_period(
                        r_grid, dt_hours=dt_hours, period_hours=dominant_period_hours
                    )
                    g_bp = _bandpass_for_period(
                        g_grid, dt_hours=dt_hours, period_hours=dominant_period_hours
                    )
                    z_r = hilbert(r_bp)
                    z_g = hilbert(g_bp)
                    delta = np.angle(z_r) - np.angle(z_g)
                    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
                    ax_phase.hist(
                        np.degrees(delta),
                        bins=36,
                        color="#9467bd",
                        alpha=0.8,
                        edgecolor="none",
                    )
                    ax_phase.set_xlabel("Phase difference (deg, ratio - growth)")
                    ax_phase.set_ylabel("Count")
                    ax_phase.set_title(
                        f"Phase differences (PLV={plv:.2f}, mean={mean_phase_deg:.1f}°)"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"Warning: phase histogram failed: {exc}")

            # Light-phase binned curves.
            if phase_centers.size > 0:
                ax_bins.plot(
                    phase_centers,
                    r_binned,
                    color="#1f77b4",
                    linewidth=1.6,
                    label="Ratio (binned)",
                )
                ax_bins.plot(
                    phase_centers,
                    g_binned,
                    color="#d62728",
                    linewidth=1.6,
                    label="Growth rate (binned)",
                )
                ax_bins.set_xlabel("Local phase since lights-on (hours)")
                ax_bins.set_ylabel("Mean amplitude (arb. units)")
                ax_bins.set_title(
                    "Binned cycles vs local phase "
                    f"(best shift≈{best_shift_hours:.2f} h, r≈{best_phase_corr:.2f})"
                )
                ax_bins.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
                ax_bins.set_xlim(0.0, 24.0)
                ax_bins.legend(fontsize=8, loc="best")

            diag_out = stack_out.with_name("ratio_growth_cycle_sync_diagnostics.png")
            diag_fig.savefig(diag_out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
            plt.close(diag_fig)
            print(f"Wrote {diag_out}")

    else:
        fig = plot_height_and_derivative(stem_df, light_cycle=args.light_cycle)
        if args.output is None:
            out = args.csv.parent / "stem_height_smooth_and_derivative_vs_timestamp.png"
        else:
            out = args.output.resolve()

        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        print(f"Wrote {out}")

    if not args.no_show:
        plt.show()
    plt.close("all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

