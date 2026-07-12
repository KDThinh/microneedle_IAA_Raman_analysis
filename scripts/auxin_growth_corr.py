"""
Auxin (Raman F/G ratio) vs plant growth-rate correlation -- a simple, honest pipeline.

WHY THIS SCRIPT EXISTS
----------------------
We want to know: does the continuously-monitored auxin signal co-vary with how fast
the plant is growing? That sounds like "just compute a correlation", but two traps
make a naive correlation misleading:

  Trap 1 -- Spurious trend correlation.
      Two signals that both drift over the week (auxin creeps up, plant gets taller)
      will show a high correlation even if unrelated. FIX: don't correlate the raw
      levels; correlate the *fluctuations* after removing each signal's slow trend.
      (Growth *rate* is already a change, which helps; we still detrend both.)

  Trap 2 -- The points aren't independent.
      A reading at 4:00pm is nearly identical to 3:30pm, so ~325 time points carry
      the information of only a few dozen independent ones. A textbook p-value assumes
      independence and will look far too significant. FIX: judge significance with a
      "time-shift" permutation test, which respects the autocorrelation.

THE UNIT OF REPLICATION IS THE PLANT, NOT THE TIME POINT.
    One plant is n = 1 for "does auxin relate to growth in this species", no matter
    how many time points it has. So this script produces ONE summary number per plant
    (`analyze`), and a second step (`aggregate`) pools those per-plant numbers across
    all your plants -- that pooled test is where the real statistical evidence lives.

WHAT IT COMPUTES (per plant)
    1. Growth rate  = time-derivative of smoothed stem height (px/hour).
    2. Both signals put on a common, native-cadence time grid (NOT oversampled),
       with real gaps left as gaps (not interpolated across).
    3. Both signals detrended (slow drift removed, daily cycle kept).
    4. Spearman correlation of the two detrended signals (rank-based = robust).
    5. A time-shift permutation p-value for that correlation.
    6. A lead/lag scan: does auxin lead or lag growth, and by how much?

USAGE
    # one plant:
    python scripts/auxin_growth_corr.py analyze \
        --csv "...\\stem_height_run4.csv" \
        --raman-csv "...\\processed_data.csv" \
        --ratio-column "Fluorescence_to_Gband_Ratio_BaselineCorrected"

    # pool across plants once you have several metrics CSVs:
    python scripts/auxin_growth_corr.py aggregate out\\*_auxin_growth_metrics.csv
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Day/night shading is a nice-to-have for the context plot; degrade gracefully if the
# analysis package isn't importable in whatever env you run this from.
try:
    from swnt_iaa_analysis.core.utils import add_day_night_shading
except Exception:  # pragma: no cover
    add_day_night_shading = None


# --------------------------------------------------------------------------- #
# Plot style (kept local so this script is standalone)
# --------------------------------------------------------------------------- #
PUBLICATION_DPI = 300


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": PUBLICATION_DPI,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
        }
    )


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_height_series(csv_path: Path, column: str = "height_smooth") -> pd.Series:
    """Load stem height as a time-indexed Series (index = timestamp, values = px)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Stem CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    for need in ("timestamp", column):
        if need not in df.columns:
            raise ValueError(f"Expected column {need!r} in {csv_path}")
    t = pd.to_datetime(df["timestamp"], errors="coerce")
    y = pd.to_numeric(df[column], errors="coerce")
    s = pd.Series(y.values, index=t).dropna().sort_index()
    # Collapse any exact-duplicate timestamps by averaging.
    s = s.groupby(level=0).mean()
    if s.empty:
        raise ValueError(f"No valid height rows in {csv_path}")
    return s


def load_ratio_series(csv_path: Path, ratio_col: str) -> pd.Series:
    """Load the Raman F/G ratio as a time-indexed Series (index = Datetime)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Raman CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "Datetime" not in df.columns:
        raise ValueError(f"Expected 'Datetime' column in {csv_path}")
    if ratio_col not in df.columns:
        raise ValueError(f"Expected ratio column {ratio_col!r} in {csv_path}")
    t = pd.to_datetime(df["Datetime"], errors="coerce")
    y = pd.to_numeric(df[ratio_col], errors="coerce")
    s = pd.Series(y.values, index=t).dropna().sort_index()
    s = s.groupby(level=0).mean()
    if s.empty:
        raise ValueError(f"No valid ratio rows in {csv_path}")
    return s


# --------------------------------------------------------------------------- #
# Core signal steps
# --------------------------------------------------------------------------- #
def growth_rate(height: pd.Series) -> pd.Series:
    """
    Growth rate (px/hour) = time-derivative of the height series.

    `np.gradient` uses central differences and accepts the actual time coordinate,
    so it handles the slightly irregular capture cadence and the two end points
    correctly in one call.
    """
    t_hours = (height.index - height.index[0]).total_seconds().to_numpy() / 3600.0
    dydt = np.gradient(height.to_numpy(dtype=float), t_hours)
    return pd.Series(dydt, index=height.index)


def regrid_gap_aware(series: pd.Series, grid: pd.DatetimeIndex, max_gap_minutes: float) -> np.ndarray:
    """
    Linearly interpolate `series` onto `grid`, but leave NaN wherever the original
    data has a gap wider than `max_gap_minutes`.

    This is the key difference from naive resampling: we do NOT invent smooth data
    across long gaps (which would fabricate fake structure and inflate correlation).
    """
    t = (series.index - grid[0]).total_seconds().to_numpy() / 60.0  # minutes from grid start
    y = series.to_numpy(dtype=float)
    tg = (grid - grid[0]).total_seconds().to_numpy() / 60.0

    out = np.interp(tg, t, y, left=np.nan, right=np.nan)  # NaN outside the data span

    # For each grid point, find the bracketing original samples; blank it if that
    # bracket is wider than max_gap (i.e., we'd be interpolating across a real gap).
    right_idx = np.searchsorted(t, tg, side="left")
    for i, tgi in enumerate(tg):
        r = right_idx[i]
        if r == 0 or r >= t.size:
            continue  # handled by the left/right=NaN above (edges)
        if (t[r] - t[r - 1]) > max_gap_minutes:
            out[i] = np.nan
    return out


def smooth_gaussian_gap_aware(values: np.ndarray, dt_hours: float, sigma_hours: float) -> np.ndarray:
    """
    Optional light Gaussian denoise that RESPECTS gaps.

    `sigma_hours` is the smoothing width in hours (0 = off). Unlike a plain Gaussian
    filter, this smooths each continuous run of data on its own and leaves NaN gaps
    untouched -- so it never smears real data across a missing span (the same reason
    we don't interpolate across gaps).

    Reminder: this removes fast *noise*, not the slow trend (that's `detrend_rolling`).
    Apply it BEFORE detrending and identically to BOTH signals. Keep it small (1-2 h):
    heavier smoothing further shrinks the already-small effective sample size.
    """
    if sigma_hours <= 0:
        return values
    sigma_steps = sigma_hours / dt_hours
    out = values.copy()
    idx = np.where(np.isfinite(values))[0]
    if idx.size == 0:
        return out
    # Split the valid indices into contiguous runs (breaks wherever a gap sits).
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    for run in runs:
        if run.size >= 1:
            out[run] = gaussian_filter1d(values[run], sigma=sigma_steps, mode="nearest")
    return out


def detrend_rolling(values: np.ndarray, dt_hours: float, window_hours: float) -> np.ndarray:
    """
    Remove slow drift by subtracting a centered rolling mean.

    IMPORTANT: choose `window_hours` LONGER than any cycle you consider real signal.
    We default to 48 h so the multi-day growth ramp is removed but the ~24 h daily
    fluctuation is preserved. NaNs are tolerated (min_periods lets edges through).
    """
    win = max(3, int(round(window_hours / dt_hours)))
    s = pd.Series(values)
    baseline = s.rolling(win, center=True, min_periods=max(2, win // 3)).mean()
    return (s - baseline).to_numpy()


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _spearman_finite(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho over the positions where both arrays are finite."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    return float(stats.spearmanr(a[m], b[m]).statistic)


def shift_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    observed: float,
    n_shuffles: int,
    min_shift: int,
    rng: np.random.Generator,
) -> float:
    """
    Time-shift permutation p-value (two-sided) for a correlation.

    Idea: if the two signals are truly linked, sliding one in time should DESTROY the
    correlation. So we circularly shift `b` by many random offsets, recompute Spearman
    each time to build a "by chance" distribution, and ask how often chance beats the
    real value. This respects autocorrelation (unlike a textbook p-value).

    `min_shift` excludes tiny shifts that leave the signals still roughly aligned.
    """
    n = a.size
    if not np.isfinite(observed) or n < 3 * min_shift:
        return np.nan
    hits = 0
    valid = 0
    # Draw shifts uniformly from [min_shift, n - min_shift] so wrap-around is meaningful.
    for _ in range(n_shuffles):
        k = int(rng.integers(min_shift, n - min_shift))
        r = _spearman_finite(a, np.roll(b, k))
        if not np.isfinite(r):
            continue
        valid += 1
        if abs(r) >= abs(observed):
            hits += 1
    if valid == 0:
        return np.nan
    # +1 smoothing so p is never exactly 0 (you can't prove p < 1/n_shuffles).
    return (hits + 1) / (valid + 1)


def lagged_correlation(
    ratio: np.ndarray,
    growth: np.ndarray,
    *,
    dt_hours: float,
    max_lag_hours: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Spearman correlation as a function of lag.

    CONVENTION (verify with a synthetic if unsure): lag > 0 means AUXIN LEADS GROWTH.
    At lag L we correlate ratio[t] with growth[t + L]; so if a rise in auxin is
    followed L hours later by faster growth, the curve peaks at +L.

    Returns (lags_hours, rho_at_each_lag). Each lag uses a proper per-lag Spearman on
    the overlapping portion, so values are genuine correlations in [-1, 1] -- this is
    what the old np.correlate/(sx*sy) version got wrong.
    """
    n = ratio.size
    max_lag = int(round(max_lag_hours / dt_hours))
    max_lag = min(max_lag, n - 5)
    lags = np.arange(-max_lag, max_lag + 1)
    rho = np.full(lags.size, np.nan)
    for j, lag in enumerate(lags):
        if lag >= 0:
            a = ratio[: n - lag]
            g = growth[lag:]
        else:
            a = ratio[-lag:]
            g = growth[: n + lag]
        rho[j] = _spearman_finite(a, g)
    return lags * dt_hours, rho


def effective_n(x: np.ndarray) -> float:
    """
    Rough effective sample size given lag-1 autocorrelation r1:
        n_eff = n * (1 - r1) / (1 + r1)
    Just a reminder of how little independent information smoothed signals carry;
    the shift test is the real significance check.
    """
    v = x[np.isfinite(x)]
    if v.size < 3:
        return float(v.size)
    r1 = np.corrcoef(v[:-1], v[1:])[0, 1]
    r1 = min(max(r1, 0.0), 0.999)
    return float(v.size * (1 - r1) / (1 + r1))


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def make_figures(
    grid: pd.DatetimeIndex,
    ratio_dt: np.ndarray,
    growth_dt: np.ndarray,
    lags_h: np.ndarray,
    rho_lag: np.ndarray,
    lag_ci: tuple[np.ndarray, np.ndarray] | None,
    *,
    rho0: float,
    best_lag: float,
    dataset: str,
    out_stub: Path,
    dpi: int,
    light_cycle: str,
) -> list[Path]:
    _apply_style()
    written: list[Path] = []

    # (1) Context: the two detrended signals we actually correlate.
    fig, (axr, axg) = plt.subplots(2, 1, sharex=True, figsize=(6.5, 5.2), layout="constrained")
    fig.patch.set_facecolor("white")
    if add_day_night_shading is not None:
        for ax in (axr, axg):
            add_day_night_shading(ax, grid[0].to_pydatetime(), grid[-1].to_pydatetime(), light_cycle=light_cycle)
    axr.plot(grid, ratio_dt, color="#1f77b4", linewidth=1.4)
    axr.set_ylabel("Auxin ratio\n(detrended)", fontweight="bold")
    axr.set_title(f"{dataset}: detrended signals being correlated", fontweight="bold")
    axg.plot(grid, growth_dt, color="#d62728", linewidth=1.4)
    axg.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    axg.set_ylabel("Growth rate\n(detrended, px/h)", fontweight="bold")
    axg.set_xlabel("Time")
    axg.xaxis.set_major_locator(mdates.DayLocator())
    axg.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    for ax in (axr, axg):
        _clean_axes(ax)
    for lbl in axg.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    p = out_stub.with_name(out_stub.stem + "_signals.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(p)

    # (2) Scatter with the headline Spearman rho.
    fig, ax = plt.subplots(figsize=(4.8, 4.6), layout="constrained")
    fig.patch.set_facecolor("white")
    m = np.isfinite(ratio_dt) & np.isfinite(growth_dt)
    ax.scatter(ratio_dt[m], growth_dt[m], s=12, alpha=0.5, color="#6a51a3", edgecolor="none")
    ax.set_xlabel("Auxin ratio (detrended)", fontweight="bold")
    ax.set_ylabel("Growth rate (detrended, px/h)", fontweight="bold")
    ax.set_title(f"{dataset}: Spearman ρ = {rho0:.2f} (lag 0)", fontweight="bold")
    _clean_axes(ax)
    p = out_stub.with_name(out_stub.stem + "_scatter.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(p)

    # (3) Lead/lag curve with the shuffled-null band.
    fig, ax = plt.subplots(figsize=(6.5, 3.6), layout="constrained")
    fig.patch.set_facecolor("white")
    if lag_ci is not None:
        lo, hi = lag_ci
        ax.fill_between(lags_h, lo, hi, color="0.85", label="95% by-chance band")
    ax.plot(lags_h, rho_lag, color="#1f77b4", linewidth=1.6)
    ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    if np.isfinite(best_lag):
        ax.axvline(best_lag, color="#d62728", linewidth=1.1, linestyle="--",
                   label=f"peak lag = {best_lag:+.1f} h")
    ax.set_xlabel("Lag (hours)   —   auxin leads growth  →", fontweight="bold")
    ax.set_ylabel("Spearman ρ", fontweight="bold")
    ax.set_title(f"{dataset}: lead/lag between auxin and growth rate", fontweight="bold")
    _clean_axes(ax)
    ax.legend(fontsize=8, loc="best")
    p = out_stub.with_name(out_stub.stem + "_lag.png")
    fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    written.append(p)

    return written


# --------------------------------------------------------------------------- #
# analyze (one plant)
# --------------------------------------------------------------------------- #
def run_analyze(args: argparse.Namespace) -> int:
    dataset = args.name or Path(args.csv).stem.replace("stem_height_", "")
    rng = np.random.default_rng(args.seed)

    height = load_height_series(Path(args.csv).resolve(), column=args.height_column)
    ratio = load_ratio_series(Path(args.raman_csv).resolve(), args.ratio_column)
    growth = growth_rate(height)

    # Common grid over the overlapping window, at native cadence (no oversampling).
    t_start = max(height.index.min(), ratio.index.min())
    t_end = min(height.index.max(), ratio.index.max())
    if t_end <= t_start:
        raise ValueError("No temporal overlap between height and Raman data.")
    grid = pd.date_range(t_start, t_end, freq=f"{args.resample_minutes}min")
    dt_hours = args.resample_minutes / 60.0

    ratio_grid = regrid_gap_aware(ratio, grid, args.max_gap_minutes)
    growth_grid = regrid_gap_aware(growth, grid, args.max_gap_minutes)

    # Optional denoise BEFORE detrending (smooth-then-detrend), gap-aware, applied
    # identically to BOTH signals. Off by default (--smooth-hours 0).
    ratio_grid = smooth_gaussian_gap_aware(ratio_grid, dt_hours, args.smooth_hours)
    growth_grid = smooth_gaussian_gap_aware(growth_grid, dt_hours, args.smooth_hours)

    # Detrend both (remove slow drift, keep daily fluctuation).
    ratio_dt = detrend_rolling(ratio_grid, dt_hours, args.detrend_window_hours)
    growth_dt = detrend_rolling(growth_grid, dt_hours, args.detrend_window_hours)

    # Headline: lag-0 Spearman + shift-test p-value.
    rho0 = _spearman_finite(ratio_dt, growth_dt)
    min_shift = max(3, int(round(args.detrend_window_hours / dt_hours)))
    p_shift = shift_test(
        ratio_dt, growth_dt, observed=rho0, n_shuffles=args.n_shuffles,
        min_shift=min_shift, rng=rng,
    )

    # Lead/lag curve, and a by-chance band from shuffles (for the figure + peak test).
    lags_h, rho_lag = lagged_correlation(
        ratio_dt, growth_dt, dt_hours=dt_hours, max_lag_hours=args.max_lag_hours
    )
    null_curves = []
    for _ in range(min(args.n_shuffles, 400)):  # 400 is plenty for a 95% band
        k = int(rng.integers(min_shift, ratio_dt.size - min_shift))
        _, r_null = lagged_correlation(
            ratio_dt, np.roll(growth_dt, k), dt_hours=dt_hours, max_lag_hours=args.max_lag_hours
        )
        null_curves.append(r_null)
    # Peak lag = the lag with the strongest |correlation|.
    best_j = int(np.nanargmax(np.abs(rho_lag)))
    best_lag = float(lags_h[best_j])
    best_lag_rho = float(rho_lag[best_j])

    # Significance of that peak, CORRECTED for having scanned many lags. We compare
    # the observed peak |rho| against the distribution of peak |rho| from shuffled
    # (time-shifted) data. This "max-statistic" test is the honest way to ask
    # "could the best-looking lag arise by chance?" -- picking the best of ~100 lags
    # without this correction almost always looks significant.
    if null_curves:
        null_stack = np.vstack(null_curves)
        lag_ci = (np.nanpercentile(null_stack, 2.5, axis=0),
                  np.nanpercentile(null_stack, 97.5, axis=0))
        null_peak = np.nanmax(np.abs(null_stack), axis=1)
        obs_peak = float(np.nanmax(np.abs(rho_lag)))
        p_lag = float((np.sum(null_peak >= obs_peak) + 1) / (null_peak.size + 1))
    else:
        lag_ci = None
        p_lag = np.nan

    n_pairs = int((np.isfinite(ratio_dt) & np.isfinite(growth_dt)).sum())
    n_eff = min(effective_n(ratio_dt), effective_n(growth_dt))

    out_dir = Path(args.outdir).resolve() if args.outdir else Path(args.csv).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_stub = out_dir / f"{dataset}_auxin_growth"

    figs = make_figures(
        grid, ratio_dt, growth_dt, lags_h, rho_lag, lag_ci,
        rho0=rho0, best_lag=best_lag, dataset=dataset, out_stub=out_stub,
        dpi=args.dpi, light_cycle=args.light_cycle,
    )

    metrics = pd.DataFrame(
        {
            "dataset": [dataset],
            "n_grid_pairs": [n_pairs],
            "n_effective_approx": [round(n_eff, 1)],
            "spearman_rho_lag0": [round(rho0, 4)],
            "p_shift_test": [round(p_shift, 4) if np.isfinite(p_shift) else np.nan],
            "best_lag_hours_auxin_leads": [round(best_lag, 2)],
            "best_lag_rho": [round(best_lag_rho, 4)],
            "best_lag_p_corrected": [round(p_lag, 4) if np.isfinite(p_lag) else np.nan],
            "resample_minutes": [args.resample_minutes],
            "detrend_window_hours": [args.detrend_window_hours],
            "smooth_hours": [args.smooth_hours],
            "max_gap_minutes": [args.max_gap_minutes],
            "n_shuffles": [args.n_shuffles],
        }
    )
    metrics_csv = out_stub.with_name(out_stub.stem + "_metrics.csv")
    metrics.to_csv(metrics_csv, index=False)

    # Human-readable summary.
    print(f"\n=== {dataset} ===")
    print(f"  paired grid points:        {n_pairs}  (~{n_eff:.0f} effectively independent)")
    print(f"  Spearman rho (lag 0):      {rho0:+.3f}")
    print(f"  shift-test p-value:        {p_shift:.3f}"
          + ("   <- unlikely by chance" if np.isfinite(p_shift) and p_shift < 0.05 else "   <- consistent with chance"))
    print(f"  best lag (auxin leads +):  {best_lag:+.1f} h  (rho={best_lag_rho:+.2f}, "
          f"corrected p={p_lag:.3f})"
          + ("   <- peak survives multi-lag correction" if np.isfinite(p_lag) and p_lag < 0.05
             else "   <- peak is consistent with chance"))
    print(f"  wrote: {metrics_csv}")
    for f in figs:
        print(f"  wrote: {f}")
    print("\nNote: this is ONE plant. Run the others, then `aggregate` the metrics CSVs "
          "for the population-level answer.")
    return 0


# --------------------------------------------------------------------------- #
# aggregate (across plants)
# --------------------------------------------------------------------------- #
def run_aggregate(args: argparse.Namespace) -> int:
    files: list[str] = []
    for pat in args.metrics:
        files.extend(glob.glob(pat))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No metrics CSVs matched: {args.metrics}")

    rows = [pd.read_csv(f) for f in files]
    df = pd.concat(rows, ignore_index=True)
    r = df["spearman_rho_lag0"].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    print(f"\n=== aggregate across {n} plants ===")
    print(df[["dataset", "spearman_rho_lag0", "p_shift_test", "best_lag_hours_auxin_leads"]].to_string(index=False))

    if n >= 2:
        mean_r = float(np.mean(r))
        # One-sample Wilcoxon: is the set of per-plant correlations centered off zero?
        # (Wilcoxon is the robust, non-normal-assuming choice; needs a handful of plants.)
        if n >= 6:
            try:
                w = stats.wilcoxon(r)
                ptxt = f"Wilcoxon p = {w.pvalue:.3f}"
            except ValueError as exc:
                ptxt = f"Wilcoxon n/a ({exc})"
        else:
            t = stats.ttest_1samp(r, 0.0)
            ptxt = f"t-test p = {t.pvalue:.3f} (few plants; treat as provisional)"
        n_pos = int((r > 0).sum())
        print(f"\n  mean per-plant Spearman rho: {mean_r:+.3f}")
        print(f"  plants with positive rho:    {n_pos}/{n}")
        print(f"  centered off zero?           {ptxt}")
        print("\n  Interpretation: consistent same-sign rho across plants (and a small p) is the")
        print("  real evidence that auxin and growth co-vary in this species -- not any single plant.")

    out = Path(args.output).resolve() if args.output else Path(files[0]).parent / "auxin_growth_aggregate.csv"
    df.to_csv(out, index=False)
    print(f"\n  wrote: {out}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="Analyze one plant (stem CSV + Raman CSV).")
    a.add_argument("--csv", required=True, help="stem_height_*.csv (has 'timestamp' + height column).")
    a.add_argument("--raman-csv", required=True, help="Raman processed_data.csv (has 'Datetime').")
    a.add_argument("--ratio-column", default="Fluorescence_to_Gband_Ratio_BaselineCorrected")
    a.add_argument("--height-column", default="height_smooth")
    a.add_argument("--name", default=None, help="Dataset label (default: derived from --csv).")
    a.add_argument("--resample-minutes", type=int, default=30,
                   help="Common-grid step; use your native cadence, don't oversample (default 30).")
    a.add_argument("--max-gap-minutes", type=float, default=90.0,
                   help="Don't interpolate across gaps wider than this (default 90).")
    a.add_argument("--detrend-window-hours", type=float, default=48.0,
                   help="Remove drift slower than this; keep it > any cycle you count as signal (default 48).")
    a.add_argument("--smooth-hours", type=float, default=0.0,
                   help="Optional Gaussian denoise (sigma in HOURS) applied to BOTH signals BEFORE "
                        "detrending; 0 = off (default). Keep small (1-2 h): it removes fast noise, not "
                        "the trend, and shrinks the effective sample size.")
    a.add_argument("--max-lag-hours", type=float, default=24.0, help="Lead/lag scan range (default +/-24).")
    a.add_argument("--n-shuffles", type=int, default=1000, help="Time-shift permutations (default 1000).")
    a.add_argument("--seed", type=int, default=0, help="RNG seed for reproducible shuffles.")
    a.add_argument("--light-cycle", default="6to22")
    a.add_argument("--outdir", default=None, help="Where to write outputs (default: next to --csv).")
    a.add_argument("--dpi", type=int, default=PUBLICATION_DPI)
    a.set_defaults(func=run_analyze)

    g = sub.add_parser("aggregate", help="Pool per-plant metrics CSVs into a population result.")
    g.add_argument("metrics", nargs="+", help="One or more *_metrics.csv paths or globs.")
    g.add_argument("--output", default=None, help="Combined CSV output path.")
    g.set_defaults(func=run_aggregate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
