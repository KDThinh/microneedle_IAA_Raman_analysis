"""Slide figure: all in-planta PL/G-band profiles, one row per experimental condition.

Layout (agreed for the DiSTAP slide 6):
  - one ROW per condition, replicates left-to-right across columns (ragged; up to max n)
  - coloured by treatment (blue Control / red Drought / green Shade)
  - figure sized for the LEFT portion of a 16:9 slide at full usable height; the right of the
    slide is left blank for title / takeaway text added in PowerPoint
  - per-panel elapsed-time x (each run its full duration) and independent y, since absolute
    level and duration differ several-fold between runs

Standalone (does not touch aggregate_results.py's QC overview.png).

    python scripts/slide_overview.py --output "G:/.../_aggregate"
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.size"] = 11
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swnt_iaa_analysis.core.loader import resolve_path  # noqa: E402
from swnt_iaa_analysis.core.baseline import apply_gaussian_smoothing, apply_als_baseline  # noqa: E402
from swnt_iaa_analysis.analysis.fourier import compute_fourier_transform  # noqa: E402
from swnt_iaa_analysis.io.config import load_profile_config  # noqa: E402

RESULTS_DIR_RE = re.compile(r"results_v4_(\d{8})_(\d{6})")
FFT_PERIOD_MIN, FFT_PERIOD_MAX = 2.0, 48.0     # hours; range shown on the FFT panel
RATIO = "Fluorescence_to_Gband_Ratio_BaselineCorrected"
TREATMENT_COLORS = {"Control": "tab:blue", "Drought": "tab:red", "Shade": "tab:green"}

# Row order: plant (Nb, Bok Choy) -> treatment (Control, Shade, Drought)
# -> light cycle (Constant, 6to22, 8to24). Unknown values sort last.
PLANT_ORDER = {"Nb": 0, "Bok Choy": 1}
TREATMENT_ORDER = {"Control": 0, "Shade": 1, "Drought": 2}
LIGHT_ORDER = {"Constant": 0, "6to22": 1, "8to24": 2}


def condition_sort_key(key):
    plant, treatment, cycle = key
    return (PLANT_ORDER.get(str(plant), 9), TREATMENT_ORDER.get(str(treatment), 9),
            LIGHT_ORDER.get(str(cycle), 9), str(plant), str(treatment), str(cycle))


BASELINE_HOURS = 24.0


def find_latest_results(data_dir: Path):
    cand = []
    for p in data_dir.rglob("results_v4_*"):
        if p.is_dir() and (p / "processed_data.csv").exists():
            m = RESULTS_DIR_RE.search(p.name)
            if m:
                try:
                    cand.append((datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"), p))
                except ValueError:
                    pass
    return max(cand, key=lambda x: x[0])[1] if cand else None


def _fft_params(config_path, name):
    """Resolve this profile's FFT settings (inheritance applied), matching the pipeline."""
    c = load_profile_config(str(config_path), name)
    pc = c.get("processing", {}) or c.get("sections", {}).get("processing", {})

    def gp(k, d):
        return pc[k] if k in pc else (c[k] if k in c else d)

    return dict(sigma=float(gp("fft_gaussian_sigma", 50)),
                lam=float(gp("fft_als_lambda", 1e8)),
                p=float(gp("fft_als_p", 1e-4)),
                niter=int(gp("fft_als_iterations", 20)),
                top=int(gp("fft_top_peaks", 10)))


def _fourier_of(y_raw, time_hours, fp):
    """FFT via the same steps as pipeline._compute_fft: gaussian -> ALS detrend -> FFT.

    Returns (period_hours, normalised_magnitude, peak_period) restricted to the display window,
    or (None, None, nan) on failure.
    """
    try:
        smoothed = apply_gaussian_smoothing(y_raw, sigma=fp["sigma"])
        als = apply_als_baseline(smoothed, lam=fp["lam"], p=fp["p"], niter=fp["niter"])
        ft = compute_fourier_transform(smoothed - als, time_hours, top_peaks=fp["top"])
        freqs, mag = ft["frequencies"], ft["magnitude"]
        pos = freqs > 0
        period = 1.0 / freqs[pos]
        m = mag[pos]
        sel = (period >= FFT_PERIOD_MIN) & (period <= FFT_PERIOD_MAX)
        period, m = period[sel], m[sel]
        order = np.argsort(period)
        period, m = period[order], m[order]
        if len(m) == 0 or not np.isfinite(m).any():
            return None, None, np.nan
        peak_period = float(period[np.nanargmax(m)])
        return period, m / (np.nanmax(m) or 1.0), peak_period
    except Exception as exc:
        print(f"  ! FFT failed: {type(exc).__name__}")
        return None, None, np.nan


def load_profiles(config_path: Path):
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs = []
    for name, prof in (cfg.get("profiles") or {}).items():
        if not isinstance(prof, dict) or name == "processing_default":
            continue
        meta = prof.get("metadata") or {}
        if str(meta.get("experiment", "")).lower() != "in planta":
            continue
        raman = (prof.get("data_source") or {}).get("raman_relative_path")
        resolved = resolve_path(raman) if raman else None
        rdir = find_latest_results(Path(resolved).parent) if resolved else None
        if rdir is None:
            continue
        df = pd.read_csv(rdir / "processed_data.csv")
        if RATIO not in df.columns or "Datetime" not in df.columns:
            continue
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime", RATIO]).sort_values("Datetime")
        if len(df) < 50:
            continue
        elapsed = (df["Datetime"] - df["Datetime"].min()).dt.total_seconds() / 86400.0
        y_raw = df[RATIO].to_numpy(float)
        # normalise each run to its own first-24 h baseline (% of baseline); y-axes stay
        # independent per panel so the SHAPE of each curve is visible, not a shared scale.
        baseline = np.nanmedian(y_raw[elapsed.to_numpy() <= BASELINE_HOURS / 24.0])
        if not np.isfinite(baseline) or baseline == 0:
            continue

        fp = _fft_params(config_path, name)
        y_norm = 100.0 * y_raw / baseline
        # black-dashed overlay = the Gaussian smoothing that also feeds the FFT (same sigma)
        smooth_norm = apply_gaussian_smoothing(y_norm, sigma=fp["sigma"])
        # FFT computed on the raw baseline-corrected ratio, exactly like the pipeline
        ft_period, ft_mag, peak_period = _fourier_of(y_raw, elapsed.to_numpy() * 24.0, fp)

        runs.append(dict(
            key=(str(meta.get("plant_type")), str(meta.get("treatment")), str(meta.get("light_cycle"))),
            plant=meta.get("plant_type"), treatment=meta.get("treatment"),
            cycle=meta.get("light_cycle"), replicate=meta.get("replicate_number"),
            temp_hum=("ctrl" if meta.get("temp_hum_control") else "var"),
            days=elapsed.to_numpy(), y=y_norm, y_smooth=smooth_norm,
            ft_period=ft_period, ft_mag=ft_mag, peak_period=peak_period))
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--output", default=str(REPO_ROOT / "aggregated_results"))
    ap.add_argument("--width-frac", type=float, default=0.98,
                    help="Fraction of the 13.33in slide width the figure occupies (default 0.98)")
    ap.add_argument("--strip-titles", action="store_true",
                    help="Remove all auto text except the condition row labels (per-panel run "
                         "labels, FFT title, axis labels, baseline note) for external annotation")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_profiles(Path(args.config))
    by_cond = {}
    for r in runs:
        by_cond.setdefault(r["key"], []).append(r)
    for members in by_cond.values():
        members.sort(key=lambda m: (m["replicate"] is None, m["replicate"]))

    ordered = [(k, by_cond[k]) for k in sorted(by_cond, key=condition_sort_key)]

    n_rows = len(ordered)
    n_rep = max(len(v) for _, v in ordered)
    n_cols = n_rep + 1                              # + rightmost FFT column

    from matplotlib.ticker import MaxNLocator
    fig_w = args.width_frac * 13.33
    fig_h = 6.9                                     # ~full usable height under an empty title strip
    width_ratios = [1.0] * n_rep + [1.6]            # FFT column a little wider
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False,
                             gridspec_kw={"width_ratios": width_ratios})
    left, right, top, bottom = 0.13, 0.995, 0.955, 0.06
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom, wspace=0.28, hspace=0.5)

    for row, (key, members) in enumerate(ordered):
        plant, treatment, cycle = key
        color = TREATMENT_COLORS.get(str(treatment), "tab:gray")
        rep_counts = {}
        for m in members:
            rep_counts[m["replicate"]] = rep_counts.get(m["replicate"], 0) + 1

        # replicate PL/G panels with the Gaussian-smoothing overlay
        for col in range(n_rep):
            ax = axes[row][col]
            if col < len(members):
                m = members[col]
                ax.plot(m["days"], m["y"], lw=0.7, color=color)
                ax.plot(m["days"], m["y_smooth"], lw=1.0, color="black", ls="--")  # gaussian
                ax.axhline(100, color="0.6", lw=0.5, ls=":")
                ax.margins(x=0.02, y=0.10)
                ax.tick_params(labelsize=6, length=2, pad=1)
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_major_locator(MaxNLocator(4))
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                if not args.strip_titles:
                    tag = f"Run{m['replicate']}"
                    if rep_counts[m["replicate"]] > 1:
                        tag += f" (T/H {m['temp_hum']})"
                    ax.set_title(tag, fontsize=6.5, color="0.4", pad=1)
            else:
                ax.axis("off")

        # rightmost FFT panel: every run in the condition overlaid (period on x)
        axf = axes[row][n_rep]
        any_ft = False
        for m in members:
            if m["ft_period"] is None:
                continue
            any_ft = True
            axf.plot(m["ft_period"], m["ft_mag"], lw=0.9, color=color, alpha=0.75)
        if any_ft:
            axf.axvline(24, color="0.4", lw=0.6, ls=":")
            axf.axvline(12, color="0.7", lw=0.5, ls=":")
            axf.set_xlim(0, FFT_PERIOD_MAX)
            axf.set_ylim(0, 1.05)
            axf.set_yticks([])
            axf.set_xticks([0, 12, 24, 36, 48])
            axf.tick_params(labelsize=6, length=2, pad=1)
            for s in ("top", "right", "left"):
                axf.spines[s].set_visible(False)
            if row == 0 and not args.strip_titles:
                axf.set_title("FFT (norm.)", fontsize=6.5, color="0.4", pad=1)
        else:
            axf.axis("off")

        # condition row label (kept in both modes), one line, left-aligned in the gutter
        p0 = axes[row][0].get_position()
        fig.text(0.004, (p0.y0 + p0.y1) / 2, f"{plant} - {treatment} - {cycle}\nn = {len(members)}",
                 ha="left", va="center", fontsize=7, fontweight="bold", color=color)

    if not args.strip_titles:
        xf = axes[-1][n_rep].get_position()
        fig.text((left + right) / 2 * 0.9, 0.02, "days since start of run", ha="center", fontsize=9)
        fig.text((xf.x0 + xf.x1) / 2, 0.02, "period (h)", ha="center", fontsize=8, color="0.35")
        fig.text(left * 0.44, top + 0.012, "% of own\nbaseline (24 h)", ha="center", va="bottom",
                 fontsize=7, color="0.4")

    svg_dir = out_dir / "svg"; svg_dir.mkdir(exist_ok=True)
    png = out_dir / "slide_overview_by_condition.png"
    fig.savefig(str(png), dpi=300, facecolor="white")        # no tight bbox: keep exact aspect
    fig.savefig(str(svg_dir / "slide_overview_by_condition.svg"), facecolor="white")
    plt.close(fig)
    print(f"{n_rows} conditions x up to {n_rep} runs + FFT column; "
          f"figure {fig_w:.2f} x {fig_h:.2f} in (aspect {fig_w/fig_h:.2f})")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
