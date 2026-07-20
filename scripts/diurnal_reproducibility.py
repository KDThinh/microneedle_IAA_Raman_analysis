"""Diurnal-cycle reproducibility across runs, for every experimental condition.

For each run the multi-day drift is removed (24 h rolling median) so only the within-day
component remains, the residual is folded onto time-of-day, and the per-day profiles are
averaged into one mean diurnal profile. Runs are then compared within their condition.

Detrending matters: without it a shared multi-day decline inflates every correlation and
unrelated runs look alike for the wrong reason.

Two reproducibility measures are reported:
  within-run  -- day-to-day correlation inside one run (is the plant's own cycle stable?)
  between-run -- correlation of mean profiles across runs (does the cycle replicate?)

The observed lighting-transition hours are detected from the data and reported alongside,
because runs sharing a config `light_cycle` label do not always share an actual schedule --
and reproducibility tracks the real schedule, not the label.

Usage
-----
    python scripts/diurnal_reproducibility.py
    python scripts/diurnal_reproducibility.py --output "G:/.../_aggregate/diurnal"
    python scripts/diurnal_reproducibility.py --min-days 3
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
# Match the plotting conventions used in swnt_iaa_analysis/visualization/plotting.py
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["axes.labelsize"] = 12
matplotlib.rcParams["axes.titlesize"] = 13
matplotlib.rcParams["xtick.labelsize"] = 11
matplotlib.rcParams["ytick.labelsize"] = 11
matplotlib.rcParams["legend.fontsize"] = 10
matplotlib.rcParams["figure.titlesize"] = 14
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swnt_iaa_analysis.core.loader import resolve_path            # noqa: E402
from swnt_iaa_analysis.core.transition_baseline import detect_transition_ramps  # noqa: E402

RESULTS_DIR_RE = re.compile(r"results_v4_(\d{8})_(\d{6})")
RATIO = "Fluorescence_to_Gband_Ratio_BaselineCorrected"
FLUOR = "Normalized_Fluorescence_Intensity"
TREATMENT_COLORS = {"Control": "tab:blue", "Drought": "tab:red", "Shade": "tab:green"}


def find_latest_results(data_dir: Path):
    candidates = []
    for path in data_dir.rglob("results_v4_*"):
        if not path.is_dir() or not (path / "processed_data.csv").exists():
            continue
        m = RESULTS_DIR_RE.search(path.name)
        if m:
            try:
                candidates.append((datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S"), path))
            except ValueError:
                continue
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def observed_transition_hours(df, bin_minutes):
    """Hours of day where lighting transitions are actually detected in this run."""
    if FLUOR not in df.columns:
        return []
    values = df[FLUOR].astype(float).to_numpy()
    interval = np.median(np.diff(df["Datetime"].to_numpy()).astype("timedelta64[s]").astype(float)) / 60
    if not np.isfinite(interval) or interval <= 0:
        return []
    level = float(np.nanmedian(np.abs(values)))
    regions = detect_transition_ramps(
        values, slope_halfwindow=max(2, int(round(25 / interval))),
        min_displacement=0.03 * level, min_rate_per_scan=0.05 * level * (interval / 60))
    if not regions:
        return []
    hours = pd.Series([df["Datetime"].iloc[a].hour for a, _ in regions])
    return sorted(hours.value_counts().head(2).index.tolist())


def load_run(name, prof, bin_minutes, min_days):
    """Mean diurnal profile plus per-day profiles for one run, or None if unusable."""
    raman = (prof.get("data_source") or {}).get("raman_relative_path")
    resolved = resolve_path(raman) if raman else None
    results_dir = find_latest_results(Path(resolved).parent) if resolved else None
    if results_dir is None:
        return None

    df = pd.read_csv(results_dir / "processed_data.csv")
    if RATIO not in df.columns or "Datetime" not in df.columns:
        return None
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime", RATIO]).sort_values("Datetime")
    if len(df) < 100:
        return None

    nbins = 24 * 60 // bin_minutes
    series = pd.Series(df[RATIO].to_numpy(float), index=df["Datetime"]).resample(f"{bin_minutes}min").mean()
    trend = series.rolling(nbins, center=True, min_periods=nbins // 3).median()
    resid = (series - trend).dropna()
    if len(resid) < nbins * min_days:
        return None

    tod = (resid.index.hour * 60 + resid.index.minute) // bin_minutes
    daily = resid.groupby([resid.index.normalize(), tod]).mean().unstack(level=1)
    daily = daily.reindex(columns=range(nbins)).interpolate(axis=1, limit_direction="both")
    daily = daily.dropna(how="any")
    if len(daily) < min_days:
        return None

    values = daily.to_numpy(float)
    mean_profile = values.mean(axis=0)
    within = [np.corrcoef(values[i], values[j])[0, 1]
              for i in range(len(values)) for j in range(i + 1, len(values))]

    meta = prof.get("metadata") or {}
    return dict(
        profile=name, plant=meta.get("plant_type"), treatment=meta.get("treatment"),
        cycle=meta.get("light_cycle"), replicate=meta.get("replicate_number"),
        temp_hum=("TempHum-ctrl" if meta.get("temp_hum_control") else "TempHum-var"),
        condition=(f"{meta.get('plant_type')} | {meta.get('treatment')} | {meta.get('light_cycle')}"
                   f" | {'TempHum-ctrl' if meta.get('temp_hum_control') else 'TempHum-var'}"),
        mean_profile=mean_profile, n_days=len(daily),
        amplitude=float(mean_profile.max() - mean_profile.min()),
        peak_hour=float(np.argmax(mean_profile) * bin_minutes / 60.0),
        within_r=float(np.nanmedian(within)) if within else np.nan,
        observed=observed_transition_hours(df, bin_minutes),
    )



def largest_agreeing_set(members, threshold):
    """Largest subset of runs where EVERY pair correlates at >= threshold.

    A simple mean correlation would let one run with two good partners hide a pair that
    disagree, so agreement is required pairwise across the whole subset.
    """
    import itertools
    n = len(members)
    if n < 2:
        return list(range(n)), np.nan, np.nan
    M = np.array([[np.corrcoef(a["mean_profile"], b["mean_profile"])[0, 1]
                   for b in members] for a in members])
    for size in range(n, 1, -1):
        for combo in itertools.combinations(range(n), size):
            pairs = [M[i, j] for i, j in itertools.combinations(combo, 2)]
            if min(pairs) >= threshold:
                return list(combo), float(min(pairs)), float(max(pairs))
    return [], np.nan, np.nan



def shade_daynight_tod(ax, light_cycle):
    """Day/night shading on a 0-24 h time-of-day axis (yellow day, blue night).

    Unlike the elapsed-time condition plots, the phase here is unambiguous: every run in a
    condition is folded onto the same time-of-day axis, so one overlay is correct for all.
    Same 6to22 / 8to24 / Constant convention as swnt_iaa_analysis.core.utils.
    """
    cycle = str(light_cycle)
    ax.axvspan(0, 24, facecolor="blue", alpha=0.06, zorder=0, lw=0)
    if cycle == "Constant":
        ax.axvspan(0, 24, facecolor="yellow", alpha=0.13, zorder=0, lw=0)
    else:
        on_h = 8 if cycle == "8to24" else 6
        off_h = 24 if cycle == "8to24" else 22
        ax.axvspan(on_h, off_h, facecolor="yellow", alpha=0.13, zorder=0, lw=0)


def plot_condition(condition, members, out_path, bin_minutes):
    nbins = 24 * 60 // bin_minutes
    hours = np.arange(nbins) * bin_minutes / 60.0
    color = TREATMENT_COLORS.get(str(members[0]["treatment"]), "tab:gray")
    n = len(members)

    fig = plt.figure(figsize=(15, 4.4))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    cycle = members[0]["cycle"]
    shade_daynight_tod(ax0, cycle)
    shade_daynight_tod(ax1, cycle)

    for m in members:
        label = f"Run{m['replicate']} (peak {m['peak_hour']:.1f}h)"
        line, = ax0.plot(hours, m["mean_profile"], lw=1.6, label=label)
        z = m["mean_profile"] - m["mean_profile"].mean()
        z = z / (z.std() or 1)
        ax1.plot(hours, z, lw=1.6, color=line.get_color(), label=f"Run{m['replicate']}")
        # Where this run's lighting transitions were actually detected. Shading above follows
        # the configured light_cycle, so a marker away from a shading edge means the metadata
        # and the data disagree for that run.
        for h in m["observed"]:
            for ax in (ax0, ax1):
                ax.axvline(h, color=line.get_color(), ls=":", lw=1.1, alpha=0.65)
    if len({m["replicate"] for m in members}) < len(members):   # collision guard
        for ax in (ax0, ax1):
            ax.legend([f"{m['profile'][-22:]}" for m in members], fontsize=6)
    for ax, title in ((ax0, "Mean diurnal profile"), (ax1, "Z-scored (shape only)")):
        ax.axhline(0, color="k", lw=0.7, ls=":")
        ax.set_xlabel("time of day (h)"); ax.set_xticks(range(0, 25, 4))
        ax.set_xlim(0, 24)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)
        ax.tick_params(axis="both", labelsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax0.set_ylabel("PL/G-band deviation from 24 h trend", fontsize=11, fontweight="bold")
    ax0.text(0.01, 0.02, "yellow = lights on (config), dotted = detected transitions",
             transform=ax0.transAxes, fontsize=7, color="0.35")

    if n >= 2:
        M = np.array([[np.corrcoef(a["mean_profile"], b["mean_profile"])[0, 1]
                       for b in members] for a in members])
        im = ax2.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        labels = [f"R{m['replicate']}" for m in members]
        ax2.set_xticks(range(n)); ax2.set_xticklabels(labels, fontsize=7)
        ax2.set_yticks(range(n)); ax2.set_yticklabels(labels, fontsize=7)
        for i in range(n):
            for j in range(n):
                ax2.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                         color="white" if abs(M[i, j]) > 0.6 else "black")
        fig.colorbar(im, ax=ax2, fraction=0.046)
        off = M[np.triu_indices(n, 1)]
        ax2.set_title(f"Between-run r (median {np.median(off):.2f})", fontsize=9)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "n = 1\nno between-run comparison",
                 ha="center", va="center", fontsize=10, color="0.4")

    fig.suptitle(f"{condition}   (n={n})   diurnal reproducibility", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    svg_dir = out_path.parent / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(svg_dir / (out_path.stem + ".svg")), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--experiment", default="in planta")
    ap.add_argument("--output", default=str(REPO_ROOT / "aggregated_results" / "diurnal"))
    ap.add_argument("--bin-minutes", type=int, default=5)
    ap.add_argument("--agree-threshold", type=float, default=0.5,
                    help="Pairwise r required for runs to count as agreeing (default 0.5)")
    ap.add_argument("--min-days", type=int, default=2,
                    help="Minimum clean days required to build a diurnal profile")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runs = []
    for name, prof in (cfg.get("profiles") or {}).items():
        if not isinstance(prof, dict) or name == "processing_default":
            continue
        meta = prof.get("metadata") or {}
        if args.experiment and str(meta.get("experiment", "")).lower() != args.experiment.lower():
            continue
        r = load_run(name, prof, args.bin_minutes, args.min_days)
        if r is None:
            print(f"  ! {name}: skipped (no results, or fewer than {args.min_days} clean days)")
        else:
            runs.append(r)

    print(f"\n{len(runs)} run(s) with a usable diurnal profile\n")

    conditions = {}
    for r in runs:
        conditions.setdefault(r["condition"], []).append(r)
    conditions = dict(sorted(conditions.items(), key=lambda kv: (-len(kv[1]), kv[0])))

    run_rows, cond_rows = [], []
    for condition, members in conditions.items():
        members.sort(key=lambda m: (m["replicate"] is None, m["replicate"]))
        slug = re.sub(r"[^A-Za-z0-9]+", "_", condition).strip("_")
        plot_condition(condition, members, out_dir / f"diurnal_{slug}.png", args.bin_minutes)

        for m in members:
            run_rows.append({k: m[k] for k in
                             ("profile", "condition", "plant", "treatment", "cycle", "temp_hum",
                              "replicate", "n_days", "amplitude", "peak_hour", "within_r")}
                            | {"observed_transitions": ",".join(f"{h:02d}h" for h in m["observed"])})

        if len(members) >= 2:
            M = np.array([[np.corrcoef(a["mean_profile"], b["mean_profile"])[0, 1]
                           for b in members] for a in members])
            off = M[np.triu_indices(len(members), 1)]
            # do these runs actually share a lighting schedule?
            morns = {m["observed"][0] for m in members if m["observed"]}
            cond_rows.append(dict(
                condition=condition, n=len(members),
                between_r_median=round(float(np.median(off)), 2),
                between_r_min=round(float(off.min()), 2),
                between_r_max=round(float(off.max()), 2),
                within_r_median=round(float(np.nanmedian([m["within_r"] for m in members])), 2),
                schedules_agree=(len(morns) <= 1),
                morning_transitions=",".join(f"{h:02d}h" for h in sorted(morns)) or "-"))
        else:
            cond_rows.append(dict(condition=condition, n=1, between_r_median=np.nan,
                                  between_r_min=np.nan, between_r_max=np.nan,
                                  within_r_median=round(members[0]["within_r"], 2),
                                  schedules_agree=None,
                                  morning_transitions=",".join(f"{h:02d}h" for h in members[0]["observed"]) or "-"))

    agree_rows = []
    for condition, members in conditions.items():
        idx, rmin, rmax = largest_agreeing_set(members, args.agree_threshold)
        for k, m in enumerate(members):
            agree_rows.append(dict(
                profile=m["profile"], condition=condition,
                in_agreeing_set=(k in idx),
                cluster_size=len(idx), cluster_r_min=rmin, cluster_r_max=rmax,
                note=("only run in condition" if len(members) == 1
                      else ("" if k in idx else "outlier"))))
    agree_df = pd.DataFrame(agree_rows)
    agree_df.to_csv(out_dir / "agreeing_runs.csv", index=False)

    runs_df = pd.DataFrame(run_rows)
    cond_df = pd.DataFrame(cond_rows)
    runs_df.to_csv(out_dir / "diurnal_runs.csv", index=False)
    cond_df.to_csv(out_dir / "diurnal_reproducibility.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", None):
        print("PER RUN")
        print(runs_df.round(2).to_string(index=False))
        print("\nPER CONDITION")
        print(cond_df.to_string(index=False))

    bad = cond_df[(cond_df.n >= 2) & (cond_df.schedules_agree == False)]  # noqa: E712
    if len(bad):
        print("\nConditions whose runs do NOT share a morning transition hour "
              "(reproducibility here is not a fair test - they are not the same condition):")
        for _, r in bad.iterrows():
            print(f"  {r.condition:32s} transitions at {r.morning_transitions}")

    print(f"\nWrote -> {out_dir}")


if __name__ == "__main__":
    main()
