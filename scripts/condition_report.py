"""Aggregate the PL/G-band ratio by experimental condition, one figure per condition.

Draft reporting script. Three things have to be handled before runs can be averaged:

  1. Absolute level differs ~6x between runs, so an un-normalised mean would track whichever
     run sits highest rather than the biology -> each run is normalised to its own baseline.
  2. Runs start on different calendar dates (spanning >1 year) -> aligned on elapsed time.
  3. Runs differ in sampling interval (1-5 min) and in duration -> resampled onto a common
     grid at the coarsest native interval, and averaged using every run available at each
     time point rather than truncating everything to the shortest run.

Spread is shown as SD (variability among plants), with every replicate also drawn
individually -- at n of 2-6 the individual traces are more informative than any band.

Usage
-----
    python scripts/condition_report.py
    python scripts/condition_report.py --bin-minutes 5 --baseline-hours 24
    python scripts/condition_report.py --output "G:/.../_aggregate/conditions"
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

from swnt_iaa_analysis.core.loader import resolve_path  # noqa: E402

RESULTS_DIR_RE = re.compile(r"results_v4_(\d{8})_(\d{6})")
RATIO = "Fluorescence_to_Gband_Ratio_BaselineCorrected"
RATIO_FALLBACK = "Fluorescence_to_Gband_Ratio"
TREATMENT_COLORS = {"Control": "tab:blue", "Drought": "tab:red", "Shade": "tab:green"}
REPORT_DAYS = [1, 2, 3, 5, 7, 10, 14]


def find_latest_results(data_dir: Path):
    """Most recent results folder containing a processed_data.csv, searched recursively."""
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


def load_runs(config_path: Path, experiment: str, bin_minutes: float, baseline_hours: float,
              include: set | None = None, align: str = "clock", align_hour: int = 0):
    """Load each profile, normalise to its own baseline, and bin onto a common grid.

    ``align`` controls where elapsed time is measured from:
      clock (default) each run is trimmed forward to the first ``align_hour`` (default 00:00),
            so a given elapsed time is the same CLOCK time in every run and diurnal phases line
            up when averaged. Costs up to 24 h of data at the head of each run.
      start each run starts at its own first sample. Keeps all data, but runs begin at
            different times of day -- here the spread reaches 12 h, i.e. two runs in antiphase
            -- so averaging mixes opposite points of the daily cycle and smears the rhythm.
    """
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs = []
    for name, prof in (cfg.get("profiles") or {}).items():
        if not isinstance(prof, dict) or name == "processing_default":
            continue
        meta = prof.get("metadata") or {}
        if experiment and str(meta.get("experiment", "")).lower() != experiment.lower():
            continue
        if include is not None and name not in include:
            continue
        raman = (prof.get("data_source") or {}).get("raman_relative_path")
        if not raman:
            continue
        resolved = resolve_path(raman)
        if not resolved:
            continue
        results_dir = find_latest_results(Path(resolved).parent)
        if results_dir is None:
            print(f"  ! {name}: no results found, skipped")
            continue

        df = pd.read_csv(results_dir / "processed_data.csv")
        col = RATIO if RATIO in df.columns else (RATIO_FALLBACK if RATIO_FALLBACK in df.columns else None)
        if col is None or "Datetime" not in df.columns:
            print(f"  ! {name}: no ratio column, skipped")
            continue

        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        df = df.dropna(subset=["Datetime", col]).sort_values("Datetime")
        if len(df) < 50:
            print(f"  ! {name}: too few points, skipped")
            continue

        raw_start = df["Datetime"].min()
        trimmed_h = 0.0
        if align == "clock":
            target = raw_start.normalize() + pd.Timedelta(hours=int(align_hour))
            if target < raw_start:
                target += pd.Timedelta(days=1)
            df = df[df["Datetime"] >= target]
            if len(df) < 50:
                print(f"  ! {name}: too little data after aligning to {align_hour:02d}:00, skipped")
                continue
            trimmed_h = (target - raw_start).total_seconds() / 3600.0

        start = df["Datetime"].min()
        elapsed_min = (df["Datetime"] - start).dt.total_seconds() / 60.0
        values = df[col].to_numpy(dtype=float)

        baseline = np.nanmedian(values[elapsed_min <= baseline_hours * 60.0])
        if not np.isfinite(baseline) or baseline == 0:
            print(f"  ! {name}: baseline not usable, skipped")
            continue

        # Bin onto the common grid, averaging within each bin.
        grid = (elapsed_min // bin_minutes) * bin_minutes
        series = pd.Series(100.0 * values / baseline, index=grid).groupby(level=0).mean()
        series.index.name = "elapsed_min"

        # Event markers (relative to run start) for annotation only -- they record when stress
        # was *observed*, not when treatment began, and only some runs have them, so they are
        # not used to define the baseline.
        events = []
        for ev in (prof.get("treatment_events") or []):
            try:
                when = pd.Timestamp(ev.get("datetime"))
            except Exception:
                continue
            days = (when - start).total_seconds() / 86400.0
            if 0 <= days <= (elapsed_min.max() / 1440.0):
                events.append((days, ev.get("event_type", "event")))

        runs.append(dict(
            profile=name, series=series, events=events, start=start,
            plant=meta.get("plant_type"), treatment=meta.get("treatment"),
            cycle=meta.get("light_cycle"), replicate=meta.get("replicate_number"),
            condition=f"{meta.get('plant_type')} | {meta.get('treatment')} | {meta.get('light_cycle')}",
            native_min=float(np.median(np.diff(elapsed_min))) if len(elapsed_min) > 1 else np.nan,
            days=float(elapsed_min.max() / 1440.0), trimmed_h=trimmed_h,
        ))
    return runs



def daynight_day_spans(start_dt, max_days, light_cycle):
    """Daylight spans in elapsed-day coordinates, for shading a non-datetime axis.

    The condition plots use elapsed time rather than calendar date, so the shading helper in
    swnt_iaa_analysis.core.utils cannot be reused directly; the same 6to22 / 8to24 / Constant
    convention is reproduced here against each run's own start time.
    """
    cycle = str(light_cycle)
    if cycle == "Constant":
        return [(0.0, float(max_days))]
    on_h = 8 if cycle == "8to24" else 6
    off_h = 24 if cycle == "8to24" else 22

    spans, midnight = [], pd.Timestamp(start_dt).normalize()
    for day in range(int(max_days) + 2):
        base = midnight + pd.Timedelta(days=day)
        on = (base + pd.Timedelta(hours=on_h) - start_dt).total_seconds() / 86400.0
        off = (base + pd.Timedelta(hours=off_h) - start_dt).total_seconds() / 86400.0
        lo, hi = max(0.0, on), min(float(max_days), off)
        if hi > lo:
            spans.append((lo, hi))
    return spans


def shade_daynight(ax, start_dt, max_days, light_cycle):
    """Blue night across the axis, yellow daylight on top (same colours as the pipeline plots)."""
    ax.axvspan(0, max_days, facecolor="blue", alpha=0.06, zorder=0, lw=0)
    for lo, hi in daynight_day_spans(start_dt, max_days, light_cycle):
        ax.axvspan(lo, hi, facecolor="yellow", alpha=0.13, zorder=0, lw=0)


def harmonise_lengths(members: list, mode: str, bin_minutes: float):
    """Make runs in a condition comparable in length. Returns (members, note).

    ragged   (default) every run contributes for as long as it lasts; n falls over time and
             the mean past the balanced window reflects a changing set of runs.
    truncate every run is cut to the shortest in the condition. Fully balanced and the trend
             is preserved, but data past that point is discarded.
    fold     runs longer than the shortest are wrapped onto it and their segments averaged,
             so every run still yields ONE series covering the common window. Balanced, and
             no data is thrown away.

    Folding never turns one run into several replicates: segments from the same plant share
    the plant, sensor and batch, so counting them separately would be pseudoreplication and
    would shrink the SD for a reason that is not biological. n stays equal to the run count.

    IMPORTANT: folding averages together times that are days apart, so any trend slower than
    the fold period is averaged away. Use it when the repeating within-window pattern is what
    matters; do NOT use it when the multi-day trend IS the signal (for example a drought
    response building over a week) -- use truncate or ragged there.
    """
    if mode == "ragged" or len(members) < 2:
        return members, "every run contributes for its full duration"

    span = min(float(m["series"].index.max()) for m in members)   # minutes
    out = []
    for m in members:
        s = m["series"]
        if mode == "truncate":
            s2 = s[s.index <= span]
        elif mode == "fold":
            pos = (s.index % span) if span > 0 else s.index
            s2 = s.groupby(pos).mean().sort_index()
            s2 = s2[s2.index <= span]
        else:
            raise ValueError(f"unknown length mode: {mode}")
        out.append({**m, "series": s2, "folded_from_days": m["days"]})

    if mode == "truncate":
        note = f"all runs truncated to the shortest ({span/1440.0:.1f} d)"
    else:
        note = (f"runs folded onto the shortest ({span/1440.0:.1f} d); "
                f"segments averaged within each run, n unchanged")
    return out, note


def plot_condition(condition: str, members: list, out_path: Path, bin_minutes: float, min_n: int,
                   length_note: str = ""):
    """One figure per condition, vertically stacked: a row per replicate, then mean +/- SD, then n.

    Replicates are stacked rather than overlaid -- with 5-6 runs an overlay is unreadable, and
    stacking makes each run legible while a shared x and y axis keeps them directly comparable.
    """
    wide = pd.concat({m["profile"]: m["series"] for m in members}, axis=1).sort_index()
    n_t = wide.notna().sum(axis=1)
    mean, sd = wide.mean(axis=1), wide.std(axis=1, ddof=1)
    days = wide.index / 1440.0

    color = TREATMENT_COLORS.get(str(members[0]["treatment"]), "tab:gray")
    n_runs = len(members)

    # rows: one per replicate, one for the mean, one for n(t)
    heights = [1.0] * n_runs + [1.6, 0.55]
    fig, axes = plt.subplots(n_runs + 2, 1, sharex=True, sharey=False,
                             figsize=(11, 1.35 * n_runs + 3.0),
                             gridspec_kw={"height_ratios": heights})
    run_axes = axes[:n_runs]
    ax_mean, ax_n = axes[n_runs], axes[n_runs + 1]

    # Shared y across replicate rows and the mean, so panels are comparable at a glance.
    lo = float(np.nanpercentile(wide.to_numpy(), 0.5))
    hi = float(np.nanpercentile(wide.to_numpy(), 99.5))
    pad = 0.08 * (hi - lo or 1)
    ylim = (lo - pad, hi + pad)

    balanced_to = None
    full = n_t[n_t == n_runs]
    if len(full):
        balanced_to = full.index.max() / 1440.0

    xmax = float(days.max())

    def decorate(ax, start_dt=None, cycle=None):
        if start_dt is not None:
            shade_daynight(ax, start_dt, xmax, cycle)
        if balanced_to is not None:
            ax.axvspan(0, balanced_to, color="0.35", alpha=0.07, zorder=0, lw=0)
        ax.axhline(100, color="k", lw=0.7, ls=":")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax, m in zip(run_axes, members):
        d = m["series"].index / 1440.0
        ax.plot(d, m["series"].values, lw=0.8, color=color)
        decorate(ax, m["start"], m["cycle"])
        ax.set_ylim(*ylim)
        ax.set_ylabel(f"Run{m['replicate']}", fontsize=10, fontweight="bold")
        ax.text(0.995, 0.9, f"{m['profile']}  ({m['days']:.1f} d)", transform=ax.transAxes,
                fontsize=6.5, color="0.35", ha="right", va="top")
        for d_ev, kind in m["events"]:
            ax.axvline(d_ev, color="darkorange", ls="-.", lw=1.1)
            ax.text(d_ev, ylim[1], f" {kind}", fontsize=6, color="darkorange",
                    va="top", rotation=90)

    ok = (n_t >= min_n).to_numpy()
    if ok.any():
        ax_mean.plot(days[ok], mean.to_numpy()[ok], lw=2.0, color=color, label=f"mean (n>={min_n})")
        if n_runs >= 3:
            ax_mean.fill_between(days[ok], (mean - sd).to_numpy()[ok], (mean + sd).to_numpy()[ok],
                                 color=color, alpha=0.22, lw=0, label="+/- SD")
        ax_mean.legend(fontsize=7, loc="best")
    tod = np.array([pd.Timestamp(m["start"]).hour + pd.Timestamp(m["start"]).minute / 60.0
                    for m in members])
    phase_spread = float(np.ptp(tod)) if len(tod) > 1 else 0.0
    if phase_spread <= 2.0:
        decorate(ax_mean, members[0]["start"], members[0]["cycle"])
    else:
        decorate(ax_mean)
        ax_mean.text(0.005, 0.90, f"no shading: runs start {phase_spread:.1f} h apart in "
                                  f"time-of-day, so day/night phase is not shared",
                     transform=ax_mean.transAxes, fontsize=7, color="0.35")
    ax_mean.set_ylim(*ylim)
    ax_mean.set_ylabel("mean", fontsize=10, fontweight="bold")
    if balanced_to is not None:
        ax_mean.text(0.005, 0.06, f"grey = balanced window (all n={n_runs})",
                     transform=ax_mean.transAxes, fontsize=6.5, color="0.35")

    ax_n.step(days, n_t.to_numpy(), where="mid", color="0.35", lw=1.2)
    ax_n.set_ylabel("n", fontsize=10, fontweight="bold")
    ax_n.set_xlabel("days since start of run")
    ax_n.set_ylim(0, n_runs + 0.5)
    ax_n.set_yticks(range(0, n_runs + 1))
    ax_n.grid(True, alpha=0.3, linestyle="--")
    ax_n.tick_params(axis="both", labelsize=9)
    ax_n.spines["top"].set_visible(False)
    ax_n.spines["right"].set_visible(False)

    subtitle = f"n={n_runs}, {bin_minutes:.0f}-min bins, normalised to own first 24 h"
    if length_note:
        subtitle += "\n" + length_note
    fig.suptitle(f"{condition}   ({subtitle})", fontsize=10)
    fig.supylabel("PL / G-band, % of own baseline", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0.01, 0, 1, 0.985])
    svg_dir = out_path.parent / "svg"
    svg_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(svg_dir / (out_path.stem + ".svg")), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    plant, treatment, cycle = (members[0]["plant"], members[0]["treatment"], members[0]["cycle"])
    n_full = len(members)

    # snapshot rows at the reporting days
    rows = []
    for d in REPORT_DAYS:
        key = d * 1440.0
        if key in wide.index and n_t[key] >= min_n:
            rows.append(dict(condition=condition, plant=plant, treatment=treatment, cycle=cycle,
                             day=d, n=int(n_t[key]),
                             mean_pct=round(float(mean[key]), 1),
                             sd=round(float(sd[key]), 1),
                             sem=round(float(sd[key] / np.sqrt(n_t[key])), 1),
                             balanced=bool(n_t[key] == n_full)))

    # full averaged timeseries: the condition mean at every time bin (n >= min_n)
    sem_all = sd / np.sqrt(n_t.where(n_t > 0))
    ts = pd.DataFrame({
        "condition": condition, "plant": plant, "treatment": treatment, "cycle": cycle,
        "elapsed_days": wide.index / 1440.0, "n": n_t.astype(int).to_numpy(),
        "mean_pct": mean.to_numpy(), "sd": sd.to_numpy(), "sem": sem_all.to_numpy(),
        "balanced": (n_t == n_full).to_numpy(),
    })
    ts = ts[ts["n"] >= min_n].round({"mean_pct": 2, "sd": 2, "sem": 2, "elapsed_days": 4})
    return rows, ts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--experiment", default="in planta")
    ap.add_argument("--output", default=str(REPO_ROOT / "aggregated_results" / "conditions"))
    ap.add_argument("--bin-minutes", type=float, default=5.0,
                    help="Common time grid; use the coarsest native sampling interval (default 5)")
    ap.add_argument("--baseline-hours", type=float, default=24.0)
    ap.add_argument("--include", default=None,
                    help="agreeing_runs.csv from diurnal_reproducibility.py; keeps only runs "
                         "whose diurnal cycle agrees with the rest of their condition")
    ap.add_argument("--align", choices=["clock", "start"], default="clock",
                    help="clock (default): trim each run forward to --align-hour so elapsed time "
                         "means the same clock time in every run and diurnal phases line up. "
                         "start: measure from each run's first sample (keeps all data, but runs "
                         "begin at different times of day and the daily cycle gets smeared)")
    ap.add_argument("--align-hour", type=int, default=0,
                    help="Clock hour to align runs to when --align clock (default 0 = midnight)")
    ap.add_argument("--length-mode", choices=["ragged", "truncate", "fold"], default="ragged",
                    help="How to handle runs of different length within a condition: "
                         "'ragged' (default) each run contributes for its full duration; "
                         "'truncate' cut all to the shortest; "
                         "'fold' wrap longer runs onto the shortest and average their segments "
                         "(balanced and keeps all data, but averages away trends slower than "
                         "the fold period - do not use when the multi-day trend is the signal)")
    ap.add_argument("--min-n", type=int, default=2,
                    help="Mean is drawn only where at least this many runs contribute")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    include = None
    if args.include:
        inc = pd.read_csv(args.include)
        if "in_agreeing_set" in inc.columns:
            inc = inc[inc["in_agreeing_set"].astype(bool)]
        include = set(inc["profile"].astype(str))
        print(f"Filtering to {len(include)} agreeing run(s) from {args.include}")

    runs = load_runs(Path(args.config), args.experiment, args.bin_minutes, args.baseline_hours,
                     include=include, align=args.align, align_hour=args.align_hour)
    print(f"\nLoaded {len(runs)} run(s)")

    native = sorted({round(r["native_min"], 1) for r in runs if np.isfinite(r["native_min"])})
    print(f"native sampling intervals present: {native} min  ->  binned to {args.bin_minutes:.0f} min")
    if native and args.bin_minutes < max(native):
        print(f"  ! WARNING: bin ({args.bin_minutes:.0f} min) is finer than the coarsest run "
              f"({max(native):.0f} min); those runs will have empty bins")

    # Aggregation of ALL in-planta profiles, long-format: every run (including single-run
    # conditions that the condition means drop) with the same baseline-normalisation and
    # alignment applied. This is the per-profile table to re-aggregate however you like.
    if runs:
        per_profile = pd.concat([
            pd.DataFrame({
                "profile": r["profile"], "condition": r["condition"],
                "plant": r["plant"], "treatment": r["treatment"], "cycle": r["cycle"],
                "replicate": r["replicate"],
                "elapsed_days": (r["series"].index / 1440.0).round(4),
                "pct_of_baseline": r["series"].to_numpy().round(2),
            })
            for r in runs
        ], ignore_index=True)
        per_profile.to_csv(out_dir / "all_profiles_timeseries.csv", index=False)
        print(f"all_profiles_timeseries.csv: {len(per_profile)} rows "
              f"({per_profile['profile'].nunique()} profiles)")

    conditions = {}
    for r in runs:
        conditions.setdefault(r["condition"], []).append(r)
    conditions = dict(sorted(conditions.items(), key=lambda kv: (-len(kv[1]), kv[0])))

    all_rows, all_ts = [], []
    print()
    for condition, members in conditions.items():
        slug = re.sub(r"[^A-Za-z0-9]+", "_", condition).strip("_")
        path = out_dir / f"condition_{slug}.png"
        members, length_note = harmonise_lengths(members, args.length_mode, args.bin_minutes)
        rows, ts = plot_condition(condition, members, path, args.bin_minutes, args.min_n,
                                  length_note)
        all_rows += rows
        all_ts.append(ts)
        durs = ", ".join(f"{m['days']:.1f}" for m in sorted(members, key=lambda m: -m["days"]))
        print(f"  {condition:32s} n={len(members)}  durations(d): {durs}")

    if all_rows:
        summary = pd.DataFrame(all_rows)
        summary.to_csv(out_dir / "condition_summary.csv", index=False)
        print("\nPL/G-band as % of own baseline "
              "(balanced=True means every run in the condition still contributes):")
        print(summary.to_string(index=False))

    if all_ts:
        combined = pd.concat(all_ts, ignore_index=True)
        combined.to_csv(out_dir / "condition_mean_timeseries.csv", index=False)
        print(f"\ncondition_mean_timeseries.csv: {len(combined)} rows "
              f"({combined['condition'].nunique()} conditions)")

    print(f"\nWrote -> {out_dir}")


if __name__ == "__main__":
    main()
