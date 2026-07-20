"""Aggregate the latest analysis results across profiles into one place.

For each profile it finds the most recent results folder (analyze *or* reprocess,
selected by the timestamp in the folder name, requiring a real processed_data.csv),
then writes:

  summary.csv             one QC row per profile, with experiment metadata
  combined_timeseries.csv all profiles' timeseries stacked long-format, with metadata
  overview.png            small-multiple plot of the corrected ratio per profile

Usage
-----
    python scripts/aggregate_results.py                          # all in-planta profiles
    python scripts/aggregate_results.py --experiment "in vitro"
    python scripts/aggregate_results.py --profiles A,B,C
    python scripts/aggregate_results.py --output "D:/somewhere"
    python scripts/aggregate_results.py --downsample 10          # every 10th scan in combined
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
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

# Running `python scripts/aggregate_results.py` puts scripts/ on sys.path, not the repo
# root, so the package would only import if it happened to be pip-installed in whichever
# interpreter is active. Add the repo root so the script works from any environment.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swnt_iaa_analysis.core.loader import resolve_path  # noqa: E402  (needs sys.path above)
RESULTS_DIR_RE = re.compile(r"results_v4_(\d{8})_(\d{6})")

# Metadata fields lifted from config for categorisation.
META_FIELDS = ["plant_type", "treatment", "light_cycle", "temp_hum_control", "replicate_number"]

# Timeseries columns carried into the combined export, when present.
TIMESERIES_COLUMNS = [
    "Raw_Fluorescence_Intensity",
    "Raw_Gband_Area",
    "Raw_Raman_Peak_850_Area",
    "Raw_Fluorescence_Intensity_BaselineCorrected",
    "Raw_Gband_Area_BaselineCorrected",
    "Raw_Raman_Peak_850_Area_BaselineCorrected",
    "Normalized_Fluorescence_Intensity",
    "Normalized_Gband_Area",
    "Normalized_Raman_Peak_850_Area",
    "Fluorescence_to_Gband_Ratio",
    "Fluorescence_to_Gband_Ratio_BaselineCorrected",
    "Fluorescence_to_Raman_Peak_850_Ratio",
    "Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected",
]

RATIO_BC = "Fluorescence_to_Gband_Ratio_BaselineCorrected"

# One overview grid per quantity. Each entry lists candidate columns in priority order, so a
# profile corrected in raw space uses Raw_*, while one corrected in normalized space (or an
# older result) falls back to Normalized_*.
OVERVIEW_PLOTS = [
    ("overview.png",
     ["Fluorescence_to_Gband_Ratio_BaselineCorrected"],
     "Baseline-corrected PL / G-band ratio"),
    ("overview_ratio_PL_850.png",
     ["Fluorescence_to_Raman_Peak_850_Ratio_BaselineCorrected"],
     "Baseline-corrected PL / 850 cm-1 Raman ratio"),
    ("overview_corrected_PL.png",
     ["Raw_Fluorescence_Intensity_BaselineCorrected",
      "Normalized_Fluorescence_Intensity_BaselineCorrected"],
     "Baseline-corrected PL (fluorescence)"),
    ("overview_corrected_gband.png",
     ["Raw_Gband_Area_BaselineCorrected", "Normalized_Gband_Area_BaselineCorrected"],
     "Baseline-corrected G-band area"),
    ("overview_corrected_850.png",
     ["Raw_Raman_Peak_850_Area_BaselineCorrected",
      "Normalized_Raman_Peak_850_Area_BaselineCorrected"],
     "Baseline-corrected 850 cm-1 Raman peak area"),
]


def parse_results_timestamp(name: str):
    """Datetime encoded in a results folder name, or None."""
    m = RESULTS_DIR_RE.search(name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def find_latest_results(data_dir: Path):
    """Most recent results folder containing a processed_data.csv, searched recursively.

    Recurses because `reprocess` nests its output inside the analyze folder it read from.
    Ordering uses the timestamp in the folder name rather than mtime, which cloud sync
    can rewrite. Folders without a CSV (failed or aborted runs) are skipped.
    """
    candidates = []
    for path in data_dir.rglob("results_v4_*"):
        if not path.is_dir() or not (path / "processed_data.csv").exists():
            continue
        ts = parse_results_timestamp(path.name)
        if ts is not None:
            candidates.append((ts, path))
    if not candidates:
        return None, None
    ts, path = max(candidates, key=lambda x: x[0])
    return path, ts


def detect_method(columns) -> str:
    """Infer which baseline method produced the file, from the columns present."""
    cols = set(columns)
    if "Raw_Fluorescence_Intensity_BaselineCorrected" in cols:
        return "transition_stitch (raw)"
    if "Normalized_Fluorescence_Intensity_BaselineCorrected" in cols:
        return "normalized-space"
    return "no baseline correction"


def _stats(series: pd.Series) -> dict:
    """Robust min/median/max/negative/NaN summary for a numeric column."""
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return dict(min=np.nan, median=np.nan, max=np.nan, n_negative=0, pct_nan=100.0)
    return dict(
        min=float(s.min()),
        median=float(s.median()),
        max=float(s.max()),
        n_negative=int((s < 0).sum()),
        pct_nan=round(100.0 * s.isna().mean(), 2),
    )


def load_profiles(config_path: Path, experiment: str | None, only: list[str] | None) -> list[dict]:
    """Config profiles matching the experiment / explicit name filter."""
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selected = []
    for name, prof in (cfg.get("profiles") or {}).items():
        if not isinstance(prof, dict) or name == "processing_default":
            continue
        meta = prof.get("metadata") or {}
        raman = (prof.get("data_source") or {}).get("raman_relative_path")
        if not raman:
            continue
        if only is not None:
            if name not in only:
                continue
        elif experiment and str(meta.get("experiment", "")).lower() != experiment.lower():
            continue
        selected.append({"profile": name, "raman_relative_path": raman,
                         **{k: meta.get(k) for k in META_FIELDS}})
    return selected


def build_summary(profiles: list[dict]) -> tuple[pd.DataFrame, dict]:
    """QC row per profile, plus the loaded frames keyed by profile name."""
    rows, frames = [], {}
    for entry in profiles:
        row = {k: entry.get(k) for k in ["profile"] + META_FIELDS}
        resolved = resolve_path(entry["raman_relative_path"])
        data_dir = Path(resolved).parent if resolved else None

        if data_dir is None or not data_dir.exists():
            rows.append({**row, "status": "DATA PATH MISSING"})
            continue

        results_dir, ts = find_latest_results(data_dir)
        if results_dir is None:
            rows.append({**row, "status": "NO RESULTS"})
            continue

        row["results_folder"] = results_dir.name
        row["run_timestamp"] = ts.strftime("%Y-%m-%d %H:%M")
        row["mode"] = "reprocess" if results_dir.name.endswith("_reprocess") else "analyze"

        try:
            df = pd.read_csv(results_dir / "processed_data.csv")
        except Exception as exc:  # unreadable CSV should not abort the whole sweep
            rows.append({**row, "status": f"READ ERROR: {type(exc).__name__}"})
            continue

        row["status"] = "OK"
        row["baseline_method"] = detect_method(df.columns)
        row["n_scans"] = len(df)

        if "Datetime" in df.columns:
            dt = pd.to_datetime(df["Datetime"], errors="coerce")
            if dt.notna().any():
                row["start"] = dt.min().strftime("%Y-%m-%d %H:%M")
                row["end"] = dt.max().strftime("%Y-%m-%d %H:%M")
                row["duration_days"] = round((dt.max() - dt.min()).total_seconds() / 86400, 2)
                df = df.assign(Datetime=dt)

        if RATIO_BC in df.columns:
            st = _stats(df[RATIO_BC])
            row.update({f"ratio_{k}": (round(v, 2) if isinstance(v, float) else v)
                        for k, v in st.items()})

        for col, tag in [("Raw_Fluorescence_Intensity_BaselineCorrected", "fluorBC"),
                         ("Raw_Gband_Area_BaselineCorrected", "gbandBC")]:
            if col in df.columns:
                st = _stats(df[col])
                row[f"{tag}_min"] = round(st["min"], 1)
                row[f"{tag}_negatives"] = st["n_negative"]

        # Flag anything that warrants a closer look.
        flags = []
        if row.get("baseline_method") != "transition_stitch (raw)":
            flags.append("not-new-pipeline")
        if row.get("ratio_n_negative", 0):
            flags.append("negative-ratio")
        if row.get("gbandBC_negatives", 0) or row.get("fluorBC_negatives", 0):
            flags.append("negative-channel")
        if row.get("ratio_pct_nan", 0) and row["ratio_pct_nan"] > 5:
            flags.append("many-NaN")
        row["flags"] = ",".join(flags)

        rows.append(row)
        frames[entry["profile"]] = (df, entry)

    return pd.DataFrame(rows), frames


def build_combined(frames: dict, downsample: int) -> pd.DataFrame:
    """Stack every profile's timeseries long-format with its metadata attached."""
    parts = []
    for profile, (df, entry) in frames.items():
        keep = [c for c in (["Datetime", "Scan Number"] + TIMESERIES_COLUMNS) if c in df.columns]
        if not keep:
            continue
        sub = df[keep].copy()
        if downsample > 1:
            sub = sub.iloc[::downsample]
        sub.insert(0, "profile", profile)
        for i, field in enumerate(META_FIELDS, start=1):
            sub.insert(i, field, entry.get(field))
        parts.append(sub)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _first_present(df, candidates):
    """First candidate column present in df, or None."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def plot_overview(frames: dict, out_path: Path, candidates, title) -> bool:
    """Small multiples of one quantity across profiles, ordered by condition.

    Each panel keeps its own y-axis: the profiles differ in level by more than an order of
    magnitude, so a shared axis would flatten most of them into a line. Compare magnitudes
    via summary.csv rather than by eye across panels.
    """
    items = []
    for profile, (df, entry) in frames.items():
        col = _first_present(df, candidates)
        if col is not None and "Datetime" in df.columns:
            items.append((profile, df, entry, col))
    if not items:
        return False
    items.sort(key=lambda x: (str(x[2].get("plant_type")), str(x[2].get("treatment")),
                              str(x[2].get("light_cycle")), str(x[2].get("replicate_number"))))

    ncols = 3
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 2.6 * nrows), squeeze=False)
    colors = {"Control": "tab:blue", "Drought": "tab:red", "Shade": "tab:green"}

    for ax, (profile, df, entry, col) in zip(axes.flat, items):
        color = colors.get(str(entry.get("treatment")), "tab:gray")
        ax.plot(df["Datetime"], pd.to_numeric(df[col], errors="coerce"), lw=0.7, color=color)
        # Flag panels that fell back to the normalized column, whose scale differs from Raw_*.
        space = " [normalized]" if col.startswith("Normalized_") else ""
        ax.set_title(f"{profile}{space}\n{entry.get('plant_type')} | {entry.get('treatment')} | "
                     f"{entry.get('light_cycle')}", fontsize=7)
        ax.tick_params(labelsize=6)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30); lbl.set_ha("right")
    for ax in axes.flat[len(items):]:
        ax.axis("off")

    fig.suptitle(f"{title} - latest results per profile", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO_ROOT / "config.yaml"))
    ap.add_argument("--experiment", default="in planta",
                    help="Filter by metadata experiment (default: 'in planta')")
    ap.add_argument("--profiles", default=None,
                    help="Comma-separated profile names; overrides --experiment")
    ap.add_argument("--output", default=str(REPO_ROOT / "aggregated_results"),
                    help="Output directory")
    ap.add_argument("--downsample", type=int, default=1,
                    help="Keep every Nth scan in combined_timeseries.csv (default: 1 = all)")
    args = ap.parse_args()

    only = [p.strip() for p in args.profiles.split(",")] if args.profiles else None
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles = load_profiles(Path(args.config), args.experiment, only)
    print(f"Matched {len(profiles)} profile(s)\n")

    summary, frames = build_summary(profiles)
    summary.to_csv(out_dir / "summary.csv", index=False)

    display_cols = [c for c in ["profile", "plant_type", "treatment", "light_cycle",
                                "replicate_number", "status", "run_timestamp", "mode",
                                "n_scans", "duration_days", "ratio_median", "flags"]
                    if c in summary.columns]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(summary[display_cols].to_string(index=False))

    combined = build_combined(frames, args.downsample)
    if not combined.empty:
        combined.to_csv(out_dir / "combined_timeseries.csv", index=False)

    made_plots = []
    for filename, candidates, title in OVERVIEW_PLOTS:
        if plot_overview(frames, out_dir / filename, candidates, title):
            made_plots.append(filename)
        else:
            print(f"  (skipped {filename}: no profile has {candidates[0]})")

    ok = int((summary["status"] == "OK").sum()) if "status" in summary else 0
    print(f"\n{ok}/{len(summary)} profiles loaded OK")
    flagged = summary[summary.get("flags", "").astype(bool)] if "flags" in summary else pd.DataFrame()
    if not flagged.empty:
        print(f"{len(flagged)} profile(s) flagged:")
        for _, r in flagged.iterrows():
            print(f"  - {r['profile']}: {r['flags']}")
    print(f"\nWrote -> {out_dir}")
    print(f"  summary.csv             ({len(summary)} rows)")
    if not combined.empty:
        print(f"  combined_timeseries.csv ({len(combined):,} rows)")
    for filename in made_plots:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
