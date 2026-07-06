"""
Compare four monotonic post-processing corrections on stem_height_analysis_v2 output.

Methods (applied to stem_h_px_filtered):
  1. cummax      — running maximum (never decreases)
  2. isotonic    — pool-adjacent-violators least-squares non-decreasing fit
  3. threshold   — keep measured value unless drop exceeds --threshold px
  4. gompertz    — parametric growth curve h(t) = h0 + A*(1 - exp(-k*t))

Writes:
  stem_height_monotonic_comparison.csv
  stem_height_monotonic_comparison.png   (all curves + GT)
  stem_height_monotonic_metrics.json     (summary stats vs GT)

Example:
  python scripts/compare_stem_height_monotonic.py
  python scripts/compare_stem_height_monotonic.py --threshold 15
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

RUN4 = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1"
)
DEFAULT_STEM_CSV = RUN4 / "DEV_1AB22C05B465" / "stem_height_analysis_v2" / "stem_height_timeseries.csv"
DEFAULT_ANNOT_DIR = Path(r"C:\Users\ryank\Desktop\stem height examples")
DEFAULT_OUT_DIR = RUN4 / "DEV_1AB22C05B465" / "stem_height_analysis_v2" / "monotonic_comparison"
TS_FORMAT = "%Y-%m-%d_%H-%M-%S"

METHODS = ("cummax", "isotonic", "threshold", "gompertz")


def _ts_to_datetime(ts: str) -> pd.Timestamp | pd.NaT:
    try:
        return pd.Timestamp(dt.datetime.strptime(ts, TS_FORMAT))
    except ValueError:
        return pd.NaT


def monotonic_cummax(y: np.ndarray) -> np.ndarray:
    return np.maximum.accumulate(y)


def monotonic_isotonic(y: np.ndarray) -> np.ndarray:
    """Non-decreasing isotonic regression (PAV), O(n)."""
    y = np.asarray(y, float)
    n = len(y)
    if n == 0:
        return y
    blocks: list[list[float]] = [[i, float(y[i]), 1.0] for i in range(n)]
    i = 0
    while i < len(blocks) - 1:
        avg_i = blocks[i][1] / blocks[i][2]
        avg_j = blocks[i + 1][1] / blocks[i + 1][2]
        if avg_i <= avg_j:
            i += 1
            continue
        blocks[i][1] += blocks[i + 1][1]
        blocks[i][2] += blocks[i + 1][2]
        del blocks[i + 1]
        if i:
            i -= 1
    out = np.empty(n)
    for start, s, w in blocks:
        out[int(start): int(start + w)] = s / w
    return out


def monotonic_threshold(y: np.ndarray, threshold: float) -> np.ndarray:
    out = np.asarray(y, float).copy()
    for i in range(1, len(out)):
        if out[i] < out[i - 1] - threshold:
            out[i] = out[i - 1]
    return out


def _gompertz(t, h0, a, k):
    """Monotonic increasing for a>0, k>0."""
    return h0 + a * (1.0 - np.exp(-k * np.maximum(t, 0.0)))


def monotonic_gompertz(hours: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    """Fit Gompertz-style curve; return evaluated heights + fit params."""
    t = np.asarray(hours, float)
    y = np.asarray(y, float)
    h0_0 = max(0.0, float(y[0]) - 20.0)
    a_0 = max(50.0, float(y.max() - y.min()))
    k_0 = 0.05
    try:
        popt, _ = curve_fit(
            _gompertz,
            t,
            y,
            p0=(h0_0, a_0, k_0),
            bounds=([0.0, 1.0, 1e-6], [float(y.max()), float(y.max() * 3), 5.0]),
            maxfev=20000,
        )
        fitted = _gompertz(t, *popt)
        params = {"h0": float(popt[0]), "A": float(popt[1]), "k": float(popt[2])}
    except Exception as exc:
        print(f"Warning: Gompertz fit failed ({exc}); falling back to cummax.")
        fitted = monotonic_cummax(y)
        params = {"fallback": "cummax"}
    return fitted, params


def extract_red_line(img_bgr):
    b, g, r = (img_bgr[..., 0].astype(int), img_bgr[..., 1].astype(int), img_bgr[..., 2].astype(int))
    red = (r > 90) & (g < 70) & (b < 70)
    ys, xs = np.where(red)
    if len(ys) < 20:
        return None
    return int(ys.min()), int(ys.max())  # apex_y, base_y


def load_ground_truth(annot_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(annot_dir.glob("*.png")):
        if "_stem_height" in p.name:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        line = extract_red_line(img)
        if line is None:
            continue
        apex_y, base_y = line
        ts = p.name.split("tl_", 1)[1].split("_mask_refined", 1)[0]
        rows.append({"timestamp": ts, "gt_h_px": base_y - apex_y})
    if not rows:
        return pd.DataFrame(columns=["timestamp", "gt_h_px", "Datetime"])
    gt = pd.DataFrame(rows)
    gt["Datetime"] = gt["timestamp"].map(_ts_to_datetime)
    return gt.dropna(subset=["Datetime"])


def series_stats(y: np.ndarray) -> dict:
    d = np.diff(y)
    return {
        "start_px": float(y[0]),
        "end_px": float(y[-1]),
        "n_decreases": int(np.sum(d < -0.5)),
        "max_drop_px": float(d.min()) if len(d) else 0.0,
        "mean_abs_step_px": float(np.mean(np.abs(d))) if len(d) else 0.0,
    }


def gt_metrics(df: pd.DataFrame, col: str, gt: pd.DataFrame, max_delta_s: float = 60.0) -> dict | None:
    if gt.empty:
        return None
    preds, truths = [], []
    for _, row in gt.iterrows():
        when = row["Datetime"]
        idx = (df["Datetime"] - when).abs().idxmin()
        if (df.loc[idx, "Datetime"] - when).total_seconds() > max_delta_s:
            continue
        preds.append(df.loc[idx, col])
        truths.append(row["gt_h_px"])
    if len(preds) < 3:
        return None
    preds = np.array(preds, float)
    truths = np.array(truths, float)
    err = preds - truths
    return {
        "n_matched": int(len(truths)),
        "mae_px": float(np.mean(np.abs(err))),
        "median_ae_px": float(np.median(np.abs(err))),
        "rmse_px": float(np.sqrt(np.mean(err ** 2))),
        "corr": float(np.corrcoef(preds, truths)[0, 1]) if len(truths) > 2 else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare monotonic stem-height corrections.")
    ap.add_argument("--stem-csv", type=Path, default=DEFAULT_STEM_CSV)
    ap.add_argument("--annot-dir", type=Path, default=DEFAULT_ANNOT_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="Drop threshold (px) for the threshold method.")
    ap.add_argument("--input-column", default="stem_h_px_filtered")
    args = ap.parse_args(argv)

    stem_csv = args.stem_csv.resolve()
    if not stem_csv.exists():
        print(f"Error: {stem_csv} not found", file=sys.stderr)
        return 1

    df = pd.read_csv(stem_csv)
    df["Datetime"] = pd.to_datetime(df["datetime_iso"])
    y_in = df[args.input_column].values.astype(float)
    hours = df["hours_from_start"].values.astype(float)

    out = df.copy()
    out["stem_h_cummax"] = monotonic_cummax(y_in)
    out["stem_h_isotonic"] = monotonic_isotonic(y_in)
    out["stem_h_threshold"] = monotonic_threshold(y_in, args.threshold)
    gomp, gomp_params = monotonic_gompertz(hours, y_in)
    out["stem_h_gompertz"] = gomp

    gt = load_ground_truth(args.annot_dir.resolve())
    if not gt.empty:
        t0 = df["Datetime"].iloc[0]
        gt["hours"] = (gt["Datetime"] - t0).dt.total_seconds() / 3600.0

    metrics = {
        "input_column": args.input_column,
        "threshold_px": args.threshold,
        "gompertz_fit": gomp_params,
        "series": {},
        "vs_ground_truth": {},
    }
    cols = {
        "input": args.input_column,
        "cummax": "stem_h_cummax",
        "isotonic": "stem_h_isotonic",
        "threshold": "stem_h_threshold",
        "gompertz": "stem_h_gompertz",
    }
    for name, col in cols.items():
        z = out[col].values.astype(float)
        metrics["series"][name] = series_stats(z)
        gm = gt_metrics(out, col, gt)
        if gm:
            metrics["vs_ground_truth"][name] = gm

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "stem_height_monotonic_comparison.csv"
    out.to_csv(csv_path, index=False)

    json_path = out_dir / "stem_height_monotonic_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- plot ---
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1]})
    ax = axes[0]
    styles = [
        ("input (filtered)", args.input_column, dict(color="0.75", lw=1.0, alpha=0.9)),
        ("1. cummax", "stem_h_cummax", dict(color="#1f77b4", lw=2.0)),
        ("2. isotonic", "stem_h_isotonic", dict(color="#ff7f0e", lw=2.0)),
        (f"3. threshold (>{args.threshold:.0f}px)", "stem_h_threshold",
         dict(color="#2ca02c", lw=2.0)),
        ("4. Gompertz fit", "stem_h_gompertz", dict(color="#d62728", lw=2.0, ls="--")),
    ]
    for label, col, kw in styles:
        ax.plot(df["hours_from_start"], out[col], label=label, **kw)
    if not gt.empty:
        ax.plot(gt["hours"], gt["gt_h_px"], "r*", ms=12, label=f"manual GT ({len(gt)})", zorder=5)
    ax.set_ylabel("stem height (px)")
    ax.set_title("Monotonic post-processing comparison (stem height should only increase)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    ax2 = axes[1]
    for label, col, kw in styles[1:]:
        delta = out[col].values - y_in
        ax2.plot(df["hours_from_start"], delta, label=label.split(". ", 1)[-1], lw=1.5)
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.set_xlabel("hours from start")
    ax2.set_ylabel("correction (px)\n(method − input)")
    ax2.set_title("How much each method shifts the filtered curve")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    png_path = out_dir / "stem_height_monotonic_comparison.png"
    fig.savefig(str(png_path), dpi=150)
    plt.close(fig)

    print(f"Input: {stem_csv}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {json_path}")
    if gomp_params.get("h0") is not None:
        print(f"Gompertz: h0={gomp_params['h0']:.1f}  A={gomp_params['A']:.1f}  k={gomp_params['k']:.4f}")
    print("\nSeries summary:")
    for name in cols:
        s = metrics["series"][name]
        print(f"  {name:10s}  end={s['end_px']:.0f}px  decreases={s['n_decreases']}  "
              f"max_drop={s['max_drop_px']:.0f}px  mean|step|={s['mean_abs_step_px']:.1f}px")
    if metrics["vs_ground_truth"]:
        print("\nVs manual annotations (lower MAE is better):")
        for name in cols:
            g = metrics["vs_ground_truth"].get(name)
            if g:
                print(f"  {name:10s}  MAE={g['mae_px']:.1f}px  median={g['median_ae_px']:.1f}px  "
                      f"corr={g['corr']:+.3f}  n={g['n_matched']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
