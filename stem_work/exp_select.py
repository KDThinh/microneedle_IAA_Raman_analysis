"""
Final selection: for each candidate proxy, (1) accuracy vs GT (corr, LOO 1D fit),
(2) robustness after a Hampel outlier filter + rolling median. Plot best calibrated.
"""
import csv, datetime as dt
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(r"C:\Users\ryank\Code\microneedle_IAA_Raman_analysis")
CSV = REPO / "stem_work" / "full_series.csv"
CACHE = REPO / "stem_work" / "annotated_cache.npz"
OUT = REPO / "stem_work"


def rolling_median(z, k):
    h = k // 2; out = z.copy()
    for i in range(len(z)):
        w = z[max(0, i-h):i+h+1]; w = w[~np.isnan(w)]
        out[i] = np.median(w) if len(w) else np.nan
    return out


def hampel(z, k=7, nsig=3.0):
    """Replace points that deviate > nsig*MAD from local median with NaN."""
    h = k // 2; out = z.copy()
    for i in range(len(z)):
        w = z[max(0, i-h):i+h+1]; w = w[~np.isnan(w)]
        if len(w) < 3:
            continue
        med = np.median(w); mad = np.median(np.abs(w - med)) + 1e-9
        if abs(z[i] - med) > nsig * 1.4826 * mad:
            out[i] = np.nan
    return out


def main():
    cols = {k: [] for k in ["edge_canopy", "raw_canopy", "raw_canopy_cont", "sil_corr", "sil_top"]}
    ts, when = [], []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            ts.append(r["ts"])
            when.append(dt.datetime.fromisoformat(r["datetime"]) if r["datetime"] else None)
            for k in cols:
                cols[k].append(float(r[k]) if r.get(k) else np.nan)
    for k in cols:
        cols[k] = np.array(cols[k])
    ts = np.array(ts)
    t0 = next(w for w in when if w)
    hrs = np.array([(w - t0).total_seconds()/3600 if w else np.nan for w in when])

    d = np.load(CACHE, allow_pickle=True)
    gtmap = {t: int(b)-int(a) for t, a, b in zip(d["ts"], d["apex"], d["base"])}
    is_gt = np.array([t in gtmap for t in ts])
    gt_y = np.array([gtmap.get(t, np.nan) for t in ts])

    print(f"{'proxy':16s} {'corr':>6} {'LOO-MAE':>8} {'LOOmax':>7} | robust jump p95/max")
    for name in ["edge_canopy", "sil_corr", "sil_top"]:
        feat = cols[name]
        x = feat[is_gt]; y = gt_y[is_gt]; ok = ~np.isnan(x)
        x, y = x[ok], y[ok]
        corr = np.corrcoef(x, y)[0, 1]
        # LOO 1D linear
        loo = []
        for i in range(len(x)):
            tr = [j for j in range(len(x)) if j != i]
            sl, ic = np.polyfit(x[tr], y[tr], 1)
            loo.append(abs(sl*x[i]+ic - y[i]))
        loo = np.array(loo)
        # robust filtered full series jumpiness (calibrated px)
        sl, ic = np.polyfit(x, y, 1)
        cal = sl*feat + ic
        filt = hampel(cal, 7, 3.0)
        filt = rolling_median(filt, 5)
        diffs = np.abs(np.diff(filt[~np.isnan(filt)]))
        print(f"{name:16s} {corr:+.3f} {loo.mean():8.1f} {loo.max():7.0f} | "
              f"{np.percentile(diffs,95):.1f}/{diffs.max():.0f}px")

    # final plot: sil_corr and sil_top calibrated+filtered vs GT
    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, name in zip(axs, ["sil_corr", "sil_top"]):
        feat = cols[name]
        x = feat[is_gt]; y = gt_y[is_gt]; ok = ~np.isnan(x)
        sl, ic = np.polyfit(x[ok], y[ok], 1)
        cal = sl*feat + ic
        filt = rolling_median(hampel(cal, 7, 3.0), 5)
        ax.plot(hrs, cal, ".", color="0.78", ms=4, label="calibrated raw")
        ax.plot(hrs, filt, "-", color="C0", lw=1.8, label="Hampel + rolling median")
        ax.plot(hrs[is_gt], gt_y[is_gt], "r*", ms=12, label="GT")
        ax.set_title(f"{name} calibrated (slope={sl:.2f})"); ax.set_ylabel("stem h (px)")
        ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
    axs[-1].set_xlabel("hours from start")
    fig.tight_layout(); fig.savefig(str(OUT / "exp_select.png"), dpi=110)
    print("saved", OUT / "exp_select.png")


if __name__ == "__main__":
    main()
