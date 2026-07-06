"""Compare temporal smoothness of edge vs raw proxies on the full series and
overlay GT annotations. Quantify jumpiness via median abs successive diff."""
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
    if k <= 1:
        return z.copy()
    h = k // 2; out = z.copy()
    for i in range(len(z)):
        w = z[max(0, i-h):i+h+1]; w = w[~np.isnan(w)]
        out[i] = np.median(w) if len(w) else np.nan
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
    t0 = next(w for w in when if w)
    hrs = np.array([(w - t0).total_seconds()/3600 if w else np.nan for w in when])

    # GT
    d = np.load(CACHE, allow_pickle=True)
    gts = {t: int(b)-int(a) for t, a, b in zip(d["ts"], d["apex"], d["base"])}
    gx, gy = [], []
    for t, w in zip(ts, when):
        if t in gts and w:
            gx.append((w - t0).total_seconds()/3600); gy.append(gts[t])

    # calibrate each feature to GT (1D linear) so they're comparable in px
    def calib(feat):
        m = np.array([t in gts for t in ts])
        x = feat[m]; y = np.array([gts[t] for t in np.array(ts)[m]])
        ok = ~np.isnan(x)
        sl, ic = np.polyfit(x[ok], y[ok], 1)
        return sl*feat + ic, sl, ic

    order = ["edge_canopy", "raw_canopy", "sil_corr", "sil_top"]
    fig, axs = plt.subplots(len(order), 1, figsize=(14, 3.4*len(order)), sharex=True)
    for ax, name in zip(axs, order):
        feat = cols[name]
        cal, sl, ic = calib(feat)
        sm = rolling_median(cal, 5)
        # jumpiness = median abs successive difference (raw, calibrated px)
        diffs = np.abs(np.diff(cal[~np.isnan(cal)]))
        jump = np.median(diffs); p95 = np.percentile(diffs, 95); mx = diffs.max()
        ax.plot(hrs, cal, ".", color="0.7", ms=4, label="raw")
        ax.plot(hrs, sm, "-", color="C0", lw=1.5, label="rolling median k=5")
        ax.plot(gx, gy, "r*", ms=12, label="GT")
        ax.set_title(f"{name}: jump med={jump:.1f}px p95={p95:.1f}px max={mx:.0f}px  (slope={sl:.2f})")
        ax.set_ylabel("stem h (px)"); ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=8)
        print(f"{name:16s} jump_med={jump:5.1f} p95={p95:5.1f} max={mx:6.0f}px")
    axs[-1].set_xlabel("hours from start")
    fig.tight_layout(); fig.savefig(str(OUT / "exp_smoothness.png"), dpi=110)
    print("saved", OUT / "exp_smoothness.png")


if __name__ == "__main__":
    main()
