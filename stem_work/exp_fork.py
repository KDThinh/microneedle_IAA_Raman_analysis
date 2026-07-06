"""
Fork detector: trace the single bright stem column up from the base; the apical
meristem is the lowest row where the column splits into >=2 bright petiole runs
separated by a dark wedge (sustained). Compare candidate apex to GT.

Also tests a 'dark-wedge' variant: row where central narrow band brightness drops
while flanking bands stay bright (the V opens).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(r"C:\Users\ryank\Code\microneedle_IAA_Raman_analysis")
CACHE = REPO / "stem_work" / "annotated_cache.npz"
OUT = REPO / "stem_work"
BRIGHT_T = 55


def derive_cx(g8, base_y, seed_cx, band=160, nrows=120, bright_t=BRIGHT_T):
    H, W = g8.shape
    y0 = max(0, base_y - nrows)
    lo, hi = max(0, seed_cx - band), min(W, seed_cx + band)
    sub = g8[y0:base_y, lo:hi].astype(float)
    colsum = (sub * (sub > bright_t)).sum(axis=0)
    if colsum.sum() == 0:
        return seed_cx
    return lo + int(round((np.arange(len(colsum)) * colsum).sum() / colsum.sum()))


def runs(row_bool):
    """Return list of (start, end) inclusive runs of True."""
    idx = np.where(row_bool)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    out = []
    s = idx[0]
    for sp in splits:
        out.append((s, idx[sp]))
        s = idx[sp + 1]
    out.append((s, idx[-1]))
    return out


def apex_by_fork(g8, base_y, cx, bright_t=BRIGHT_T, search=140, top_limit=1000,
                 min_run=8, min_gap=10, sustain=12, frac=0.6):
    """Walk up; track stem run nearest center; detect sustained split into 2 runs."""
    H, W = g8.shape
    bright = g8 > bright_t
    top = max(0, base_y - top_limit)
    cur = cx
    split_count = 0
    cand = None
    for y in range(base_y, top, -1):
        lo, hi = max(0, cur - search), min(W, cur + search)
        rs = runs(bright[y, lo:hi])
        rs = [(a + lo, b + lo) for (a, b) in rs if (b - a + 1) >= min_run]
        if not rs:
            split_count = 0
            continue
        # stem run nearest current center
        centers = [(a + b) / 2 for (a, b) in rs]
        k = int(np.argmin([abs(c - cur) for c in centers]))
        cur = int(round(0.6 * cur + 0.4 * centers[k]))
        # is there a split? >=2 runs within +-search, separated by dark gap>=min_gap
        big = [r for r in rs if (r[1] - r[0] + 1) >= min_run]
        is_split = False
        if len(big) >= 2:
            big.sort()
            gaps = [big[j + 1][0] - big[j][1] - 1 for j in range(len(big) - 1)]
            if max(gaps) >= min_gap:
                is_split = True
        if is_split:
            split_count += 1
            if cand is None:
                cand = y
            if split_count >= int(sustain * frac):
                return float(base_y - cand)
        else:
            split_count = 0
            cand = None
    return np.nan


def main():
    d = np.load(CACHE, allow_pickle=True)
    ts, cx, apex, base, raw = d["ts"], d["cx"], d["apex"], d["base"], d["raw"]
    n = len(ts)
    gt_h = (base - apex).astype(float)

    best = None
    grid = []
    for min_run in (6, 10, 16):
        for min_gap in (6, 12, 20):
            for sustain in (8, 16, 28):
                ph = np.array([apex_by_fork(raw[i], int(base[i]),
                               derive_cx(raw[i], int(base[i]), int(cx[i])),
                               min_run=min_run, min_gap=min_gap, sustain=sustain) for i in range(n)])
                ok = ~np.isnan(ph)
                if ok.sum() < 10:
                    continue
                corr = np.corrcoef(ph[ok], gt_h[ok])[0, 1]
                sl, ic = np.polyfit(ph[ok], gt_h[ok], 1)
                fit_mae = np.mean(np.abs(sl * ph[ok] + ic - gt_h[ok]))
                raw_mae = np.mean(np.abs(ph[ok] - gt_h[ok]))
                grid.append((fit_mae, corr, min_run, min_gap, sustain, ok.sum(), raw_mae, sl, ic, ph))
    grid.sort(key=lambda r: r[0])
    print("Top fork configs by fit-MAE:")
    print(f"{'fitMAE':>7} {'corr':>6} {'run':>4} {'gap':>4} {'sus':>4} {'nok':>4} {'rawMAE':>7} {'slope':>6} {'int':>7}")
    for r in grid[:10]:
        print(f"{r[0]:7.1f} {r[1]:+.3f} {r[2]:4d} {r[3]:4d} {r[4]:4d} {r[5]:4d} {r[6]:7.1f} {r[7]:6.3f} {r[8]:7.1f}")

    best = grid[0]
    ph, sl, ic = best[9], best[7], best[8]
    ok = ~np.isnan(ph)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(ph[ok], gt_h[ok], s=20)
    xx = np.linspace(np.nanmin(ph[ok]), np.nanmax(ph[ok]), 10)
    ax[0].plot(xx, sl * xx + ic, "r-")
    ax[0].set_title(f"fork run={best[2]} gap={best[3]} sus={best[4]}\n"
                    f"corr={best[1]:+.3f} fitMAE={best[0]:.0f} (n={best[5]})")
    ax[0].set_xlabel("feature px"); ax[0].set_ylabel("GT h px"); ax[0].grid(alpha=0.3)
    order = np.arange(n)
    ax[1].plot(order, gt_h, "g*-", label="GT")
    ax[1].plot(order, sl * ph + ic, "b.-", label="fork pred (calib)")
    ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_xlabel("frame (time order)")
    ax[1].set_title("calibrated fork prediction vs GT")
    fig.tight_layout(); fig.savefig(str(OUT / "exp_fork.png"), dpi=110)
    print("\nsaved", OUT / "exp_fork.png")


if __name__ == "__main__":
    main()
