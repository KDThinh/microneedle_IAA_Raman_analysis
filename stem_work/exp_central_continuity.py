"""
Test: apex = topmost row reachable by walking UP from the base along a narrow
central band that stays 'plant-bright', tolerating small gaps. The petiole fork
opens a dark notch in the center, so continuity should break at the meristem.

Sweeps band half-width, brightness threshold, continuity threshold, allowed gap.
Reports corr / MAE vs GT and the best config. Also tests robustness of base/cx
by re-deriving cx from the lower stem instead of using GT cx.
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


def derive_cx(g8, base_y, seed_cx, bright_t, band=160, nrows=120):
    """Stem x near base = brightness-weighted center in a band over rows just above base."""
    H, W = g8.shape
    y0 = max(0, base_y - nrows)
    lo, hi = max(0, seed_cx - band), min(W, seed_cx + band)
    sub = g8[y0:base_y, lo:hi].astype(float)
    colsum = (sub * (sub > bright_t)).sum(axis=0)
    if colsum.sum() == 0:
        return seed_cx
    return lo + int(round((np.arange(len(colsum)) * colsum).sum() / colsum.sum()))


def apex_by_continuity(g8, base_y, cx, bw, bright_t, cont_t, max_gap, top_limit=1000):
    H, W = g8.shape
    lo, hi = max(0, cx - bw), min(W, cx + bw)
    prof = (g8[:, lo:hi] > bright_t).mean(axis=1)
    top = max(0, base_y - top_limit)
    gap = 0
    best = base_y
    for y in range(base_y, top, -1):
        if prof[y] >= cont_t:
            best = y
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
    return float(base_y - best)


def main():
    d = np.load(CACHE, allow_pickle=True)
    ts, cx, apex, base, raw = d["ts"], d["cx"], d["apex"], d["base"], d["raw"]
    n = len(ts)
    gt_h = (base - apex).astype(float)

    bright_t = 55
    configs = []
    for bw in (6, 10, 16, 24, 35):
        for cont_t in (0.5, 0.7, 0.85, 0.95):
            for max_gap in (5, 12, 25):
                configs.append((bw, cont_t, max_gap))

    results = []
    for (bw, cont_t, max_gap) in configs:
        ph = np.array([apex_by_continuity(raw[i], int(base[i]),
                                          derive_cx(raw[i], int(base[i]), int(cx[i]), bright_t),
                                          bw, bright_t, cont_t, max_gap) for i in range(n)])
        ok = ~np.isnan(ph)
        if ok.sum() < 5:
            continue
        corr = np.corrcoef(ph[ok], gt_h[ok])[0, 1]
        sl, ic = np.polyfit(ph[ok], gt_h[ok], 1)
        fit_mae = np.mean(np.abs(sl * ph[ok] + ic - gt_h[ok]))
        results.append((corr, fit_mae, bw, cont_t, max_gap, sl, ic, ph))

    results.sort(key=lambda r: r[1])  # by fit MAE
    print("Top 8 configs by fit-MAE (continuity rule):")
    print(f"{'corr':>6} {'fitMAE':>7} {'bw':>3} {'cont_t':>6} {'gap':>4} {'slope':>6} {'int':>7}")
    for r in results[:8]:
        print(f"{r[0]:+.3f} {r[1]:7.1f} {r[2]:3d} {r[3]:6.2f} {r[4]:4d} {r[5]:6.3f} {r[6]:7.1f}")

    best = results[0]
    ph = best[7]
    sl, ic = best[5], best[6]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ok = ~np.isnan(ph)
    ax[0].scatter(ph[ok], gt_h[ok], s=20)
    xx = np.linspace(ph[ok].min(), ph[ok].max(), 10)
    ax[0].plot(xx, sl * xx + ic, "r-", lw=1)
    ax[0].set_xlabel("predicted feature (px)"); ax[0].set_ylabel("GT h (px)")
    ax[0].set_title(f"best continuity: bw={best[2]} cont_t={best[3]} gap={best[4]}\n"
                    f"corr={best[0]:+.3f} fitMAE={best[1]:.0f}px"); ax[0].grid(alpha=0.3)
    # time order residual
    order = np.argsort(ts)
    calib_h = sl * ph + ic
    ax[1].plot(np.arange(n), gt_h[order], "g*-", label="GT")
    ax[1].plot(np.arange(n), calib_h[order], "b.-", label="pred (calibrated)")
    ax[1].set_xlabel("annotated frame (time order)"); ax[1].set_ylabel("stem h (px)")
    ax[1].legend(); ax[1].grid(alpha=0.3); ax[1].set_title("calibrated prediction vs GT over time")
    fig.tight_layout(); fig.savefig(str(OUT / "exp_central_continuity.png"), dpi=110)
    print("\nsaved", OUT / "exp_central_continuity.png")


if __name__ == "__main__":
    main()
