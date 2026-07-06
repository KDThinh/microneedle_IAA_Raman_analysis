"""
Trace the central bright stem column from base upward (following connectivity),
get its width profile w(y), and test whether a width-transition rule predicts the
GT apex (meristem = where the narrow stem widens into the petiole crown).

Outputs: console stats (corr, MAE for several rules) + a scatter/overlay PNG.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(r"C:\Users\ryank\Code\microneedle_IAA_Raman_analysis")
CACHE = REPO / "stem_work" / "annotated_cache.npz"
OUT = REPO / "stem_work"

BRIGHT_T = 55
WIN = 130          # search half-window when following the column
SMOOTH = 9


def smooth1d(a, k):
    if k <= 1:
        return a
    ker = np.ones(k) / k
    return np.convolve(a, ker, mode="same")


def trace_column(g8, base_y, cx0, bright_t=BRIGHT_T, win=WIN, top_limit=900):
    """From base_y upward, follow the bright run nearest the running center.
    Returns ys (top->base order not guaranteed), center x, width arrays indexed by row."""
    H, W = g8.shape
    bright = g8 > bright_t
    top = max(0, base_y - top_limit)
    cx = cx0
    rows = np.arange(base_y, top, -1)
    cxs = np.full(H, np.nan)
    wid = np.full(H, np.nan)
    miss = 0
    for y in rows:
        row = bright[y]
        lo = max(0, cx - win)
        hi = min(W, cx + win)
        local = row[lo:hi]
        if not local.any():
            miss += 1
            if miss > 25:
                break
            continue
        miss = 0
        idx = np.where(local)[0] + lo
        c2 = idx[np.argmin(np.abs(idx - cx))]
        l = c2
        while l > 0 and row[l - 1]:
            l -= 1
        r = c2
        while r < W - 1 and row[r + 1]:
            r += 1
        cen = (l + r) // 2
        cxs[y] = cen
        wid[y] = r - l + 1
        # update center using a damped move toward the run center (stay near stem)
        cx = int(round(0.5 * cx + 0.5 * cen))
    return cxs, wid


def main():
    d = np.load(CACHE, allow_pickle=True)
    ts, cx, apex, base, raw = d["ts"], d["cx"], d["apex"], d["base"], d["raw"]
    n = len(ts)

    gt_h = (base - apex).astype(float)
    # for each frame compute width profile and several apex candidates
    rule_mult = [1.5, 2.0, 2.5, 3.0]
    pred = {f"w>{m}xstem": [] for m in rule_mult}
    w_at_gt = []
    stemw_list = []

    for i in range(n):
        g = raw[i]
        bi, ai = int(base[i]), int(apex[i])
        # cx near base: use column center over lower 40 rows from base via GT cx as seed
        cxs, wid = trace_column(g, bi, int(cx[i]))
        ys = np.where(~np.isnan(wid))[0]
        if len(ys) < 20:
            for m in rule_mult:
                pred[f"w>{m}xstem"].append(np.nan)
            w_at_gt.append(np.nan); stemw_list.append(np.nan)
            continue
        ws = smooth1d(np.nan_to_num(wid, nan=0.0), SMOOTH)
        # stem width = median width over lower 30% of traced height
        h_traced = bi - ys.min()
        low_band = ys[ys >= bi - int(0.30 * h_traced)]
        stem_w = np.median(wid[low_band])
        stemw_list.append(stem_w)
        # width at GT apex
        w_at_gt.append(wid[ai] if (0 <= ai < len(wid) and not np.isnan(wid[ai])) else np.nan)
        # rule: going up from base, first row where smoothed width > m*stem_w (sustained)
        for m in rule_mult:
            thr = m * stem_w
            apex_pred = np.nan
            for y in range(bi, ys.min() - 1, -1):
                if 0 <= y < len(ws) and ws[y] > thr and not np.isnan(wid[y]):
                    # require sustained: next 15 rows up also > thr*0.8
                    yy = np.arange(max(0, y - 15), y)
                    if np.nanmean(ws[yy] > 0.8 * thr) > 0.6:
                        apex_pred = y
                        break
            pred[f"w>{m}xstem"].append(apex_pred)

    print(f"GT stem_h range: {gt_h.min():.0f}-{gt_h.max():.0f} px")
    print(f"width at GT apex / stem_width: median ratio = "
          f"{np.nanmedian(np.array(w_at_gt)/np.array(stemw_list)):.2f}")
    print()
    fig, axs = plt.subplots(1, len(rule_mult), figsize=(4.5 * len(rule_mult), 4.3))
    for k, m in enumerate(rule_mult):
        ap = np.array(pred[f"w>{m}xstem"], float)
        ph = base - ap
        ok = ~np.isnan(ph)
        if ok.sum() >= 3:
            corr = np.corrcoef(ph[ok], gt_h[ok])[0, 1]
            sl, ic = np.polyfit(ph[ok], gt_h[ok], 1)
            mae = np.mean(np.abs(sl * ph[ok] + ic - gt_h[ok]))
            # residual after fit, and raw error
            raw_mae = np.mean(np.abs(ph[ok] - gt_h[ok]))
        else:
            corr = mae = raw_mae = np.nan
        print(f"rule w>{m}xstem: n_ok={ok.sum():2d}  corr={corr:+.3f}  "
              f"fit_MAE={mae:.1f}px  raw_MAE={raw_mae:.1f}px")
        axs[k].scatter(ph[ok], gt_h[ok], s=18)
        axs[k].plot([0, gt_h.max()], [0, gt_h.max()], "k--", lw=0.8)
        axs[k].set_title(f"w>{m}xstem  corr={corr:+.2f}\nfitMAE={mae:.0f} rawMAE={raw_mae:.0f}", fontsize=9)
        axs[k].set_xlabel("predicted h (px)"); axs[k].set_ylabel("GT h (px)")
        axs[k].grid(alpha=0.3)
    fig.suptitle("Width-transition apex vs GT (each point = annotated frame)")
    fig.tight_layout()
    fig.savefig(str(OUT / "exp_column_trace.png"), dpi=110)
    print("\nsaved", OUT / "exp_column_trace.png")


if __name__ == "__main__":
    main()
