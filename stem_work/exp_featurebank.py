"""
Build a bank of candidate features per frame (on raw grayscale + edge mask),
then evaluate:
  (a) each single feature's correlation with GT stem height,
  (b) leave-one-out cross-validated ridge regression combining features.
Goal: find the most accurate, generalizable estimator of meristem height.
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(r"C:\Users\ryank\Code\microneedle_IAA_Raman_analysis")
CACHE = REPO / "stem_work" / "annotated_cache.npz"
BRIGHT_T = 55


def smooth1d(a, k):
    return a if k <= 1 else np.convolve(a, np.ones(k) / k, mode="same")


def derive_cx(g8, base_y, seed_cx, band=160, nrows=120, bright_t=BRIGHT_T):
    H, W = g8.shape
    y0 = max(0, base_y - nrows)
    lo, hi = max(0, seed_cx - band), min(W, seed_cx + band)
    sub = g8[y0:base_y, lo:hi].astype(float)
    colsum = (sub * (sub > bright_t)).sum(axis=0)
    if colsum.sum() == 0:
        return seed_cx
    return lo + int(round((np.arange(len(colsum)) * colsum).sum() / colsum.sum()))


def feats_for_frame(g8, base_y, cx, edge):
    H, W = g8.shape
    out = {}
    # canopy proxy on EDGE mask (reproduce current method): topmost corridor row w/ density>0.2
    for chalf, dt, name in [(90, 0.20, "edge_canopy")]:
        lo, hi = max(0, cx - chalf), min(W, cx + chalf)
        dens = smooth1d((edge[:, lo:hi] > 127).mean(axis=1), 15)
        top = max(0, base_y - 1000)
        hits = np.where(dens[top:base_y] > dt)[0]
        out[name] = float(base_y - (top + hits.min())) if len(hits) else 0.0
    # canopy proxy on RAW brightness
    for chalf, dt, name in [(90, 0.20, "raw_canopy")]:
        lo, hi = max(0, cx - chalf), min(W, cx + chalf)
        dens = smooth1d((g8[:, lo:hi] > BRIGHT_T).mean(axis=1), 15)
        top = max(0, base_y - 1000)
        hits = np.where(dens[top:base_y] > dt)[0]
        out[name] = float(base_y - (top + hits.min())) if len(hits) else 0.0
    # narrow-band bright continuity break (center wedge)
    for bw, ct, name in [(10, 0.85, "narrow_cont"), (20, 0.8, "mid_cont")]:
        lo, hi = max(0, cx - bw), min(W, cx + bw)
        prof = (g8[:, lo:hi] > BRIGHT_T).mean(axis=1)
        top = max(0, base_y - 1000); gap = 0; best = base_y
        for y in range(base_y, top, -1):
            if prof[y] >= ct:
                best = y; gap = 0
            else:
                gap += 1
                if gap > 12:
                    break
        out[name] = float(base_y - best)
    # column width transition
    bright = g8 > BRIGHT_T
    cur = cx; widths = np.full(H, np.nan)
    for y in range(base_y, max(0, base_y - 1000), -1):
        row = bright[y]; lo = max(0, cur - 130); hi = min(W, cur + 130)
        idx = np.where(row[lo:hi])[0] + lo
        if len(idx) == 0:
            continue
        c2 = idx[np.argmin(np.abs(idx - cur))]
        l = c2
        while l > 0 and row[l-1]:
            l -= 1
        r = c2
        while r < W-1 and row[r+1]:
            r += 1
        widths[y] = r - l + 1
        cur = int(round(0.5*cur + 0.5*((l+r)//2)))
    ys = np.where(~np.isnan(widths))[0]
    if len(ys) > 10:
        ws = smooth1d(np.nan_to_num(widths), 9)
        htr = base_y - ys.min()
        stem_w = np.median(widths[ys[ys >= base_y - int(0.3*htr)]])
        for m, name in [(2.0, "wtrans2"), (2.5, "wtrans25")]:
            thr = m*stem_w; a = 0.0
            for y in range(base_y, ys.min()-1, -1):
                if ws[y] > thr:
                    a = float(base_y - y); break
            out[name] = a
    else:
        out["wtrans2"] = 0.0; out["wtrans25"] = 0.0
    # total bright top (canopy top, full width) on raw
    rows = np.where((g8 > BRIGHT_T).sum(axis=1) > 30)[0]
    out["raw_top"] = float(base_y - rows.min()) if len(rows) else 0.0
    return out


def main():
    d = np.load(CACHE, allow_pickle=True)
    ts, cx, apex, base, raw, edge = d["ts"], d["cx"], d["apex"], d["base"], d["raw"], d["edge"]
    n = len(ts)
    gt = (base - apex).astype(float)

    rows = []
    for i in range(n):
        cxi = derive_cx(raw[i], int(base[i]), int(cx[i]))
        rows.append(feats_for_frame(raw[i], int(base[i]), cxi, edge[i]))
    names = list(rows[0].keys())
    X = np.array([[r[k] for k in names] for r in rows], float)

    print("Single-feature correlation with GT:")
    for j, nm in enumerate(names):
        xj = X[:, j]
        c = np.corrcoef(xj, gt)[0, 1]
        sl, ic = np.polyfit(xj, gt, 1)
        mae = np.mean(np.abs(sl*xj+ic-gt))
        print(f"  {nm:14s} corr={c:+.3f}  1D-fitMAE={mae:5.1f}px")

    # standardize
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = (X - mu) / sd

    def loo_ridge(cols, lam=5.0):
        Xc = Xs[:, cols]
        errs = []
        for i in range(n):
            tr = [j for j in range(n) if j != i]
            A = Xc[tr]; y = gt[tr]
            A1 = np.hstack([A, np.ones((len(tr), 1))])
            P = A1.T @ A1 + lam * np.eye(A1.shape[1]); P[-1, -1] = 0
            w = np.linalg.solve(P, A1.T @ y)
            xi = np.hstack([Xc[i], 1.0])
            errs.append(xi @ w - gt[i])
        errs = np.abs(errs)
        return errs.mean(), np.median(errs), errs.max()

    print("\nLOO-CV ridge regressions:")
    # all features
    allc = list(range(len(names)))
    m, md, mx = loo_ridge(allc)
    print(f"  ALL feats        LOO meanMAE={m:5.1f} medMAE={md:5.1f} maxMAE={mx:5.1f}px")
    # canopy-only baselines
    for nm in ["edge_canopy", "raw_canopy"]:
        j = names.index(nm)
        m, md, mx = loo_ridge([j])
        print(f"  {nm:14s}   LOO meanMAE={m:5.1f} medMAE={md:5.1f} maxMAE={mx:5.1f}px")
    # greedy forward selection
    chosen = []
    remaining = list(range(len(names)))
    best_prev = 1e9
    while remaining:
        scored = []
        for j in remaining:
            m, _, _ = loo_ridge(chosen + [j])
            scored.append((m, j))
        scored.sort()
        if scored[0][0] < best_prev - 0.3:
            best_prev = scored[0][0]; chosen.append(scored[0][1]); remaining.remove(scored[0][1])
        else:
            break
    m, md, mx = loo_ridge(chosen)
    print(f"\n  greedy subset = {[names[j] for j in chosen]}")
    print(f"  greedy LOO meanMAE={m:5.1f} medMAE={md:5.1f} maxMAE={mx:5.1f}px")


if __name__ == "__main__":
    main()
