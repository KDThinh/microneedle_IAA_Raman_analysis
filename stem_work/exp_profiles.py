"""
For several annotated frames, show the zoomed raw stem crop next to vertical
profiles computed in a central corridor, with the GT apex marked. Goal: find
which signal has a clear, consistent 'knee' at the apical meristem.

Profiles (y from base upward):
  - colwidth : width (px) of the central bright connected run at each row
  - brightfrac : fraction of corridor pixels that are 'plant-bright'
  - meanbright : mean corridor intensity
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

CORR_HALF = 70     # corridor half width (px) around stem x
BRIGHT_T = 60      # 8-bit intensity considered plant (bg is dark)


def main():
    d = np.load(CACHE, allow_pickle=True)
    ts, cx, apex, base, raw = d["ts"], d["cx"], d["apex"], d["base"], d["raw"]
    n = len(ts)
    sel = np.linspace(0, n - 1, 6).round().astype(int)

    fig, axs = plt.subplots(2, len(sel), figsize=(3.4 * len(sel), 10))
    for j, i in enumerate(sel):
        g = raw[i]
        cxi, ai, bi = int(cx[i]), int(apex[i]), int(base[i])
        H, W = g.shape
        x0, x1 = max(0, cxi - CORR_HALF), min(W, cxi + CORR_HALF)
        corr = g[:, x0:x1]
        bright = corr > BRIGHT_T
        brightfrac = bright.mean(axis=1)
        meanbright = corr.mean(axis=1) / 255.0

        # central bright run width: at each row, the contiguous bright run that
        # contains the corridor center
        colwidth = np.zeros(H)
        cc = corr.shape[1] // 2
        for y in range(H):
            rowb = bright[y]
            if not rowb[cc]:
                # nearest bright to center
                idx = np.where(rowb)[0]
                if len(idx) == 0:
                    continue
                c2 = idx[np.argmin(np.abs(idx - cc))]
            else:
                c2 = cc
            # expand run around c2
            l = c2
            while l > 0 and rowb[l - 1]:
                l -= 1
            r = c2
            while r < len(rowb) - 1 and rowb[r + 1]:
                r += 1
            colwidth[y] = r - l + 1

        # crop region for display
        yb0, yb1 = max(0, ai - 250), min(H, bi + 60)
        crop = cv2.cvtColor(g[yb0:yb1, max(0, cxi-220):min(W, cxi+220)], cv2.COLOR_GRAY2BGR)
        # GT apex/base in crop coords
        cv2.line(crop, (0, ai - yb0), (crop.shape[1]-1, ai - yb0), (0, 255, 0), 2)
        cv2.line(crop, (0, bi - yb0), (crop.shape[1]-1, bi - yb0), (0, 165, 255), 2)
        axs[0, j].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        axs[0, j].axis("off")
        axs[0, j].set_title(f"{ts[i][5:]}\nGT h={bi-ai}", fontsize=8)

        # profiles vs height-above-base (so up is positive)
        yy = np.arange(H)
        habove = bi - yy
        m = (habove >= -20) & (habove <= (bi - ai) + 300)
        ax = axs[1, j]
        ax.plot(brightfrac[m], habove[m], label="brightfrac")
        ax.plot(meanbright[m], habove[m], label="meanbright")
        ax.plot(colwidth[m] / (2 * CORR_HALF), habove[m], label="colwidth/corr")
        ax.axhline(bi - ai, color="g", lw=2, label="GT apex")
        ax.axhline(0, color="orange", lw=1.5, label="base")
        ax.set_ylim(-30, (bi - ai) + 300)
        ax.set_xlim(0, 1)
        if j == 0:
            ax.legend(fontsize=7)
            ax.set_ylabel("height above base (px)")
        ax.grid(alpha=0.3)

    fig.suptitle("Stem crop (top) and corridor profiles vs height (bottom). GT apex = green.", fontsize=11)
    fig.tight_layout()
    outp = OUT / "exp_profiles.png"
    fig.savefig(str(outp), dpi=110)
    print("saved", outp)


if __name__ == "__main__":
    main()
