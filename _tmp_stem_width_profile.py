"""
Strategy B: Width profile analysis for stem node and meristem detection.

Builds on v3 corridor detection from _tmp_stem_diagnostic.py:
  Corridor centre = coarse anchor (bottom BOTTOM_FRAC strip median-x)
                  + max-run-length column projection within ±ANCHOR_MARGIN px
  Corridor width  = ±CORRIDOR_HALF_PX px

Width profile definition
  width(y) = horizontal span (rightmost - leftmost white pixel + 1) of mask
             pixels inside the corridor at image row y.
             0 if fewer than MIN_STEM_WIDTH white pixels are present.

Derived quantities
  Nodes     = local maxima in the SMOOTHED width profile whose prominence
              exceeds NODE_MIN_PROM px. A node is where a petiole branches off
              the stem; the petiole base adds extra white pixels on one side,
              locally widening the span.
  Meristem  = topmost image row where MERISTEM_MIN_ROWS consecutive rows all
              have width > 0 (scanning downward from y0).
  Base      = bottommost image row where MERISTEM_MIN_ROWS consecutive rows all
              have width > 0 (scanning upward from y1).
  Stem height = base_y - meristem_y  (px, in image coordinates).

Visualisation (per sampled frame)
  Left  — mask crop with corridor boundaries (yellow), node rows (blue dashes),
           meristem row (red), base row (green).
  Right — width profile: x = width (px), y = height-from-base (px) so pot is
           at the bottom and meristem at the top. Raw profile (light fill) +
           smoothed profile (solid line). Node peaks (red dots), meristem (red
           star) and base (green star) annotated.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.signal import find_peaks
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
MASKS_DIR = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465\masks"
)
OUT_PATH = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465"
    r"\stem_width_profile.png"
)

# ── Corridor detection (v3 — identical to _tmp_stem_diagnostic.py) ─────────────
CORRIDOR_HALF_PX = 40
BOTTOM_FRAC      = 0.15
PROJ_BAND        = (0.20, 0.85)
ANCHOR_MARGIN    = 150
SMOOTH_K         = 15

# ── Width profile tuning ───────────────────────────────────────────────────────
MIN_STEM_WIDTH    = 3   # px: minimum span to count as stem-present (not noise)
GAP_INTERP_MAX    = 30  # rows: interpolate across gaps ≤ this size before peak search
PROFILE_SMOOTH_K  = 9   # rows: moving-average kernel for profile smoothing
NODE_MIN_PROM     = 6   # px: minimum width prominence for a node peak
NODE_MIN_DIST     = 20  # rows: minimum vertical separation between node peaks
MERISTEM_MIN_ROWS = 3   # consecutive non-zero rows to confirm meristem / base

# ── Display ────────────────────────────────────────────────────────────────────
N_SAMPLES = 6
DPI       = 200


# ══════════════════════════════════════════════════════════════════════════════
# Corridor detection (v3)
# ══════════════════════════════════════════════════════════════════════════════

def col_max_run_lengths(binary_2d: np.ndarray) -> np.ndarray:
    """Max consecutive vertical run of True per column."""
    max_runs     = np.zeros(binary_2d.shape[1], dtype=np.int32)
    current_runs = np.zeros(binary_2d.shape[1], dtype=np.int32)
    for row in binary_2d:
        current_runs = np.where(row, current_runs + 1, 0)
        np.maximum(max_runs, current_runs, out=max_runs)
    return max_runs


def estimate_stem_cx(mask: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> int:
    """
    Step 1 — coarse anchor: median-x of white pixels in the bottom BOTTOM_FRAC
             of the plant bbox.
    Step 2 — fine search: within ±ANCHOR_MARGIN of anchor, find column whose
             max vertical run length is greatest (middle PROJ_BAND of height).
    """
    plant_h  = y1 - y0
    y_cut    = int(y1 - BOTTOM_FRAC * plant_h)
    strip    = mask[max(0, y_cut) : y1 + 1, x0 : x1 + 1]
    _, xs    = np.where(strip > 127)
    cx_anch  = (x0 + int(np.median(xs))) if len(xs) else ((x0 + x1) // 2)

    sx0 = max(x0, cx_anch - ANCHOR_MARGIN)
    sx1 = min(x1, cx_anch + ANCHOR_MARGIN)
    bt  = int(y0 + PROJ_BAND[0] * plant_h)
    bb  = int(y0 + PROJ_BAND[1] * plant_h)
    band = mask[bt : bb + 1, sx0 : sx1 + 1] > 127

    if band.size == 0 or not band.any():
        return cx_anch

    proj   = col_max_run_lengths(band).astype(np.float64)
    kernel = np.ones(SMOOTH_K) / SMOOTH_K
    smooth = np.convolve(proj, kernel, mode="same")
    return sx0 + int(np.argmax(smooth))


# ══════════════════════════════════════════════════════════════════════════════
# Width profile functions
# ══════════════════════════════════════════════════════════════════════════════

def compute_width_profile(
    mask: np.ndarray,
    y0: int, y1: int,
    cx0: int, cx1: int,
) -> np.ndarray:
    """
    width[i] = rightmost - leftmost white pixel + 1 in mask[y0+i, cx0:cx1+1].
    0 if fewer than MIN_STEM_WIDTH white pixels present in that row.
    """
    n = y1 - y0 + 1
    widths = np.zeros(n, dtype=np.float32)
    for i in range(n):
        row     = mask[y0 + i, cx0 : cx1 + 1]
        white_x = np.where(row > 127)[0]
        if len(white_x) >= MIN_STEM_WIDTH:
            widths[i] = float(white_x[-1] - white_x[0] + 1)
    return widths


def interpolate_gaps(widths: np.ndarray, max_gap: int = GAP_INTERP_MAX) -> np.ndarray:
    """
    Linearly interpolate across zero-runs of length ≤ max_gap.
    Larger gaps (genuine breaks) are left as zero.
    """
    w = widths.copy()
    n = len(w)
    i = 0
    while i < n:
        if w[i] == 0:
            j = i
            while j < n and w[j] == 0:
                j += 1
            gap = j - i
            if gap <= max_gap and i > 0 and j < n:
                v0, v1 = w[i - 1], w[j]
                for k in range(gap):
                    w[i + k] = v0 + (v1 - v0) * (k + 1) / (gap + 1)
            i = j
        else:
            i += 1
    return w


def smooth_profile(widths: np.ndarray, k: int = PROFILE_SMOOTH_K) -> np.ndarray:
    kernel = np.ones(k) / k
    return np.convolve(widths, kernel, mode="same")


def find_stem_landmarks(
    widths_raw: np.ndarray,
    y0: int,
    y1: int,
) -> tuple[list[int], int | None, int | None]:
    """
    Returns (node_ys, meristem_y, base_y) in image (pixel) coordinates.

    node_ys    — image y of each detected node (local width-profile maximum).
    meristem_y — image y of the topmost sustained non-zero row (scan top→bottom).
    base_y     — image y of the bottommost sustained non-zero row (scan bottom→top).
    """
    # Work on gap-interpolated then smoothed profile for peak detection
    w_interp = interpolate_gaps(widths_raw)
    w_smooth = smooth_profile(w_interp)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    peaks, _ = find_peaks(w_smooth, prominence=NODE_MIN_PROM, distance=NODE_MIN_DIST)
    node_ys  = [y0 + int(p) for p in peaks]

    # ── Meristem: topmost run of MERISTEM_MIN_ROWS consecutive non-zero rows ───
    meristem_y, run = None, 0
    for i, w in enumerate(widths_raw):
        if w > 0:
            run += 1
            if run >= MERISTEM_MIN_ROWS:
                meristem_y = y0 + (i - MERISTEM_MIN_ROWS + 1)
                break
        else:
            run = 0

    # ── Base: bottommost run (scan from bottom) ────────────────────────────────
    base_y, run = None, 0
    for i, w in enumerate(reversed(widths_raw)):
        if w > 0:
            run += 1
            if run >= MERISTEM_MIN_ROWS:
                base_y = y1 - (i - MERISTEM_MIN_ROWS + 1)
                break
        else:
            run = 0

    return node_ys, meristem_y, base_y


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

masks   = sorted(MASKS_DIR.glob("*_mask_refined.png"))
if not masks:
    raise FileNotFoundError(f"No masks in {MASKS_DIR}")

step    = max(1, len(masks) // N_SAMPLES)
sampled = masks[::step][:N_SAMPLES]
print(f"Total masks: {len(masks)}, sampling {len(sampled)} (every {step}th)")

fig, axes = plt.subplots(N_SAMPLES, 2, figsize=(12, N_SAMPLES * 4.5))

for row_idx, mask_path in enumerate(sampled):
    ax_img = axes[row_idx, 0]
    ax_prf = axes[row_idx, 1]

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        ax_img.set_title("READ ERROR"); ax_img.axis("off")
        ax_prf.axis("off"); continue

    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        ax_img.set_title("EMPTY MASK"); ax_img.axis("off")
        ax_prf.axis("off"); continue

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # ── Corridor ────────────────────────────────────────────────────────────
    cx  = estimate_stem_cx(mask, y0, y1, x0, x1)
    cx0 = max(0,              cx - CORRIDOR_HALF_PX)
    cx1 = min(mask.shape[1]-1, cx + CORRIDOR_HALF_PX)

    # ── Width profile ────────────────────────────────────────────────────────
    widths_raw  = compute_width_profile(mask, y0, y1, cx0, cx1)
    w_interp    = interpolate_gaps(widths_raw)
    w_smooth    = smooth_profile(w_interp)

    node_ys, meristem_y, base_y = find_stem_landmarks(widths_raw, y0, y1)

    stem_h = (base_y - meristem_y) if (meristem_y is not None and base_y is not None) else None

    # ── Height-from-base coordinates (y-axis of profile plot) ───────────────
    # row i in widths_raw → image y = y0 + i
    # height from base = base_y - (y0 + i)  → 0 at base, positive upward
    n_rows  = len(widths_raw)
    hfb_raw    = np.array([base_y - (y0 + i) if base_y is not None else 0
                            for i in range(n_rows)], dtype=np.float32)
    # for scipy peaks, peaks are indices in widths array
    node_hfb   = [(base_y - ny) for ny in node_ys if base_y is not None]
    mer_hfb    = (base_y - meristem_y) if (meristem_y is not None and base_y is not None) else None
    base_hfb   = 0

    # ══════════════════════════════════════════
    # LEFT PANEL: mask crop with annotations
    # ══════════════════════════════════════════
    margin = 40
    yc0 = max(0, y0 - margin);  yc1 = min(mask.shape[0], y1 + margin)
    xc0 = max(0, x0 - margin);  xc1 = min(mask.shape[1], x1 + margin)

    vis = np.stack([(mask[yc0:yc1, xc0:xc1] // 5)] * 3, axis=2).astype(np.uint8)

    # Corridor borders (yellow)
    for cy in range(yc0, yc1):
        if 0 <= cx0 - xc0 < vis.shape[1]:
            vis[cy - yc0, cx0 - xc0] = [255, 220, 0]
        if 0 <= cx1 - xc0 < vis.shape[1]:
            vis[cy - yc0, cx1 - xc0] = [255, 220, 0]

    # Node rows (blue)
    for ny in node_ys:
        if yc0 <= ny <= yc1:
            vis[ny - yc0, max(0, cx0-xc0) : min(vis.shape[1], cx1-xc0+1)] = [80, 140, 255]

    # Meristem row (red)
    if meristem_y is not None and yc0 <= meristem_y <= yc1:
        vis[meristem_y - yc0, max(0, cx0-xc0) : min(vis.shape[1], cx1-xc0+1)] = [230, 40, 40]

    # Base row (green)
    if base_y is not None and yc0 <= base_y <= yc1:
        vis[base_y - yc0, max(0, cx0-xc0) : min(vis.shape[1], cx1-xc0+1)] = [40, 200, 80]

    ax_img.imshow(vis)
    ts = mask_path.stem.replace("tl_", "").replace("_mask_refined", "")
    stem_h_str = f"{stem_h}px" if stem_h is not None else "N/A"
    ax_img.set_title(
        f"{ts}\n"
        f"nodes: {len(node_ys)}   stem h: {stem_h_str}",
        fontsize=8,
    )
    ax_img.axis("off")

    # Legend patches for mask panel
    leg = [
        mpatches.Patch(color=[1,.86,0], label="corridor"),
        mpatches.Patch(color=[.31,.55,1], label="nodes"),
        mpatches.Patch(color=[.9,.16,.16], label="meristem"),
        mpatches.Patch(color=[.16,.78,.31], label="base"),
    ]
    ax_img.legend(handles=leg, fontsize=6, loc="lower left",
                  framealpha=0.6, handlelength=1)

    # ══════════════════════════════════════════
    # RIGHT PANEL: width profile
    # ══════════════════════════════════════════
    # x = width (px), y = height from base (px)
    ax_prf.fill_betweenx(hfb_raw, 0, widths_raw, alpha=0.25, color="green",
                          label="raw width")
    ax_prf.plot(w_smooth, hfb_raw, color="green", lw=1.5, label="smoothed")

    # Node peaks
    if node_hfb:
        for ny, nh in zip(node_ys, node_hfb):
            i_peak = ny - y0
            ax_prf.plot(w_smooth[i_peak], nh, "o", color="royalblue",
                        ms=7, zorder=5)
            ax_prf.axhline(nh, color="royalblue", lw=0.8, ls="--", alpha=0.5)

    # Meristem
    if mer_hfb is not None:
        ax_prf.axhline(mer_hfb, color="crimson", lw=1.2, ls="--",
                       label=f"meristem ({mer_hfb:.0f}px from base)")
        ax_prf.plot(0, mer_hfb, "*", color="crimson", ms=10, zorder=6)

    # Base
    ax_prf.axhline(0, color="forestgreen", lw=1.2, ls="--",
                   label="base (0 px)")
    ax_prf.plot(0, 0, "*", color="forestgreen", ms=10, zorder=6)

    ax_prf.set_xlabel("Corridor span — width (px)", fontsize=8)
    ax_prf.set_ylabel("Height from base (px)", fontsize=8)
    ax_prf.set_xlim(left=0, right=CORRIDOR_HALF_PX * 2 + 10)
    ax_prf.tick_params(labelsize=7)
    ax_prf.legend(fontsize=6, loc="lower right")
    n_nodes_str = f"{len(node_ys)} node(s) detected"
    ax_prf.set_title(n_nodes_str, fontsize=8)
    ax_prf.grid(axis="x", ls=":", alpha=0.4)

plt.suptitle(
    "Strategy B — Width profile: stem corridor span per row\n"
    f"Corridor: ±{CORRIDOR_HALF_PX}px | min width: {MIN_STEM_WIDTH}px | "
    f"gap interp ≤{GAP_INTERP_MAX}rows | node prom ≥{NODE_MIN_PROM}px dist ≥{NODE_MIN_DIST}rows",
    fontsize=9, y=1.005,
)
plt.tight_layout()
plt.savefig(str(OUT_PATH), dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_PATH}")

print(f"\n{'Frame':<45} {'nodes':>6} {'meristem_y':>11} {'base_y':>7} {'stem_h':>8}")
print("-" * 82)
for mask_path in sampled:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None: continue
    ys, xs = np.where(mask > 127)
    if len(ys) == 0: continue
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cx  = estimate_stem_cx(mask, y0, y1, x0, x1)
    cx0 = max(0, cx - CORRIDOR_HALF_PX)
    cx1 = min(mask.shape[1]-1, cx + CORRIDOR_HALF_PX)
    widths_raw = compute_width_profile(mask, y0, y1, cx0, cx1)
    node_ys, meristem_y, base_y = find_stem_landmarks(widths_raw, y0, y1)
    stem_h = (base_y - meristem_y) if (meristem_y and base_y) else None
    ts = mask_path.stem.replace("tl_", "").replace("_mask_refined", "")
    print(
        f"{ts:<45} {len(node_ys):>6} {str(meristem_y):>11} "
        f"{str(base_y):>7} {str(stem_h):>8}"
    )
