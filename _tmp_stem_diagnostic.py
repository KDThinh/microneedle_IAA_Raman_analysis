"""
Stem corridor continuity diagnostic.

Stem centre estimated by combining two approaches:
  1. Coarse anchor  — median-x of white pixels in the bottom BOTTOM_FRAC of the
                      plant bounding box (reliable near the pot where only the
                      stem base is present).
  2. Fine search    — within ANCHOR_MARGIN px of that anchor, find the column
                      whose longest consecutive vertical run of white pixels is
                      greatest (max-run-length projection).  This rewards columns
                      where the stem edge is a single long unbroken line rather
                      than scattered leaf-crossing pixels.
  Both steps operate on the middle PROJ_BAND of the plant height so the pot and
  the very top of the canopy do not bias the result.

For each sampled frame the corridor (cx ± CORRIDOR_HALF_PX) is checked row-by-row;
green = pixel present, red = empty row (gap).
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

MASKS_DIR = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465\masks"
)
OUT_PATH = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465"
    r"\stem_continuity_diagnostic.png"
)

N_SAMPLES     = 12
CORRIDOR_HALF_PX = 40    # half-width of final corridor (px)
MIN_GAP_SIZE  = 5        # gaps shorter than this are ignored (noise)
BOTTOM_FRAC   = 0.15     # fraction of plant height used for coarse anchor
PROJ_BAND     = (0.20, 0.85)  # vertical band for column projection (skip pot + canopy top)
ANCHOR_MARGIN = 150      # restrict column-projection search to ±this px from anchor
SMOOTH_K      = 15       # moving-average kernel width for projection smoothing
DPI           = 220


def col_max_run_lengths(binary_2d: np.ndarray) -> np.ndarray:
    """
    For each column of a boolean 2-D array, return the length of the longest
    consecutive run of True values (max vertical run length).
    """
    max_runs     = np.zeros(binary_2d.shape[1], dtype=np.int32)
    current_runs = np.zeros(binary_2d.shape[1], dtype=np.int32)
    for row in binary_2d:
        current_runs = np.where(row, current_runs + 1, 0)
        np.maximum(max_runs, current_runs, out=max_runs)
    return max_runs


def estimate_stem_cx(mask: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> int:
    """
    Combined coarse-anchor + continuity-weighted column-projection approach.
    """
    plant_h = y1 - y0

    # --- Step 1: coarse anchor from bottom strip ---
    y_cut = int(y1 - BOTTOM_FRAC * plant_h)
    strip = mask[max(0, y_cut) : y1 + 1, x0 : x1 + 1]
    _, xs_strip = np.where(strip > 127)
    cx_anchor = (x0 + int(np.median(xs_strip))) if len(xs_strip) else ((x0 + x1) // 2)

    # --- Step 2: restrict search window around anchor ---
    search_x0 = max(x0, cx_anchor - ANCHOR_MARGIN)
    search_x1 = min(x1, cx_anchor + ANCHOR_MARGIN)

    # --- Step 3: max-run-length projection in search window, middle of plant ---
    band_top    = int(y0 + PROJ_BAND[0] * plant_h)
    band_bottom = int(y0 + PROJ_BAND[1] * plant_h)
    band = mask[band_top : band_bottom + 1, search_x0 : search_x1 + 1] > 127

    if band.size == 0 or not band.any():
        return cx_anchor

    run_proj = col_max_run_lengths(band).astype(np.float64)

    # Smooth to reduce noise
    kernel = np.ones(SMOOTH_K) / SMOOTH_K
    run_proj_smooth = np.convolve(run_proj, kernel, mode="same")

    peak_rel = int(np.argmax(run_proj_smooth))
    return search_x0 + peak_rel


# ---------------------------------------------------------------------------
masks = sorted(MASKS_DIR.glob("*_mask_refined.png"))
if not masks:
    raise FileNotFoundError(f"No masks in {MASKS_DIR}")

step    = max(1, len(masks) // N_SAMPLES)
sampled = masks[::step][:N_SAMPLES]
print(f"Total masks: {len(masks)}, sampling {len(sampled)} (every {step}th)")

ncols = 4
nrows = (len(sampled) + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 7))
axes_flat = np.array(axes).flat

results = []

for ax, mask_path in zip(axes_flat, sampled):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        ax.set_title("READ ERROR"); ax.axis("off"); continue

    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        ax.set_title(f"{mask_path.stem}\nEMPTY"); ax.axis("off"); continue

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    plant_h = y1 - y0

    cx  = estimate_stem_cx(mask, y0, y1, x0, x1)
    cx0 = max(0, cx - CORRIDOR_HALF_PX)
    cx1 = min(mask.shape[1] - 1, cx + CORRIDOR_HALF_PX)

    # Row-by-row gap check within plant bbox
    corridor    = mask[y0 : y1 + 1, cx0 : cx1 + 1]
    row_has_fg  = (corridor > 127).any(axis=1)

    gap_lengths, run = [], 0
    for has in row_has_fg:
        if not has:
            run += 1
        else:
            if run >= MIN_GAP_SIZE:
                gap_lengths.append(run)
            run = 0
    if run >= MIN_GAP_SIZE:
        gap_lengths.append(run)

    n_gaps          = len(gap_lengths)
    max_gap         = max(gap_lengths) if gap_lengths else 0
    continuity_pct  = float(row_has_fg.sum()) / len(row_has_fg) * 100

    results.append(dict(
        name=mask_path.stem,
        n_gaps=n_gaps,
        max_gap_px=max_gap,
        continuity_pct=continuity_pct,
        plant_h_px=plant_h,
        cx=cx,
    ))

    # Visualisation
    vis = np.stack([(mask // 6)] * 3, axis=2).astype(np.uint8)
    for row_idx, has in enumerate(row_has_fg):
        y = y0 + row_idx
        if has:
            vis[y, cx0 : cx1 + 1] = np.maximum(vis[y, cx0 : cx1 + 1], [0, 110, 0])
        else:
            vis[y, cx0 : cx1 + 1] = [160, 0, 0]
    vis[y0 : y1 + 1, cx0] = [255, 220, 0]
    vis[y0 : y1 + 1, cx1] = [255, 220, 0]

    margin = 40
    yc0 = max(0, y0 - margin);  yc1 = min(mask.shape[0], y1 + margin)
    xc0 = max(0, x0 - margin);  xc1 = min(mask.shape[1], x1 + margin)

    ts    = mask_path.stem.replace("tl_", "").replace("_mask_refined", "")
    color = "red" if n_gaps > 0 else "darkgreen"
    ax.imshow(vis[yc0:yc1, xc0:xc1])
    ax.set_title(
        f"{ts}\n"
        f"gaps ≥{MIN_GAP_SIZE}px: {n_gaps}   max gap: {max_gap}px\n"
        f"continuity: {continuity_pct:.1f}%   plant h: {plant_h}px\n"
        f"corridor cx: {cx}px   ±{CORRIDOR_HALF_PX}px",
        fontsize=8, color=color,
    )
    ax.axis("off")

for ax in list(axes_flat)[len(sampled):]:
    ax.axis("off")

plt.suptitle(
    "Stem corridor continuity diagnostic\n"
    f"Centre: coarse anchor (bottom {int(BOTTOM_FRAC*100)}% strip median-x) "
    f"+ max-run-length column projection within ±{ANCHOR_MARGIN}px of anchor "
    f"(band {int(PROJ_BAND[0]*100)}–{int(PROJ_BAND[1]*100)}% of plant height)\n"
    f"Corridor width: ±{CORRIDOR_HALF_PX}px  |  "
    "Green = pixel present  |  Red = gap  |  Yellow = corridor boundary\n"
    f"Gaps counted only if ≥ {MIN_GAP_SIZE} consecutive empty rows",
    fontsize=9, y=1.01,
)
plt.tight_layout()
plt.savefig(str(OUT_PATH), dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_PATH}")

print(f"\n{'Frame':<45} {'gaps':>5} {'max_gap':>8} {'continuity':>11} {'plant_h':>8} {'cx':>6}")
print("-" * 85)
for r in results:
    flag = " <-- BROKEN" if r["n_gaps"] > 0 else ""
    print(
        f"{r['name']:<45} {r['n_gaps']:>5} {r['max_gap_px']:>8} "
        f"{r['continuity_pct']:>10.1f}% {r['plant_h_px']:>8} {r['cx']:>6}{flag}"
    )
