"""
_tmp_skeleton_pruning_demo.py
Demonstrates iterative spur pruning on the raw Sobel skeleton.

A "spur" is a dead-end branch (starts at an endpoint, ends at a junction or
another endpoint) shorter than MIN_SPUR_LEN pixels.  Removing spurs reduces
noise junctions while preserving long structural branches (stem, petioles,
leaves).

Algorithm (applied iteratively until convergence):
  1. Remove isolated connected components smaller than MIN_COMPONENT_PX.
  2. Find all skeleton endpoints (degree 1).
  3. From each endpoint, trace the branch pixel-by-pixel until:
       - A junction (degree >= 3) is reached   -> spur candidate
       - Another endpoint is reached            -> isolated short branch
       - Path length exceeds MIN_SPUR_LEN       -> keep (long enough)
  4. If path length < MIN_SPUR_LEN: delete those pixels from the skeleton.
  5. Repeat from step 2 until no more spurs are found in a full pass.
     (Removing a spur may demote its junction to a path pixel, exposing new
      short spurs in the next iteration.)

Layout: N_SAMPLES rows x 5 columns

  Col 1  RAW SKELETON        no ops at all (baseline)
  Col 2  PRUNED  20 px       spur threshold = 20 px
  Col 3  PRUNED  40 px       spur threshold = 40 px
  Col 4  PRUNED  80 px       spur threshold = 80 px
  Col 5  CORRIDOR STRIP      zoomed 80px corridor:
                              RAW | 20px | 40px | 80px side-by-side

Colour coding (all columns):
  Cyan   = skeleton pixel (uniform — no degree markers)
  Yellow = corridor boundary lines (cx +/- 40 px)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from skimage.morphology import skeletonize as sk_skeletonize
from scipy.ndimage import label as scipy_label

# ── Paths ──────────────────────────────────────────────────────────────────
MASKS_DIR = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465\masks"
)
OUT_PATH = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465"
    r"\skeleton_pruning_demo.png"
)

# ── Tunable parameters ─────────────────────────────────────────────────────
N_SAMPLES        = 3
CORRIDOR_HALF_PX = 40
EXPAND_PX        = 200
BOTTOM_FRAC      = 0.15
PROJ_BAND        = (0.20, 0.85)
ANCHOR_MARGIN    = 150
SMOOTH_K         = 15

SPUR_LENGTHS     = [20, 40, 80]   # three pruning thresholds to compare
MIN_COMPONENT_PX = 10             # remove isolated fragments smaller than this
MAX_PRUNE_ITER   = 20             # safety cap on iterations

MARGIN = 20
DPI    = 180


# ══════════════════════════════════════════════════════════════════════════
# Corridor centre estimation (unchanged from other scripts)
# ══════════════════════════════════════════════════════════════════════════

def col_max_run_lengths(b):
    mx = np.zeros(b.shape[1], np.int32)
    cur = np.zeros(b.shape[1], np.int32)
    for row in b:
        cur = np.where(row, cur + 1, 0)
        np.maximum(mx, cur, out=mx)
    return mx


def estimate_stem_cx(mask, y0, y1, x0, x1):
    ph = y1 - y0
    yc = int(y1 - BOTTOM_FRAC * ph)
    st = mask[max(0, yc):y1+1, x0:x1+1]
    _, xs = np.where(st > 127)
    ca = (x0 + int(np.median(xs))) if len(xs) else (x0 + x1) // 2
    sx0 = max(x0, ca - ANCHOR_MARGIN); sx1 = min(x1, ca + ANCHOR_MARGIN)
    bt  = int(y0 + PROJ_BAND[0] * ph); bb  = int(y0 + PROJ_BAND[1] * ph)
    band = mask[bt:bb+1, sx0:sx1+1] > 127
    if not band.any():
        return ca
    proj = col_max_run_lengths(band).astype(float)
    sm   = np.convolve(proj, np.ones(SMOOTH_K) / SMOOTH_K, mode='same')
    return sx0 + int(np.argmax(sm))


# ══════════════════════════════════════════════════════════════════════════
# Spur pruning
# ══════════════════════════════════════════════════════════════════════════

def _degree_map(pos_set):
    """Return {(y,x): degree} for all pixels in pos_set."""
    return {
        (y, x): sum(
            1 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0) and (y + dy, x + dx) in pos_set
        )
        for (y, x) in pos_set
    }


def _trace_spur(start_yx, pos_set, deg_map, min_spur_len):
    """
    Trace from an endpoint (degree 1) along the skeleton.
    Returns the list of pixels in the branch (including start) if the
    branch is shorter than min_spur_len and ends at a junction or another
    endpoint; otherwise returns an empty list (branch is long enough).
    """
    path = [start_yx]
    prev = None
    cur  = start_yx

    while len(path) < min_spur_len:
        cy, cx = cur
        nexts = [
            (cy + dy, cx + dx)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0)
            and (cy + dy, cx + dx) in pos_set
            and (cy + dy, cx + dx) != prev
        ]

        if not nexts:
            # Reached a dead-end (isolated fragment or dangling pixel)
            return path          # short -> remove

        nxt = nexts[0]

        if deg_map.get(nxt, 0) >= 3:
            # Next pixel is a junction -> this IS a spur (don't include junction)
            return path          # short -> remove

        prev = cur
        cur  = nxt
        path.append(cur)

    return []   # branch >= min_spur_len -> keep


def prune_skeleton(skel_u8, min_spur_len=20, min_component_px=10,
                   max_iter=MAX_PRUNE_ITER):
    """
    Remove short spur branches from a skeleton image (uint8, 0/255).

    Steps:
      1. Remove isolated connected components < min_component_px pixels.
      2. Iteratively find and remove endpoint branches < min_spur_len px.

    Returns pruned skeleton as uint8 (0/255).
    """
    # ── Step 1: remove tiny isolated fragments ────────────────────────────
    binary = skel_u8 > 127
    labeled, n_comp = scipy_label(binary, structure=np.ones((3, 3)))
    sizes = np.bincount(labeled.ravel())
    keep_mask = sizes >= min_component_px
    keep_mask[0] = False   # background
    binary = keep_mask[labeled]

    # ── Step 2: iterative spur pruning ────────────────────────────────────
    pos_set = set(zip(*np.where(binary))) if binary.any() else set()

    for iteration in range(max_iter):
        if not pos_set:
            break

        deg_map = _degree_map(pos_set)

        # Collect all endpoints
        endpoints = [p for p, d in deg_map.items() if d == 1]
        if not endpoints:
            break

        spurs_to_remove = set()
        for ep in endpoints:
            if ep not in pos_set:
                continue   # already removed in this pass
            branch = _trace_spur(ep, pos_set, deg_map, min_spur_len)
            spurs_to_remove.update(branch)

        if not spurs_to_remove:
            break

        pos_set -= spurs_to_remove

    # ── Reconstruct image ──────────────────────────────────────────────────
    result = np.zeros_like(skel_u8)
    for (y, x) in pos_set:
        result[y, x] = 255
    return result


# ══════════════════════════════════════════════════════════════════════════
# Skeleton visualisation helpers
# ══════════════════════════════════════════════════════════════════════════

def color_skeleton(skel_u8, mask_bg=None):
    """Paint every skeleton pixel cyan — uniform colour, no junction markers."""
    H, W = skel_u8.shape
    vis  = np.zeros((H, W, 3), dtype=np.uint8)
    if mask_bg is not None:
        dim = (mask_bg // 8).astype(np.uint8)
        vis[..., 0] = dim; vis[..., 1] = dim; vis[..., 2] = dim

    ys, xs = np.where(skel_u8 > 127)
    vis[ys, xs] = [0, 200, 200]   # uniform cyan

    return vis


def draw_corridor_lines(vis, y0, y1, cx0, cx1):
    vis[y0:y1+1, cx0] = [255, 220, 0]
    vis[y0:y1+1, cx1] = [255, 220, 0]


def skel_stats(skel_u8, y0, y1, cx0, cx1):
    """Return (total_px, total_junctions, corridor_px, corridor_junctions)."""
    pts     = np.argwhere(skel_u8 > 127)
    pos_set = set((int(p[0]), int(p[1])) for p in pts)
    total_px  = len(pts)
    total_jct = sum(
        1 for (y, x) in pos_set
        if sum(1 for dy in (-1,0,1) for dx in (-1,0,1)
               if not (dy==0 and dx==0) and (y+dy,x+dx) in pos_set) >= 3
    )
    corr_pts = [(y, x) for (y, x) in pos_set
                if y0 <= y <= y1 and cx0 <= x <= cx1]
    corr_set = set(corr_pts)
    corr_jct = sum(
        1 for (y, x) in corr_set
        if sum(1 for dy in (-1,0,1) for dx in (-1,0,1)
               if not (dy==0 and dx==0) and (y+dy,x+dx) in pos_set) >= 3
    )
    return total_px, total_jct, len(corr_pts), corr_jct


def make_corridor_strip(skels_u8, y0, y1, cx0, cx1):
    """
    Side-by-side corridor strip for each skeleton in skels_u8.
    Returns RGB image (plant_h, n*(corr_w+3)-3, 3).
    """
    plant_h = y1 - y0 + 1
    corr_w  = cx1 - cx0 + 1
    DIV     = 3
    n       = len(skels_u8)
    total_w = n * corr_w + (n - 1) * DIV

    strip = np.zeros((plant_h, total_w, 3), dtype=np.uint8)

    for k, skel in enumerate(skels_u8):
        x_off   = k * (corr_w + DIV)
        c_skel  = skel[y0:y1+1, cx0:cx1+1]
        pos_set = set(zip(*np.where(c_skel > 127))) if (c_skel > 127).any() else set()

        for (r, c) in pos_set:
            strip[r, x_off + c] = [0, 200, 200]   # uniform cyan

        if k < n - 1:
            strip[:, x_off + corr_w : x_off + corr_w + DIV] = [60, 60, 60]

    return strip


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

masks   = sorted(MASKS_DIR.glob("*_mask_refined.png"))
if not masks:
    raise FileNotFoundError(f"No masks in {MASKS_DIR}")
step    = max(1, len(masks) // N_SAMPLES)
sampled = masks[::step][:N_SAMPLES]
tss     = [m.stem.replace("tl_","").replace("_mask_refined","") for m in sampled]
print(f"Total masks: {len(masks)}  |  sampling {len(sampled)}")
print(f"Frames: {tss}\n")

N_COLS  = 5   # raw + 3 pruned + corridor strip
fig = plt.figure(figsize=(34, N_SAMPLES * 9))
gs  = gridspec.GridSpec(
    N_SAMPLES, N_COLS,
    figure=fig,
    width_ratios=[4, 4, 4, 4, 2],
    wspace=0.05, hspace=0.24,
)
axes = np.array([[fig.add_subplot(gs[r, c])
                  for c in range(N_COLS)]
                 for r in range(N_SAMPLES)])

spur_labels = ["RAW\n(no pruning)"] + [f"PRUNED {s}px" for s in SPUR_LENGTHS]

COL_TITLES = [
    f"Col 1 — RAW SKELETON\n"
    f"skeletonize(raw Sobel), zero ops\n"
    f"Baseline — maximum noise",

    f"Col 2 — PRUNED  {SPUR_LENGTHS[0]} px\n"
    f"Remove spurs < {SPUR_LENGTHS[0]} px\n"
    f"Removes tiny Sobel noise stubs",

    f"Col 3 — PRUNED  {SPUR_LENGTHS[1]} px\n"
    f"Remove spurs < {SPUR_LENGTHS[1]} px\n"
    f"Retains petioles, removes small leaf fragments",

    f"Col 4 — PRUNED  {SPUR_LENGTHS[2]} px\n"
    f"Remove spurs < {SPUR_LENGTHS[2]} px\n"
    f"Retains only major structural branches",

    f"Col 5\nCORRIDOR STRIP\n(80px wide)\n\n"
    + "\n".join(f"[{l}]" for l in spur_labels),
]

for row, (mask_path, ts) in enumerate(zip(sampled, tss)):
    print(f"[{row+1}/{N_SAMPLES}] {ts}")
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("  READ ERROR"); continue

    ys, xs = np.where(mask > 127)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cx      = estimate_stem_cx(mask, y0, y1, x0, x1)
    cx0     = max(0,               cx - CORRIDOR_HALF_PX)
    cx1     = min(mask.shape[1]-1, cx + CORRIDOR_HALF_PX)
    exp_x0  = max(0,               cx0 - EXPAND_PX)
    exp_x1  = min(mask.shape[1]-1, cx1 + EXPAND_PX)

    # Raw skeleton (no ops, restricted to expanded zone)
    raw_zone = np.zeros_like(mask)
    raw_zone[:, exp_x0:exp_x1+1] = mask[:, exp_x0:exp_x1+1]
    skel_raw = sk_skeletonize(raw_zone > 127).astype(np.uint8) * 255

    # Pruned skeletons
    skels = [skel_raw]
    for spur_len in SPUR_LENGTHS:
        pruned = prune_skeleton(skel_raw, min_spur_len=spur_len,
                                min_component_px=MIN_COMPONENT_PX)
        skels.append(pruned)

    # Stats
    print(f"  {'Approach':<20}  {'total_px':>8}  {'total_jct':>10}  "
          f"{'corr_px':>8}  {'corr_jct':>9}")
    labels = ["raw"] + [f"pruned_{s}px" for s in SPUR_LENGTHS]
    for lbl, sk in zip(labels, skels):
        tp, tj, cp, cj = skel_stats(sk, y0, y1, cx0, cx1)
        print(f"  {lbl:<20}  {tp:>8}  {tj:>10}  {cp:>8}  {cj:>9}")
    print()

    # Colour each skeleton
    vis_skels = [color_skeleton(sk, mask_bg=mask) for sk in skels]
    for vs in vis_skels:
        draw_corridor_lines(vs, y0, y1, cx0, cx1)

    # Corridor strip
    strip = make_corridor_strip(skels, y0, y1, cx0, cx1)

    # Crop window
    yc0 = max(0,             y0     - MARGIN)
    yc1 = min(mask.shape[0], y1     + MARGIN)
    xc0 = max(0,             exp_x0 - MARGIN)
    xc1 = min(mask.shape[1], exp_x1 + MARGIN)

    def crop(img):
        return img[yc0:yc1, xc0:xc1]

    for col, (vs, ax) in enumerate(zip(vis_skels, axes[row, :4])):
        ax.imshow(crop(vs))
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(ts, fontsize=9, rotation=0, labelpad=135,
                          va="center", fontweight="bold")
        if row == 0:
            ax.set_title(COL_TITLES[col], fontsize=7.5, pad=5,
                         loc="center", linespacing=1.5)
        # Per-panel stats subtitle
        tp, tj, cp, cj = skel_stats(skels[col], y0, y1, cx0, cx1)
        ax.set_xlabel(
            f"total: {tp}px  {tj}jct  |  corridor: {cp}px  {cj}jct",
            fontsize=6.5, labelpad=3
        )

    # Col 5: corridor strip
    ax5 = axes[row, 4]
    ax5.imshow(strip, aspect="auto")
    ax5.axis("off")
    corr_w  = cx1 - cx0 + 1
    DIV     = 3
    strip_labels = ["RAW", f"{SPUR_LENGTHS[0]}px", f"{SPUR_LENGTHS[1]}px", f"{SPUR_LENGTHS[2]}px"]
    colors        = ["white", "cyan", "cyan", "orange"]
    for k, (lbl, col_lbl) in enumerate(zip(strip_labels, colors)):
        mid_x = k * (corr_w + DIV) + corr_w // 2
        ax5.text(mid_x, -4, lbl, ha="center", va="bottom",
                 fontsize=5, color=col_lbl)
    if row == 0:
        ax5.set_title(COL_TITLES[4], fontsize=7, pad=5,
                      loc="center", linespacing=1.3)

# ── Legend ─────────────────────────────────────────────────────────────────
fig.subplots_adjust(bottom=0.07)
leg = [
    mpatches.Patch(color=(0, 200/255, 200/255),
                   label="Skeleton pixel (uniform cyan)"),
    mpatches.Patch(color=(255/255, 220/255, 0),
                   label=f"Corridor boundary  cx +/- {CORRIDOR_HALF_PX}px  [yellow]"),
]
fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=8,
           framealpha=0.85, handlelength=1.4, columnspacing=2.0,
           bbox_to_anchor=(0.5, 0.0))

plt.suptitle(
    "Skeleton spur pruning demo — raw Sobel skeleton with iterative endpoint-branch removal\n"
    f"Algorithm: (1) remove isolated components < {MIN_COMPONENT_PX}px,  "
    f"(2) iteratively remove dead-end branches < threshold  "
    f"(up to {MAX_PRUNE_ITER} passes until convergence)\n"
    f"Spur thresholds compared: 0 (raw) | {SPUR_LENGTHS[0]}px | "
    f"{SPUR_LENGTHS[1]}px | {SPUR_LENGTHS[2]}px",
    fontsize=9, y=1.003,
)

plt.savefig(str(OUT_PATH), dpi=DPI, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PATH}")
