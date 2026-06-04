"""
_tmp_raw_skeleton_demo.py
Illustrates what skimage.morphology.skeletonize produces on the raw Sobel
edge mask, with zero / minimal / H-Bridge pre-processing.

Layout: N_SAMPLES rows x 5 columns

  Col 1  RAW MASK       : original Sobel mask + corridor (yellow)
  Col 2  RAW SKELETON   : skeletonize on the raw mask — NO morph ops at all
                          Two disconnected parallel lines in the corridor gap
  Col 3  3x1 PATCH      : skeletonize after MORPH_RECT(3,1) noise seal only
                          Gap still 25-39px; corridor still disconnected
  Col 4  H-BRIDGE 43x1  : skeletonize after MORPH_RECT(43,1) horizontal close
                          Gap filled; corridor becomes one connected path
  Col 5  CORRIDOR STRIP : zoomed 80px-wide corridor (full plant height)
                          three sub-strips side by side:  RAW | 3x1 | H-Bridge
                          Clearly shows the disconnected vs connected result

Skeleton colour coding (Cols 2-4 and strip):
  Cyan  = path pixel     (degree 2 — interior of a branch)
  Red   = junction pixel (degree >= 3 — where branches meet)
  Blue  = endpoint pixel (degree 1 — tip of a branch)
  Yellow lines = corridor boundaries (cx +/- 40 px)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from skimage.morphology import skeletonize as sk_skeletonize

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
    r"\raw_skeleton_demo.png"
)

# ── Parameters ─────────────────────────────────────────────────────────────
N_SAMPLES        = 3
CORRIDOR_HALF_PX = 40      # cx +/- this = corridor (80 px wide)
EXPAND_PX        = 200     # flanks shown in Cols 1-4
BOTTOM_FRAC      = 0.15
PROJ_BAND        = (0.20, 0.85)
ANCHOR_MARGIN    = 150
SMOOTH_K         = 15
MARGIN           = 20      # border around the plant bbox in plots
DPI              = 180


# ══════════════════════════════════════════════════════════════════════════
# Corridor centre estimation
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
    sx0 = max(x0, ca - ANCHOR_MARGIN)
    sx1 = min(x1, ca + ANCHOR_MARGIN)
    bt  = int(y0 + PROJ_BAND[0] * ph)
    bb  = int(y0 + PROJ_BAND[1] * ph)
    band = mask[bt:bb+1, sx0:sx1+1] > 127
    if not band.any():
        return ca
    proj = col_max_run_lengths(band).astype(float)
    sm   = np.convolve(proj, np.ones(SMOOTH_K) / SMOOTH_K, mode='same')
    return sx0 + int(np.argmax(sm))


# ══════════════════════════════════════════════════════════════════════════
# Mask builders
# ══════════════════════════════════════════════════════════════════════════

def build_expanded_zone(mask, exp_x0, exp_x1):
    """Return a mask restricted to the expanded zone (corridor + flanks)."""
    out = np.zeros_like(mask)
    out[:, exp_x0:exp_x1+1] = mask[:, exp_x0:exp_x1+1]
    return out


def apply_rect_close(zone, y0, y1, cx0, cx1, kw):
    """
    Apply MORPH_CLOSE with a horizontal MORPH_RECT(kw, 1) kernel to the
    corridor region only.  The flanks are left untouched.
    kw=3  -> seals 1-px noise gaps (no effect on the ~25-39 px stem gap)
    kw=43 -> bridges the full stem gap without any vertical expansion
    """
    out = zone.copy()
    k   = kw if kw % 2 == 1 else kw + 1
    el  = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    corr = zone[y0:y1+1, cx0:cx1+1]
    out[y0:y1+1, cx0:cx1+1] = cv2.morphologyEx(corr, cv2.MORPH_CLOSE, el)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Skeleton visualisation
# ══════════════════════════════════════════════════════════════════════════

def color_skeleton(skel_u8, mask_bg=None):
    """
    Return an RGB image with skeleton pixels coloured by their degree:
      degree 1  -> blue  [80, 80, 255]   endpoint / tip
      degree 2  -> cyan  [0, 200, 200]   simple path pixel
      degree 3+ -> red   [255, 50, 30]   junction (3x3 blob for visibility)
    Non-skeleton pixels: dim background from mask_bg (optional).
    """
    H, W = skel_u8.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    if mask_bg is not None:
        dim = (mask_bg // 8).astype(np.uint8)
        vis[:, :, 0] = dim
        vis[:, :, 1] = dim
        vis[:, :, 2] = dim

    pts = np.argwhere(skel_u8 > 127)
    if len(pts) == 0:
        return vis

    # Build position lookup for fast degree computation
    pos_set = set((int(p[0]), int(p[1])) for p in pts)

    for p in pts:
        y, x = int(p[0]), int(p[1])
        deg = sum(
            1 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0) and (y + dy, x + dx) in pos_set
        )
        if deg == 1:
            vis[y, x] = [80, 80, 255]      # endpoint
        elif deg >= 3:
            # Mark a 3x3 blob so junctions are visible at small scale
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    vy, vx = y + dy, x + dx
                    if 0 <= vy < H and 0 <= vx < W:
                        vis[vy, vx] = [255, 50, 30]   # junction
        else:
            vis[y, x] = [0, 200, 200]      # path


    return vis


def draw_corridor_lines(vis, y0, y1, cx0, cx1):
    """Overlay yellow vertical lines at the corridor boundaries."""
    vis[y0:y1+1, cx0] = [255, 220, 0]
    vis[y0:y1+1, cx1] = [255, 220, 0]


def make_corridor_strip(skels_u8, labels, y0, y1, cx0, cx1):
    """
    Build a composite corridor-strip image showing all skeletons side by side.
    Each sub-strip is the corridor region (cx0..cx1) for one skeleton approach,
    coloured by degree.  Sub-strips are separated by a 3-px grey divider.

    Returns an RGB image of shape (plant_h, n * corr_w + (n-1)*3, 3).
    """
    plant_h = y1 - y0 + 1
    corr_w  = cx1 - cx0 + 1
    DIVIDER = 3
    n       = len(skels_u8)
    total_w = n * corr_w + (n - 1) * DIVIDER

    strip = np.zeros((plant_h, total_w, 3), dtype=np.uint8)

    for k, skel in enumerate(skels_u8):
        x_off = k * (corr_w + DIVIDER)

        # Colour skeleton pixels in the corridor slice
        corr_skel = skel[y0:y1+1, cx0:cx1+1]
        pos_set   = set(zip(*np.where(corr_skel > 127)))

        for r in range(plant_h):
            for c in range(corr_w):
                if corr_skel[r, c] <= 127:
                    continue
                deg = sum(
                    1 for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if not (dr == 0 and dc == 0) and (r + dr, c + dc) in pos_set
                )
                col = (
                    [80, 80, 255]  if deg == 1 else
                    [255, 50, 30]  if deg >= 3 else
                    [0, 200, 200]
                )
                # For junctions: 3x3 blob
                if deg >= 3:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            sr = r + dr; sc = x_off + c + dc
                            if 0 <= sr < plant_h and 0 <= sc < total_w:
                                strip[sr, sc] = col
                else:
                    strip[r, x_off + c] = col

        # Grey divider after each strip (except last)
        if k < n - 1:
            strip[:, x_off + corr_w : x_off + corr_w + DIVIDER] = [70, 70, 70]

    return strip


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

masks   = sorted(MASKS_DIR.glob("*_mask_refined.png"))
if not masks:
    raise FileNotFoundError(f"No masks found in {MASKS_DIR}")
step    = max(1, len(masks) // N_SAMPLES)
sampled = masks[::step][:N_SAMPLES]
tss     = [m.stem.replace("tl_", "").replace("_mask_refined", "") for m in sampled]
print(f"Total masks: {len(masks)}  |  sampling {len(sampled)}")
print(f"Frames: {tss}\n")

# ── Figure layout ──────────────────────────────────────────────────────────
# 5 columns; Col 5 (corridor strip) is narrower than the others
fig = plt.figure(figsize=(34, N_SAMPLES * 9))
gs  = gridspec.GridSpec(
    N_SAMPLES, 5,
    figure=fig,
    width_ratios=[4, 4, 4, 4, 1.5],   # Col 5 is narrower (strip zoom)
    wspace=0.05, hspace=0.22
)
axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(5)]
                 for r in range(N_SAMPLES)])

COL_TITLES = [
    "Col 1 — RAW MASK\n"
    "Original Sobel edge mask\n"
    f"Corridor +/-{CORRIDOR_HALF_PX}px (yellow lines)\n"
    f"Flanks +/-{EXPAND_PX}px shown",

    "Col 2 — RAW SKELETON\n"
    "skeletonize(raw mask) — zero morph ops\n"
    "Corridor: two DISCONNECTED parallel lines\n"
    "Gap (~25-39px) remains dark / empty",

    "Col 3 — 3x1 NOISE PATCH\n"
    "MORPH_RECT(3,1) close, then skeletonize\n"
    "Seals only 1-2px noise gaps\n"
    "Corridor still DISCONNECTED (gap too wide)",

    "Col 4 — H-BRIDGE 43x1\n"
    "MORPH_RECT(43,1) close, then skeletonize\n"
    "Bridges full stem gap horizontally\n"
    "Corridor: one CONNECTED path",

    "Col 5\nCORRIDOR\nSTRIP\nZOOM\n\n"
    "[RAW]\n[3x1]\n[43x1]",
]

for row, (mask_path, ts) in enumerate(zip(sampled, tss)):
    print(f"[{row+1}/{N_SAMPLES}] {ts} ...", end=" ", flush=True)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("READ ERROR"); continue

    ys, xs = np.where(mask > 127)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cx      = estimate_stem_cx(mask, y0, y1, x0, x1)
    cx0     = max(0,               cx - CORRIDOR_HALF_PX)
    cx1     = min(mask.shape[1]-1, cx + CORRIDOR_HALF_PX)
    exp_x0  = max(0,               cx0 - EXPAND_PX)
    exp_x1  = min(mask.shape[1]-1, cx1 + EXPAND_PX)

    # ── Build three mask variants ──────────────────────────────────────────
    zone_raw  = build_expanded_zone(mask, exp_x0, exp_x1)
    zone_3x1  = apply_rect_close(zone_raw,  y0, y1, cx0, cx1, kw=3)
    zone_43x1 = apply_rect_close(zone_raw,  y0, y1, cx0, cx1, kw=43)

    # ── Skeletonise each ───────────────────────────────────────────────────
    skel_raw  = sk_skeletonize(zone_raw  > 127).astype(np.uint8) * 255
    skel_3x1  = sk_skeletonize(zone_3x1  > 127).astype(np.uint8) * 255
    skel_43x1 = sk_skeletonize(zone_43x1 > 127).astype(np.uint8) * 255

    # Count corridor skeleton pixels for each approach
    def corr_stats(skel):
        c = skel[y0:y1+1, cx0:cx1+1]
        n_px = int((c > 127).sum())
        pts = np.argwhere(c > 127)
        if len(pts) == 0:
            return n_px, 0
        pos_s = set((int(p[0]), int(p[1])) for p in pts)
        n_jct = sum(
            1 for p in pts
            if sum(1 for dr in (-1,0,1) for dc in (-1,0,1)
                   if not (dr==0 and dc==0) and (int(p[0])+dr, int(p[1])+dc) in pos_s) >= 3
        )
        return n_px, n_jct

    px_r, jct_r = corr_stats(skel_raw)
    px_3, jct_3 = corr_stats(skel_3x1)
    px_h, jct_h = corr_stats(skel_43x1)
    print(f"corridor skel px/jct:  raw={px_r}/{jct_r}  3x1={px_3}/{jct_3}  43x1={px_h}/{jct_h}")

    # ── Colour skeletons ───────────────────────────────────────────────────
    vis_raw  = color_skeleton(skel_raw,  mask_bg=mask)
    vis_3x1  = color_skeleton(skel_3x1,  mask_bg=mask)
    vis_43x1 = color_skeleton(skel_43x1, mask_bg=mask)

    for vis in (vis_raw, vis_3x1, vis_43x1):
        draw_corridor_lines(vis, y0, y1, cx0, cx1)

    # ── Corridor strip zoom (Col 5) ────────────────────────────────────────
    strip = make_corridor_strip(
        [skel_raw, skel_3x1, skel_43x1],
        ["RAW", "3x1", "43x1"],
        y0, y1, cx0, cx1
    )

    # ── Crop window (same for Cols 1-4) ───────────────────────────────────
    yc0 = max(0,             y0     - MARGIN)
    yc1 = min(mask.shape[0], y1     + MARGIN)
    xc0 = max(0,             exp_x0 - MARGIN)
    xc1 = min(mask.shape[1], exp_x1 + MARGIN)

    def crop(img):
        return img[yc0:yc1, xc0:xc1]

    # ── Col 1: raw mask ───────────────────────────────────────────────────
    vis_mask = np.stack([(mask // 5)] * 3, axis=2).astype(np.uint8)
    draw_corridor_lines(vis_mask, y0, y1, cx0, cx1)
    # dashed expand boundary
    for y in range(y0, y1+1):
        if (y // 8) % 2 == 0:
            for bx in (exp_x0, exp_x1):
                if 0 <= bx < vis_mask.shape[1]:
                    vis_mask[y, bx] = [160, 160, 160]

    # ── Plot ──────────────────────────────────────────────────────────────
    panels = [crop(vis_mask), crop(vis_raw), crop(vis_3x1), crop(vis_43x1)]
    for col, (panel, ax) in enumerate(zip(panels, axes[row, :4])):
        ax.imshow(panel)
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(ts, fontsize=9, rotation=0, labelpad=135,
                          va="center", fontweight="bold")
        if row == 0:
            ax.set_title(COL_TITLES[col], fontsize=7.5, pad=5,
                         loc="center", linespacing=1.5)

    # Col 5: corridor strip
    ax5 = axes[row, 4]
    ax5.imshow(strip, aspect="auto")
    ax5.axis("off")
    # Label the three sub-strips at the top
    corr_w  = cx1 - cx0 + 1
    DIVIDER = 3
    mid_raw = corr_w // 2
    mid_3x1 = corr_w + DIVIDER + corr_w // 2
    mid_43x = 2 * (corr_w + DIVIDER) + corr_w // 2
    ax5.text(mid_raw, -2, "RAW",   ha="center", va="bottom", fontsize=5.5, color="cyan")
    ax5.text(mid_3x1, -2, "3x1",   ha="center", va="bottom", fontsize=5.5, color="cyan")
    ax5.text(mid_43x, -2, "43x1",  ha="center", va="bottom", fontsize=5.5, color="orange")
    if row == 0:
        ax5.set_title(COL_TITLES[4], fontsize=7, pad=5, loc="center", linespacing=1.4)

# ── Legend ─────────────────────────────────────────────────────────────────
fig.subplots_adjust(bottom=0.07)
leg = [
    mpatches.Patch(color=(0,   200/255, 200/255), label="Path pixel  (degree 2 — interior of branch)"),
    mpatches.Patch(color=(255/255, 50/255,  30/255), label="Junction     (degree >= 3 — branch point / petiole attachment)"),
    mpatches.Patch(color=(80/255,  80/255, 255/255), label="Endpoint     (degree 1 — tip of branch)"),
    mpatches.Patch(color=(255/255, 220/255, 0),      label="Corridor boundary  cx +/- 40 px  [yellow]"),
    mpatches.Patch(color=(160/255, 160/255, 160/255),label="Expand boundary  +/- 200 px  [grey dashed, Col 1 only]"),
]
fig.legend(handles=leg, loc="lower center", ncol=5, fontsize=7.5,
           framealpha=0.85, handlelength=1.4, columnspacing=1.8,
           bbox_to_anchor=(0.5, 0.0))

plt.suptitle(
    "Raw Sobel skeletonization demo\n"
    "skimage.morphology.skeletonize (Lee 1994) applied with zero / minimal / H-Bridge pre-processing\n"
    f"Corridor: cx +/-{CORRIDOR_HALF_PX}px  |  "
    f"Flanks: +/-{EXPAND_PX}px  |  "
    "Gap measured at 25-39px (May 7) -> MORPH_RECT(43,1) needed to bridge",
    fontsize=9, y=1.003
)

plt.savefig(str(OUT_PATH), dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_PATH}")
