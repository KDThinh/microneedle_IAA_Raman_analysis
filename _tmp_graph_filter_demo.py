"""
_tmp_graph_filter_demo.py

Graph-level branch length filtering on raw Sobel skeletons, with optional
morphological opening pre-processing to separate fused Sobel bands.

After skeletonization the skeleton is treated as a graph:
  • Nodes  = junction pixels (degree >= 3) or endpoint pixels (degree 1)
  • Edges  = chains of degree-2 pixels connecting two nodes
  • Loops  = closed cycles of degree-2 pixels with no node (pure rings)

Any edge / loop shorter than MIN_BRANCH_LEN px is removed entirely.
This removes the web mesh (short junction→junction arcs) that spur pruning
cannot touch, because those arcs have no free endpoint to start from.

Layout: N_SAMPLES rows × 5 columns
  Col 1  Raw skeleton (baseline)
  Col 2  Open OPEN_SIZES[0] disk → skel → graph filter   (1 = no opening)
  Col 3  Open OPEN_SIZES[1] disk → skel → graph filter
  Col 4  Open OPEN_SIZES[2] disk → skel → graph filter
  Col 5  OR union of Col 2 + Col 3 + Col 4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  TUNABLE PARAMETERS  (edit the block below — everything else is fixed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  MIN_BRANCH_LEN        Graph-filter threshold applied to each individual
                        variant (Col 2, 3, 4) before the union.
                        Lower  → more permissive, retains short petioles.
                        Higher → cleaner, but risks erasing short petioles.
                        Suggested range: 5 – 30 px.

  MIN_BRANCH_LEN_UNION  Graph-filter threshold applied to the OR union
                        (Col 5) as a second cleanup pass.
                        The union can introduce short reconnection artefacts
                        where two slightly offset skeletons meet; this pass
                        removes them.  Can be set independently — e.g. keep
                        MIN_BRANCH_LEN small to preserve short petioles in
                        individual variants, then use a larger value here to
                        clean up the merged result.
                        Suggested range: 10 – 50 px.

  OPEN_SIZES       Three disk diameters (px) for Col 2, Col 3, Col 4.
                   Use 1 for "no opening" (identity — raw Sobel input).
                   Larger values separate fused Sobel bands before
                   skeletonizing, recovering features lost when nearby
                   edges merge, but risk erasing very thin petioles.
                   Example: [1, 3, 7]  →  no-op | small | large
                   Suggested range: 1 – 9 px.

  CORRIDOR_HALF_PX Half-width of the stem corridor used for the corridor
                   strip (Col 5). Should match your actual stem width / 2.

  N_SAMPLES        How many evenly-spaced frames to sample from the dataset.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
    r"\graph_filter_demo.png"
)

# ══════════════════════════════════════════════════════════════════════════
# ▼▼▼  EDIT THESE  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════

MIN_BRANCH_LEN       = 10   # px — graph-filter threshold for each variant (Col 2-4)
MIN_BRANCH_LEN_UNION = 10   # px — graph-filter threshold for the OR union (Col 5)
                             #      can be larger than MIN_BRANCH_LEN to clean up
                             #      reconnection artefacts introduced by the union
OPEN_SIZES       = [3, 5, 7]  # disk diameters for Col 2, 3, 4  (1 = no opening)
CORRIDOR_HALF_PX = 40      # px — half-width of stem corridor
N_SAMPLES        = 3       # number of frames to sample

# ══════════════════════════════════════════════════════════════════════════
# ▲▲▲  END OF USER SETTINGS  ▲▲▲  (no need to touch anything below)
# ══════════════════════════════════════════════════════════════════════════

EXPAND_PX     = 200
BOTTOM_FRAC   = 0.15
PROJ_BAND     = (0.20, 0.85)
ANCHOR_MARGIN = 150
SMOOTH_K      = 15
MARGIN        = 20
DPI           = 180


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
    sx0 = max(x0, ca - ANCHOR_MARGIN); sx1 = min(x1, ca + ANCHOR_MARGIN)
    bt  = int(y0 + PROJ_BAND[0] * ph); bb  = int(y0 + PROJ_BAND[1] * ph)
    band = mask[bt:bb+1, sx0:sx1+1] > 127
    if not band.any():
        return ca
    proj = col_max_run_lengths(band).astype(float)
    sm   = np.convolve(proj, np.ones(SMOOTH_K) / SMOOTH_K, mode='same')
    return sx0 + int(np.argmax(sm))


# ══════════════════════════════════════════════════════════════════════════
# Core: graph-level branch length filter
# ══════════════════════════════════════════════════════════════════════════

def _nb8(y, x, pos_set):
    """8-connected neighbours of (y,x) that exist in pos_set."""
    return [(y+dy, x+dx)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0) and (y+dy, x+dx) in pos_set]


def filter_by_branch_length(skel_u8, min_branch_len=50):
    """
    Remove all skeleton branches (endpoint→junction, junction→junction arcs,
    and pure degree-2 loop cycles) shorter than min_branch_len pixels.

    Algorithm:
      1. Compute degree for every skeleton pixel.
      2. 'Node' pixels = degree != 2  (endpoints=1, junctions>=3, isolated=0).
      3. From every node, trace along degree-2 pixels to the next node.
         Keep the branch if len >= min_branch_len.
      4. Separately: find connected components of degree-2 pixels that form
         closed loops (no node). Keep if circumference >= min_branch_len.
      5. Keep a node pixel only if it connects to at least one kept branch.
    """
    binary = skel_u8 > 127
    if not binary.any():
        return np.zeros_like(skel_u8)

    pos_set = set(zip(*np.where(binary)))

    # ── Degree map ────────────────────────────────────────────────────────
    deg = {p: len(_nb8(*p, pos_set)) for p in pos_set}

    # ── Node pixels (degree != 2) ─────────────────────────────────────────
    node_set = {p for p, d in deg.items() if d != 2}

    keep         = set()
    visited_edge = set()   # (node, first_step) — avoid tracing same branch twice

    # ── Trace every branch from each node ─────────────────────────────────
    for start in node_set:
        for nb in _nb8(*start, pos_set):
            key = (start, nb)
            if key in visited_edge:
                continue
            visited_edge.add(key)

            path = [start]
            prev, cur = start, nb

            while True:
                path.append(cur)

                if cur in node_set:
                    # reached the other end — mark reverse direction visited
                    visited_edge.add((cur, prev))
                    break

                nxts = [p for p in _nb8(*cur, pos_set) if p != prev]
                if not nxts:
                    # dangling tip — cur is effectively an endpoint
                    break

                prev, cur = cur, nxts[0]

            if len(path) >= min_branch_len:
                keep.update(path)

    # ── Handle pure degree-2 loops (closed rings with no node) ────────────
    deg2 = {p for p, d in deg.items() if d == 2}
    visited_loop = set()
    for seed in deg2:
        if seed in visited_loop or seed in keep:
            continue
        # BFS over the degree-2 connected component
        comp  = set()
        stack = [seed]
        while stack:
            p = stack.pop()
            if p in visited_loop:
                continue
            visited_loop.add(p)
            comp.add(p)
            for nb in _nb8(*p, pos_set):
                if nb in deg2 and nb not in visited_loop:
                    stack.append(nb)
        if len(comp) >= min_branch_len:
            keep.update(comp)

    # ── Keep node pixels that still connect to at least one kept pixel ────
    for node in node_set:
        if any(nb in keep for nb in _nb8(*node, pos_set)):
            keep.add(node)

    result = np.zeros_like(skel_u8)
    for (y, x) in keep:
        result[y, x] = 255
    return result


# ══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════

def color_skeleton(skel_u8, mask_bg=None):
    """Uniform cyan skeleton on dimmed mask background."""
    H, W = skel_u8.shape
    vis  = np.zeros((H, W, 3), dtype=np.uint8)
    if mask_bg is not None:
        dim = (mask_bg // 8).astype(np.uint8)
        vis[..., 0] = dim; vis[..., 1] = dim; vis[..., 2] = dim
    ys, xs = np.where(skel_u8 > 127)
    vis[ys, xs] = [0, 200, 200]
    return vis


def draw_corridor_lines(vis, y0, y1, cx0, cx1):
    vis[y0:y1+1, cx0] = [255, 220, 0]
    vis[y0:y1+1, cx1] = [255, 220, 0]


def skel_stats(skel_u8, y0, y1, cx0, cx1):
    """Return (total_px, total_junctions, corridor_px, corridor_junctions)."""
    pos_set  = set(zip(*np.where(skel_u8 > 127)))
    total_px = len(pos_set)
    total_jct = sum(
        1 for (y, x) in pos_set
        if sum(1 for dy in (-1,0,1) for dx in (-1,0,1)
               if not (dy==0 and dx==0) and (y+dy,x+dx) in pos_set) >= 3
    )
    corr_set = {(y, x) for (y, x) in pos_set if y0 <= y <= y1 and cx0 <= x <= cx1}
    corr_jct = sum(
        1 for (y, x) in corr_set
        if sum(1 for dy in (-1,0,1) for dx in (-1,0,1)
               if not (dy==0 and dx==0) and (y+dy,x+dx) in pos_set) >= 3
    )
    return total_px, total_jct, len(corr_set), corr_jct


def make_corridor_strip(skels_u8, y0, y1, cx0, cx1):
    """Side-by-side corridor strips for each skeleton. Returns RGB image."""
    plant_h = y1 - y0 + 1
    corr_w  = cx1 - cx0 + 1
    DIV     = 3
    n       = len(skels_u8)
    total_w = n * corr_w + (n - 1) * DIV
    strip   = np.zeros((plant_h, total_w, 3), dtype=np.uint8)

    for k, skel in enumerate(skels_u8):
        x_off  = k * (corr_w + DIV)
        c_skel = skel[y0:y1+1, cx0:cx1+1]
        ys, xs = np.where(c_skel > 127)
        for r, c in zip(ys, xs):
            strip[r, x_off + c] = [0, 200, 200]
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

N_COLS = 6
fig = plt.figure(figsize=(50, N_SAMPLES * 9))
gs  = gridspec.GridSpec(
    N_SAMPLES, N_COLS,
    figure=fig,
    width_ratios=[4, 4, 4, 4, 4, 4],
    wspace=0.05, hspace=0.24,
)
axes = np.array([[fig.add_subplot(gs[r, c])
                  for c in range(N_COLS)]
                 for r in range(N_SAMPLES)])

def _col_title(i, sz):
    label = f"no opening (sz=1)" if sz <= 1 else f"open {sz}x{sz} disk"
    return (f"Col {i+2} — {label}\n"
            f"+ graph filter {MIN_BRANCH_LEN} px\n"
            f"OPEN_SIZES[{i}] = {sz}")

COL_TITLES = (
    ["Col 1 — RAW SKELETON\nno preprocessing\nbaseline (maximum noise / web)"]
    + [_col_title(i, sz) for i, sz in enumerate(OPEN_SIZES)]
    + [f"Col 5 — UNION (before cleanup)\nOR of Col 2+3+4\nreconnection artefacts still present"]
    + [f"Col 6 — UNION + 2nd GF {MIN_BRANCH_LEN_UNION}px\nOR union after second graph filter\nreconnection artefacts removed"]
)

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

    # Restrict to expanded zone (same as other demo scripts)
    zone = np.zeros_like(mask)
    zone[:, exp_x0:exp_x1+1] = mask[:, exp_x0:exp_x1+1]

    # ── Col 1: raw skeleton ───────────────────────────────────────────────
    skel_raw = sk_skeletonize(zone > 127).astype(np.uint8) * 255

    # ── Cols 2-4: open[sz] → skeletonize → graph filter (unified loop) ───
    filtered = []
    for sz in OPEN_SIZES:
        lbl_sz = f"sz={sz}" if sz > 1 else "no-open"
        print(f"  open {sz}x{sz} -> skel -> graph filter {MIN_BRANCH_LEN}px  [{lbl_sz}]")
        k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
        zone_op  = cv2.morphologyEx(zone, cv2.MORPH_OPEN, k)
        skel_op  = sk_skeletonize(zone_op > 127).astype(np.uint8) * 255
        skel_gf  = filter_by_branch_length(skel_op, min_branch_len=MIN_BRANCH_LEN)
        filtered.append(skel_gf)

    # ── Col 5: OR union (raw, before cleanup) ────────────────────────────
    union_raw = np.zeros_like(skel_raw)
    for sk in filtered:
        union_raw = ((union_raw > 127) | (sk > 127)).astype(np.uint8) * 255

    # ── Col 6: second graph filter pass on the union ──────────────────────
    print(f"  graph filter {MIN_BRANCH_LEN_UNION}px on union (2nd pass) ...")
    union_clean = filter_by_branch_length(union_raw, min_branch_len=MIN_BRANCH_LEN_UNION)

    skels = [skel_raw] + filtered + [union_raw, union_clean]

    # ── Stats ─────────────────────────────────────────────────────────────
    print(f"\n  {'Approach':<28}  {'total_px':>8}  {'total_jct':>10}  "
          f"{'corr_px':>8}  {'corr_jct':>9}")
    lbls = (["raw"]
            + [f"open{sz}+gf{MIN_BRANCH_LEN}" for sz in OPEN_SIZES]
            + ["union_raw", f"union_gf{MIN_BRANCH_LEN_UNION}"])
    for lbl, sk in zip(lbls, skels):
        tp, tj, cp, cj = skel_stats(sk, y0, y1, cx0, cx1)
        print(f"  {lbl:<28}  {tp:>8}  {tj:>10}  {cp:>8}  {cj:>9}")
    print()

    # ── Colour & draw ─────────────────────────────────────────────────────
    vis_skels = [color_skeleton(sk, mask_bg=mask) for sk in skels]
    for vs in vis_skels:
        draw_corridor_lines(vs, y0, y1, cx0, cx1)

    yc0 = max(0,             y0     - MARGIN)
    yc1 = min(mask.shape[0], y1     + MARGIN)
    xc0 = max(0,             exp_x0 - MARGIN)
    xc1 = min(mask.shape[1], exp_x1 + MARGIN)

    def crop(img):
        return img[yc0:yc1, xc0:xc1]

    for col, (vs, ax) in enumerate(zip(vis_skels, axes[row])):
        ax.imshow(crop(vs))
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(ts, fontsize=9, rotation=0, labelpad=135,
                          va="center", fontweight="bold")
        if row == 0:
            ax.set_title(COL_TITLES[col], fontsize=7.5, pad=5,
                         loc="center", linespacing=1.5)
        tp, tj, cp, cj = skel_stats(skels[col], y0, y1, cx0, cx1)
        ax.set_xlabel(
            f"total: {tp}px  {tj}jct  |  corridor: {cp}px  {cj}jct",
            fontsize=6.5, labelpad=3,
        )

# ── Legend & suptitle ──────────────────────────────────────────────────────
fig.subplots_adjust(bottom=0.07)
leg = [
    mpatches.Patch(color=(0, 200/255, 200/255),
                   label="Skeleton pixel (uniform cyan)"),
    mpatches.Patch(color=(255/255, 220/255, 0),
                   label=f"Corridor boundary  cx ± {CORRIDOR_HALF_PX}px"),
]
fig.legend(handles=leg, loc="lower center", ncol=2, fontsize=8,
           framealpha=0.85, handlelength=1.4, columnspacing=2.0,
           bbox_to_anchor=(0.5, 0.0))

sz_str = " | ".join(f"{sz}x{sz}" if sz > 1 else "no-open" for sz in OPEN_SIZES)
plt.suptitle(
    f"Graph-level branch length filtering  |  threshold = {MIN_BRANCH_LEN} px  |  "
    f"OPEN_SIZES = [{', '.join(str(s) for s in OPEN_SIZES)}]  ({sz_str})\n"
    "Col 5 = OR union of all filtered variants — pixels present in ANY filtered skeleton.\n"
    "Opening (sz>1) pre-separates fused Sobel bands; sz=1 means no opening (raw Sobel input).",
    fontsize=9, y=1.003,
)

plt.savefig(str(OUT_PATH), dpi=DPI, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PATH}")
