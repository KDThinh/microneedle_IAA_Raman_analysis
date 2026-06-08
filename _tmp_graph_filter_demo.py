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

MIN_BRANCH_LEN       = 10   # px — graph-filter threshold for each variant
MIN_BRANCH_LEN_UNION = 10   # px — graph-filter threshold for the OR union (last col)
                             #      can be larger than MIN_BRANCH_LEN to clean up
                             #      reconnection artefacts introduced by the union

# ── Mode A: USE_ASF = False  (original behaviour) ─────────────────────────
# Cols 2-N show: single close(CLOSE_R) → open(OPEN_SIZES[i]) → skel → GF
OPEN_SIZES       = [3, 5, 7]  # disk diameters for each variant column (1 = no opening)
CLOSE_R          = 5        # single closing radius before opening (0 = skip)
                            # fills gaps <= 2*CLOSE_R px.  Kernel = 2*CLOSE_R+1.

# ── Mode B: USE_ASF = True  (alternating sequential filter) ───────────────
# Cols 2-N show the binary mask after each CUMULATIVE C→O step.
# Final col shows skeleton + graph filter on the last binary result.
# OPEN_SIZES and CLOSE_R are both ignored in this mode.
#
# ASF_SIZES  list of kernel DIAMETERS (px, must be odd) for each iteration.
#   Each entry = one column in the figure.
#   Values should be increasing for a proper ASF (e.g. 3, 5, 7, 9, 11).
#   Fewer entries → fewer columns.  Larger values → more aggressive smoothing.
ASF_SIZES        = [1, 3, 5]  # kernel diameters for each ASF iteration

USE_ASF          = True     # False = single close+open variants (Mode A)
                            # True  = progressive ASF, one col per iteration (Mode B)

CORRIDOR_HALF_PX = 40      # px — half-width of stem corridor
N_SAMPLES        = 3       # number of frames to sample

USE_ARC_LENGTH   = False   # False = pixel count (default, fast)
                            # True  = arc length: diagonal step = sqrt(2) ~ 1.414
                            #         instead of 1.0 — more physically accurate.

# ── Endpoint-bridging gap repair ──────────────────────────────────────────
# Connects disconnected skeleton components by finding their nearest
# endpoints and drawing a bridge line on the binary mask BEFORE any
# morphological processing.  Col 1 shows the raw skeleton WITHOUT the
# bridge so you can see the before/after side by side.
BRIDGE_GAP         = True  # True  = bridge disconnected skeleton components
BRIDGE_THICKNESS   = 8     # px — bridge line width drawn on the binary mask
BRIDGE_MAX_DIST    = 500   # px — max endpoint-to-endpoint distance to bridge
                            #      raise if gaps are wider; lower to avoid wrong bridges
BRIDGE_ANGLE_DEG   = 75    # degrees — each endpoint's outward tangent must point within
                            #      this angle toward the partner endpoint.
                            #      180 = no constraint   90 = strict   60 = very strict
BRIDGE_REQUIRE_BOTH = True  # True  = BOTH endpoints must satisfy BRIDGE_ANGLE_DEG
                             #         (recommended — avoids spurious leaf-to-stem bridges)
                             # False = only ONE endpoint needs to satisfy it (old behaviour)
BRIDGE_TANGENT_K   = 15    # skeleton pixels to walk back for tangent direction estimate
                            # higher → smoother estimate on curved branches near gap
BRIDGE_MIN_COMP_PX = 50    # px — ignore skeleton components smaller than this (noise)
BRIDGE_CORRIDOR_MARGIN = 60 # px — only endpoints whose x-coordinate falls within
                             #      [cx0 - margin, cx1 + margin] are considered for
                             #      bridging (cx0/cx1 = corridor boundaries).
                             #      Keeps bridging focused on the stem column;
                             #      raise to allow endpoints slightly off-centre.
BRIDGE_SHOW_MARKERS = True  # True = draw a distinct-coloured marker at each
                             #        bridged endpoint pair (both endpoints get
                             #        the same colour so you can trace each pair)
BRIDGE_MARKER_R    = 3      # px — radius of the filled endpoint circles

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


def _path_length(path):
    """
    Branch length according to USE_ARC_LENGTH:
      False → pixel count  (len of path list)
      True  → arc length   (sum of per-step Euclidean distances;
                            axis-aligned step = 1.0, diagonal step = √2 ≈ 1.414)
    """
    if not USE_ARC_LENGTH:
        return len(path)
    total = 0.0
    for i in range(len(path) - 1):
        dy = path[i+1][0] - path[i][0]
        dx = path[i+1][1] - path[i][1]
        total += (1.4142135623730951 if dy != 0 and dx != 0 else 1.0)
    return total


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

            if _path_length(path) >= min_branch_len:
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
        if _path_length(list(comp)) >= min_branch_len:
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


def color_binary(binary_u8, original_mask=None):
    """White binary-mask foreground on dimmed original mask background.
    Used to visualise intermediate ASF morphological results."""
    H, W = binary_u8.shape
    vis  = np.zeros((H, W, 3), dtype=np.uint8)
    if original_mask is not None:
        dim = (original_mask // 8).astype(np.uint8)
        vis[..., 0] = dim; vis[..., 1] = dim; vis[..., 2] = dim
    ys, xs = np.where(binary_u8 > 127)
    vis[ys, xs] = [255, 255, 255]
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
# Corridor-and-Bridge gap repair
# ══════════════════════════════════════════════════════════════════════════

def bridge_skeleton_endpoints(mask, skel,
                               max_dist=500, angle_deg=75, tangent_k=15,
                               bridge_thickness=8, min_comp_px=50,
                               require_both_aligned=True,
                               corridor_xlim=None):
    """
    Bridge disconnected skeleton components by connecting their nearest
    skeleton endpoints.

    Algorithm
    ---------
    1. Label 8-connected components of the skeleton (cv2).
    2. For every component >= min_comp_px pixels, find its degree-1
       (endpoint) pixels and compute each endpoint's outward tangent
       direction (look-back over tangent_k skeleton pixels).
    3. Build candidate bridges: all endpoint pairs across different
       components whose distance <= max_dist AND whose outward directions
       satisfy the angle constraint (at least one endpoint points within
       angle_deg of the vector toward the other).
    4. Sort candidates by distance; greedily select bridges using
       union-find (shortest first, skip pairs already transitively
       connected).
    5. Draw each selected bridge as a thick line on the binary mask.

    Parameters
    ----------
    mask          : uint8 binary mask to draw bridges on
    skel          : uint8 raw skeleton for component / endpoint analysis
    max_dist      : px — max endpoint-to-endpoint distance to bridge
    angle_deg     : degrees — directional constraint (180 = none)
    tangent_k     : skeleton pixels to walk back for tangent estimation
    bridge_thickness : line width drawn on mask
    min_comp_px   : ignore components smaller than this (noise filter)

    Returns
    -------
    (result_mask, bridges)
        bridges : list of (dist, y_a, x_a, y_b, x_b) for each bridge drawn
    """
    result    = mask.copy()
    binary_u8 = (skel > 127).astype(np.uint8)
    if not binary_u8.any():
        return result, []

    pos_set = set(zip(*np.where(binary_u8)))

    # ── Connected components ──────────────────────────────────────────────
    n_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(binary_u8)
    valid_labels = {lbl for lbl in range(1, n_labels)
                    if stats[lbl, cv2.CC_STAT_AREA] >= min_comp_px}
    if len(valid_labels) < 2:
        return result, []

    # ── Degree map → endpoints (degree == 1) ─────────────────────────────
    # If corridor_xlim=(x_lo, x_hi) is given, only endpoints within that
    # x-range are eligible — focuses bridging on the stem column.
    x_lo = corridor_xlim[0] if corridor_xlim is not None else None
    x_hi = corridor_xlim[1] if corridor_xlim is not None else None
    ep_by_comp = {}
    for (y, x) in pos_set:
        if x_lo is not None and not (x_lo <= x <= x_hi):
            continue                        # outside corridor+margin — skip
        d = sum(1 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if (dy or dx) and (y + dy, x + dx) in pos_set)
        if d == 1:
            lbl = int(label_img[y, x])
            if lbl in valid_labels:
                ep_by_comp.setdefault(lbl, []).append((y, x))

    if len(ep_by_comp) < 2:
        return result, []

    # ── Outward tangent direction at each endpoint ────────────────────────
    def outward_dir(ep):
        path = [ep]
        prev = ep
        nbs  = _nb8(*ep, pos_set)
        if not nbs:
            return np.zeros(2)
        cur = nbs[0]
        for _ in range(tangent_k - 1):
            path.append(cur)
            nxts = [p for p in _nb8(*cur, pos_set) if p != prev]
            if not nxts:
                break
            prev, cur = cur, nxts[0]
        if len(path) < 2:
            return np.zeros(2)
        dy = path[0][0] - path[-1][0]
        dx = path[0][1] - path[-1][1]
        L  = np.hypot(dy, dx)
        return np.array([dy / L, dx / L]) if L > 0 else np.zeros(2)

    all_eps  = [(ep, lbl)
                for lbl, eps in ep_by_comp.items() for ep in eps]
    ep_dirs  = {ep: outward_dir(ep) for ep, _ in all_eps}

    # ── Candidate bridges ─────────────────────────────────────────────────
    # Scored by alignment quality (sum of both cosines) so well-aligned pairs
    # (e.g. stem-to-stem facing each other across a gap) are tried before
    # merely-close pairs.  Sort key: (-cos_a - cos_b, dist).
    cos_thr    = np.cos(np.radians(angle_deg))
    candidates = []

    for i, (ep_a, lbl_a) in enumerate(all_eps):
        ya, xa   = ep_a
        dir_a    = ep_dirs[ep_a]
        for ep_b, lbl_b in all_eps[i + 1:]:
            if lbl_a == lbl_b:
                continue
            yb, xb = ep_b
            dy, dx = yb - ya, xb - xa
            dist   = np.hypot(dy, dx)
            if dist == 0 or dist > max_dist:
                continue
            if angle_deg < 180:
                ab    = np.array([dy / dist, dx / dist])
                dir_b = ep_dirs[ep_b]
                cos_a = float(dir_a @ ab)    if dir_a.any() else 1.0
                cos_b = float(dir_b @ (-ab)) if dir_b.any() else 1.0
                ok_a  = cos_a > cos_thr
                ok_b  = cos_b > cos_thr
                # AND: both endpoints must face each other (strict, avoids leaf spurious)
                # OR:  at least one must (permissive — original behaviour)
                if require_both_aligned:
                    if not (ok_a and ok_b):
                        continue
                else:
                    if not (ok_a or ok_b):
                        continue
            else:
                cos_a = cos_b = 1.0
            # Primary sort key = best alignment; secondary = shorter distance
            candidates.append((-cos_a - cos_b, dist, ep_a, ep_b, lbl_a, lbl_b))

    if not candidates:
        return result, []
    candidates.sort()

    # ── Greedy bridge selection (union-find) ──────────────────────────────
    parent = {lbl: lbl for lbl in valid_labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    bridges = []
    for _neg_align, dist, ep_a, ep_b, lbl_a, lbl_b in candidates:
        if find(lbl_a) == find(lbl_b):
            continue
        cv2.line(result,
                 (ep_a[1], ep_a[0]),   # cv2 takes (x, y)
                 (ep_b[1], ep_b[0]),
                 255, bridge_thickness)
        parent[find(lbl_a)] = find(lbl_b)
        bridges.append((dist, ep_a[0], ep_a[1], ep_b[0], ep_b[1]))

    return result, bridges


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

# ── Dynamic column count ──────────────────────────────────────────────────
if USE_ASF:
    N_COLS = len(ASF_SIZES) + 2       # raw_skel + len(ASF_SIZES) binary images + final_skel+GF
else:
    N_COLS = 1 + len(OPEN_SIZES) + 2  # raw + variants + union_raw + union_clean

fig = plt.figure(figsize=(8 * N_COLS, N_SAMPLES * 9))
gs  = gridspec.GridSpec(
    N_SAMPLES, N_COLS,
    figure=fig,
    width_ratios=[4] * N_COLS,
    wspace=0.05, hspace=0.24,
)
axes = np.array([[fig.add_subplot(gs[r, c])
                  for c in range(N_COLS)]
                 for r in range(N_SAMPLES)])

def _col_title_asf_bin(i, sz):
    """Title for ASF binary-mask column i (1-indexed), kernel diameter sz."""
    steps = "  ".join(f"C{s}O{s}" for s in ASF_SIZES[:i])
    return (f"Col {i+1} — Binary after ASF iter {i}\n"
            f"{steps}  (this step: {sz}x{sz})\n"
            f"cumulative morphological result")

def _col_title_open(i, sz):
    open_lbl  = "no opening" if sz <= 1 else f"open {sz}x{sz}"
    close_lbl = f"close r={CLOSE_R} -> " if CLOSE_R > 0 else ""
    return (f"Col {i+2} — {close_lbl}{open_lbl}\n"
            f"+ graph filter {MIN_BRANCH_LEN} px\n"
            f"CLOSE_R={CLOSE_R}  OPEN_SIZES[{i}]={sz}")

_col1_title = (
    "Col 1 — RAW SKELETON\nno bridge / no preprocessing\ngap visible — baseline"
    if BRIDGE_GAP else
    "Col 1 — RAW SKELETON\nno preprocessing\nbaseline (maximum noise / web)"
)
_align_mode  = "both-aligned" if BRIDGE_REQUIRE_BOTH else "one-aligned"
_bridge_note = (f"BRIDGE_GAP=True  (max_dist={BRIDGE_MAX_DIST}px, "
                f"angle<={BRIDGE_ANGLE_DEG}deg [{_align_mode}], "
                f"corridor±{BRIDGE_CORRIDOR_MARGIN}px, thick={BRIDGE_THICKNESS}px)\n"
                f"applied to all cols >= 2"
                if BRIDGE_GAP else "BRIDGE_GAP=False")

if USE_ASF:
    _asf_n        = len(ASF_SIZES)
    _asf_final_col = _asf_n + 1   # 0-indexed
    _asf_size_str  = " ".join(f"C{sz}O{sz}" for sz in ASF_SIZES)
    COL_TITLES = (
        [_col1_title]
        + [_col_title_asf_bin(i, sz) for i, sz in enumerate(ASF_SIZES, 1)]
        + [f"Col {_asf_final_col+1} — FINAL skeleton (GF {MIN_BRANCH_LEN}px)\n"
           f"After all {_asf_n} ASF iterations\n"
           f"{_asf_size_str}  ->  skel  ->  graph filter"]
    )
else:
    _n_ov      = len(OPEN_SIZES)
    _union_col = _n_ov + 1   # 0-indexed column of union_raw
    _clean_col = _n_ov + 2   # 0-indexed column of union_clean
    COL_TITLES = (
        [_col1_title]
        + [_col_title_open(i, sz) for i, sz in enumerate(OPEN_SIZES)]
        + [f"Col {_union_col+1} — UNION (before cleanup)\nOR of all variant cols\nreconnection artefacts still present"]
        + [f"Col {_clean_col+1} — UNION + 2nd GF {MIN_BRANCH_LEN_UNION}px\nOR union after second graph filter\nreconnection artefacts removed"]
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

    # ── Col 1: raw skeleton (no bridge — baseline shows the gap) ─────────
    skel_raw = sk_skeletonize(zone > 127).astype(np.uint8) * 255

    # ── Optional endpoint bridge (applied to zone before all further steps)
    bridges = []   # populated below when BRIDGE_GAP=True; always defined
    if BRIDGE_GAP:
        zone, bridges = bridge_skeleton_endpoints(
            zone, skel_raw,
            max_dist=BRIDGE_MAX_DIST,
            angle_deg=BRIDGE_ANGLE_DEG,
            tangent_k=BRIDGE_TANGENT_K,
            bridge_thickness=BRIDGE_THICKNESS,
            min_comp_px=BRIDGE_MIN_COMP_PX,
            require_both_aligned=BRIDGE_REQUIRE_BOTH,
            corridor_xlim=(cx0 - BRIDGE_CORRIDOR_MARGIN,
                           cx1 + BRIDGE_CORRIDOR_MARGIN),
        )
        if bridges:
            print(f"  bridge: {len(bridges)} connection(s) drawn:")
            for dist, ya, xa, yb, xb in bridges:
                print(f"    dist={dist:.1f}px  ({ya},{xa})->({yb},{xb})")
        else:
            print(f"  bridge: no endpoint pairs within {BRIDGE_MAX_DIST}px found")

    # ── Variant columns: ASF or single-close+open ─────────────────────────
    if USE_ASF:
        # Cumulative ASF: store the BINARY MASK after each C_sz→O_sz step.
        # Kernel sizes are taken from ASF_SIZES (user-configurable).
        # Skeletonisation + graph filter are applied ONCE on the final result.
        zone_cur = zone.copy()
        asf_bins = []                        # binary images, one per iteration
        for i, sz in enumerate(ASF_SIZES, 1):
            kr       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
            zone_cur = cv2.morphologyEx(zone_cur, cv2.MORPH_CLOSE, kr)
            zone_cur = cv2.morphologyEx(zone_cur, cv2.MORPH_OPEN,  kr)
            asf_bins.append(zone_cur.copy())
            print(f"  ASF iter {i}: C{sz}O{sz} (kernel {sz}x{sz})")
        skel_final = sk_skeletonize(zone_cur > 127).astype(np.uint8) * 255
        skel_gf    = filter_by_branch_length(skel_final, min_branch_len=MIN_BRANCH_LEN)
        print(f"  skel + GF {MIN_BRANCH_LEN}px on final ASF result")

        # images: [raw_skel | binary_1 .. binary_N | final_skel+GF]
        images  = [skel_raw] + asf_bins + [skel_gf]
        is_skel = [True]     + [False] * len(ASF_SIZES) + [True]

    else:
        # Single pre-close then multiple opening variants + union
        if CLOSE_R > 0:
            kc        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                  (2*CLOSE_R+1, 2*CLOSE_R+1))
            zone_base = cv2.morphologyEx(zone, cv2.MORPH_CLOSE, kc)
            print(f"  pre-close r={CLOSE_R} (kernel {2*CLOSE_R+1}x{2*CLOSE_R+1})"
                  f" — fills gaps <= {2*CLOSE_R}px")
        else:
            zone_base = zone
        filtered = []
        for sz in OPEN_SIZES:
            lbl_sz    = f"sz={sz}" if sz > 1 else "no-open"
            close_lbl = f"close(r={CLOSE_R})->" if CLOSE_R > 0 else ""
            print(f"  {close_lbl}open {sz}x{sz} -> skel -> GF {MIN_BRANCH_LEN}px  [{lbl_sz}]")
            k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
            zone_op  = cv2.morphologyEx(zone_base, cv2.MORPH_OPEN, k)
            skel_op  = sk_skeletonize(zone_op > 127).astype(np.uint8) * 255
            skel_gf  = filter_by_branch_length(skel_op, min_branch_len=MIN_BRANCH_LEN)
            filtered.append(skel_gf)

        union_raw = np.zeros_like(skel_raw)
        for sk in filtered:
            union_raw = ((union_raw > 127) | (sk > 127)).astype(np.uint8) * 255
        print(f"  graph filter {MIN_BRANCH_LEN_UNION}px on union (2nd pass) ...")
        union_clean = filter_by_branch_length(union_raw, min_branch_len=MIN_BRANCH_LEN_UNION)

        images  = [skel_raw] + filtered + [union_raw, union_clean]
        is_skel = [True] * len(images)

    # ── Stats ─────────────────────────────────────────────────────────────
    if USE_ASF:
        lbls = (["raw_skel"]
                + [f"ASF_bin_sz{sz}" for sz in ASF_SIZES]
                + [f"ASF_final+gf{MIN_BRANCH_LEN}"])
        print(f"\n  {'Approach':<28}  info")
        for lbl, img, sk_flag in zip(lbls, images, is_skel):
            if sk_flag:
                tp, tj, cp, cj = skel_stats(img, y0, y1, cx0, cx1)
                print(f"  {lbl:<28}  skel: {tp}px  {tj}jct  corr:{cp}px  {cj}jct")
            else:
                area = int(np.sum(img > 127))
                print(f"  {lbl:<28}  binary: {area}px foreground")
    else:
        lbls = (["raw"]
                + [f"open{sz}+gf{MIN_BRANCH_LEN}" for sz in OPEN_SIZES]
                + ["union_raw", f"union_gf{MIN_BRANCH_LEN_UNION}"])
        print(f"\n  {'Approach':<28}  {'total_px':>8}  {'total_jct':>10}  "
              f"{'corr_px':>8}  {'corr_jct':>9}")
        for lbl, img in zip(lbls, images):
            tp, tj, cp, cj = skel_stats(img, y0, y1, cx0, cx1)
            print(f"  {lbl:<28}  {tp:>8}  {tj:>10}  {cp:>8}  {cj:>9}")
    print()

    # ── Colour & draw ─────────────────────────────────────────────────────
    vis_list = []
    for img, sk_flag in zip(images, is_skel):
        if sk_flag:
            vis = color_skeleton(img, mask_bg=mask)
            draw_corridor_lines(vis, y0, y1, cx0, cx1)
        else:
            vis = color_binary(img, original_mask=zone)
        vis_list.append(vis)

    # ── Bridge endpoint markers ────────────────────────────────────────────
    # Draw coloured filled circles at each bridged endpoint pair.
    # Each pair gets a unique colour from tab10; both endpoints share it so
    # you can see which endpoint connected to which.  A white halo ring gives
    # contrast against both bright and dark backgrounds.
    # Markers are applied to every post-bridge column (index >= 1); col 0 is
    # the raw skeleton (pre-bridge) and intentionally left unmarked.
    _TAB10 = [                      # matplotlib tab10 palette, RGB uint8
        ( 31, 119, 180),  # blue
        (255, 127,  14),  # orange
        ( 44, 160,  44),  # green
        (214,  39,  40),  # red
        (148, 103, 189),  # purple
        (140,  86,  75),  # brown
        (227, 119, 194),  # pink
        (127, 127, 127),  # grey
        (188, 189,  34),  # olive
        ( 23, 190, 207),  # teal
    ]
    if BRIDGE_GAP and BRIDGE_SHOW_MARKERS and bridges:
        for k, (dist, ya, xa, yb, xb) in enumerate(bridges):
            rgb = _TAB10[k % len(_TAB10)]
            for vis in vis_list[1:]:          # skip col 0 (raw, pre-bridge)
                # Solid fill in pair colour
                cv2.circle(vis, (xa, ya), BRIDGE_MARKER_R, rgb, -1)
                cv2.circle(vis, (xb, yb), BRIDGE_MARKER_R, rgb, -1)
                # Thin connector line so the pairing is obvious at a glance
                cv2.line(vis, (xa, ya), (xb, yb), rgb, 2)

    yc0 = max(0,             y0     - MARGIN)
    yc1 = min(mask.shape[0], y1     + MARGIN)
    xc0 = max(0,             exp_x0 - MARGIN)
    xc1 = min(mask.shape[1], exp_x1 + MARGIN)

    def crop(img):
        return img[yc0:yc1, xc0:xc1]

    for col, (vs, img, sk_flag, ax) in enumerate(
            zip(vis_list, images, is_skel, axes[row])):
        ax.imshow(crop(vs))
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(ts, fontsize=9, rotation=0, labelpad=135,
                          va="center", fontweight="bold")
        if row == 0:
            ax.set_title(COL_TITLES[col], fontsize=7.5, pad=5,
                         loc="center", linespacing=1.5)
        if sk_flag:
            tp, tj, cp, cj = skel_stats(img, y0, y1, cx0, cx1)
            ax.set_xlabel(
                f"total: {tp}px  {tj}jct  |  corridor: {cp}px  {cj}jct",
                fontsize=6.5, labelpad=3,
            )
        else:
            area = int(np.sum(img > 127))
            ax.set_xlabel(f"foreground: {area}px", fontsize=6.5, labelpad=3)

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

len_mode = "arc length (diag=sqrt2)" if USE_ARC_LENGTH else "pixel count"
if USE_ASF:
    asf_seq = "  ".join(f"C{sz}O{sz}" for sz in ASF_SIZES)
    plt.suptitle(
        f"Pipeline: [{_bridge_note}]  ->  ASF ({len(ASF_SIZES)} iter): {asf_seq}\n"
        f"Intermediate cols = binary mask.  Col {len(ASF_SIZES)+2} = skeleton + GF {MIN_BRANCH_LEN}px  |  "
        f"length mode: {len_mode}",
        fontsize=9, y=1.003,
    )
else:
    sz_str    = " | ".join(f"{sz}x{sz}" if sz > 1 else "no-open" for sz in OPEN_SIZES)
    close_str = f"close r={CLOSE_R} (fills gaps <={2*CLOSE_R}px) -> " if CLOSE_R > 0 else "no closing -> "
    plt.suptitle(
        f"Pipeline: [{_bridge_note}]  ->  {close_str}open({sz_str}) -> skel -> GF\n"
        f"GF thresholds: variants={MIN_BRANCH_LEN}px  |  union 2nd pass={MIN_BRANCH_LEN_UNION}px  |  "
        f"length mode: {len_mode}\n"
        f"Col {_union_col+1} = raw OR union.  Col {_clean_col+1} = union after 2nd graph filter pass.",
        fontsize=9, y=1.003,
    )

plt.savefig(str(OUT_PATH), dpi=DPI, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT_PATH}")
