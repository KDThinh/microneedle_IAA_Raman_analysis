"""
_tmp_stem_width_profile_c.py
Strategy C: skeleton branch tracing + angle classification for node / meristem detection.

Corridor detection: v3 (identical to _tmp_stem_diagnostic.py).

Pipeline per frame
──────────────────
1. Detect stem corridor (v3: anchor + max-run-length column projection).
2. Build expanded mask  = original edge mask clipped to corridor ± EXPAND_PX.
   Within the corridor itself, apply morphological close (kernel CLOSE_K) to
   bridge the two parallel Sobel edge lines into a solid stem region → the
   skeleton of this region is a single clean medial axis rather than two
   noisy parallel lines.
3. Skeletonise the expanded mask (iterative morphological thinning).
4. Build adjacency graph on skeleton pixels (8-connectivity).
5. Find junction pixels (≥ 3 skeleton neighbours).
6. Trace each branch from each junction (up to BRANCH_TRACE_LEN px).
7. Classify each branch by its angle from vertical (PCA on path pixels):
     angle < STEM_ANGLE_THRESH °  →  stem / vertical segment     (blue)
     angle ≥ STEM_ANGLE_THRESH °  →  petiole / side branch       (orange)
8. Node     = junction within / near the corridor that has ≥ 1 petiole branch
              of length ≥ MIN_BRANCH_LEN.
9. Meristem = topmost skeleton pixel within the corridor.
10. Base    = bottommost skeleton pixel within the corridor.

On the disconnected-leaf problem
─────────────────────────────────
A leaf that appears disconnected from the corridor still has its petiole
rooted at a junction ON the corridor. The branch trace from that junction
follows the petiole out, computes its angle, and marks the junction as a node
— regardless of whether the leaf blob itself is connected further along.
Widening the corridor is not needed; the node is found at the corridor boundary.

Visualisation (N_SAMPLES frames, 2 panels each)
  Left  — dimmed mask + colour-coded skeleton + node/meristem/base markers.
  Right — simplified plant diagram: vertical stem line + node ticks + meristem star.
"""

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage.morphology import skeletonize as sk_skeletonize

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
    r"\stem_width_profile_c.png"
)

# ── Corridor detection v3 (same as _tmp_stem_diagnostic.py) ───────────────────
CORRIDOR_HALF_PX = 40
BOTTOM_FRAC      = 0.15
PROJ_BAND        = (0.20, 0.85)
ANCHOR_MARGIN    = 150
SMOOTH_K         = 15

# ── Strategy C parameters ──────────────────────────────────────────────────────
EXPAND_PX         = 200   # px beyond corridor on each side to capture petioles
CLOSE_K           = 39    # morph-close kernel for filling stem interior in corridor
                           # must be ≥ interior gap between the two Sobel edge lines
BRANCH_TRACE_LEN  = 200   # max skeleton pixels to follow per branch from a junction
STEM_ANGLE_THRESH = 25    # degrees from vertical: < this = stem, ≥ this = petiole
MIN_BRANCH_LEN    = 50    # branches shorter than this ignored (skeleton noise / rungs)
                           # young top-leaf petioles can be 50-80 px; noise branches <20 px
CORRIDOR_TOL      = 20    # junction on medial axis if |x − cx_centre| ≤ this (px)
MIN_APEX_SEP      = 80    # exclude junctions within this many px BELOW the plant apex
                           # (the curved apex region produces false T-junctions)

# ── Display ────────────────────────────────────────────────────────────────────
N_SAMPLES = 6
DPI       = 200


# ══════════════════════════════════════════════════════════════════════════════
# Corridor detection v3
# ══════════════════════════════════════════════════════════════════════════════

def col_max_run_lengths(binary_2d: np.ndarray) -> np.ndarray:
    """Longest consecutive True run per column of a 2-D boolean array."""
    max_runs = np.zeros(binary_2d.shape[1], dtype=np.int32)
    cur_runs = np.zeros(binary_2d.shape[1], dtype=np.int32)
    for row in binary_2d:
        cur_runs = np.where(row, cur_runs + 1, 0)
        np.maximum(max_runs, cur_runs, out=max_runs)
    return max_runs


def estimate_stem_cx(mask: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> int:
    """
    v3: coarse anchor (bottom BOTTOM_FRAC strip median-x) +
        max-run-length column projection within ±ANCHOR_MARGIN of anchor.
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
    if not band.any():
        return cx_anch
    proj   = col_max_run_lengths(band).astype(np.float64)
    smooth = np.convolve(proj, np.ones(SMOOTH_K) / SMOOTH_K, mode="same")
    return sx0 + int(np.argmax(smooth))


# ══════════════════════════════════════════════════════════════════════════════
# Mask preparation
# ══════════════════════════════════════════════════════════════════════════════

def build_expanded_mask(
    mask: np.ndarray,
    y0: int, y1: int,
    cx0: int, cx1: int,
    exp_x0: int, exp_x1: int,
) -> np.ndarray:
    """
    Expanded mask for skeleton analysis.

    Outside corridor : original Sobel edge pixels (preserves petiole edges).
    Inside corridor  : morphologically closed to fill the stem interior.
                       The two parallel Sobel edge lines are bridged into one
                       solid region → single medial axis after skeletonising.
    """
    out = np.zeros_like(mask)
    # Paste original edges in the expanded flanks
    out[:, exp_x0 : cx0]      = mask[:, exp_x0 : cx0]
    out[:, cx1 + 1 : exp_x1 + 1] = mask[:, cx1 + 1 : exp_x1 + 1]

    # Fill corridor region with morphological close
    k  = CLOSE_K if CLOSE_K % 2 == 1 else CLOSE_K + 1
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    corr_strip = mask[y0 : y1 + 1, cx0 : cx1 + 1]
    filled     = cv2.morphologyEx(corr_strip, cv2.MORPH_CLOSE, el)
    out[y0 : y1 + 1, cx0 : cx1 + 1] = filled

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Skeletonisation
# ══════════════════════════════════════════════════════════════════════════════

def morphological_skeleton(mask_u8: np.ndarray) -> np.ndarray:
    """
    True 1-px-wide skeleton using skimage.morphology.skeletonize (Lee 1994).
    Returns uint8 array with 255 at skeleton pixels, 0 elsewhere.
    """
    binary = (mask_u8 > 127)
    skel   = sk_skeletonize(binary)      # bool array, exactly 1-px wide
    return skel.astype(np.uint8) * 255


# ══════════════════════════════════════════════════════════════════════════════
# Skeleton graph
# ══════════════════════════════════════════════════════════════════════════════

def build_skeleton_graph(
    skel: np.ndarray,
) -> tuple[dict, dict, set, set]:
    """
    8-connected adjacency graph on skeleton pixels.

    Returns
    -------
    adj       : {pixel_idx: [neighbour_idx, ...]}
    yx_coords : {pixel_idx: (y, x)}
    junctions : pixel indices with ≥ 3 neighbours
    endpoints : pixel indices with exactly 1 neighbour
    """
    pts = np.argwhere(skel > 127)
    if len(pts) == 0:
        return {}, {}, set(), set()

    pos_map  = {(int(p[0]), int(p[1])): i for i, p in enumerate(pts)}
    yx_coords = {i: (int(pts[i, 0]), int(pts[i, 1])) for i in range(len(pts))}
    adj       = {i: [] for i in range(len(pts))}

    for i, (y, x) in enumerate(pts):
        y, x = int(y), int(x)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = pos_map.get((y + dy, x + dx))
                if j is not None:
                    adj[i].append(j)

    junctions = {i for i, nb in adj.items() if len(nb) >= 3}
    endpoints = {i for i, nb in adj.items() if len(nb) == 1}
    return adj, yx_coords, junctions, endpoints


# ══════════════════════════════════════════════════════════════════════════════
# Branch tracing and angle classification
# ══════════════════════════════════════════════════════════════════════════════

def trace_branch(
    start: int,
    came_from: int,
    adj: dict,
    junctions: set,
    max_len: int = BRANCH_TRACE_LEN,
) -> list[int]:
    """
    Trace a skeleton branch from `start` (neighbour of a junction),
    coming from `came_from` (the junction itself).
    Stops at another junction, a dead end, or max_len pixels.
    Returns the ordered list of pixel indices (including came_from).
    """
    path = [came_from, start]
    while len(path) <= max_len:
        cur   = path[-1]
        prev  = path[-2]
        nexts = [n for n in adj[cur] if n != prev]
        if not nexts or cur in junctions:
            break
        path.append(nexts[0])
    return path


def angle_from_vertical(path: list[int], yx_coords: dict) -> float:
    """
    Angle of a branch from vertical (0° = vertical, 90° = horizontal).
    Uses PCA on path pixel coordinates (≥ 3 points) or start-to-end vector.
    """
    if len(path) < 2:
        return 90.0
    pts = np.array([yx_coords[i] for i in path], dtype=np.float64)
    if len(pts) < 3:
        dy = pts[-1, 0] - pts[0, 0]
        dx = pts[-1, 1] - pts[0, 1]
        return 90.0 if abs(dy) < 1e-6 else float(np.degrees(np.arctan2(abs(dx), abs(dy))))
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    dy, dx   = vt[0]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


# ══════════════════════════════════════════════════════════════════════════════
# Node / meristem / base detection
# ══════════════════════════════════════════════════════════════════════════════

def find_landmarks(
    adj       : dict,
    yx_coords : dict,
    junctions : set,
    endpoints : set,
    cx0: int,
    cx1: int,
) -> tuple[list[tuple[int, int]], int | None, int | None, int | None]:
    """
    Returns (nodes_yx, top_node_y, base_y, apex_y).

    Node       — junction ON the corridor medial axis (|x − cx_centre| ≤ CORRIDOR_TOL)
                 that has ≥1 branch with angle ≥ STEM_ANGLE_THRESH
                 and length ≥ MIN_BRANCH_LEN.
    top_node_y — y-coordinate of the topmost detected node (the "meristem" in the
                 practical sense used here: highest visible leaf attachment point).
                 None if no nodes detected.
    base_y     — bottommost skeleton pixel in the full corridor (≈ pot rim / stem base).
    apex_y     — topmost skeleton pixel in the full corridor (true plant tip, above
                 the top node); kept for reference only.
    """
    cx_centre = (cx0 + cx1) // 2

    def on_axis(idx: int) -> bool:
        """Pixel is on (or very close to) the corridor medial axis."""
        _, x = yx_coords[idx]
        return abs(x - cx_centre) <= CORRIDOR_TOL

    def in_corridor(idx: int) -> bool:
        """Pixel is anywhere within the full corridor width."""
        _, x = yx_coords[idx]
        return (cx0 - CORRIDOR_TOL) <= x <= (cx1 + CORRIDOR_TOL)

    nodes_yx: list[tuple[int, int]] = []

    for jct in junctions:
        if not on_axis(jct):
            continue
        jy, jx = yx_coords[jct]
        for nb in adj[jct]:
            path  = trace_branch(nb, jct, adj, junctions)
            angle = angle_from_vertical(path, yx_coords)
            if angle >= STEM_ANGLE_THRESH and len(path) >= MIN_BRANCH_LEN:
                nodes_yx.append((jy, jx))
                break   # one qualifying petiole branch is enough

    nodes_yx.sort(key=lambda p: p[0])   # sort top → bottom (smallest y = highest in image)

    # apex_y / base_y first (needed for apex-exclusion filter below)
    corr_ys = [yx_coords[i][0] for i in yx_coords if in_corridor(i)]
    apex_y = int(min(corr_ys)) if corr_ys else None
    base_y = int(max(corr_ys)) if corr_ys else None

    # Drop nodes within MIN_APEX_SEP px of the plant apex.
    # The curved stem tip generates false T-junctions that survive the angle/length
    # filter but sit only 30–50 px below the actual apex — far above any real leaf.
    if apex_y is not None:
        nodes_yx = [(ny, nx) for ny, nx in nodes_yx
                    if (ny - apex_y) >= MIN_APEX_SEP]

    # top_node_y: y of the TOPMOST remaining node ("meristem" per user definition)
    top_node_y = nodes_yx[0][0] if nodes_yx else None

    return nodes_yx, top_node_y, base_y, apex_y


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════════════

def render_skeleton_overlay(
    mask       : np.ndarray,
    skel       : np.ndarray,
    adj        : dict,
    yx_coords  : dict,
    junctions  : set,
    nodes_yx   : list,
    top_node_y : int | None,
    apex_y     : int | None,
    base_y     : int | None,
    cx0: int, cx1: int,
    y0:  int, y1:  int,
) -> np.ndarray:
    """
    Dimmed mask with colour-coded skeleton:
      blue   = near-vertical branch (stem)
      orange = angled branch (petiole)
      cyan   = node (leaf attachment junction)
      red ✕  = top node ("meristem" per user definition)
      purple ✕= plant apex (true tip, above top node)
      green ✕ = base (pot rim)
      yellow  = corridor boundaries
    """
    vis = np.stack([(mask // 7)] * 3, axis=2).astype(np.uint8)

    # Colour skeleton branches
    painted = set()
    for jct in junctions:
        for nb in adj[jct]:
            key = (min(jct, nb), max(jct, nb))
            if key in painted:
                continue
            path  = trace_branch(nb, jct, adj, junctions)
            ang   = angle_from_vertical(path, yx_coords)
            color = (80, 120, 255) if ang < STEM_ANGLE_THRESH else (255, 140, 30)
            for idx in path:
                py, px = yx_coords[idx]
                if 0 <= py < vis.shape[0] and 0 <= px < vis.shape[1]:
                    vis[py, px] = color
            for a, b in zip(path[:-1], path[1:]):
                painted.add((min(a, b), max(a, b)))

    # Remaining skeleton pixels not covered by any branch trace
    for i, (py, px) in yx_coords.items():
        if vis[py, px, 0] <= mask[py, px] // 7 + 2:
            vis[py, px] = (180, 180, 180)   # light grey

    # Corridor boundaries (yellow)
    vis[y0 : y1 + 1, cx0] = [255, 220, 0]
    vis[y0 : y1 + 1, cx1] = [255, 220, 0]

    # Node circles (cyan, r=5)
    for ny, nx in nodes_yx:
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if dy * dy + dx * dx <= 25:
                    py, px = ny + dy, nx + dx
                    if 0 <= py < vis.shape[0] and 0 <= px < vis.shape[1]:
                        vis[py, px] = [0, 230, 230]

    # Cross marker helper (drawn on corridor centre x)
    def draw_cross(cy: int, col: tuple, r: int = 7) -> None:
        cx_m = (cx0 + cx1) // 2
        for d in range(-r, r + 1):
            for py, px in [(cy + d, cx_m), (cy, cx_m + d)]:
                if 0 <= py < vis.shape[0] and 0 <= px < vis.shape[1]:
                    vis[py, px] = col

    # Top node (= "meristem"): red cross
    if top_node_y is not None:
        draw_cross(top_node_y, (255, 50,  50),  r=8)
    # Plant apex (true tip): purple cross
    if apex_y is not None:
        draw_cross(apex_y,     (200, 80, 200),  r=6)
    # Base: green cross
    if base_y is not None:
        draw_cross(base_y,     (50,  220, 80),  r=8)

    return vis


def draw_plant_diagram(
    ax         : plt.Axes,
    nodes_yx   : list,
    top_node_y : int | None,
    apex_y     : int | None,
    base_y     : int | None,
    stem_h     : int | None,
    plant_h    : int | None,
) -> None:
    """
    Simplified plant diagram (all heights measured from base):
      brown line  = main stem (base → plant apex)
      blue ticks  = nodes (top → bottom, numbered 1 = topmost)
      red star    = top node ("meristem" per user definition)
      purple star = plant apex (true tip above top node)
      green star  = base
    """
    if base_y is None:
        ax.text(0.5, 0.5, "base not detected",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.axis("off")
        return

    # y-axis extent: 0 (base) → plant_h (apex)
    diagram_h = plant_h if plant_h else (stem_h if stem_h else 1000)

    # Stem line: base (0) to apex
    ax.plot([0, 0], [0, diagram_h], color="saddlebrown", lw=4, solid_capstyle="round")

    # Nodes (blue ticks, numbered top → bottom)
    for i, (ny, _) in enumerate(nodes_yx, start=1):
        h = base_y - ny
        if -10 <= h <= diagram_h + 10:
            ax.plot([-22, 22], [h, h], "-", color="royalblue", lw=2.0)
            ax.text(27, h, f"N{i}", fontsize=7, va="center", color="royalblue")

    # Top node ("meristem")
    if top_node_y is not None:
        mn_h = base_y - top_node_y
        ax.plot(0, mn_h, "*", color="crimson", ms=15, zorder=6)
        ax.text(10, mn_h + diagram_h * 0.02,
                f"Top node\n(meristem)\n{mn_h} px", fontsize=7, color="crimson")

    # Plant apex (true tip)
    if apex_y is not None:
        ap_h = base_y - apex_y
        ax.plot(0, ap_h, "*", color="mediumpurple", ms=12, zorder=6)
        ax.text(10, ap_h + diagram_h * 0.01,
                f"Apex\n{ap_h} px", fontsize=7, color="mediumpurple")

    # Base
    ax.plot(0, 0, "*", color="forestgreen", ms=15, zorder=6)
    ax.text(10, diagram_h * 0.02, "Base", fontsize=7, color="forestgreen")

    ax.set_xlim(-70, 160)
    ax.set_ylim(-diagram_h * 0.08, diagram_h * 1.12)
    ax.set_ylabel("Height from base (px)", fontsize=8)
    ax.set_title(
        f"{len(nodes_yx)} node(s) detected\n"
        f"stem_h (top-node→base) = {stem_h} px\n"
        f"plant_h (apex→base)    = {plant_h} px",
        fontsize=8,
    )
    ax.yaxis.grid(True, ls=":", alpha=0.4)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════════

masks   = sorted(MASKS_DIR.glob("*_mask_refined.png"))
if not masks:
    raise FileNotFoundError(f"No masks in {MASKS_DIR}")

step    = max(1, len(masks) // N_SAMPLES)
sampled = masks[::step][:N_SAMPLES]
print(f"Total masks: {len(masks)} | sampling {len(sampled)} (every {step}th)\n")

fig, axes = plt.subplots(N_SAMPLES, 2, figsize=(14, N_SAMPLES * 5.5))

summary_rows = []

for row_idx, mask_path in enumerate(sampled):
    ax_img = axes[row_idx, 0]
    ax_dia = axes[row_idx, 1]
    ts = mask_path.stem.replace("tl_", "").replace("_mask_refined", "")
    t0 = time.time()
    print(f"[{row_idx+1}/{len(sampled)}] {ts} ...", end=" ", flush=True)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("READ ERROR")
        ax_img.set_title("READ ERROR"); ax_img.axis("off")
        ax_dia.axis("off"); continue

    ys, xs = np.where(mask > 127)
    if len(ys) == 0:
        print("EMPTY")
        ax_img.set_title("EMPTY"); ax_img.axis("off")
        ax_dia.axis("off"); continue

    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    # ── Corridor ──────────────────────────────────────────────────────────────
    cx  = estimate_stem_cx(mask, y0, y1, x0, x1)
    cx0 = max(0,               cx - CORRIDOR_HALF_PX)
    cx1 = min(mask.shape[1]-1, cx + CORRIDOR_HALF_PX)

    exp_x0 = max(0,               cx0 - EXPAND_PX)
    exp_x1 = min(mask.shape[1]-1, cx1 + EXPAND_PX)

    # ── Expanded + filled mask ─────────────────────────────────────────────────
    mask_exp = build_expanded_mask(mask, y0, y1, cx0, cx1, exp_x0, exp_x1)

    # ── Skeleton ───────────────────────────────────────────────────────────────
    skel = morphological_skeleton(mask_exp)

    # ── Graph ─────────────────────────────────────────────────────────────────
    adj, yx_coords, junctions, endpoints = build_skeleton_graph(skel)

    # ── Graph diagnostics ──────────────────────────────────────────────────────
    cx_c = (cx0 + cx1) // 2
    total_skel_px = int((skel > 127).sum())
    jct_xs = [yx_coords[j][1] for j in junctions]
    on_axis_jcts = [j for j in junctions if abs(yx_coords[j][1] - cx_c) <= CORRIDOR_TOL]
    print(f"  skel_px={total_skel_px}  junctions={len(junctions)}  "
          f"  on_axis_jcts={len(on_axis_jcts)}  "
          f"  cx_centre={cx_c}  "
          f"  jct_x_range=[{min(jct_xs) if jct_xs else 'N/A'}, {max(jct_xs) if jct_xs else 'N/A'}]")

    # ── Landmarks ──────────────────────────────────────────────────────────────
    nodes_yx, top_node_y, base_y, apex_y = find_landmarks(
        adj, yx_coords, junctions, endpoints, cx0, cx1
    )
    # stem_h = base to topmost node  (user-defined "meristem" = top leaf attachment)
    stem_h  = (base_y - top_node_y) if (top_node_y is not None and base_y is not None) else None
    # plant_h = base to plant apex   (full plant height including tip above top node)
    plant_h = (base_y - apex_y)     if (apex_y     is not None and base_y is not None) else None

    elapsed = time.time() - t0
    print(f"nodes={len(nodes_yx)}  stem_h(top-node->base)={stem_h}  "
          f"plant_h(apex->base)={plant_h}  ({elapsed:.1f}s)")
    # Print all node heights from base (for debugging)
    if nodes_yx and base_y is not None:
        node_hts = [base_y - ny for ny, _ in nodes_yx]
        print(f"  node heights from base (top->bottom): {node_hts}")
    summary_rows.append(dict(ts=ts, nodes=len(nodes_yx),
                             top_node_y=top_node_y, base_y=base_y,
                             stem_h=stem_h, plant_h=plant_h, apex_y=apex_y))

    # ── LEFT PANEL: mask + skeleton overlay ────────────────────────────────────
    vis = render_skeleton_overlay(
        mask, skel, adj, yx_coords, junctions,
        nodes_yx, top_node_y, apex_y, base_y,
        cx0, cx1, y0, y1,
    )
    margin = 30
    yc0 = max(0, y0 - margin);       yc1 = min(mask.shape[0], y1 + margin)
    xc0 = max(0, exp_x0 - margin);   xc1 = min(mask.shape[1], exp_x1 + margin)

    ax_img.imshow(vis[yc0:yc1, xc0:xc1])
    ax_img.set_title(
        f"{ts}\n"
        f"nodes={len(nodes_yx)}  stem_h(top-node→base)={stem_h}px  "
        f"plant_h(apex→base)={plant_h}px\n"
        f"angle_thresh={STEM_ANGLE_THRESH}°  min_branch={MIN_BRANCH_LEN}px",
        fontsize=7,
    )
    ax_img.axis("off")

    leg = [
        mpatches.Patch(color=(80/255, 120/255, 1),      label=f"stem  (angle < {STEM_ANGLE_THRESH}°)"),
        mpatches.Patch(color=(1, 140/255, 30/255),      label=f"petiole (angle ≥ {STEM_ANGLE_THRESH}°)"),
        mpatches.Patch(color=(0, 230/255, 230/255),     label="node (junction w/ petiole)"),
        mpatches.Patch(color=(1, 50/255, 50/255),       label="top node = 'meristem'"),
        mpatches.Patch(color=(200/255, 80/255, 200/255),label="plant apex (tip)"),
        mpatches.Patch(color=(50/255, 220/255, 80/255), label="base (pot rim)"),
        mpatches.Patch(color=(1, 220/255, 0),           label="corridor boundary"),
    ]
    ax_img.legend(handles=leg, fontsize=5.5, loc="lower left",
                  framealpha=0.65, handlelength=1, ncol=2)

    # ── RIGHT PANEL: plant diagram ─────────────────────────────────────────────
    draw_plant_diagram(ax_dia, nodes_yx, top_node_y, apex_y, base_y, stem_h, plant_h)

plt.suptitle(
    "Strategy C — skeleton branch tracing + angle classification\n"
    f"Corridor ±{CORRIDOR_HALF_PX}px | expand ±{EXPAND_PX}px | "
    f"close kernel {CLOSE_K}px | stem angle < {STEM_ANGLE_THRESH}° | "
    f"min branch {MIN_BRANCH_LEN}px",
    fontsize=9, y=1.005,
)
plt.tight_layout()
plt.savefig(str(OUT_PATH), dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_PATH}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Frame':<35} {'nodes':>6} {'top_node_y':>11} {'apex_y':>8} "
      f"{'base_y':>8} {'stem_h':>8} {'plant_h':>9}")
print("-" * 90)
for r in summary_rows:
    print(
        f"{r['ts']:<35} {r['nodes']:>6} {str(r['top_node_y']):>11} "
        f"{str(r['apex_y']):>8} {str(r['base_y']):>8} "
        f"{str(r['stem_h']):>8} {str(r['plant_h']):>9}"
    )
