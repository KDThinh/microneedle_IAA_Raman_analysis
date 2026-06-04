"""
_tmp_stem_width_profile_cc.py  -  Step-by-step diagnostic, comparing two
approaches to the corridor mask and their effect on skeleton + node detection.

Layout: N_SAMPLES rows x 5 columns

  Col 1  INPUT          : original mask + corridor (yellow) + expand zone (dashed grey)

  Col 2  MORPH-ELLIPSE  : current approach -- MORPH_ELLIPSE (k x k) fills the corridor
                          in all directions. Reliable but over-fills vertically.

  Col 3  H-BRIDGE       : proposed approach --
                          - In corridor: MORPH_RECT(k, 1) closes only HORIZONTALLY,
                            bridging the gap between the two Sobel stem lines without
                            any vertical expansion.
                          - Outside corridor: original Sobel edges (unchanged).
                          Advantage: preserves exact petiole-stem junction positions,
                          including short petioles that stay entirely inside the corridor
                          (which the stub/midpoint-axis approach misses entirely).

  Col 4  SKELETON        : Lee 1994 skeleton of the H-BRIDGE expanded mask.
                          Cyan=path  Red 3x3=junction  Blue=endpoint

  Col 5  BRANCHES+NODES  : branch classification + node detection on the H-Bridge skeleton.
                          Blue=stem  Orange=petiole
                          Magenta dot=in-corridor junction candidate
                          Cyan circle=qualified node
                          Red/Purple/Green crosses=top node/apex/base

Measured stem gap (from _tmp_gap_measure.py):
  May 7 : 18-39 px  |  May 9 : 2-25 px  |  May 11 : 8-30 px
  => H_BRIDGE_K=43 (MORPH_RECT width, covers max gap + margin)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from skimage.morphology import skeletonize as sk_skeletonize
from scipy.interpolate import interp1d

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
    r"\stem_width_profile_cc.png"
)

# ── Corridor detection ─────────────────────────────────────────────────────
CORRIDOR_HALF_PX = 40
BOTTOM_FRAC      = 0.15
PROJ_BAND        = (0.20, 0.85)
ANCHOR_MARGIN    = 150
SMOOTH_K         = 15

# ── Step 2 params ──────────────────────────────────────────────────────────
EXPAND_PX    = 200
CLOSE_K      = 41   # MORPH_ELLIPSE kernel (Col 2 comparison only)
H_BRIDGE_K   = 43   # MORPH_RECT width for horizontal-only gap close (Col 3)
               # Covers max measured gap (39 px) + 4 px margin

# ── Step 4 params ──────────────────────────────────────────────────────────
BRANCH_TRACE_LEN  = 200
STEM_ANGLE_THRESH = 25   # deg: branches steeper than this are "stem" (not petiole)
MIN_BRANCH_LEN    = 50   # px: minimum skeleton branch length to qualify as petiole
CORRIDOR_TOL      = 35   # px: how far from cx_centre a junction can be and still be
                          # considered "on the stem".  35 covers the stem Sobel lines
                          # which are ~20 px from centre (with H-Bridge skeleton
                          # the junction can land on the rail, not just the centre)
MIN_APEX_SEP      = 250  # px below apex to exclude (young unexpanded leaf junctions
                          # cluster 90-220 px below apex; real meristem is 300-500 px)
NODE_CLUSTER_PX   = 40   # merge qualified nodes within this y-range into one

N_SAMPLES = 3
MARGIN    = 20
DPI       = 180


# ══════════════════════════════════════════════════════════════════════════
# Corridor centre estimation (v3, unchanged)
# ══════════════════════════════════════════════════════════════════════════

def col_max_run_lengths(b):
    mx  = np.zeros(b.shape[1], np.int32)
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
    ca = (x0 + int(np.median(xs))) if len(xs) else (x0+x1)//2
    sx0 = max(x0, ca - ANCHOR_MARGIN); sx1 = min(x1, ca + ANCHOR_MARGIN)
    bt  = int(y0 + PROJ_BAND[0] * ph); bb = int(y0 + PROJ_BAND[1] * ph)
    band = mask[bt:bb+1, sx0:sx1+1] > 127
    if not band.any():
        return ca
    proj = col_max_run_lengths(band).astype(float)
    sm   = np.convolve(proj, np.ones(SMOOTH_K)/SMOOTH_K, mode='same')
    return sx0 + int(np.argmax(sm))


# ══════════════════════════════════════════════════════════════════════════
# Step 2 — two mask variants
# ══════════════════════════════════════════════════════════════════════════

def build_morph_close_mask(mask, y0, y1, cx0, cx1, exp_x0, exp_x1):
    """Col 2: MORPH_ELLIPSE (k x k) — current approach, over-fills vertically."""
    out = np.zeros_like(mask)
    out[:, exp_x0:cx0]        = mask[:, exp_x0:cx0]
    out[:, cx1+1:exp_x1+1]    = mask[:, cx1+1:exp_x1+1]
    k  = CLOSE_K if CLOSE_K % 2 == 1 else CLOSE_K + 1
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    filled = cv2.morphologyEx(mask[y0:y1+1, cx0:cx1+1], cv2.MORPH_CLOSE, el)
    out[y0:y1+1, cx0:cx1+1] = filled
    return out


def build_hbridge_mask(mask, y0, y1, cx0, cx1, exp_x0, exp_x1):
    """
    Col 3: MORPH_RECT (H_BRIDGE_K x 1) inside the corridor -- closes only
    horizontally, bridging the gap between the two Sobel stem lines without
    any vertical expansion.

    Key advantage: the original petiole-stem junction positions (where the
    petiole Sobel edge meets the stem Sobel line inside the corridor) are
    preserved exactly.  Petioles that are short and stay entirely inside the
    corridor are still captured, which the midpoint-axis/stub approach misses.

    Outside the corridor: original Sobel edges unchanged.
    """
    out = np.zeros_like(mask)
    out[:, exp_x0:cx0]        = mask[:, exp_x0:cx0]
    out[:, cx1+1:exp_x1+1]    = mask[:, cx1+1:exp_x1+1]
    k  = H_BRIDGE_K if H_BRIDGE_K % 2 == 1 else H_BRIDGE_K + 1
    el = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    filled = cv2.morphologyEx(mask[y0:y1+1, cx0:cx1+1], cv2.MORPH_CLOSE, el)
    out[y0:y1+1, cx0:cx1+1] = filled
    return out


# ══════════════════════════════════════════════════════════════════════════
# Steps 3 & 4 — skeleton graph + branch analysis
# ══════════════════════════════════════════════════════════════════════════

def morphological_skeleton(mask_u8):
    return sk_skeletonize(mask_u8 > 127).astype(np.uint8) * 255


def build_skeleton_graph(skel):
    pts = np.argwhere(skel > 127)
    if len(pts) == 0:
        return {}, {}, set(), set()
    pos_map   = {(int(p[0]), int(p[1])): i for i, p in enumerate(pts)}
    yx_coords = {i: (int(pts[i,0]), int(pts[i,1])) for i in range(len(pts))}
    adj       = {i: [] for i in range(len(pts))}
    for i, (y, x) in enumerate(pts):
        y, x = int(y), int(x)
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                if dy == 0 and dx == 0:
                    continue
                j = pos_map.get((y+dy, x+dx))
                if j is not None:
                    adj[i].append(j)
    junctions = {i for i, nb in adj.items() if len(nb) >= 3}
    endpoints = {i for i, nb in adj.items() if len(nb) == 1}
    return adj, yx_coords, junctions, endpoints


def trace_branch(start, came_from, adj, junctions):
    path = [came_from, start]
    while len(path) <= BRANCH_TRACE_LEN:
        cur = path[-1]; prev = path[-2]
        nexts = [n for n in adj[cur] if n != prev]
        if not nexts or cur in junctions:
            break
        path.append(nexts[0])
    return path


def angle_from_vertical(path, yx_coords):
    if len(path) < 2:
        return 90.0
    pts = np.array([yx_coords[i] for i in path], dtype=float)
    if len(pts) < 3:
        dy = pts[-1,0]-pts[0,0]; dx = pts[-1,1]-pts[0,1]
        return 90.0 if abs(dy)<1e-6 else float(np.degrees(np.arctan2(abs(dx), abs(dy))))
    c = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    dy, dx = vt[0]
    return float(np.degrees(np.arctan2(abs(dx), abs(dy))))


def find_landmarks(adj, yx_coords, junctions, cx0, cx1):
    """
    Find nodes (leaf attachment junctions) + plant apex + base.

    For the H-Bridge mask, junctions on the stem skeleton land on the skeleton
    centreline which lies between the two Sobel rails (roughly cx +/- 20 px
    from the corridor centre).  CORRIDOR_TOL=35 covers this range.

    apex_y  : topmost skeleton pixel anywhere in the corridor (plant tip)
    base_y  : bottommost skeleton pixel anywhere in the corridor (pot rim)
    nodes_yx: qualified leaf-node junctions (in corridor, far enough from apex,
              with at least one branch that is angled + long enough)
    """
    cx_c = (cx0 + cx1) // 2

    def in_corr(i):
        x = yx_coords[i][1]
        return (cx0 - CORRIDOR_TOL) <= x <= (cx1 + CORRIDOR_TOL)

    def near_axis(i):
        return abs(yx_coords[i][1] - cx_c) <= CORRIDOR_TOL

    corr_ys = [yx_coords[i][0] for i in yx_coords if in_corr(i)]
    apex_y  = int(min(corr_ys)) if corr_ys else None
    base_y  = int(max(corr_ys)) if corr_ys else None

    nodes_yx = []
    for jct in junctions:
        if not near_axis(jct):
            continue
        jy, jx = yx_coords[jct]
        # Exclude junctions too close to the apex (young/unexpanded leaf tips)
        if apex_y is not None and (jy - apex_y) < MIN_APEX_SEP:
            continue
        # Require at least one branch with petiole-like angle and length
        for nb in adj[jct]:
            path  = trace_branch(nb, jct, adj, junctions)
            angle = angle_from_vertical(path, yx_coords)
            if angle >= STEM_ANGLE_THRESH and len(path) >= MIN_BRANCH_LEN:
                nodes_yx.append((jy, jx))
                break

    nodes_yx.sort(key=lambda p: p[0])

    # Cluster nearby nodes: the same petiole can produce T-junctions over
    # several consecutive rows; merge them into one node per petiole.
    if nodes_yx:
        clustered = []
        cluster   = [nodes_yx[0]]
        for node in nodes_yx[1:]:
            if node[0] - cluster[-1][0] <= NODE_CLUSTER_PX:
                cluster.append(node)
            else:
                cy    = int(np.median([p[0] for p in cluster]))
                cx_cl = int(np.median([p[1] for p in cluster]))
                clustered.append((cy, cx_cl))
                cluster = [node]
        cy    = int(np.median([p[0] for p in cluster]))
        cx_cl = int(np.median([p[1] for p in cluster]))
        clustered.append((cy, cx_cl))
        nodes_yx = clustered

    top_node_y = nodes_yx[0][0] if nodes_yx else None
    return nodes_yx, top_node_y, apex_y, base_y


# ══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════

def _draw_boundaries(vis, y0, y1, cx0, cx1, exp_x0, exp_x1):
    for bx in (cx0, cx1):
        vis[y0:y1+1, bx] = [255, 220, 0]
    for y in range(y0, y1+1):
        if (y // 8) % 2 == 0:
            for bx in (exp_x0, exp_x1):
                if 0 <= bx < vis.shape[1]:
                    vis[y, bx] = [160, 160, 160]


def vis_col1_input(mask, y0, y1, cx0, cx1, exp_x0, exp_x1):
    vis = np.stack([(mask // 5)] * 3, axis=2).astype(np.uint8)
    _draw_boundaries(vis, y0, y1, cx0, cx1, exp_x0, exp_x1)
    return vis


def vis_col2_morph(mask, mask_exp, y0, y1, cx0, cx1, exp_x0, exp_x1):
    vis = np.full((*mask.shape, 3), 15, dtype=np.uint8)
    lf = mask_exp[:, exp_x0:cx0] > 127
    vis[:, exp_x0:cx0][lf]          = [200, 200, 200]
    rf = mask_exp[:, cx1+1:exp_x1+1] > 127
    vis[:, cx1+1:exp_x1+1][rf]      = [200, 200, 200]
    cf = mask_exp[y0:y1+1, cx0:cx1+1] > 127
    vis[y0:y1+1, cx0:cx1+1][cf]     = [50, 200, 80]
    vis[y0:y1+1, cx0:cx1+1][~cf]    = [18, 50, 22]
    _draw_boundaries(vis, y0, y1, cx0, cx1, exp_x0, exp_x1)
    return vis


def vis_col3_hbridge(mask, mask_hb, y0, y1, cx0, cx1, exp_x0, exp_x1):
    """
    H-Bridge mask.  In the corridor:
    - pixels from original Sobel edges shown in white
    - pixels added by horizontal close shown in yellow-green
    Outside corridor: original Sobel edges (white).
    """
    vis = np.full((*mask.shape, 3), 15, dtype=np.uint8)
    # Flanks: original Sobel edges
    lf = mask_hb[:, exp_x0:cx0] > 127
    vis[:, exp_x0:cx0][lf]          = [200, 200, 200]
    rf = mask_hb[:, cx1+1:exp_x1+1] > 127
    vis[:, cx1+1:exp_x1+1][rf]      = [200, 200, 200]
    # Corridor: original edges (white) vs added bridge pixels (yellow-green)
    orig_corr = mask[y0:y1+1, cx0:cx1+1] > 127
    hb_corr   = mask_hb[y0:y1+1, cx0:cx1+1] > 127
    added     = hb_corr & ~orig_corr
    vis[y0:y1+1, cx0:cx1+1][orig_corr] = [220, 220, 220]  # original
    vis[y0:y1+1, cx0:cx1+1][added]     = [200, 230, 50]   # added by bridge
    vis[y0:y1+1, cx0:cx1+1][~hb_corr]  = [18, 50, 22]     # empty (dark green)
    _draw_boundaries(vis, y0, y1, cx0, cx1, exp_x0, exp_x1)
    return vis


def vis_col4_skeleton(mask_hb, adj, yx_coords, junctions, endpoints,
                      y0, y1, cx0, cx1):
    vis = np.stack([(mask_hb // 8)] * 3, axis=2).astype(np.uint8)
    for idx, (py, px) in yx_coords.items():
        if idx not in junctions and idx not in endpoints:
            vis[py, px] = [0, 180, 180]
    for idx in endpoints:
        py, px = yx_coords[idx]
        vis[py, px] = [80, 80, 255]
    for idx in junctions:
        py, px = yx_coords[idx]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                vy, vx = py+dy, px+dx
                if 0 <= vy < vis.shape[0] and 0 <= vx < vis.shape[1]:
                    vis[vy, vx] = [255, 70, 30]
    vis[y0:y1+1, cx0] = [255, 220, 0]
    vis[y0:y1+1, cx1] = [255, 220, 0]
    return vis


def vis_col5_branches(mask, adj, yx_coords, junctions, cx0, cx1,
                      nodes_yx, top_node_y, apex_y, base_y, y0, y1):
    cx_c = (cx0 + cx1) // 2
    vis  = np.stack([(mask // 7)] * 3, axis=2).astype(np.uint8)
    painted = set()
    for jct in junctions:
        for nb in adj[jct]:
            key = (min(jct,nb), max(jct,nb))
            if key in painted: continue
            path  = trace_branch(nb, jct, adj, junctions)
            ang   = angle_from_vertical(path, yx_coords)
            color = (80,120,255) if ang < STEM_ANGLE_THRESH else (255,140,30)
            for idx in path:
                py, px = yx_coords[idx]
                if 0<=py<vis.shape[0] and 0<=px<vis.shape[1]:
                    vis[py,px] = color
            for a,b in zip(path[:-1],path[1:]):
                painted.add((min(a,b), max(a,b)))
    for idx, (py, px) in yx_coords.items():
        rv = int(vis[py,px,0]); gv = int(vis[py,px,1]); bv = int(vis[py,px,2])
        dv = int(mask[py,px])//7
        if rv<=dv+3 and gv<=dv+3 and bv<=dv+3:
            vis[py,px] = [155,155,155]
    # Highlight corridor zone
    bx0 = max(0, cx_c - CORRIDOR_TOL)
    bx1 = min(vis.shape[1]-1, cx_c + CORRIDOR_TOL)
    vis[y0:y1+1, bx0:bx1+1] = np.clip(
        vis[y0:y1+1, bx0:bx1+1].astype(np.int32)+[0,0,25], 0, 255).astype(np.uint8)
    # Mark junction candidates (in corridor zone)
    for jct in junctions:
        py, px = yx_coords[jct]
        is_near = abs(px - cx_c) <= CORRIDOR_TOL
        vis[py,px] = [230,0,230] if is_near else [90,70,50]
    vis[y0:y1+1, cx0] = [255,220,0]; vis[y0:y1+1, cx1] = [255,220,0]
    # Qualified nodes (cyan circles)
    for ny, nx in nodes_yx:
        for dy in range(-6,7):
            for dx in range(-6,7):
                if dy*dy+dx*dx<=36:
                    vy,vx = ny+dy, nx+dx
                    if 0<=vy<vis.shape[0] and 0<=vx<vis.shape[1]:
                        vis[vy,vx] = [0,230,230]
    def draw_cross(cy, col, r=8):
        for d in range(-r,r+1):
            for vy,vx in [(cy+d,cx_c),(cy,cx_c+d)]:
                if 0<=vy<vis.shape[0] and 0<=vx<vis.shape[1]:
                    vis[vy,vx] = col
    if top_node_y is not None: draw_cross(top_node_y, [255,50,50],  r=9)
    if apex_y     is not None: draw_cross(apex_y,     [200,80,200], r=7)
    if base_y     is not None: draw_cross(base_y,     [50,220,80],  r=9)
    return vis


# ══════════════════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════════════════

masks   = sorted(MASKS_DIR.glob("*_mask_refined.png"))
if not masks:
    raise FileNotFoundError(f"No masks in {MASKS_DIR}")
step    = max(1, len(masks) // N_SAMPLES)
sampled = masks[::step][:N_SAMPLES]
tss     = [m.stem.replace("tl_","").replace("_mask_refined","") for m in sampled]
print(f"Total masks: {len(masks)} | sampling {len(sampled)}")
print(f"Frames: {tss}\n")

fig, axes = plt.subplots(N_SAMPLES, 5,
                         figsize=(30, N_SAMPLES * 9),
                         gridspec_kw={"wspace": 0.04, "hspace": 0.22})

COL_TITLES = [
    ("Col 1 -- INPUT\n"
     "Original Sobel edge mask\n"
     f"Corridor +/-{CORRIDOR_HALF_PX}px (yellow)\n"
     f"Expand +/-{EXPAND_PX}px (grey dashed)"),

    ("Col 2 -- MORPH-ELLIPSE  (current)\n"
     f"MORPH_ELLIPSE {CLOSE_K}x{CLOSE_K}px\n"
     "Fills in all directions -- over-fills vertically\n"
     "Green=filled  Dark=empty in corridor"),

    ("Col 3 -- H-BRIDGE  (proposed)\n"
     f"MORPH_RECT {H_BRIDGE_K}x1px -- horizontal gap close only\n"
     "Bridges stem gap without vertical expansion\n"
     "White=original Sobel  Yellow-green=added bridge"),

    ("Col 4 -- SKELETON of H-Bridge mask\n"
     "Lee 1994 thinning (1-px wide)\n"
     "Cyan=path  Red 3x3=junction  Blue=tip\n"
     "No direct params -- quality from Col 3"),

    (f"Col 5 -- BRANCHES + NODES (H-Bridge)\n"
     f"Blue=stem(<{STEM_ANGLE_THRESH}deg) Orange=petiole(>={STEM_ANGLE_THRESH}deg)\n"
     f"Magenta=in-corridor junction (|x-cx|<={CORRIDOR_TOL}px)\n"
     f"Cyan circle=node  Red/Purple/Green=top-node/apex/base"),
]

for row, (mask_path, ts) in enumerate(zip(sampled, tss)):
    print(f"[{row+1}/{len(sampled)}] {ts} ...", end=" ", flush=True)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("READ ERROR"); continue

    ys, xs = np.where(mask > 127)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cx     = estimate_stem_cx(mask, y0, y1, x0, x1)
    cx0    = max(0,               cx - CORRIDOR_HALF_PX)
    cx1    = min(mask.shape[1]-1, cx + CORRIDOR_HALF_PX)
    exp_x0 = max(0,               cx0 - EXPAND_PX)
    exp_x1 = min(mask.shape[1]-1, cx1 + EXPAND_PX)

    # Step 2 -- two mask variants
    mask_close   = build_morph_close_mask(mask, y0, y1, cx0, cx1, exp_x0, exp_x1)
    mask_hb      = build_hbridge_mask(mask, y0, y1, cx0, cx1, exp_x0, exp_x1)

    # Step 3 -- skeleton of the H-Bridge mask
    skel = morphological_skeleton(mask_hb)

    # Step 4 -- graph + landmarks
    adj, yx_coords, junctions, endpoints = build_skeleton_graph(skel)
    nodes_yx, top_node_y, apex_y, base_y = find_landmarks(adj, yx_coords, junctions, cx0, cx1)
    stem_h  = (base_y - top_node_y) if (top_node_y and base_y) else None
    plant_h = (base_y - apex_y)     if (apex_y and base_y)     else None

    cx_c    = (cx0+cx1)//2
    on_corr = sum(1 for j in junctions if abs(yx_coords[j][1]-cx_c)<=CORRIDOR_TOL)
    node_hts = [base_y - ny for ny, _ in nodes_yx] if (nodes_yx and base_y) else []

    print(f"skel={int((skel>127).sum())}px  junctions={len(junctions)}  "
          f"on-corridor={on_corr}  nodes={len(nodes_yx)}  "
          f"stem_h={stem_h}  plant_h={plant_h}")
    if node_hts:
        print(f"  node heights from base (top->bottom): {node_hts}")

    # Crop (same for all 5 panels)
    yc0 = max(0,             y0     - MARGIN)
    yc1 = min(mask.shape[0], y1     + MARGIN)
    xc0 = max(0,             exp_x0 - MARGIN)
    xc1 = min(mask.shape[1], exp_x1 + MARGIN)

    def crop(img):
        return img[yc0:yc1, xc0:xc1]

    panels = [
        vis_col1_input(mask, y0, y1, cx0, cx1, exp_x0, exp_x1),
        vis_col2_morph(mask, mask_close, y0, y1, cx0, cx1, exp_x0, exp_x1),
        vis_col3_hbridge(mask, mask_hb, y0, y1, cx0, cx1, exp_x0, exp_x1),
        vis_col4_skeleton(mask_hb, adj, yx_coords, junctions, endpoints, y0, y1, cx0, cx1),
        vis_col5_branches(mask, adj, yx_coords, junctions, cx0, cx1,
                          nodes_yx, top_node_y, apex_y, base_y, y0, y1),
    ]

    for col, (panel, ax) in enumerate(zip(panels, axes[row])):
        ax.imshow(crop(panel))
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(ts, fontsize=8.5, rotation=0, labelpad=130,
                          va="center", fontweight="bold")
        if row == 0:
            ax.set_title(COL_TITLES[col], fontsize=7.5, pad=5,
                         loc="center", linespacing=1.5)
        if col == 4:
            axes[row, 4].set_title(
                f"nodes={len(nodes_yx)}  stem_h={stem_h}px  plant_h={plant_h}px\n"
                f"node hts from base: {node_hts}",
                fontsize=7, loc="center", pad=4)

# ── Legend ──────────────────────────────────────────────────────────────────
fig.subplots_adjust(bottom=0.09)
leg = [
    mpatches.Patch(color=(50/255, 200/255, 80/255),   label="MORPH_ELLIPSE filled corridor  [Col 2 green]"),
    mpatches.Patch(color=(220/255, 220/255, 220/255),  label="Original Sobel edges in corridor  [Col 3 white]"),
    mpatches.Patch(color=(200/255, 230/255, 50/255),   label="Pixels added by horizontal bridge  [Col 3 yellow-green]"),
    mpatches.Patch(color=(200/255, 200/255, 200/255),  label="Original Sobel flank edges  [Cols 2 & 3]"),
    mpatches.Patch(color=(0, 180/255, 180/255),        label="Skeleton path pixel (degree 2)  [Col 4 cyan]"),
    mpatches.Patch(color=(255/255, 70/255, 30/255),    label="Skeleton junction (degree>=3)  [Col 4 red 3x3]"),
    mpatches.Patch(color=(80/255, 80/255, 255/255),    label="Skeleton endpoint  [Col 4 blue]"),
    mpatches.Patch(color=(80/255, 120/255, 255/255),   label=f"Stem branch  angle<{STEM_ANGLE_THRESH}deg  [Col 5 blue]"),
    mpatches.Patch(color=(255/255, 140/255, 30/255),   label=f"Petiole branch  angle>={STEM_ANGLE_THRESH}deg  [Col 5 orange]"),
    mpatches.Patch(color=(230/255, 0, 230/255),        label=f"In-corridor junction |x-cx|<={CORRIDOR_TOL}px  [magenta]"),
    mpatches.Patch(color=(0, 230/255, 230/255),        label="Qualified node (angle+len+apex filters)  [cyan circle]"),
    mpatches.Patch(color=(255/255, 50/255, 50/255),    label="Top node = meristem  [red cross]"),
    mpatches.Patch(color=(200/255, 80/255, 200/255),   label="Plant apex  [purple cross]"),
    mpatches.Patch(color=(50/255, 220/255, 80/255),    label="Base  [green cross]"),
    mpatches.Patch(color=(255/255, 220/255, 0),        label="Corridor boundaries  [yellow]"),
]
fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=6.5,
           framealpha=0.85, handlelength=1.2, columnspacing=1.2,
           bbox_to_anchor=(0.5, 0.0))

plt.suptitle(
    "Strategy C intermediate diagnostic  |  "
    f"CORRIDOR_HALF_PX={CORRIDOR_HALF_PX}px  EXPAND_PX={EXPAND_PX}px\n"
    f"Col 2: MORPH_ELLIPSE {CLOSE_K}x{CLOSE_K}px  |  "
    f"Col 3: MORPH_RECT (H-Bridge) {H_BRIDGE_K}x1px\n"
    f"STEM_ANGLE_THRESH={STEM_ANGLE_THRESH}deg  MIN_BRANCH_LEN={MIN_BRANCH_LEN}px  "
    f"CORRIDOR_TOL={CORRIDOR_TOL}px  MIN_APEX_SEP={MIN_APEX_SEP}px  "
    f"NODE_CLUSTER_PX={NODE_CLUSTER_PX}px",
    fontsize=8.5, y=1.003,
)

plt.savefig(str(OUT_PATH), dpi=DPI, bbox_inches="tight")
plt.close()
print(f"\nSaved: {OUT_PATH}")
