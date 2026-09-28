"""
stem_analysis_pipeline.py

End-to-end batch analysis pipeline for binary Sobel edge masks of Nb plants.

Merges the approaches from:
  _tmp_freq_fill.py         (temporal frequency-voting gap repair)
  _tmp_graph_filter_demo.py (skeleton graph-level branch filtering)

Both source scripts were removed once merged here; see archive/README.md for the
git refs if you need to consult the originals.

PIPELINE (per frame)
--------------------
  1. Frequency-fill temporal gaps      sliding-window voting (±WINDOW_HALF frames)
  2. Morphological post-processing     optional close/open on the filled mask
  3. Lee skeletonisation               1-px medial axis (skimage)
  4. Graph-level branch filter         removes short skeleton branches
  5. Measurements                      plant_h, stem_h, junction count

All frames are always loaded into RAM so the sliding-window frequency map
is computed correctly.  Results are written to a CSV file.

MEASUREMENTS (image-coordinate convention: y=0 at image top)
-------------------------------------------------------------
  plant_h_px       y_bottom − y_top of all skeleton pixels
  stem_h_px        y-extent of skeleton inside MEAS_CORRIDOR_HALF_PX around cx
  skel_total_px    total skeleton pixel count (after graph filter)
  skel_corridor_px skeleton pixels inside the measurement corridor
  n_junctions      junction pixel count (degree ≥ 3)

OUTPUTS
-------
  <DEVICE_DIR>/pipeline_measurements.csv    one row per processed frame
  <DEVICE_DIR>/pipeline_preview.png         N_PREVIEW rows × 3 columns
      Col 1 – original mask
      Col 2 – freq-filled mask  (green = added pixels)
      Col 3 – final skeleton with plant_h (green) and stem_h (yellow) rulers
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from skimage.morphology import skeletonize as sk_skeletonize

# ── Paths ─────────────────────────────────────────────────────────────────────
MASKS_DIR  = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465\masks"
)
DEVICE_DIR = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465"
)
OUT_SKEL_DIR     = DEVICE_DIR / "masks_skeleton"         # coloured overlay
OUT_SKEL_BIN_DIR = DEVICE_DIR / "masks_skeleton_binary"  # skeleton-only binary mask
PREVIEW_PATH     = DEVICE_DIR / "pipeline_preview.png"

# ══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  EDIT THESE  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════════

# ── Frequency fill ────────────────────────────────────────────────────────────
WINDOW_HALF          = 20    # frames each side of target (total window = 2*W+1)
FREQ_THRESHOLD       = 0.2   # pixel must be foreground in this fraction of window frames
RESTRICT_TO_CORRIDOR = True  # True = fill only within corridor x-range
CORRIDOR_HALF_PX     = 40    # px — fill corridor half-width
CORRIDOR_MARGIN      = 80    # px — extra margin beyond corridor for fill region

# ── Morphological post-processing (AFTER freq fill, BEFORE skeletonisation) ───
# Recommended: MORPH_CLOSE only (MORPH_OPEN_R = 0) to avoid erasing thin edges.
MORPH_POSTPROCESS = True      # True = apply ops before skeletonising
MORPH_MODE        = "simple"  # "simple" = close then open  |  "ASF" = alternating
MORPH_ASF_SIZES   = [3, 5]   # ASF kernel diameters per iteration
MORPH_CLOSE_R     = 2        # closing disk radius; fills gaps ≤ 2*R px  (0 = skip)
MORPH_OPEN_R      = 0        # opening disk radius  (0 recommended — avoids erasing thin edges)

# ── Skeletonisation + graph filter ────────────────────────────────────────────
MIN_BRANCH_LEN = 15    # px — skeleton branches shorter than this are removed
USE_ARC_LENGTH = False  # False = pixel count  True = Euclidean arc length (slower)

# ── Measurement corridor ──────────────────────────────────────────────────────
MEAS_CORRIDOR_HALF_PX = 60  # px — corridor half-width for stem_h extraction
                              # (may be wider than the fill corridor)

# ── Overlay appearance ────────────────────────────────────────────────────────
MASK_ALPHA = 0.35   # brightness of the binary mask in the overlay image
                     # 0.0 = invisible  1.0 = fully white  0.35 = dim grey
                     # lower = skeleton (red) stands out more

# ── Processing scope ──────────────────────────────────────────────────────────
PROCESS_ALL = True   # True  = every frame
                      # False = N_SAMPLES evenly-spaced frames
N_SAMPLES   = 20     # used when PROCESS_ALL = False
N_PREVIEW   = 6      # rows shown in the preview PNG

# ══════════════════════════════════════════════════════════════════════════════
# ▲▲▲  END OF USER SETTINGS  ▲▲▲
# ══════════════════════════════════════════════════════════════════════════════

DPI           = 150
MARGIN        = 30          # px border around crop in preview
EXPAND_PX     = 200         # px added to x-extent of preview crop (leaf context)
BOTTOM_FRAC   = 0.15
PROJ_BAND     = (0.20, 0.85)
ANCHOR_MARGIN = 150
SMOOTH_K      = 15
CORRUPT_Y_MAX = 50          # frames with all foreground pixels above this row
                             # are flagged as corrupt (artifact stripe at top)


# ══════════════════════════════════════════════════════════════════════════════
# Corridor centre estimation
# ══════════════════════════════════════════════════════════════════════════════

def _col_max_run_lengths(b):
    mx  = np.zeros(b.shape[1], np.int32)
    cur = np.zeros(b.shape[1], np.int32)
    for row in b:
        cur = np.where(row, cur + 1, 0)
        np.maximum(mx, cur, out=mx)
    return mx


def estimate_stem_cx(mask, y0, y1, x0, x1):
    ph   = y1 - y0
    yc   = int(y1 - BOTTOM_FRAC * ph)
    st   = mask[max(0, yc): y1 + 1, x0: x1 + 1]
    _, xs = np.where(st > 127)
    ca   = (x0 + int(np.median(xs))) if len(xs) else (x0 + x1) // 2
    sx0  = max(x0, ca - ANCHOR_MARGIN)
    sx1  = min(x1, ca + ANCHOR_MARGIN)
    bt   = int(y0 + PROJ_BAND[0] * ph)
    bb   = int(y0 + PROJ_BAND[1] * ph)
    band = mask[bt: bb + 1, sx0: sx1 + 1] > 127
    if not band.any():
        return ca
    proj = _col_max_run_lengths(band).astype(float)
    sm   = np.convolve(proj, np.ones(SMOOTH_K) / SMOOTH_K, mode='same')
    return sx0 + int(np.argmax(sm))


# ══════════════════════════════════════════════════════════════════════════════
# Frequency fill
# ══════════════════════════════════════════════════════════════════════════════

def freq_fill_frame(t, binary_stack, corridor_xlim=None):
    """
    Sliding-window frequency fill for frame t.

    Returns (filled_u8, n_added):
      filled_u8  uint8 (H, W) — OR of original and structural template
      n_added    number of pixels newly set by the template
    """
    N    = binary_stack.shape[0]
    t_lo = max(0,     t - WINDOW_HALF)
    t_hi = min(N - 1, t + WINDOW_HALF)
    freq_map = binary_stack[t_lo: t_hi + 1].mean(axis=0).astype(np.float32)
    template = freq_map >= FREQ_THRESHOLD
    orig     = binary_stack[t]
    filled   = orig.copy()
    if corridor_xlim is not None:
        x_lo, x_hi = corridor_xlim
        filled[:, x_lo: x_hi + 1] |= template[:, x_lo: x_hi + 1]
    else:
        filled |= template
    n_added = int((filled & ~orig).sum())
    return filled.astype(np.uint8) * 255, n_added


# ══════════════════════════════════════════════════════════════════════════════
# Morphological post-processing
# ══════════════════════════════════════════════════════════════════════════════

def morph_postprocess(frame_u8):
    """Apply close/open (simple) or alternating sequential filter (ASF)."""
    out = frame_u8.copy()
    if MORPH_MODE == "ASF":
        for sz in MORPH_ASF_SIZES:
            kr  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kr)
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  kr)
    else:
        if MORPH_CLOSE_R > 0:
            d  = 2 * MORPH_CLOSE_R + 1
            kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kc)
        if MORPH_OPEN_R > 0:
            d  = 2 * MORPH_OPEN_R + 1
            ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ko)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Graph-level branch length filter
# ══════════════════════════════════════════════════════════════════════════════

def _nb8(y, x, pos_set):
    """8-connected neighbours of (y, x) present in pos_set."""
    return [(y + dy, x + dx)
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if (dy or dx) and (y + dy, x + dx) in pos_set]


def _branch_len(path):
    if not USE_ARC_LENGTH:
        return len(path)
    s = 0.0
    for i in range(len(path) - 1):
        dy = path[i + 1][0] - path[i][0]
        dx = path[i + 1][1] - path[i][1]
        s += 1.4142135623730951 if (dy and dx) else 1.0
    return s


def filter_by_branch_length(skel_u8):
    """
    Remove all skeleton branches shorter than MIN_BRANCH_LEN pixels.

    Handles endpoint→junction branches, junction→junction arcs,
    and pure degree-2 closed loops.
    """
    binary = skel_u8 > 127
    if not binary.any():
        return np.zeros_like(skel_u8)

    pos_set  = set(zip(*np.where(binary)))
    deg      = {p: len(_nb8(*p, pos_set)) for p in pos_set}
    node_set = {p for p, d in deg.items() if d != 2}
    keep     = set()
    visited  = set()

    for start in node_set:
        for nb in _nb8(*start, pos_set):
            key = (start, nb)
            if key in visited:
                continue
            visited.add(key)
            path = [start]
            prev, cur = start, nb
            while True:
                path.append(cur)
                if cur in node_set:
                    visited.add((cur, prev))
                    break
                nxts = [p for p in _nb8(*cur, pos_set) if p != prev]
                if not nxts:
                    break
                prev, cur = cur, nxts[0]
            if _branch_len(path) >= MIN_BRANCH_LEN:
                keep.update(path)

    # Pure degree-2 loops (closed rings with no node pixel)
    deg2     = {p for p, d in deg.items() if d == 2}
    vis_loop = set()
    for seed in deg2:
        if seed in vis_loop or seed in keep:
            continue
        comp  = set()
        stack = [seed]
        while stack:
            p = stack.pop()
            if p in vis_loop:
                continue
            vis_loop.add(p)
            comp.add(p)
            for nb in _nb8(*p, pos_set):
                if nb in deg2 and nb not in vis_loop:
                    stack.append(nb)
        if _branch_len(list(comp)) >= MIN_BRANCH_LEN:
            keep.update(comp)

    # Retain node pixels that connect to at least one kept branch pixel
    for node in node_set:
        if any(nb in keep for nb in _nb8(*node, pos_set)):
            keep.add(node)

    out = np.zeros_like(skel_u8)
    for (y, x) in keep:
        out[y, x] = 255
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Measurements
# ══════════════════════════════════════════════════════════════════════════════

def measure_frame(skel_u8, cx, img_w):
    """
    Extract geometry measurements from the filtered skeleton.

    Image-coordinate convention: y = 0 at image top, y increases downward.
      y_top    = smallest y value  (topmost skeleton point = plant apex)
      y_bottom = largest  y value  (lowest  skeleton point = near pot)
      plant_h  = y_bottom − y_top

    Stem height uses skeleton pixels inside the measurement corridor
    x ∈ [cx − MEAS_CORRIDOR_HALF_PX, cx + MEAS_CORRIDOR_HALF_PX].

    Parameters
    ----------
    skel_u8 : uint8 (H, W) filtered skeleton image
    cx      : estimated stem x-centre pixel
    img_w   : image width W (used to clamp corridor bounds)

    Returns
    -------
    dict with plant_h_px, stem_h_px, skel_total_px, skel_corridor_px,
         n_junctions, y_top, y_bottom, stem_y_top, stem_y_bottom.
    """
    empty = dict(plant_h_px=0, stem_h_px=0, skel_total_px=0,
                 skel_corridor_px=0, n_junctions=0,
                 y_top=None, y_bottom=None,
                 stem_y_top=None, stem_y_bottom=None)
    if not (skel_u8 > 127).any():
        return empty

    pos_set = set(zip(*np.where(skel_u8 > 127)))
    ys_all  = [p[0] for p in pos_set]
    y_top   = min(ys_all)
    y_bot   = max(ys_all)

    x_lo = max(0,       cx - MEAS_CORRIDOR_HALF_PX)
    x_hi = min(img_w-1, cx + MEAS_CORRIDOR_HALF_PX)
    corr = [(y, x) for (y, x) in pos_set if x_lo <= x <= x_hi]

    if corr:
        cys    = [p[0] for p in corr]
        sy_top = min(cys)
        sy_bot = max(cys)
        stem_h = sy_bot - sy_top
    else:
        sy_top = sy_bot = None
        stem_h = 0

    n_jct = sum(1 for p in pos_set if len(_nb8(*p, pos_set)) >= 3)

    return dict(
        plant_h_px       = y_bot - y_top,
        stem_h_px        = stem_h,
        skel_total_px    = len(pos_set),
        skel_corridor_px = len(corr),
        n_junctions      = n_jct,
        y_top            = y_top,
        y_bottom         = y_bot,
        stem_y_top       = sy_top,
        stem_y_bottom    = sy_bot,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _dim(img_u8):
    return (img_u8 // 8).astype(np.uint8)


def vis_original(mask_u8):
    """White foreground on dimmed background."""
    H, W = mask_u8.shape
    v = np.zeros((H, W, 3), dtype=np.uint8)
    d = _dim(mask_u8)
    v[..., 0] = d; v[..., 1] = d; v[..., 2] = d
    ys, xs = np.where(mask_u8 > 127)
    v[ys, xs] = [255, 255, 255]
    return v


def vis_filled(orig_u8, filled_u8):
    """White = unchanged foreground; green = added by freq fill."""
    H, W = orig_u8.shape
    v = np.zeros((H, W, 3), dtype=np.uint8)
    d = _dim(orig_u8)
    v[..., 0] = d; v[..., 1] = d; v[..., 2] = d
    ys, xs = np.where(orig_u8 > 127)
    v[ys, xs] = [255, 255, 255]
    ys2, xs2 = np.where((filled_u8 > 127) & ~(orig_u8 > 127))
    v[ys2, xs2] = [0, 220, 80]
    return v


def vis_skeleton_with_rulers(skel_u8, mask_bg, meas, mcx0, mcx1):
    """
    Cyan skeleton on dimmed background with measurement rulers:
      green lines  = plant_h  (full image width)
      yellow lines = stem_h   (corridor width only)
    """
    H, W = skel_u8.shape
    v = np.zeros((H, W, 3), dtype=np.uint8)
    d = _dim(mask_bg)
    v[..., 0] = d; v[..., 1] = d; v[..., 2] = d
    ys, xs = np.where(skel_u8 > 127)
    v[ys, xs] = [0, 200, 200]                  # cyan skeleton

    if meas['y_top'] is not None:              # green plant_h rulers
        v[meas['y_top'],    :] = [0, 255, 0]
        v[meas['y_bottom'], :] = [0, 255, 0]

    if meas['stem_y_top'] is not None:         # yellow stem_h rulers
        v[meas['stem_y_top'],    mcx0: mcx1+1] = [255, 220, 0]
        v[meas['stem_y_bottom'], mcx0: mcx1+1] = [255, 220, 0]

    return v


def draw_corridor_lines(vis, y0, y1, cx0, cx1):
    H, W = vis.shape[:2]
    y0c = max(0, y0); y1c = min(H, y1 + 1)
    if 0 <= cx0 < W: vis[y0c:y1c, cx0] = [255, 220, 0]
    if 0 <= cx1 < W: vis[y0c:y1c, cx1] = [255, 220, 0]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Load all frames ────────────────────────────────────────────────────────
all_paths = sorted(MASKS_DIR.glob("*_mask_refined.png"))
if not all_paths:
    raise FileNotFoundError(f"No masks found in {MASKS_DIR}")
N_total = len(all_paths)
print(f"Found {N_total} masks.")

print("Loading all frames into RAM …")
first = cv2.imread(str(all_paths[0]), cv2.IMREAD_GRAYSCALE)
if first is None:
    raise IOError(f"Cannot read {all_paths[0]}")
H, W = first.shape
stack_raw = np.zeros((N_total, H, W), dtype=np.uint8)
for i, p in enumerate(all_paths):
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is not None:
        stack_raw[i] = img
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{N_total}")
print(f"Stack: {stack_raw.shape}  ({stack_raw.nbytes / 1e6:.0f} MB)")

binary_stack = stack_raw > 127   # bool array (N, H, W)

# ── 2. Estimate corridor centre (median over non-corrupt sample frames) ────────
print("Estimating corridor centre …")
samp_idx = np.linspace(0, N_total - 1, min(10, N_total), dtype=int)
cx_list  = []
for i in samp_idx:
    m = stack_raw[i]
    ys, xs = np.where(m > 127)
    if len(ys) < 100:
        continue
    if int(ys.max()) < CORRUPT_Y_MAX:
        continue                               # skip corrupt artifact frames
    cx_list.append(estimate_stem_cx(
        m, int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())))
if not cx_list:
    raise RuntimeError("Could not estimate corridor centre — check mask files.")
cx_med = int(np.median(cx_list))
print(f"  cx = {cx_med}")

# Fill corridor bounds
fill_cx0 = max(0,   cx_med - CORRIDOR_HALF_PX)
fill_cx1 = min(W-1, cx_med + CORRIDOR_HALF_PX)
if RESTRICT_TO_CORRIDOR:
    corridor_xlim = (max(0,   fill_cx0 - CORRIDOR_MARGIN),
                     min(W-1, fill_cx1 + CORRIDOR_MARGIN))
    print(f"  fill region: x=[{corridor_xlim[0]}, {corridor_xlim[1]}]")
else:
    corridor_xlim = None
    print(f"  fill region: full width ({W}px)")

# Measurement corridor bounds
mcx0 = max(0,   cx_med - MEAS_CORRIDOR_HALF_PX)
mcx1 = min(W-1, cx_med + MEAS_CORRIDOR_HALF_PX)
print(f"  meas corridor: x=[{mcx0}, {mcx1}]")

# ── 3. Select frames to process ───────────────────────────────────────────────
if PROCESS_ALL:
    proc_idx = list(range(N_total))
else:
    step     = max(1, N_total // N_SAMPLES)
    proc_idx = list(range(0, N_total, step))[:N_SAMPLES]
print(f"\nProcessing {len(proc_idx)}/{N_total} frames …")

morph_desc = (
    (f"ASF {MORPH_ASF_SIZES}") if MORPH_MODE == "ASF"
    else f"close r={MORPH_CLOSE_R} / open r={MORPH_OPEN_R}"
) if MORPH_POSTPROCESS else "none"

print(f"  freq fill : window ±{WINDOW_HALF} fr, threshold={FREQ_THRESHOLD}")
print(f"  morph     : {morph_desc}")
print(f"  graph GF  : min_branch={MIN_BRANCH_LEN}px")
print()

# ── 4. Pipeline loop ──────────────────────────────────────────────────────────
OUT_SKEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_SKEL_BIN_DIR.mkdir(parents=True, exist_ok=True)
preview_rows = []       # tuples for preview PNG

for idx, t in enumerate(proc_idx):
    orig_u8 = stack_raw[t]
    ts      = all_paths[t].stem.replace("tl_", "").replace("_mask_refined", "")

    # Corrupt-frame detection
    ys_orig = np.where(orig_u8 > 127)[0]
    corrupt = (len(ys_orig) > 0) and (int(ys_orig.max()) < CORRUPT_Y_MAX)

    # Step 1 — frequency fill
    filled_u8, n_added = freq_fill_frame(t, binary_stack, corridor_xlim)

    # Step 2 — morphological post-processing
    mask_for_skel = morph_postprocess(filled_u8) if MORPH_POSTPROCESS else filled_u8

    # Step 3 — skeletonise
    skel_u8 = sk_skeletonize(mask_for_skel > 127).astype(np.uint8) * 255

    # Step 4 — graph filter
    skel_gf = filter_by_branch_length(skel_u8)

    # Step 5 — measure (for preview labels only)
    meas = measure_frame(skel_gf, cx_med, W)

    # ── Save binary skeleton mask (grayscale, 0/255) ─────────────────────────
    cv2.imwrite(str(OUT_SKEL_BIN_DIR / all_paths[t].name), skel_gf)

    # ── Save coloured skeleton overlay ───────────────────────────────────────
    # Layer order (bottom→top): background | dim mask | dashed corridor | skeleton
    overlay  = np.zeros((H, W, 3), dtype=np.uint8)
    mask_val = int(255 * MASK_ALPHA)               # dim-grey mask level
    ys_fg, xs_fg = np.where(orig_u8 > 127)
    overlay[ys_fg, xs_fg] = [mask_val, mask_val, mask_val]
    # Dashed grey corridor edges (10 px on / 10 px off), lighter than mask
    _DASH, _GAP, _CORR_VAL = 10, 10, max(mask_val + 30, 140)
    for _y in range(0, H, _DASH + _GAP):
        _y1 = min(H, _y + _DASH)
        if 0 <= fill_cx0 < W: overlay[_y:_y1, fill_cx0] = [_CORR_VAL, _CORR_VAL, _CORR_VAL]
        if 0 <= fill_cx1 < W: overlay[_y:_y1, fill_cx1] = [_CORR_VAL, _CORR_VAL, _CORR_VAL]
    # Red skeleton drawn on top
    ys_sk, xs_sk = np.where(skel_gf > 127)
    overlay[ys_sk, xs_sk] = [220, 30, 30]
    cv2.imwrite(str(OUT_SKEL_DIR / all_paths[t].name),
                cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    preview_rows.append((t, orig_u8, filled_u8, skel_gf, meas, ts, corrupt))

    if (idx + 1) % 20 == 0 or idx == len(proc_idx) - 1:
        print(f"  [{idx+1:>3}/{len(proc_idx)}]  frame {t:>4}  {ts}"
              f"  skel={meas['skel_total_px']}px"
              f"  jct={meas['n_junctions']}"
              + ("  CORRUPT" if corrupt else ""))

print(f"\nSkeleton overlays  ({len(proc_idx)} frames)  →  {OUT_SKEL_DIR}")
print(f"Skeleton binary    ({len(proc_idx)} frames)  →  {OUT_SKEL_BIN_DIR}")

# ── 5. Preview PNG ────────────────────────────────────────────────────────────
n_prev = min(N_PREVIEW, len(preview_rows))
step_p = max(1, len(preview_rows) // n_prev)
prev_sel = preview_rows[::step_p][:n_prev]

N_COLS = 3
fig = plt.figure(figsize=(8 * N_COLS, n_prev * 9))
gs  = gridspec.GridSpec(n_prev, N_COLS, figure=fig, wspace=0.04, hspace=0.22)
axes = [[fig.add_subplot(gs[r, c]) for c in range(N_COLS)]
        for r in range(n_prev)]

COL_TITLES = [
    "Original mask\n(raw *_mask_refined.png)",
    f"Freq-filled mask\n"
    f"window ±{WINDOW_HALF} frames  |  threshold = {FREQ_THRESHOLD}\n"
    f"white = unchanged  |  green = added pixels",
    f"Final skeleton  (morph: {morph_desc}  |  GF: {MIN_BRANCH_LEN}px)\n"
    f"cyan = skeleton\n"
    f"green ruler = plant_h  |  yellow ruler = stem_h",
]

for row, (t, orig_u8, filled_u8, skel_gf, meas, ts, corrupt) in enumerate(prev_sel):

    # ── Crop bounds: use orig for full plant extent; fall back to filled ──────
    ys_o, xs_o = np.where(orig_u8   > 127)
    ys_f, xs_f = np.where(filled_u8 > 127)

    if len(ys_o) > 0:
        y0p, y1p = int(ys_o.min()), int(ys_o.max())
        x0p, x1p = int(xs_o.min()), int(xs_o.max())
    elif len(ys_f) > 0:
        y0p, y1p = int(ys_f.min()), int(ys_f.max())
        x0p, x1p = int(xs_f.min()), int(xs_f.max())
    else:
        # Completely blank frame — placeholder
        for ax in axes[row]:
            ax.set_facecolor('black')
            ax.text(0.5, 0.5, f"{ts}\n(no foreground pixels)",
                    transform=ax.transAxes, ha='center', va='center',
                    color='gray', fontsize=9)
            ax.axis('off')
        continue

    yc0 = max(0,   y0p - MARGIN)
    yc1 = min(H-1, y1p + MARGIN)
    xc0 = max(0,   x0p - EXPAND_PX - MARGIN)
    xc1 = min(W-1, x1p + EXPAND_PX + MARGIN)
    # Enforce minimum crop height so a thin frame doesn't render as a strip
    if (yc1 - yc0) < 200:
        ym  = (yc0 + yc1) // 2
        yc0 = max(0,   ym - 200)
        yc1 = min(H-1, ym + 200)

    def crop(img):
        return img[yc0:yc1, xc0:xc1]

    v_orig   = vis_original(orig_u8)
    v_filled = vis_filled(orig_u8, filled_u8)
    v_skel   = vis_skeleton_with_rulers(skel_gf, filled_u8, meas, mcx0, mcx1)

    draw_corridor_lines(v_orig,   y0p, y1p, fill_cx0, fill_cx1)
    draw_corridor_lines(v_filled, y0p, y1p, fill_cx0, fill_cx1)

    for col, (vis, ax) in enumerate(zip([v_orig, v_filled, v_skel], axes[row])):
        ax.imshow(crop(vis), aspect='auto')
        ax.axis('off')
        if col == 0:
            lbl = f"{ts}  (frame {t})"
            if corrupt:
                lbl += "  ⚠ corrupt"
            ax.set_ylabel(lbl, fontsize=8, rotation=0, labelpad=145,
                          va='center', fontweight='bold',
                          color='red' if corrupt else 'black')
        if row == 0:
            ax.set_title(COL_TITLES[col], fontsize=8, pad=5, linespacing=1.5)
        if col == 2:
            ax.set_xlabel(
                f"plant_h = {meas['plant_h_px']} px  |  "
                f"stem_h = {meas['stem_h_px']} px  |  "
                f"junctions = {meas['n_junctions']}  |  "
                f"skel = {meas['skel_total_px']} px",
                fontsize=6.5, labelpad=3,
            )

# ── Legend ─────────────────────────────────────────────────────────────────────
fig.subplots_adjust(bottom=0.07)
leg = [
    mpatches.Patch(color=(1.,    1.,    1.),    label="Unchanged foreground"),
    mpatches.Patch(color=(0, 220/255, 80/255),  label="Added by freq fill"),
    mpatches.Patch(color=(0, 200/255, 200/255), label="Skeleton (filtered)"),
    mpatches.Patch(color=(0,    1.,    0.),     label="Plant height ruler"),
    mpatches.Patch(color=(1., 220/255, 0.),     label="Stem height ruler / corridor"),
]
fig.legend(handles=leg, loc='lower center', ncol=5, fontsize=7.5,
           framealpha=0.85, handlelength=1.3, columnspacing=1.5,
           bbox_to_anchor=(0.5, 0.0))

restrict_str = (f"corridor x=[{corridor_xlim[0]},{corridor_xlim[1]}]"
                if RESTRICT_TO_CORRIDOR else "full image width")
plt.suptitle(
    f"Pipeline: freq-fill (±{WINDOW_HALF} fr, thr={FREQ_THRESHOLD}, region={restrict_str})"
    f"  →  morph ({morph_desc})"
    f"  →  skel  →  GF ({MIN_BRANCH_LEN}px)\n"
    f"{len(proc_idx)}/{N_total} frames processed  |  "
    f"meas corridor ±{MEAS_CORRIDOR_HALF_PX}px  (cx={cx_med})",
    fontsize=9, y=1.003,
)
plt.savefig(str(PREVIEW_PATH), dpi=DPI, bbox_inches='tight')
plt.close()
print(f"Preview  →  {PREVIEW_PATH}")
print("\nAll done.")
