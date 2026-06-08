"""
_tmp_freq_fill.py

Temporal frequency voting with a sliding window for binary mask gap repair.

IDEA
----
For each target frame t, look at the ±WINDOW_HALF neighbouring frames.
For every pixel (y, x), compute how often it is foreground across that window:

    freq[y, x] = (number of window frames where pixel == 1) / window_size

Pixels with freq >= FREQ_THRESHOLD are considered structurally real (the
stem is consistently there in nearby frames).  A "structural template" is
built from these high-frequency pixels and OR'd into the target frame to
fill any gaps.

WHY A SLIDING WINDOW (not the full dataset)?
--------------------------------------------
• Handles plant growth — only nearby frames share the same plant state.
• Handles leaf movement — within a small window, leaves are in similar
  positions, so their frequency stays high only when they genuinely overlap
  the stem region.  A tight FREQ_THRESHOLD filters out flickering leaves.

PIPELINE
--------
  1. Load ALL *_mask_refined.png frames into RAM (needed for window access).
  2. [Optional] Apply per-frame morphological preprocessing.
  3. For each output frame t:
       a. Extract window stack[t−W .. t+W]  (clamped to valid range)
       b. Compute per-pixel frequency  (mean along time axis)
       c. Threshold → structural template
       d. OR template into frame t  (restricted to corridor if enabled)
  4. Save selected filled masks to  <device_folder>/masks_freq_filled/
  5. Write a preview PNG showing: original | freq map | filled result.

KEY PARAMETERS
--------------
  WINDOW_HALF
        Half-size of the sliding window (frames on each side of t).
        Total window = 2*WINDOW_HALF + 1 frames.
        Larger → smoother/more complete template but less sensitive to growth.
        Start with 15 (≈ 31 frames) and adjust.

  FREQ_THRESHOLD
        Fraction of window frames a pixel must be foreground to be included
        in the structural template.
        0.5 = present in ≥ 50 % of window frames.
        Raise (e.g. 0.7) to be stricter — avoids flickering leaves.
        Lower (e.g. 0.3) to catch more of the stem edge but risks false fills.

  RESTRICT_TO_CORRIDOR
        True  → template applied only within x ∈ [cx − CORRIDOR_HALF_PX −
                CORRIDOR_MARGIN, cx + CORRIDOR_HALF_PX + CORRIDOR_MARGIN].
        False → applied to the full image width.
        Recommended: True (stem-only gap repair).
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────
MASKS_DIR = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465\masks"
)
DEVICE_DIR = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465"
)
OUT_MASKS_DIR = DEVICE_DIR / "masks_freq_filled"
PREVIEW_PATH  = DEVICE_DIR / "freq_fill_preview.png"

# ══════════════════════════════════════════════════════════════════════════
# ▼▼▼  EDIT THESE  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════

WINDOW_HALF      = 20     # frames on each side of target (total = 2*W+1)
FREQ_THRESHOLD   = 0.2    # pixel must be foreground in this fraction of
                           # window frames to be included in the template
                           # raise → stricter (fewer fills, less noise)
                           # lower → looser (more fills, more risk of errors)

RESTRICT_TO_CORRIDOR = True   # True = only fill within the corridor x-range
CORRIDOR_HALF_PX     = 40    # px — stem corridor half-width
CORRIDOR_MARGIN      = 80    # px — extra x beyond corridor on each side

# ── Per-frame morphological preprocessing (optional) ──────────────────────
# Applied to every frame before frequency computation.
# Helps merge thin double-edges into solid bands before voting.
MORPH_PREPROCESS = False     # True = apply morphological ops first
MORPH_MODE       = "simple"  # "ASF" or "simple"
MORPH_ASF_SIZES  = [3, 5, 7]
MORPH_CLOSE_R    = 3
MORPH_OPEN_R     = 1

# ── Output / preview ──────────────────────────────────────────────────────
PROCESS_ALL = True   # True  = save filled masks for ALL frames
                       # False = save only N_SAMPLES evenly-spaced frames
                       # NOTE: ALL frames are always loaded (needed for window)
N_SAMPLES   = 50       # frames to save/preview when PROCESS_ALL = False
N_PREVIEW   = 5       # frames to show in the preview PNG

# ══════════════════════════════════════════════════════════════════════════
# ▲▲▲  END OF USER SETTINGS  ▲▲▲
# ══════════════════════════════════════════════════════════════════════════

DPI           = 150
MARGIN        = 30
EXPAND_PX     = 200
BOTTOM_FRAC   = 0.15
PROJ_BAND     = (0.20, 0.85)
ANCHOR_MARGIN = 150
SMOOTH_K      = 15


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
# Per-frame morphological preprocessing
# ══════════════════════════════════════════════════════════════════════════

def morph_preprocess(frame_u8):
    out = frame_u8.copy()
    if MORPH_MODE == "ASF":
        for sz in MORPH_ASF_SIZES:
            kr  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kr)
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN,  kr)
    else:
        if MORPH_CLOSE_R > 0:
            d   = 2 * MORPH_CLOSE_R + 1
            kc  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kc)
        if MORPH_OPEN_R > 0:
            d   = 2 * MORPH_OPEN_R + 1
            ko  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ko)
    return out


# ══════════════════════════════════════════════════════════════════════════
# Core: sliding-window frequency fill
# ══════════════════════════════════════════════════════════════════════════

def freq_fill_frame(t, binary_stack, window_half, freq_threshold,
                    corridor_xlim=None):
    """
    Compute the frequency-filled mask for frame t.

    Parameters
    ----------
    t              : target frame index
    binary_stack   : bool array (N, H, W)
    window_half    : half-size of sliding window
    freq_threshold : foreground fraction required to include pixel in template
    corridor_xlim  : (x_lo, x_hi) or None — restrict fill to this x range

    Returns
    -------
    filled_u8   : uint8 (0 / 255) filled mask
    freq_map    : float32 (H, W) frequency values 0–1 for the window
    template_u8 : uint8 (0 / 255) structural template before OR
    n_added     : number of pixels newly added
    """
    N = binary_stack.shape[0]
    t_lo = max(0, t - window_half)
    t_hi = min(N - 1, t + window_half)
    window   = binary_stack[t_lo : t_hi + 1]          # (W, H, W_img)
    freq_map = window.mean(axis=0).astype(np.float32)  # (H, W), 0–1

    template = freq_map >= freq_threshold              # bool (H, W)

    orig   = binary_stack[t]                           # bool (H, W)
    filled = orig.copy()

    if corridor_xlim is not None:
        x_lo, x_hi = corridor_xlim
        filled[:, x_lo:x_hi+1] |= template[:, x_lo:x_hi+1]
    else:
        filled |= template

    n_added     = int((filled & ~orig).sum())
    filled_u8   = filled.astype(np.uint8) * 255
    template_u8 = template.astype(np.uint8) * 255
    return filled_u8, freq_map, template_u8, n_added


# ══════════════════════════════════════════════════════════════════════════
# Visualisation helpers
# ══════════════════════════════════════════════════════════════════════════

def color_binary(binary_u8, dim_bg=None):
    H, W = binary_u8.shape
    vis  = np.zeros((H, W, 3), dtype=np.uint8)
    if dim_bg is not None:
        d = (dim_bg // 8).astype(np.uint8)
        vis[..., 0] = d; vis[..., 1] = d; vis[..., 2] = d
    ys, xs = np.where(binary_u8 > 127)
    vis[ys, xs] = [255, 255, 255]
    return vis


def color_freq_map(freq_map, corridor_xlim=None):
    """Render frequency map as a viridis heatmap (0=black, 1=yellow)."""
    import matplotlib.cm as cm
    cmap   = cm.get_cmap('viridis')
    rgb    = (cmap(freq_map)[..., :3] * 255).astype(np.uint8)
    # dim out-of-corridor region if restricted
    if corridor_xlim is not None:
        x_lo, x_hi = corridor_xlim
        mask = np.zeros(freq_map.shape, dtype=bool)
        mask[:, x_lo:x_hi+1] = True
        rgb[~mask] = (rgb[~mask] * 0.15).astype(np.uint8)
    return rgb


def color_filled(orig_u8, filled_u8, dim_bg=None):
    """White = unchanged foreground; green = added by template."""
    H, W = orig_u8.shape
    vis  = np.zeros((H, W, 3), dtype=np.uint8)
    if dim_bg is not None:
        d = (dim_bg // 8).astype(np.uint8)
        vis[..., 0] = d; vis[..., 1] = d; vis[..., 2] = d
    ys, xs = np.where(orig_u8 > 127)
    vis[ys, xs] = [255, 255, 255]
    added = (filled_u8 > 127) & ~(orig_u8 > 127)
    ys2, xs2 = np.where(added)
    vis[ys2, xs2] = [0, 220, 80]
    return vis


def draw_corridor_lines(vis, y0, y1, cx0, cx1):
    vis[y0:y1+1, cx0] = [255, 220, 0]
    vis[y0:y1+1, cx1] = [255, 220, 0]


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

all_paths = sorted(MASKS_DIR.glob("*_mask_refined.png"))
if not all_paths:
    raise FileNotFoundError(f"No masks found in {MASKS_DIR}")
N_total = len(all_paths)
print(f"Found {N_total} masks.")

# ── Always load ALL frames (needed for window frequency computation) ────────
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
        print(f"  loaded {i+1}/{N_total}")
print(f"Stack: {stack_raw.shape}  ({stack_raw.nbytes / 1e6:.0f} MB)")

# ── Optional morphological preprocessing ──────────────────────────────────
if MORPH_PREPROCESS:
    if MORPH_MODE == "ASF":
        morph_desc = f"ASF {MORPH_ASF_SIZES}"
    else:
        morph_desc = f"simple close r={MORPH_CLOSE_R} / open r={MORPH_OPEN_R}"
    print(f"Morphological preprocessing ({morph_desc}) …")
    stack_proc = np.zeros_like(stack_raw)
    for i in range(N_total):
        stack_proc[i] = morph_preprocess(stack_raw[i])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{N_total}")
    print("  Done.")
else:
    morph_desc = "none"
    stack_proc = stack_raw

binary_stack = stack_proc > 127    # bool (N, H, W)

# ── Estimate corridor centre (median over ~10 evenly-spaced frames) ─────────
print("Estimating corridor centre …")
samp_idx = np.linspace(0, N_total - 1, min(10, N_total), dtype=int)
cx_list  = []
for i in samp_idx:
    m = stack_raw[i]
    ys, xs = np.where(m > 127)
    if len(ys) == 0:
        continue
    y0m, y1m = int(ys.min()), int(ys.max())
    x0m, x1m = int(xs.min()), int(xs.max())
    cx_list.append(estimate_stem_cx(m, y0m, y1m, x0m, x1m))
if not cx_list:
    raise RuntimeError("Could not estimate corridor centre.")
cx_med = int(np.median(cx_list))
cx0    = max(0,     cx_med - CORRIDOR_HALF_PX)
cx1    = min(W - 1, cx_med + CORRIDOR_HALF_PX)
print(f"  cx_median = {cx_med}  →  corridor x = [{cx0}, {cx1}]")

# ── Corridor x limits for fill ────────────────────────────────────────────
if RESTRICT_TO_CORRIDOR:
    x_lo = max(0,     cx0 - CORRIDOR_MARGIN)
    x_hi = min(W - 1, cx1 + CORRIDOR_MARGIN)
    corridor_xlim = (x_lo, x_hi)
    print(f"  fill region: x=[{x_lo}, {x_hi}]  "
          f"(corridor ± {CORRIDOR_MARGIN}px = {x_hi - x_lo + 1}px wide)")
else:
    corridor_xlim = None
    print(f"  fill region: full image width ({W}px)")

# ── Which frames to save ──────────────────────────────────────────────────
if PROCESS_ALL:
    save_idx = list(range(N_total))
    print(f"PROCESS_ALL = True  →  will save all {N_total} frames.")
else:
    step     = max(1, N_total // N_SAMPLES)
    save_idx = list(range(0, N_total, step))[:N_SAMPLES]
    print(f"PROCESS_ALL = False  →  saving {len(save_idx)} / {N_total} frames "
          f"(every {step} frames).")

# ── Compute and save filled masks ─────────────────────────────────────────
OUT_MASKS_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nComputing frequency fill  "
      f"(window = ±{WINDOW_HALF} frames = {2*WINDOW_HALF+1} total, "
      f"threshold = {FREQ_THRESHOLD}) …")

n_added_list = []
for i, t in enumerate(save_idx):
    filled_u8, freq_map, template_u8, n_added = freq_fill_frame(
        t, binary_stack,
        window_half=WINDOW_HALF,
        freq_threshold=FREQ_THRESHOLD,
        corridor_xlim=corridor_xlim,
    )
    n_added_list.append(n_added)
    out_p = OUT_MASKS_DIR / all_paths[t].name
    cv2.imwrite(str(out_p), filled_u8)
    if (i + 1) % 10 == 0 or i == len(save_idx) - 1:
        print(f"  [{i+1}/{len(save_idx)}]  frame {t:>4}  +{n_added} px")

n_changed = sum(1 for n in n_added_list if n > 0)
print(f"\nResults:")
print(f"  Frames with pixels added : {n_changed} / {len(save_idx)}")
print(f"  Total pixels added       : {sum(n_added_list):,}")
print(f"  Mean per changed frame   : "
      f"{sum(n_added_list)/max(n_changed,1):.0f} px")
print(f"  Saved to: {OUT_MASKS_DIR}")

# ── Preview PNG ───────────────────────────────────────────────────────────
n_prev = min(N_PREVIEW, len(save_idx))
print(f"\nGenerating preview ({n_prev} frames) …")

# Pick frames with most pixels added, mixed with even spacing
added_arr  = np.array(n_added_list)
most_idx   = np.argsort(added_arr)[::-1][:n_prev].tolist()
evenly_idx = list(np.linspace(0, len(save_idx)-1, n_prev, dtype=int))
cands      = sorted(set(most_idx[:n_prev//2 + 1] + evenly_idx))[:n_prev]
preview_save_pos = cands   # positions into save_idx

N_COLS = 3   # original | frequency map | filled result
fig = plt.figure(figsize=(8 * N_COLS, n_prev * 9))
gs  = gridspec.GridSpec(n_prev, N_COLS, figure=fig,
                        wspace=0.04, hspace=0.22)
axes = np.array([[fig.add_subplot(gs[r, c])
                  for c in range(N_COLS)]
                 for r in range(n_prev)])

col_titles = [
    "Original mask\n(raw *_mask_refined.png)",
    f"Frequency map  (window ±{WINDOW_HALF} frames)\n"
    f"bright = pixel present in many frames\n"
    f"dashed line = FREQ_THRESHOLD = {FREQ_THRESHOLD}",
    f"Filled result\n"
    f"white = unchanged  green = added by template",
]

for row, sp in enumerate(preview_save_pos):
    t        = save_idx[sp]
    orig_u8  = stack_raw[t]
    ts       = all_paths[t].stem.replace("tl_","").replace("_mask_refined","")
    n_added  = n_added_list[sp]

    # Recompute freq_map and filled for this frame (fast)
    filled_u8, freq_map, template_u8, _ = freq_fill_frame(
        t, binary_stack,
        window_half=WINDOW_HALF,
        freq_threshold=FREQ_THRESHOLD,
        corridor_xlim=corridor_xlim,
    )

    # Crop bounds — use orig_u8 so the full plant extent (including leaves)
    # sets the crop region.  Fall back to filled_u8 only when orig is empty
    # (e.g. genuinely blank frame), which avoids the corridor-only crop that
    # occurs when filled_u8 has pixels only inside the corridor strip.
    ys_o, xs_o = np.where(orig_u8   > 127)
    ys_f, xs_f = np.where(filled_u8 > 127)
    if len(ys_o) > 0:
        ys_bb, xs_bb = ys_o, xs_o
    elif len(ys_f) > 0:
        ys_bb, xs_bb = ys_f, xs_f
    else:
        # Completely blank frame — show a placeholder in every column
        for ax in axes[row]:
            ax.set_facecolor('black')
            ax.text(0.5, 0.5, f"Frame {t}\n(no foreground pixels)",
                    transform=ax.transAxes, ha='center', va='center',
                    color='gray', fontsize=9)
            ax.axis('off')
        continue

    y0p, y1p = int(ys_bb.min()), int(ys_bb.max())
    x0p, x1p = int(xs_bb.min()), int(xs_bb.max())
    xc0 = max(0,     x0p - EXPAND_PX - MARGIN)
    xc1 = min(W - 1, x1p + EXPAND_PX + MARGIN)
    yc0 = max(0,     y0p - MARGIN)
    yc1 = min(H - 1, y1p + MARGIN)
    # Enforce minimum crop height so a very wide/short frame doesn't render
    # as a hairline strip inside the fixed-height axes row.
    if (yc1 - yc0) < 200:
        yc_mid = (yc0 + yc1) // 2
        yc0    = max(0,     yc_mid - 200)
        yc1    = min(H - 1, yc_mid + 200)

    def crop(img):
        return img[yc0:yc1, xc0:xc1]

    vis_orig   = color_binary(orig_u8, dim_bg=orig_u8)
    vis_freq   = color_freq_map(freq_map, corridor_xlim=corridor_xlim)
    vis_filled = color_filled(orig_u8, filled_u8, dim_bg=orig_u8)

    draw_corridor_lines(vis_orig,   y0p, y1p, cx0, cx1)
    draw_corridor_lines(vis_freq,   y0p, y1p, cx0, cx1)
    draw_corridor_lines(vis_filled, y0p, y1p, cx0, cx1)

    for col, (vis, ax) in enumerate(zip(
            [vis_orig, vis_freq, vis_filled], axes[row])):
        ax.imshow(crop(vis), aspect='auto')   # aspect='auto' fills the axes row
        ax.axis("off")
        if col == 0:
            ax.set_ylabel(f"{ts}  (frame {t})", fontsize=8,
                          rotation=0, labelpad=140,
                          va="center", fontweight="bold")
        if row == 0:
            ax.set_title(col_titles[col], fontsize=8, pad=5, linespacing=1.5)
        if col == 2:
            orig_px   = int((orig_u8   > 127).sum())
            filled_px = int((filled_u8 > 127).sum())
            ax.set_xlabel(
                f"+{n_added} px added  |  "
                f"{orig_px:,} → {filled_px:,} foreground px",
                fontsize=6.5, labelpad=3,
            )

# ── Legend ────────────────────────────────────────────────────────────────
fig.subplots_adjust(bottom=0.07)
leg = [
    mpatches.Patch(color=(1.0,   1.0,  1.0), label="Unchanged foreground"),
    mpatches.Patch(color=(0, 220/255, 80/255), label="Added by frequency template"),
    mpatches.Patch(color=(1.0, 220/255, 0),
                   label=f"Corridor boundary  cx ± {CORRIDOR_HALF_PX}px"),
]
fig.legend(handles=leg, loc="lower center", ncol=3, fontsize=8,
           framealpha=0.85, handlelength=1.4, columnspacing=2.0,
           bbox_to_anchor=(0.5, 0.0))

restrict_str = (f"corridor x=[{corridor_xlim[0]},{corridor_xlim[1]}]"
                if RESTRICT_TO_CORRIDOR else "full image width")
plt.suptitle(
    f"Frequency voting gap fill  —  window ±{WINDOW_HALF} frames "
    f"({2*WINDOW_HALF+1} total)  |  threshold = {FREQ_THRESHOLD}  |  "
    f"region = {restrict_str}\n"
    f"morph preprocessing: {morph_desc}  |  "
    f"{n_changed}/{len(save_idx)} frames modified  |  "
    f"{sum(n_added_list):,} px added total  |  "
    f"saved to: {OUT_MASKS_DIR.name}/",
    fontsize=9, y=1.003,
)

plt.savefig(str(PREVIEW_PATH), dpi=DPI, bbox_inches="tight")
plt.close()
print(f"Preview saved: {PREVIEW_PATH}")
print(f"\nAll done.  Filled masks in: {OUT_MASKS_DIR}")
