"""
stem_height_analysis.py

Standalone stem-height time-series from the EXISTING frequency-filled Sobel masks
(`masks_freq_filled/`).  Fast: it only reads each mask once and computes a cheap
vertical density profile — no freq-fill, morphology, or skeletonisation — so a full
run over a few hundred frames takes seconds, not the long runtime of
`stem_analysis_pipeline.py`.

------------------------------------------------------------------------------------
METHOD  (image-coordinate convention: y = 0 at image top, y increases downward)
------------------------------------------------------------------------------------
Stem height = distance from a FIXED base (soil line) to the apical meristem.

Directly tracing the meristem from the edge masks proved unreliable once the upper
leaves overlap the stem (the dark stem channel gets occluded).  Instead we use a
robust PROXY feature that correlates ~0.95 with manually-annotated stem height and,
unlike geometric tracing, does not saturate on tall stems:

    feature_px = base_y - (topmost row where the smoothed density of the central
                           +/-CORRIDOR_HALF-px corridor exceeds DENSITY_T)
    stem_h_px  = slope * feature_px + intercept          # linear calibration

`slope` / `intercept` are fit once per run against a handful of manually annotated
frames (a red vertical line drawn from base to apical meristem on the mask).  The
slope is < 1 because the upper leaves rise faster than the meristem, so the
plant-top scales proportionally with — but is taller than — the true stem.

------------------------------------------------------------------------------------
CALIBRATION  (built into this script)
------------------------------------------------------------------------------------
Run with no arguments for an interactive menu:
    [1] Use existing calibration and run the analysis
    [2] Create a calibration from EXISTING annotation PNGs, then run
    [3] ANNOTATE frames now (pop-up clicker), then calibrate and run

Or non-interactively:
    python scripts/stem_height_analysis.py --annotate --samples 15   # pop-up clicker to label frames
    python scripts/stem_height_analysis.py --calibrate               # build & save calibration
    python scripts/stem_height_analysis.py --apply                   # run using saved calibration
    python scripts/stem_height_analysis.py --calibrate --apply       # both
    python scripts/stem_height_analysis.py --calibrate --px-per-mm 12.5   # set physical scale

Interactive annotation (--annotate): pops up N evenly-spaced frames; on the FIRST frame
you click base then apex, on the rest you click the apex only (base is reused).  The view
is auto-zoomed to the plant region (a reference crop computed from the last/largest frames),
so the plant fills the window and clicks are precise; click coordinates are mapped back to
full-image space, so the saved annotations and calibration stay in full-image coordinates.
Each frame is saved as a red-line PNG in the annotation folder, ready for --calibrate.
Needs a GUI display (Tk/Qt) — run it in a local terminal, not headless / not as a notebook cell.

NOTE: --calibrate uses EVERY annotation PNG in the folder, and --annotate ADDS to that folder
(it does not replace).  So repeated --annotate runs accumulate.  To make a run stand alone
(N samples -> exactly N annotations), either point --annot-dir at a fresh folder, or pass
--fresh-annotations (backs up existing annotations to a _backup_<timestamp>/ subfolder first).

The calibration (base_y, cx, feature params, slope/intercept, px_per_mm, CV error, and the
masks/annotation folders it was built from) is saved as parameters to a JSON file so it can
be reused or re-fit later.

------------------------------------------------------------------------------------
INPUT / OUTPUT FOLDERS
------------------------------------------------------------------------------------
Defaults point at masks_freq_filled (recommended).  Override on the command line:
    --masks-dir  PATH   analyse a different mask folder (e.g. the raw "masks" Sobel
                        edges).  Raw masks work but are noisier (lower correlation),
                        so RE-CALIBRATE whenever you switch folders.
    --annot-dir  PATH   use a different annotation set (keep separate sets per run).

Outputs (including the calibration) are written PER mask-folder under
    <masks_dir parent>/stem_height_analysis/<masks_dir name>/
so each input folder keeps its own calibration and results:
    stem_height_calibration.json   calibration parameters
    stem_height_timeseries.csv     per-frame: timestamp, hours, feature, stem_h, total_h
                                   (stem_h = calibrated base->meristem;
                                    total_h = base->topmost plant pixel, uncalibrated)
    stem_height_growth_curve.png   stem height + total plant height vs time (+ GT if available)
    stem_height_overlays.png       QC montage: base/feature/apex drawn on sample masks

Optional per-frame height images (only when --estimated-height-dir is given):
    <out>/estimated_height/*.png   one image per frame with the estimated stem height
                                   drawn on a dimmed mask (high contrast), labelled px/mm
"""

import cv2
import json
import shutil
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  PATHS — EDIT THESE  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════════
DEVICE_DIR = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465"
)
MASKS_FREQ_DIR = DEVICE_DIR / "masks_freq_filled"          # input masks (override: --masks-dir)
ANNOT_DIR      = Path(r"C:\Users\ryank\Desktop\stem height examples")  # red lines (override: --annot-dir)

MASK_GLOB = "tl_*_mask_refined.png"
TS_FORMAT = "%Y-%m-%d_%H-%M-%S"        # timestamp encoded in the filename


def _configure_paths(masks_dir=None, annot_dir=None):
    """Resolve input/output paths. Outputs live per mask-folder so each input
    folder (e.g. masks_freq_filled vs raw masks) keeps its own calibration/results."""
    global MASKS_FREQ_DIR, ANNOT_DIR, OUT_DIR, CALIB_PATH
    if masks_dir:
        MASKS_FREQ_DIR = Path(masks_dir)
    if annot_dir:
        ANNOT_DIR = Path(annot_dir)
    OUT_DIR    = MASKS_FREQ_DIR.parent / "stem_height_analysis" / MASKS_FREQ_DIR.name
    CALIB_PATH = OUT_DIR / "stem_height_calibration.json"


# initialise OUT_DIR / CALIB_PATH from the defaults above
_configure_paths()

# ══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  FEATURE / SMOOTHING PARAMETERS  ▼▼▼  (defaults validated against 20 frames)
# ══════════════════════════════════════════════════════════════════════════════
CORRIDOR_HALF   = 90      # px — half-width of the central corridor around cx
DENSITY_T       = 0.20    # corridor white-fraction threshold defining "plant content"
SMOOTH_K        = 15      # rows — vertical smoothing of the density profile
SEARCH_ABOVE    = 1000    # px — look at most this far above the base for plant content
TEMPORAL_MED_K  = 5       # frames — rolling-median window for the time series (odd)

# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def _smooth(a, k):
    if k <= 1:
        return a
    return np.convolve(a, np.ones(k) / k, mode="same")


def _timestamp_from_name(name):
    """Extract the 'YYYY-MM-DD_HH-MM-SS' timestamp from a mask/annotation filename."""
    stem = name
    if "tl_" in stem:
        stem = stem.split("tl_", 1)[1]
    stem = stem.split("_mask_refined", 1)[0]
    try:
        return stem, dt.datetime.strptime(stem, TS_FORMAT)
    except ValueError:
        return stem, None


def _rolling_median(z, k):
    if k <= 1:
        return z.copy()
    h = k // 2
    out = z.copy()
    for i in range(len(z)):
        w = z[max(0, i - h): i + h + 1]
        w = w[~np.isnan(w)]
        out[i] = np.median(w) if len(w) else np.nan
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Core feature
# ──────────────────────────────────────────────────────────────────────────────
def compute_feature(mask_gray, base_y, cx,
                    corridor_half=CORRIDOR_HALF, density_t=DENSITY_T,
                    smooth_k=SMOOTH_K, search_above=SEARCH_ABOVE):
    """
    feature_px = base_y - (topmost row where smoothed corridor density > density_t).

    Returns np.nan when the corridor holds no plant content above the base.
    """
    m = mask_gray > 127
    W = mask_gray.shape[1]
    x0, x1 = max(0, cx - corridor_half), min(W - 1, cx + corridor_half)
    dens = _smooth(m[:, x0:x1 + 1].mean(axis=1).astype(float), smooth_k)

    c_lo = max(0, base_y - search_above)
    ys = np.arange(c_lo, min(base_y, mask_gray.shape[0]))
    seg = dens[c_lo:min(base_y, mask_gray.shape[0])]
    hits = ys[seg > density_t]
    if len(hits) == 0:
        return np.nan
    return float(base_y - hits.min())


def compute_total_height(mask_gray, base_y, search_above=SEARCH_ABOVE, min_row_px=5):
    """
    Total plant height = base_y - topmost plant row, measured across the FULL image
    width (not just the corridor), so it captures the highest point of the whole plant.

    A row must hold more than `min_row_px` foreground pixels to count, which ignores
    isolated speckle.  Unlike stem height this needs NO calibration — it is a direct
    geometric measurement.  Returns np.nan if no plant rows are found.
    """
    m = mask_gray > 127
    c_lo = max(0, base_y - search_above)
    region = m[c_lo:min(base_y, mask_gray.shape[0])]
    rowcount = region.sum(axis=1)
    rows = np.where(rowcount > min_row_px)[0]
    if len(rows) == 0:
        return np.nan
    return float(base_y - (c_lo + int(rows.min())))


# ──────────────────────────────────────────────────────────────────────────────
# Annotation extraction (ground truth from red vertical lines)
# ──────────────────────────────────────────────────────────────────────────────
def extract_red_line(img_bgr):
    """Return (cx, apex_y, base_y) of a dark-red annotation line, or None."""
    b, g, r = (img_bgr[..., 0].astype(int),
               img_bgr[..., 1].astype(int),
               img_bgr[..., 2].astype(int))
    red = (r > 90) & (g < 70) & (b < 70)
    ys, xs = np.where(red)
    if len(ys) < 20:
        return None
    return int(np.median(xs)), int(ys.min()), int(ys.max())


def _find_mask_for_timestamp(ts):
    """Locate the freq-filled mask matching an annotation timestamp."""
    p = MASKS_FREQ_DIR / f"tl_{ts}_mask_refined.png"
    return p if p.exists() else None


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────
def calibrate(px_per_mm=None):
    """
    Fit feature_px -> stem_h_px from the annotated frames in ANNOT_DIR.

    For each annotation: read the red line (base_y, apex_y, cx), compute the proxy
    feature from the MATCHING freq-filled mask, then least-squares fit a line and
    report leave-one-out cross-validation error.  Returns a calibration dict.
    """
    annot_paths = sorted(ANNOT_DIR.glob("*.png"))
    if not annot_paths:
        raise FileNotFoundError(f"No annotation PNGs in {ANNOT_DIR}")

    feats, heights, base_ys, cxs, used = [], [], [], [], []
    for p in annot_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        line = extract_red_line(img)
        if line is None:
            continue
        cx, apex_y, base_y = line
        ts, _ = _timestamp_from_name(p.name)
        mask_path = _find_mask_for_timestamp(ts)
        # prefer the clean mask; fall back to the annotation image's grayscale
        if mask_path is not None:
            mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            mask_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mask_gray[extract_red_mask(img)] = 0   # drop the red line itself
        f = compute_feature(mask_gray, base_y, cx)
        if np.isnan(f):
            continue
        feats.append(f); heights.append(base_y - apex_y)
        base_ys.append(base_y); cxs.append(cx); used.append(ts)

    feats = np.array(feats, float)
    heights = np.array(heights, float)
    if len(feats) < 3:
        raise RuntimeError(f"Only {len(feats)} usable annotations — need >= 3 to calibrate.")

    slope, intercept = np.polyfit(feats, heights, 1)
    corr = float(np.corrcoef(feats, heights)[0, 1])

    # leave-one-out CV
    loo = []
    for i in range(len(feats)):
        idx = [j for j in range(len(feats)) if j != i]
        a, b = np.polyfit(feats[idx], heights[idx], 1)
        loo.append(a * feats[i] + b - heights[i])
    loo = np.abs(np.array(loo))

    calib = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "n_annotations": len(feats),
        "annotation_timestamps": used,
        "base_y": int(round(np.median(base_ys))),
        "cx": int(round(np.median(cxs))),
        "corridor_half": CORRIDOR_HALF,
        "density_t": DENSITY_T,
        "smooth_k": SMOOTH_K,
        "search_above": SEARCH_ABOVE,
        "slope": float(slope),
        "intercept": float(intercept),
        "corr": corr,
        "loo_cv_mae_px": float(np.mean(loo)),
        "loo_cv_median_px": float(np.median(loo)),
        "loo_cv_max_px": float(np.max(loo)),
        "px_per_mm": (float(px_per_mm) if px_per_mm else None),
        "masks_dir": str(MASKS_FREQ_DIR),
        "annotation_dir": str(ANNOT_DIR),
    }

    print(f"\nCalibration fit on {len(feats)} annotated frames:")
    print(f"  stem_h_px = {slope:.4f} * feature_px + {intercept:.2f}   (corr {corr:+.3f})")
    print(f"  LOO-CV:  mean|err| = {calib['loo_cv_mae_px']:.1f} px   "
          f"median = {calib['loo_cv_median_px']:.1f} px   max = {calib['loo_cv_max_px']:.0f} px")
    print(f"  base_y = {calib['base_y']}   cx = {calib['cx']}")
    if calib["px_per_mm"]:
        print(f"  px_per_mm = {calib['px_per_mm']:.4f}  -> physical units enabled")
    else:
        print("  px_per_mm = (not set)  -> pixel units only")
    return calib


def extract_red_mask(img_bgr):
    b, g, r = (img_bgr[..., 0].astype(int),
               img_bgr[..., 1].astype(int),
               img_bgr[..., 2].astype(int))
    return (r > 90) & (g < 70) & (b < 70)


def save_calibration(calib, path=None):
    path = Path(path) if path else CALIB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    prev = None
    if path.exists():                       # a new calibration overwrites the old one
        try:
            prev = json.load(open(path)).get("created")
        except Exception:
            prev = "unknown"
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)
    if prev:
        print(f"\nOverwrote existing calibration (previously created {prev})")
        print(f"Saved calibration -> {path}")
    else:
        print(f"\nSaved calibration -> {path}")


def load_calibration(path=None):
    path = Path(path) if path else CALIB_PATH
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Apply: full time series
# ──────────────────────────────────────────────────────────────────────────────
def apply_analysis(calib, make_overlays=6, est_height_dir=None):
    """Compute the stem-height time series for every mask and write CSV + plots."""
    base_y = calib["base_y"]; cx = calib["cx"]
    slope = calib["slope"];   intercept = calib["intercept"]
    px_per_mm = calib.get("px_per_mm")

    paths = sorted(MASKS_FREQ_DIR.glob(MASK_GLOB))
    if not paths:
        raise FileNotFoundError(f"No masks ({MASK_GLOB}) in {MASKS_FREQ_DIR}")
    print(f"\nProcessing {len(paths)} masks from {MASKS_FREQ_DIR.name} ...")

    rows = []
    for i, p in enumerate(paths):
        ts, when = _timestamp_from_name(p.name)
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        feat = compute_feature(mask, base_y, cx,
                               calib["corridor_half"], calib["density_t"],
                               calib["smooth_k"], calib["search_above"])
        h_px = slope * feat + intercept if not np.isnan(feat) else np.nan
        total = compute_total_height(mask, base_y, calib["search_above"])
        rows.append([i, ts, when, feat, h_px, total])
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(paths)}")

    idx = np.array([r[0] for r in rows])
    tss = [r[1] for r in rows]
    whens = [r[2] for r in rows]
    feat_px = np.array([r[3] for r in rows], float)
    h_raw = np.array([r[4] for r in rows], float)
    h_smooth = _rolling_median(h_raw, calib.get("temporal_med_k", TEMPORAL_MED_K))
    total_raw = np.array([r[5] for r in rows], float)
    total_smooth = _rolling_median(total_raw, calib.get("temporal_med_k", TEMPORAL_MED_K))

    # hours from first valid timestamp
    t0 = next((w for w in whens if w is not None), None)
    hours = np.array([((w - t0).total_seconds() / 3600.0) if (w and t0) else np.nan
                      for w in whens])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(idx, tss, whens, hours, feat_px, h_raw, h_smooth,
               total_raw, total_smooth, px_per_mm)
    _plot_growth(hours, h_raw, h_smooth, total_smooth, px_per_mm, calib, t0)
    if make_overlays:
        _plot_overlays(paths, calib, h_raw, n=make_overlays)
    if est_height_dir is not None:
        save_estimated_height_images(calib, est_height_dir)
    print("\nDone.")


def _write_csv(idx, tss, whens, hours, feat_px, h_raw, h_smooth,
               total_raw, total_smooth, px_per_mm):
    import csv
    out = OUT_DIR / "stem_height_timeseries.csv"
    header = ["index", "timestamp", "datetime_iso", "hours_from_start",
              "feature_px", "stem_h_px_raw", "stem_h_px_smooth",
              "total_h_px_raw", "total_h_px_smooth"]
    if px_per_mm:
        header += ["stem_h_mm_raw", "stem_h_mm_smooth",
                   "total_h_mm_raw", "total_h_mm_smooth"]

    def f2(v):
        return f"{v:.2f}" if not np.isnan(v) else ""

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(idx)):
            row = [idx[i], tss[i],
                   whens[i].isoformat() if whens[i] else "",
                   f"{hours[i]:.4f}" if not np.isnan(hours[i]) else "",
                   f2(feat_px[i]), f2(h_raw[i]), f2(h_smooth[i]),
                   f2(total_raw[i]), f2(total_smooth[i])]
            if px_per_mm:
                row += [f"{h_raw[i]/px_per_mm:.3f}" if not np.isnan(h_raw[i]) else "",
                        f"{h_smooth[i]/px_per_mm:.3f}" if not np.isnan(h_smooth[i]) else "",
                        f"{total_raw[i]/px_per_mm:.3f}" if not np.isnan(total_raw[i]) else "",
                        f"{total_smooth[i]/px_per_mm:.3f}" if not np.isnan(total_smooth[i]) else ""]
            w.writerow(row)
    print(f"  CSV   -> {out}")


def _gt_points():
    """Read any available red-line annotations for plotting (optional)."""
    gx, gy, gts = [], [], []
    for p in sorted(ANNOT_DIR.glob("*.png")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        line = extract_red_line(img)
        if line is None:
            continue
        _, apex_y, base_y = line
        ts, when = _timestamp_from_name(p.name)
        gts.append((ts, when, base_y - apex_y))
    return gts


def _plot_growth(hours, h_raw, h_smooth, total_smooth, px_per_mm, calib, t0):
    out = OUT_DIR / "stem_height_growth_curve.png"
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hours, total_smooth, "-", color="C2", lw=1.4, alpha=0.9,
            label="total plant height (smoothed)")
    ax.plot(hours, h_raw, ".", color="0.75", ms=4, label="stem height (raw)")
    ax.plot(hours, h_smooth, "-", color="C0", lw=1.6,
            label=f"stem height (rolling median k={calib.get('temporal_med_k', TEMPORAL_MED_K)})")

    # overlay manual annotations (red-line ground truth) for QC, placed on the
    # same time axis via their own timestamps
    gts = _gt_points()
    gx, gy = [], []
    for ts, when, h in gts:
        if when is not None and t0 is not None:
            gx.append((when - t0).total_seconds() / 3600.0); gy.append(h)
    if gx:
        ax.plot(gx, gy, "r*", ms=14, label=f"manual annotation ({len(gx)})")

    ax.set_xlabel("hours from start")
    ylab = "stem height (px)"
    ax.set_ylabel(ylab)
    if px_per_mm:
        sec = ax.secondary_yaxis('right', functions=(lambda v: v / px_per_mm,
                                                      lambda v: v * px_per_mm))
        sec.set_ylabel("stem height (mm)")
    ax.set_title(f"Stem & total plant height over time   "
                 f"(stem LOO-CV mean|err| = {calib['loo_cv_mae_px']:.0f} px)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(str(out), dpi=120); plt.close(fig)
    print(f"  plot  -> {out}")


def _plot_overlays(paths, calib, h_raw, n=6):
    out = OUT_DIR / "stem_height_overlays.png"
    base_y = calib["base_y"]; cx = calib["cx"]; W = calib["corridor_half"]
    cy0, cy1, cx0, cx1 = _reference_crop()                 # same crop as annotation
    sel = np.linspace(0, len(paths) - 1, n, dtype=int)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 7))
    if n == 1:
        axs = [axs]
    for ax, k in zip(axs, sel):
        mask = cv2.imread(str(paths[k]), cv2.IMREAD_GRAYSCALE)
        feat = compute_feature(mask, base_y, cx, W, calib["density_t"],
                               calib["smooth_k"], calib["search_above"])
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (cx - W, base_y), (cx + W, base_y), (0, 180, 255), 3)   # base
        if not np.isnan(feat):
            frow = int(base_y - feat)
            cv2.line(vis, (cx - W, frow), (cx + W, frow), (255, 220, 0), 2)   # feature row
            h = calib["slope"] * feat + calib["intercept"]
            apex = int(base_y - h)
            cv2.line(vis, (cx, apex), (cx, base_y), (0, 0, 255), 3)           # stem (pred)
            cv2.line(vis, (cx - W, apex), (cx + W, apex), (0, 255, 0), 2)     # apex
        crop = vis[cy0:cy1 + 1, cx0:cx1 + 1]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); ax.axis("off")
        ts, _ = _timestamp_from_name(paths[k].name)
        hh = calib["slope"] * feat + calib["intercept"] if not np.isnan(feat) else float("nan")
        ax.set_title(f"{ts[5:]}\nstem_h={hh:.0f}px", fontsize=8)
    fig.suptitle("orange=base  yellow=feature row  green=apex  red=stem (predicted)",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(str(out), dpi=120); plt.close(fig)
    print(f"  overlays -> {out}")


def save_estimated_height_images(calib, out_dir):
    """
    Save one image per frame with the ESTIMATED stem height drawn on it.

    The mask is dimmed to grey so the bright annotation stands out (good contrast):
      orange line  = fixed base       red line     = estimated stem (base -> apex)
      green line   = estimated apex    magenta line = total plant top
      text         = timestamp + stem height + total height (px, and mm if scaled)
    Cropped to the plant region (same reference crop as the annotator).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_y = calib["base_y"]; cx = calib["cx"]; W = calib["corridor_half"]
    slope = calib["slope"]; intercept = calib["intercept"]
    px_per_mm = calib.get("px_per_mm")
    cy0, cy1, cx0, cx1 = _reference_crop()
    paths = sorted(MASKS_FREQ_DIR.glob(MASK_GLOB))
    n = 0
    for p in paths:
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        feat = compute_feature(mask, base_y, cx, W, calib["density_t"],
                               calib["smooth_k"], calib["search_above"])
        total = compute_total_height(mask, base_y, calib["search_above"])
        # dim the edge mask to grey so the coloured annotation has strong contrast
        vis = cv2.cvtColor((mask // 3).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.line(vis, (cx - W, base_y), (cx + W, base_y), (0, 165, 255), 2)   # base (orange)
        if not np.isnan(total):                                              # plant top (magenta)
            ptop = int(round(base_y - total))
            cv2.line(vis, (0, ptop), (vis.shape[1] - 1, ptop), (255, 0, 255), 2)
        if np.isnan(feat):
            label = "no stem detected"
        else:
            h = slope * feat + intercept
            apex = int(round(base_y - h))
            cv2.line(vis, (cx, base_y), (cx, apex), (0, 0, 255), 4)           # stem (red)
            cv2.line(vis, (cx - 35, apex), (cx + 35, apex), (0, 255, 0), 3)   # apex (green)
            label = f"stem_h = {h:.0f} px"
            if px_per_mm:
                label += f"  ({h / px_per_mm:.1f} mm)"
        tlabel = "" if np.isnan(total) else (
            f"total_h = {total:.0f} px" + (f"  ({total / px_per_mm:.1f} mm)" if px_per_mm else ""))
        crop = vis[cy0:cy1 + 1, cx0:cx1 + 1].copy()
        ts, _ = _timestamp_from_name(p.name)
        cv2.putText(crop, ts, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (0, 255, 255), 2, cv2.LINE_AA)                            # timestamp (yellow)
        cv2.putText(crop, label, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.95,
                    (0, 255, 0), 2, cv2.LINE_AA)                              # stem height (green)
        cv2.putText(crop, tlabel, (12, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 0, 255), 2, cv2.LINE_AA)                           # total height (magenta)
        cv2.imwrite(str(out_dir / p.name), crop)
        n += 1
    print(f"  estimated-height images ({n}) -> {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Interactive annotation (pop-up clicker)
# ──────────────────────────────────────────────────────────────────────────────
def _pick_sample_paths(paths, n):
    """n evenly-spaced frames across the sorted sequence (always includes frame 0)."""
    if n >= len(paths):
        return list(paths)
    idx = np.linspace(0, len(paths) - 1, n).round().astype(int)
    idx = sorted(set(idx.tolist()))
    return [paths[i] for i in idx]


def _reference_crop(margin=80, n_last=10, speck=2):
    """
    Bounding box of the plant (the Sobel-edge region) for zoomed display.

    Taken from the union of the last `n_last` frames (where the plant is largest),
    ignoring isolated speckle (rows/cols with <= `speck` foreground px), padded by
    `margin`.  Returns (y0, y1, x0, x1) in full-image coordinates.
    """
    paths = sorted(MASKS_FREQ_DIR.glob(MASK_GLOB))
    if not paths:
        raise FileNotFoundError(f"No masks ({MASK_GLOB}) in {MASKS_FREQ_DIR}")
    sample = paths[-n_last:] if len(paths) >= n_last else paths
    H = W = None
    y0 = x0 = 10 ** 9
    y1 = x1 = -1
    for p in sample:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        H, W = m.shape
        fg = m > 127
        rows = np.where(fg.sum(1) > speck)[0]
        cols = np.where(fg.sum(0) > speck)[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        y0 = min(y0, int(rows.min())); y1 = max(y1, int(rows.max()))
        x0 = min(x0, int(cols.min())); x1 = max(x1, int(cols.max()))
    if y1 < 0:
        return 0, (H - 1 if H else 0), 0, (W - 1 if W else 0)
    return (max(0, y0 - margin), min(H - 1, y1 + margin),
            max(0, x0 - margin), min(W - 1, x1 + margin))


def _draw_and_save_annotation(mask_path, base_xy, apex_xy, out_dir):
    """Draw the red base->apex line on the mask and save it as a standard annotation PNG."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.line(vis, (int(base_xy[0]), int(base_xy[1])),
             (int(apex_xy[0]), int(apex_xy[1])), (0, 0, 139), 6)   # dark red (BGR)
    out = out_dir / mask_path.name
    cv2.imwrite(str(out), vis)
    return out


def _use_interactive_backend():
    """Switch matplotlib to a GUI backend for clicking; return True on success."""
    import matplotlib.pyplot as plt
    for be in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
        try:
            plt.switch_backend(be)
            return True
        except Exception:
            continue
    return False


def _backup_annotations(annot_dir):
    """Move existing annotation PNGs into a timestamped _backup_ subfolder (non-destructive)."""
    existing = sorted(annot_dir.glob("*.png"))
    if not existing:
        return 0
    backup = annot_dir / f"_backup_{dt.datetime.now():%Y%m%d_%H%M%S}"
    backup.mkdir(parents=True, exist_ok=True)
    for p in existing:
        shutil.move(str(p), str(backup / p.name))
    print(f"  moved {len(existing)} existing annotation(s) -> {backup.name}/")
    return len(existing)


def annotate_frames(n_samples, annot_dir=None, fresh=False):
    """
    Pop up N evenly-spaced masks for manual annotation.

      first frame  : click BASE then APEX  (2 clicks) — sets the base for the run
      other frames : click APEX only       (1 click)  — base is reused

    Each click set is drawn as a red line and saved as a standard annotation PNG in
    `annot_dir`, ready for `calibrate()`.  Returns the list of saved paths.

    fresh=True backs up any existing annotations first, so the new set stands alone
    (N samples -> exactly N annotations); otherwise new annotations are ADDED to the folder.
    """
    import matplotlib.pyplot as plt
    annot_dir = Path(annot_dir) if annot_dir else ANNOT_DIR
    annot_dir.mkdir(parents=True, exist_ok=True)
    if fresh:
        _backup_annotations(annot_dir)
    paths = sorted(MASKS_FREQ_DIR.glob(MASK_GLOB))
    if not paths:
        raise FileNotFoundError(f"No masks ({MASK_GLOB}) in {MASKS_FREQ_DIR}")
    sel = _pick_sample_paths(paths, n_samples)
    if not _use_interactive_backend():
        raise RuntimeError("No interactive matplotlib backend (Tk/Qt) available — "
                           "annotate on a machine with a display, or draw lines manually.")

    cy0, cy1, cx0, cx1 = _reference_crop()    # zoom display to the plant region
    print(f"\nAnnotating {len(sel)} frames -> {annot_dir}")
    print(f"  display cropped to plant region rows[{cy0}:{cy1}] cols[{cx0}:{cx1}] "
          f"(reference: last frames)")
    print("  Left-click to place a point, right-click to undo, Enter to confirm.\n")

    base_xy = None                            # stored in FULL-image coordinates
    saved = []
    for i, p in enumerate(sel):
        mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        crop = mask[cy0:cy1 + 1, cx0:cx1 + 1]
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.imshow(crop, cmap="gray")          # click coords are crop-local
        first = (i == 0)
        n_clicks = 2 if first else 1
        msg = ("Click BASE then APEX (2 clicks)" if first
               else "Click APEX only (1 click) — base reused")
        ax.set_title(f"[{i+1}/{len(sel)}] {p.name}\n{msg}", fontsize=9)
        if base_xy is not None:               # draw stored base in crop-local coords
            ax.axhline(base_xy[1] - cy0, color="orange", lw=1, ls="--")
            ax.plot([base_xy[0] - cx0], [base_xy[1] - cy0], "o", color="orange", ms=7)
        try:
            pts = plt.ginput(n_clicks, timeout=0)
        except Exception as e:
            plt.close(fig); print(f"  aborted: {e}"); break
        plt.close(fig)
        if len(pts) < n_clicks:
            print(f"  [{i+1}] skipped {p.name} (need {n_clicks} click(s))")
            continue
        # map crop-local clicks back to full-image coordinates
        if first:
            base_xy = (pts[0][0] + cx0, pts[0][1] + cy0)
            apex_xy = (pts[1][0] + cx0, pts[1][1] + cy0)
        else:
            apex_xy = (pts[0][0] + cx0, pts[0][1] + cy0)
        outp = _draw_and_save_annotation(p, base_xy, apex_xy, annot_dir)
        saved.append(outp)
        print(f"  [{i+1}] saved {outp.name}  base=({base_xy[0]:.0f},{base_xy[1]:.0f})  "
              f"apex=({apex_xy[0]:.0f},{apex_xy[1]:.0f})")

    plt.switch_backend("Agg")     # restore headless backend for plotting
    print(f"\nDone — {len(saved)} annotation(s) saved to {annot_dir}")
    return saved


# ──────────────────────────────────────────────────────────────────────────────
# CLI / interactive
# ──────────────────────────────────────────────────────────────────────────────
def _interactive():
    have = CALIB_PATH.exists()
    print("\n=== Stem height analysis ===")
    print(f"Calibration file: {CALIB_PATH}")
    print(f"  {'FOUND' if have else 'not found'}")
    print("\nChoose an option:")
    print("  [1] Use existing calibration and run the analysis"
          + ("" if have else "   (unavailable — no calibration yet)"))
    print("  [2] Create a calibration from EXISTING annotation PNGs, then run")
    print("  [3] ANNOTATE frames now (pop-up clicker), then calibrate and run")
    print("  [q] Quit")
    choice = input("> ").strip().lower()

    if choice == "1":
        calib = load_calibration()
        if not calib:
            print("No calibration found. Choose [2] or [3] first.")
            return
        apply_analysis(calib)
    elif choice in ("2", "3"):
        if choice == "3":
            ans = input("How many frames to annotate? (e.g. 12): ").strip()
            n = int(ans) if ans.isdigit() else 12
            fresh = input("Start a fresh set? (backs up existing annotations) [y/N]: ").strip().lower() == "y"
            annotate_frames(n, fresh=fresh)
        ans = input("px per mm for physical units (blank to skip): ").strip()
        px_per_mm = float(ans) if ans else None
        calib = calibrate(px_per_mm=px_per_mm)
        save_calibration(calib)
        if input("Run the analysis now? [y/N]: ").strip().lower() == "y":
            apply_analysis(calib)
    else:
        print("Bye.")


def main():
    ap = argparse.ArgumentParser(description="Standalone stem-height analysis from freq-filled masks.")
    ap.add_argument("--annotate", action="store_true", help="interactive pop-up clicker to create annotations")
    ap.add_argument("--samples", type=int, default=12, help="number of frames to annotate (with --annotate)")
    ap.add_argument("--fresh-annotations", action="store_true",
                    help="with --annotate: back up existing annotations first, so N samples = N annotations")
    ap.add_argument("--calibrate", action="store_true", help="build & save calibration from annotations")
    ap.add_argument("--apply", action="store_true", help="run the analysis using the calibration")
    ap.add_argument("--px-per-mm", type=float, default=None, help="spatial scale for mm output")
    ap.add_argument("--no-overlays", action="store_true", help="skip the QC overlay montage")
    ap.add_argument("--masks-dir", default=None, help="mask folder to analyse (default: masks_freq_filled)")
    ap.add_argument("--annot-dir", default=None, help="annotation folder (default: the examples folder)")
    ap.add_argument("--estimated-height-dir", nargs="?", const="__default__", default=None,
                    help="save a per-frame image with the estimated height drawn (optional PATH; "
                         "default: <out>/estimated_height). Only saved when this flag is given.")
    args = ap.parse_args()

    _configure_paths(args.masks_dir, args.annot_dir)
    print(f"masks : {MASKS_FREQ_DIR}")
    print(f"annot : {ANNOT_DIR}")
    print(f"out   : {OUT_DIR}")

    if not (args.annotate or args.calibrate or args.apply):
        _interactive()
        return

    if args.annotate:
        annotate_frames(args.samples, fresh=args.fresh_annotations)

    calib = None
    if args.calibrate:
        calib = calibrate(px_per_mm=args.px_per_mm)
        save_calibration(calib)
    if args.apply:
        if calib is None:
            calib = load_calibration()
            if calib is None:
                raise SystemExit("No calibration found — run with --calibrate first.")
            if args.px_per_mm is not None:            # allow overriding scale at apply time
                calib["px_per_mm"] = args.px_per_mm
        est_dir = None
        if args.estimated_height_dir is not None:
            est_dir = (OUT_DIR / "estimated_height"
                       if args.estimated_height_dir == "__default__"
                       else Path(args.estimated_height_dir))
        apply_analysis(calib, make_overlays=0 if args.no_overlays else 6,
                       est_height_dir=est_dir)


if __name__ == "__main__":
    main()
