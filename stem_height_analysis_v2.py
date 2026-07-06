"""
stem_height_analysis_v2.py
==========================================================================================
Stem-height time series for Nb plants, rebuilt to fix the sudden jumps / biologically
impossible drops produced by the original (edge-mask) approach.

WHY THIS IS DIFFERENT
-------------------------------------------------------------------------------------------
The original script measured the topmost row of a central corridor on the *Sobel edge /
freq-filled* masks.  That input is noisy (double edges, freq-fill box artifact) and the
feature follows whatever leaf is in the corridor, so a leaf drifting in/out produced
±100-400 px jumps with no real growth.

This version instead works from the RAW grayscale frames:

  1. SOLID SILHOUETTE   threshold the raw image into a clean solid plant blob
                        (Sobel -> triangle -> Gaussian blur sigma 50 -> Otsu ->
                        largest connected component -> fill holes).  This is far more
                        temporally stable than the edge masks.
  2. CORRIDOR PROXY     feature_px = base_y - (top of the CONTIGUOUS silhouette column in a
                        central +/-CORRIDOR_HALF corridor, anchored at the fixed base and
                        tolerating small gaps).  Anchoring to the base rejects detached
                        high blobs; the contiguity requirement rejects transient specks.
  3. LINEAR CALIBRATION stem_h_px = slope * feature_px + intercept, fit once against the
                        manually annotated frames (red base->apex lines).  Validated:
                        corr ~0.92, leave-one-out MAE ~27 px over 41 annotations.
  4. ROBUST TEMPORAL    the per-frame series is cleaned with a Hampel outlier filter
     FILTER             (drops single bad frames, e.g. segmentation failures) followed by
                        a rolling median.  This is what removes the spikes the old curve
                        showed.

A secondary, calibration-free measurement is also reported:
  total_h_px = base_y - topmost silhouette row over the FULL width (whole-plant height).

The existing red-line annotations (drawn on the *_mask_refined.png masks) are reused as-is:
their coordinates are in full-image space and the masks share the raw image dimensions.

USAGE
-------------------------------------------------------------------------------------------
  python stem_height_analysis_v2.py --calibrate --apply        # fit then run full series
  python stem_height_analysis_v2.py --apply                    # run with saved calibration
  python stem_height_analysis_v2.py --calibrate                # just (re)fit calibration
  python stem_height_analysis_v2.py --apply --estimated-height-dir   # also save per-frame PNGs

Outputs under  <DEVICE_DIR>/stem_height_analysis_v2/ :
  stem_height_calibration.json
  stem_height_timeseries.csv
  stem_height_growth_curve.png
  stem_height_overlays.png
  estimated_height/*.png        (only with --estimated-height-dir)
"""

import sys
import csv
import json
import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from plant_timelapse import plant_contour_v3 as v3

# ══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  PATHS — EDIT THESE  ▼▼▼
# ══════════════════════════════════════════════════════════════════════════════
DEVICE_DIR = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465"
)
ANNOT_DIR = Path(r"C:\Users\ryank\Desktop\stem height examples")

# ══════════════════════════════════════════════════════════════════════════════
# ▼▼▼  FEATURE / FILTER PARAMETERS  ▼▼▼  (validated on 41 annotations + 299 frames)
# ══════════════════════════════════════════════════════════════════════════════
CORRIDOR_HALF = 90      # px — half-width of central corridor around the stem x
DENSITY_T     = 0.20    # corridor silhouette-fill fraction marking "plant content"
SMOOTH_K      = 15      # rows — vertical smoothing of the corridor profile
MAX_GAP       = 18      # rows — gap tolerated while walking the contiguous column up
SEARCH_ABOVE  = 1200    # px — look at most this far above the base
BRIGHT_T      = 55      # 8-bit intensity treated as plant when locating the stem x
SIL_BLUR_SIGMA = 50.0   # Gaussian sigma (px) for the solid-silhouette step
# robust temporal filter
HAMPEL_K      = 7       # frames — Hampel window (odd)
HAMPEL_NSIG   = 3.0     # MAD multiples beyond which a frame is an outlier
MED_K         = 5       # frames — final rolling-median window (odd)

MASK_GLOB = "tl_*_mask_refined.png"
TS_FORMAT = "%Y-%m-%d_%H-%M-%S"


# ──────────────────────────────────────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────────────────────────────────────
def _find_raw_dir(device_dir):
    cands = sorted(p for p in device_dir.glob("timelapse_*") if p.is_dir())
    if not cands:
        raise FileNotFoundError(f"No timelapse_* folder under {device_dir}")
    return cands[0]


RAW_DIR  = _find_raw_dir(DEVICE_DIR)
MASK_DIR = DEVICE_DIR / "masks_freq_filled"      # only used to overlay GT, optional
OUT_DIR  = DEVICE_DIR / "stem_height_analysis_v2"
CALIB_PATH = OUT_DIR / "stem_height_calibration.json"


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def _smooth(a, k):
    return a if k <= 1 else np.convolve(a, np.ones(k) / k, mode="same")


def _timestamp_from_name(name):
    s = name
    if "tl_" in s:
        s = s.split("tl_", 1)[1]
    s = s.split("_mask_refined", 1)[0].replace(".tif", "").replace(".png", "")
    try:
        return s, dt.datetime.strptime(s, TS_FORMAT)
    except ValueError:
        return s, None


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


def _hampel(z, k=HAMPEL_K, nsig=HAMPEL_NSIG):
    """Replace points deviating > nsig * 1.4826 * MAD from the local median with NaN."""
    h = k // 2
    out = z.copy()
    for i in range(len(z)):
        w = z[max(0, i - h): i + h + 1]
        w = w[~np.isnan(w)]
        if len(w) < 3:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med)) + 1e-9
        if abs(z[i] - med) > nsig * 1.4826 * mad:
            out[i] = np.nan
    return out


def robust_filter(raw_series):
    """Hampel outlier rejection -> rolling median. NaNs are preserved as gaps."""
    return _rolling_median(_hampel(raw_series), MED_K)


# ──────────────────────────────────────────────────────────────────────────────
# Image -> silhouette -> feature
# ──────────────────────────────────────────────────────────────────────────────
def load_stretched_u8(raw_path):
    gray = v3.load_grayscale_working(raw_path, uint8=False)
    return gray, v3.auto_stretched_gray_u8(gray, saturated_fraction=0.0035)


def solid_silhouette(gray):
    """Clean solid plant blob from the raw grayscale (the most temporally stable input)."""
    edges = v3.sobel_find_edges(gray, blur_ksize=0, sobel_ksize=3, uint8=False)
    mask, _ = v3.threshold_binary_mask(edges, "triangle", None)
    smeared, _ = v3.gaussian_blur_binary_mask(mask, SIL_BLUR_SIGMA)
    final, _ = v3.threshold_soft_uint8(smeared, "otsu", None, otsu_margin=0)
    final = v3.largest_connected_component_mask(final)
    return v3.binary_fill_holes_u8(final)


def derive_cx(g8, base_y, seed_cx, band=160, nrows=140):
    """Stem x near the base: brightness-weighted column centre in a band above base."""
    H, W = g8.shape
    y0 = max(0, base_y - nrows)
    lo, hi = max(0, seed_cx - band), min(W, seed_cx + band)
    sub = g8[y0:base_y, lo:hi].astype(float)
    colsum = (sub * (sub > BRIGHT_T)).sum(axis=0)
    if colsum.sum() == 0:
        return int(seed_cx)
    return lo + int(round((np.arange(len(colsum)) * colsum).sum() / colsum.sum()))


def corridor_feature(sil, base_y, cx, corridor_half=CORRIDOR_HALF, density_t=DENSITY_T,
                     smooth_k=SMOOTH_K, max_gap=MAX_GAP, search_above=SEARCH_ABOVE):
    """base_y - top of the contiguous silhouette corridor column anchored at the base."""
    H, W = sil.shape
    lo, hi = max(0, cx - corridor_half), min(W, cx + corridor_half)
    prof = _smooth((sil[:, lo:hi] > 127).mean(axis=1), smooth_k)
    top = max(0, base_y - search_above)
    gap = 0
    best = base_y
    started = False
    for y in range(min(base_y, H - 1), top, -1):
        if prof[y] > density_t:
            best = y
            gap = 0
            started = True
        elif started:
            gap += 1
            if gap > max_gap:
                break
    return float(base_y - best)


def total_height(sil, base_y, search_above=SEARCH_ABOVE, min_row_px=30):
    """base_y - topmost silhouette row across the full width (whole-plant height)."""
    H, W = sil.shape
    top = max(0, base_y - search_above)
    region = (sil[top:min(base_y, H), :] > 127)
    rows = np.where(region.sum(axis=1) > min_row_px)[0]
    return float(base_y - (top + rows.min())) if len(rows) else np.nan


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth annotation extraction
# ──────────────────────────────────────────────────────────────────────────────
def extract_red_line(img_bgr):
    b, g, r = (img_bgr[..., 0].astype(int), img_bgr[..., 1].astype(int), img_bgr[..., 2].astype(int))
    red = (r > 90) & (g < 70) & (b < 70)
    ys, xs = np.where(red)
    if len(ys) < 20:
        return None
    return int(np.median(xs)), int(ys.min()), int(ys.max())  # cx, apex_y, base_y


def _read_annotations():
    """Return list of (ts, cx, apex_y, base_y) from the annotation PNGs."""
    out = []
    for p in sorted(ANNOT_DIR.glob("*.png")):
        if "_stem_height" in p.name:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        line = extract_red_line(img)
        if line is None:
            continue
        ts, _ = _timestamp_from_name(p.name)
        out.append((ts, line[0], line[1], line[2]))
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Calibration
# ──────────────────────────────────────────────────────────────────────────────
def calibrate():
    annots = _read_annotations()
    if len(annots) < 3:
        raise RuntimeError(f"Only {len(annots)} annotations found in {ANNOT_DIR} — need >= 3.")

    feats, heights, base_ys, cxs, used = [], [], [], [], []
    for ts, cx, apex_y, base_y in annots:
        raw_p = RAW_DIR / f"tl_{ts}.tif"
        if not raw_p.exists():
            print(f"  [calib] missing raw for {ts} — skipped")
            continue
        gray, g8 = load_stretched_u8(raw_p)
        sil = solid_silhouette(gray)
        cxf = derive_cx(g8, base_y, cx)
        f = corridor_feature(sil, base_y, cxf)
        if np.isnan(f):
            continue
        feats.append(f)
        heights.append(base_y - apex_y)
        base_ys.append(base_y)
        cxs.append(cxf)
        used.append(ts)

    feats = np.array(feats, float)
    heights = np.array(heights, float)
    if len(feats) < 3:
        raise RuntimeError(f"Only {len(feats)} usable annotations — need >= 3.")

    slope, intercept = np.polyfit(feats, heights, 1)
    corr = float(np.corrcoef(feats, heights)[0, 1])
    loo = []
    for i in range(len(feats)):
        idx = [j for j in range(len(feats)) if j != i]
        a, b = np.polyfit(feats[idx], heights[idx], 1)
        loo.append(abs(a * feats[i] + b - heights[i]))
    loo = np.array(loo)

    calib = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "method": "raw-silhouette corridor proxy + robust temporal filter",
        "n_annotations": len(feats),
        "annotation_timestamps": used,
        "base_y": int(round(np.median(base_ys))),
        "cx_seed": int(round(np.median(cxs))),
        "corridor_half": CORRIDOR_HALF,
        "density_t": DENSITY_T,
        "smooth_k": SMOOTH_K,
        "max_gap": MAX_GAP,
        "search_above": SEARCH_ABOVE,
        "bright_t": BRIGHT_T,
        "sil_blur_sigma": SIL_BLUR_SIGMA,
        "slope": float(slope),
        "intercept": float(intercept),
        "corr": corr,
        "loo_cv_mae_px": float(loo.mean()),
        "loo_cv_median_px": float(np.median(loo)),
        "loo_cv_max_px": float(loo.max()),
        "hampel_k": HAMPEL_K,
        "hampel_nsig": HAMPEL_NSIG,
        "med_k": MED_K,
        "px_per_mm": None,
        "raw_dir": str(RAW_DIR),
        "annotation_dir": str(ANNOT_DIR),
    }
    print(f"\nCalibration on {len(feats)} annotated frames:")
    print(f"  stem_h_px = {slope:.4f} * feature_px + {intercept:.2f}   (corr {corr:+.3f})")
    print(f"  LOO-CV: mean|err| = {loo.mean():.1f} px   median = {np.median(loo):.1f} px   max = {loo.max():.0f} px")
    print(f"  base_y = {calib['base_y']}   cx_seed = {calib['cx_seed']}")
    return calib


def save_calibration(calib):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CALIB_PATH, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"Saved calibration -> {CALIB_PATH}")


def load_calibration():
    if not CALIB_PATH.exists():
        return None
    with open(CALIB_PATH) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Apply: full time series
# ──────────────────────────────────────────────────────────────────────────────
def apply_analysis(calib, make_overlays=6, est_height_dir=None):
    base_y = calib["base_y"]
    cx_seed = calib["cx_seed"]
    slope = calib["slope"]
    intercept = calib["intercept"]

    paths = sorted(RAW_DIR.glob("tl_*.tif"))
    if not paths:
        raise FileNotFoundError(f"No raw frames (tl_*.tif) in {RAW_DIR}")
    print(f"\nProcessing {len(paths)} raw frames from {RAW_DIR.name} ...")

    idx, tss, whens, feats, totals, cxs = [], [], [], [], [], []
    for i, p in enumerate(paths):
        ts, when = _timestamp_from_name(p.name)
        gray, g8 = load_stretched_u8(p)
        sil = solid_silhouette(gray)
        cx = derive_cx(g8, base_y, cx_seed)
        feats.append(corridor_feature(sil, base_y, cx,
                                      calib["corridor_half"], calib["density_t"],
                                      calib["smooth_k"], calib["max_gap"], calib["search_above"]))
        totals.append(total_height(sil, base_y, calib["search_above"]))
        idx.append(i); tss.append(ts); whens.append(when); cxs.append(cx)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(paths)}")

    feat = np.array(feats, float)
    total_raw = np.array(totals, float)
    h_raw = slope * feat + intercept
    h_filt = robust_filter(h_raw)
    total_filt = robust_filter(total_raw)

    t0 = next((w for w in whens if w is not None), None)
    hours = np.array([((w - t0).total_seconds() / 3600.0) if (w and t0) else np.nan for w in whens])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(idx, tss, whens, hours, cxs, feat, h_raw, h_filt, total_raw, total_filt)
    _plot_growth(hours, h_raw, h_filt, total_filt, calib, t0)
    if make_overlays:
        _plot_overlays(paths, calib, n=make_overlays)
    if est_height_dir is not None:
        save_estimated_height_images(calib, est_height_dir)
    print("\nDone.")


def _write_csv(idx, tss, whens, hours, cxs, feat, h_raw, h_filt, total_raw, total_filt):
    out = OUT_DIR / "stem_height_timeseries.csv"

    def f2(v):
        return f"{v:.2f}" if not (v is None or np.isnan(v)) else ""

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "timestamp", "datetime_iso", "hours_from_start", "cx",
                    "feature_px", "stem_h_px_raw", "stem_h_px_filtered",
                    "total_h_px_raw", "total_h_px_filtered"])
        for i in range(len(idx)):
            w.writerow([idx[i], tss[i], whens[i].isoformat() if whens[i] else "",
                        f"{hours[i]:.4f}" if not np.isnan(hours[i]) else "", cxs[i],
                        f2(feat[i]), f2(h_raw[i]), f2(h_filt[i]),
                        f2(total_raw[i]), f2(total_filt[i])])
    print(f"  CSV   -> {out}")


def _gt_for_plot(t0):
    gx, gy = [], []
    for ts, cx, apex_y, base_y in _read_annotations():
        _, when = _timestamp_from_name(ts)
        if when is not None and t0 is not None:
            gx.append((when - t0).total_seconds() / 3600.0)
            gy.append(base_y - apex_y)
    return gx, gy


def _plot_growth(hours, h_raw, h_filt, total_filt, calib, t0):
    out = OUT_DIR / "stem_height_growth_curve.png"
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hours, total_filt, "-", color="C2", lw=1.4, alpha=0.9, label="total plant height (filtered)")
    ax.plot(hours, h_raw, ".", color="0.78", ms=4, label="stem height (raw)")
    ax.plot(hours, h_filt, "-", color="C0", lw=1.8, label="stem height (Hampel + rolling median)")
    gx, gy = _gt_for_plot(t0)
    if gx:
        ax.plot(gx, gy, "r*", ms=13, label=f"manual annotation ({len(gx)})")
    ax.set_xlabel("hours from start")
    ax.set_ylabel("height (px)")
    ax.set_title(f"Stem & total plant height over time "
                 f"(stem corr {calib['corr']:+.2f}, LOO-CV {calib['loo_cv_mae_px']:.0f} px)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(str(out), dpi=120); plt.close(fig)
    print(f"  plot  -> {out}")


def _reference_crop(margin=90, n_last=10):
    paths = sorted(RAW_DIR.glob("tl_*.tif"))
    sample = paths[-n_last:] if len(paths) >= n_last else paths
    H = W = None
    y0 = x0 = 10 ** 9
    y1 = x1 = -1
    for p in sample:
        gray, _ = load_stretched_u8(p)
        sil = solid_silhouette(gray)
        H, W = sil.shape
        fg = sil > 127
        rows = np.where(fg.sum(1) > 5)[0]
        cols = np.where(fg.sum(0) > 5)[0]
        if len(rows) == 0 or len(cols) == 0:
            continue
        y0 = min(y0, int(rows.min())); y1 = max(y1, int(rows.max()))
        x0 = min(x0, int(cols.min())); x1 = max(x1, int(cols.max()))
    if y1 < 0:
        return 0, (H - 1 if H else 0), 0, (W - 1 if W else 0)
    return (max(0, y0 - margin), min(H - 1, y1 + margin),
            max(0, x0 - margin), min(W - 1, x1 + margin))


def _draw_estimate(g8, sil, calib, cx):
    base_y = calib["base_y"]; W2 = calib["corridor_half"]
    feat = corridor_feature(sil, base_y, cx, calib["corridor_half"], calib["density_t"],
                            calib["smooth_k"], calib["max_gap"], calib["search_above"])
    total = total_height(sil, base_y, calib["search_above"])
    vis = cv2.cvtColor((g8 // 2).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # silhouette outline (faint blue)
    cnts, _ = cv2.findContours((sil > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (180, 120, 0), 1)
    cv2.line(vis, (cx - W2, base_y), (cx + W2, base_y), (0, 165, 255), 3)        # base (orange)
    if not np.isnan(total):
        ptop = int(round(base_y - total))
        cv2.line(vis, (0, ptop), (vis.shape[1] - 1, ptop), (255, 0, 255), 2)     # plant top (magenta)
    h = np.nan
    if not np.isnan(feat):
        h = calib["slope"] * feat + calib["intercept"]
        apex = int(round(base_y - h))
        cv2.line(vis, (cx, base_y), (cx, apex), (0, 0, 255), 4)                   # stem (red)
        cv2.line(vis, (cx - 40, apex), (cx + 40, apex), (0, 255, 0), 3)           # apex (green)
    return vis, feat, h, total


def _plot_overlays(paths, calib, n=6):
    out = OUT_DIR / "stem_height_overlays.png"
    cy0, cy1, cx0, cx1 = _reference_crop()
    sel = np.linspace(0, len(paths) - 1, n, dtype=int)
    fig, axs = plt.subplots(1, n, figsize=(4 * n, 7))
    if n == 1:
        axs = [axs]
    for ax, k in zip(axs, sel):
        gray, g8 = load_stretched_u8(paths[k])
        sil = solid_silhouette(gray)
        cx = derive_cx(g8, calib["base_y"], calib["cx_seed"])
        vis, feat, h, total = _draw_estimate(g8, sil, calib, cx)
        crop = vis[cy0:cy1 + 1, cx0:cx1 + 1]
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); ax.axis("off")
        ts, _ = _timestamp_from_name(paths[k].name)
        ax.set_title(f"{ts[5:]}\nstem_h={h:.0f}px" if not np.isnan(h) else f"{ts[5:]}\n(no stem)", fontsize=8)
    fig.suptitle("orange=base  red=stem(predicted)  green=apex  magenta=plant top", fontsize=11)
    fig.tight_layout(); fig.savefig(str(out), dpi=120); plt.close(fig)
    print(f"  overlays -> {out}")


def save_estimated_height_images(calib, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cy0, cy1, cx0, cx1 = _reference_crop()
    paths = sorted(RAW_DIR.glob("tl_*.tif"))
    n = 0
    for p in paths:
        gray, g8 = load_stretched_u8(p)
        sil = solid_silhouette(gray)
        cx = derive_cx(g8, calib["base_y"], calib["cx_seed"])
        vis, feat, h, total = _draw_estimate(g8, sil, calib, cx)
        crop = vis[cy0:cy1 + 1, cx0:cx1 + 1].copy()
        ts, _ = _timestamp_from_name(p.name)
        cv2.putText(crop, ts, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA)
        lab = "no stem detected" if np.isnan(h) else f"stem_h = {h:.0f} px"
        cv2.putText(crop, lab, (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 0), 2, cv2.LINE_AA)
        if not np.isnan(total):
            cv2.putText(crop, f"total_h = {total:.0f} px", (12, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / (p.stem + ".png")), crop)
        n += 1
    print(f"  estimated-height images ({n}) -> {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Stem-height analysis v2 (raw-silhouette corridor proxy).")
    ap.add_argument("--calibrate", action="store_true", help="fit & save calibration from annotations")
    ap.add_argument("--apply", action="store_true", help="run the full time series")
    ap.add_argument("--no-overlays", action="store_true", help="skip the QC overlay montage")
    ap.add_argument("--estimated-height-dir", nargs="?", const="__default__", default=None,
                    help="save a per-frame PNG with the estimate drawn (optional PATH)")
    args = ap.parse_args()

    print(f"raw    : {RAW_DIR}")
    print(f"annot  : {ANNOT_DIR}")
    print(f"out    : {OUT_DIR}")

    if not (args.calibrate or args.apply):
        ap.print_help()
        return

    calib = None
    if args.calibrate:
        calib = calibrate()
        save_calibration(calib)
    if args.apply:
        if calib is None:
            calib = load_calibration()
            if calib is None:
                raise SystemExit("No calibration found — run with --calibrate first.")
        est_dir = None
        if args.estimated_height_dir is not None:
            est_dir = (OUT_DIR / "estimated_height" if args.estimated_height_dir == "__default__"
                       else Path(args.estimated_height_dir))
        apply_analysis(calib, make_overlays=0 if args.no_overlays else 6, est_height_dir=est_dir)


if __name__ == "__main__":
    main()
