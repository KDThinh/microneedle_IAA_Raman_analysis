"""
Stream all raw frames; compute several candidate per-frame features and save a
local CSV so we can compare temporal smoothness offline (without re-reading G:).

Features (all = base_y - <some top row>):
  edge_canopy        : current method, on the freq-filled EDGE mask
  raw_canopy         : same threshold rule but on RAW brightness
  raw_canopy_cont    : RAW brightness, top of CONTIGUOUS (gap-tolerant) run anchored at base
"""
import sys, csv, datetime as dt
from pathlib import Path
import numpy as np
import cv2

REPO = Path(r"C:\Users\ryank\Code\microneedle_IAA_Raman_analysis")
sys.path.insert(0, str(REPO))
from plant_timelapse import plant_contour_v3 as v3

DEV = Path(r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
           r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
           r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465")
RAW_DIR  = DEV / "timelapse_2026-05-07_12-56-08"
MASK_DIR = DEV / "masks_freq_filled"
OUT = REPO / "stem_work" / "full_series.csv"

BASE_Y = 2410
CX0 = 980
CHALF = 90
DENS_T = 0.20
SMOOTH = 15
BRIGHT_T = 55
TOP_LIMIT = 1200


def smooth1d(a, k):
    return a if k <= 1 else np.convolve(a, np.ones(k) / k, mode="same")


def derive_cx(g8, base_y, seed_cx, band=160, nrows=140):
    H, W = g8.shape
    y0 = max(0, base_y - nrows)
    lo, hi = max(0, seed_cx - band), min(W, seed_cx + band)
    sub = g8[y0:base_y, lo:hi].astype(float)
    colsum = (sub * (sub > BRIGHT_T)).sum(axis=0)
    if colsum.sum() == 0:
        return seed_cx
    return lo + int(round((np.arange(len(colsum)) * colsum).sum() / colsum.sum()))


def canopy_top(profile, base_y, dt_):
    top = max(0, base_y - TOP_LIMIT)
    hits = np.where(profile[top:base_y] > dt_)[0]
    return float(base_y - (top + hits.min())) if len(hits) else np.nan


def canopy_cont(profile, base_y, dt_, max_gap=18):
    top = max(0, base_y - TOP_LIMIT); gap = 0; best = base_y
    started = False
    for y in range(base_y, top, -1):
        if profile[y] > dt_:
            best = y; gap = 0; started = True
        elif started:
            gap += 1
            if gap > max_gap:
                break
    return float(base_y - best)


def solid_silhouette(gray):
    edges = v3.sobel_find_edges(gray, blur_ksize=0, sobel_ksize=3, uint8=False)
    mask, _ = v3.threshold_binary_mask(edges, "triangle", None)
    smeared, _ = v3.gaussian_blur_binary_mask(mask, 50.0)
    final, _ = v3.threshold_soft_uint8(smeared, "otsu", None, otsu_margin=0)
    final = v3.largest_connected_component_mask(final)
    return v3.binary_fill_holes_u8(final)


def ts_from(name):
    s = name.split("tl_", 1)[1].split("_mask_refined", 1)[0].replace(".tif", "")
    try:
        return s, dt.datetime.strptime(s, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return s, None


def main():
    raws = sorted(RAW_DIR.glob("tl_*.tif"))
    print(f"{len(raws)} raw frames")
    rows = []
    for i, rp in enumerate(raws):
        ts, when = ts_from(rp.name)
        gray = v3.load_grayscale_working(rp, uint8=False)
        g8 = v3.auto_stretched_gray_u8(gray, saturated_fraction=0.0035)
        cx = derive_cx(g8, BASE_Y, CX0)
        lo, hi = max(0, cx - CHALF), min(g8.shape[1], cx + CHALF)
        raw_prof = smooth1d((g8[:, lo:hi] > BRIGHT_T).mean(axis=1), SMOOTH)
        f_raw = canopy_top(raw_prof, BASE_Y, DENS_T)
        f_raw_cont = canopy_cont(raw_prof, BASE_Y, DENS_T)
        # silhouette corridor proxy (solid blob built from raw)
        sil = solid_silhouette(gray)
        sprof = smooth1d((sil[:, lo:hi] > 127).mean(axis=1), SMOOTH)
        f_sil = canopy_cont(sprof, BASE_Y, DENS_T)
        # silhouette FULL-width top (canopy apex, no corridor) = total plant top
        srows = np.where((sil > 127).sum(axis=1) > 30)[0]
        f_sil_top = float(BASE_Y - srows.min()) if len(srows) else np.nan
        # edge mask version
        mp = MASK_DIR / f"tl_{ts}_mask_refined.png"
        f_edge = np.nan
        if mp.exists():
            edge = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            eprof = smooth1d((edge[:, lo:hi] > 127).mean(axis=1), SMOOTH)
            f_edge = canopy_top(eprof, BASE_Y, DENS_T)
        rows.append([i, ts, when.isoformat() if when else "", cx, f_edge, f_raw, f_raw_cont, f_sil, f_sil_top])
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(raws)}")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "ts", "datetime", "cx", "edge_canopy", "raw_canopy", "raw_canopy_cont", "sil_corr", "sil_top"])
        w.writerows(rows)
    print("saved", OUT)


if __name__ == "__main__":
    main()
