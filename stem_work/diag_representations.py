"""
Diagnostic: for a handful of ANNOTATED frames, render
   raw grayscale | solid silhouette | Sobel edge mask
with the ground-truth (red-line) base->apex overlaid, so we can SEE where the
apical meristem sits in each representation. Output is a single montage PNG.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(r"C:\Users\ryank\Code\microneedle_IAA_Raman_analysis")
sys.path.insert(0, str(REPO))
from plant_timelapse import plant_contour_v3 as v3

DEV = Path(r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
           r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
           r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465")
RAW_DIR  = DEV / "timelapse_2026-05-07_12-56-08"
MASK_DIR = DEV / "masks_freq_filled"
ANNOT    = Path(r"C:\Users\ryank\Desktop\stem height examples")
OUT      = REPO / "stem_work"


def extract_red_line(img_bgr):
    b, g, r = (img_bgr[..., 0].astype(int), img_bgr[..., 1].astype(int), img_bgr[..., 2].astype(int))
    red = (r > 90) & (g < 70) & (b < 70)
    ys, xs = np.where(red)
    if len(ys) < 20:
        return None
    return int(np.median(xs)), int(ys.min()), int(ys.max())  # cx, apex_y, base_y


def solid_silhouette(gray):
    """Reproduce v3 'binary after gaussian blur, largest component' plant blob."""
    edges = v3.sobel_find_edges(gray, blur_ksize=0, sobel_ksize=3, uint8=False)
    mask, _ = v3.threshold_binary_mask(edges, "triangle", None)
    smeared, _ = v3.gaussian_blur_binary_mask(mask, 50.0)
    final, _ = v3.threshold_soft_uint8(smeared, "otsu", None, otsu_margin=0)
    final = v3.largest_connected_component_mask(final)
    final = v3.binary_fill_holes_u8(final)
    return final


def main():
    annots = sorted(ANNOT.glob("*_mask_refined.png"))
    # drop the *_stem_height.png variant
    annots = [p for p in annots if "_stem_height" not in p.name]
    # ground truth per timestamp
    gts = {}
    for p in annots:
        img = cv2.imread(str(p))
        line = extract_red_line(img)
        if line:
            ts = p.name.split("tl_", 1)[1].split("_mask_refined", 1)[0]
            gts[ts] = line
    tss = sorted(gts)
    print(f"{len(tss)} annotated frames")

    # pick 6 across the timeline
    sel = [tss[i] for i in np.linspace(0, len(tss) - 1, 6).round().astype(int)]
    n = len(sel)
    fig, axs = plt.subplots(3, n, figsize=(3.2 * n, 11))

    for col, ts in enumerate(sel):
        cx, apex_y, base_y = gts[ts]
        raw_p = RAW_DIR / f"tl_{ts}.tif"
        mask_p = MASK_DIR / f"tl_{ts}_mask_refined.png"
        gray = v3.load_grayscale_working(raw_p, uint8=False)
        g8 = v3.auto_stretched_gray_u8(gray, saturated_fraction=0.0035)
        sil = solid_silhouette(gray)
        edge = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        # crop to plant region using silhouette bbox + margin
        ys, xs = np.where(sil > 127)
        if len(ys):
            y0, y1 = max(0, ys.min() - 80), min(gray.shape[0], ys.max() + 80)
            x0, x1 = max(0, xs.min() - 80), min(gray.shape[1], xs.max() + 80)
        else:
            y0, y1, x0, x1 = 0, gray.shape[0], 0, gray.shape[1]

        def draw(base_img_gray):
            v = cv2.cvtColor(base_img_gray, cv2.COLOR_GRAY2BGR)
            cv2.line(v, (cx, base_y), (cx, apex_y), (0, 0, 255), 3)        # GT stem (red)
            cv2.line(v, (cx - 60, base_y), (cx + 60, base_y), (0, 165, 255), 2)  # base
            cv2.line(v, (cx - 60, apex_y), (cx + 60, apex_y), (0, 255, 0), 2)    # apex
            return cv2.cvtColor(v[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)

        for row, (im, name) in enumerate([(g8, "raw"), (sil, "silhouette"), (edge, "edge mask")]):
            axs[row, col].imshow(draw(im))
            axs[row, col].axis("off")
            if row == 0:
                axs[row, col].set_title(f"{ts[5:]}\nGT stem_h={base_y-apex_y}px", fontsize=8)
            if col == 0:
                axs[row, col].text(-0.08, 0.5, name, transform=axs[row, col].transAxes,
                                   rotation=90, va="center", fontsize=11, fontweight="bold")

    fig.suptitle("Ground-truth apex (green) / base (orange) on raw, silhouette, edges", fontsize=12)
    fig.tight_layout()
    outp = OUT / "diag_representations.png"
    fig.savefig(str(outp), dpi=110)
    print("saved", outp)


if __name__ == "__main__":
    main()
