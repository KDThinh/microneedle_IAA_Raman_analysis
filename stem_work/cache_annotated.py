"""Cache the 41 annotated frames locally: auto-stretched 8-bit raw + edge mask + GT.
Saves a single npz so experiments run fast without hitting the slow G: drive."""
import sys
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
ANNOT    = Path(r"C:\Users\ryank\Desktop\stem height examples")
OUT      = REPO / "stem_work"
CACHE    = OUT / "annotated_cache.npz"


def extract_red_line(img_bgr):
    b, g, r = (img_bgr[..., 0].astype(int), img_bgr[..., 1].astype(int), img_bgr[..., 2].astype(int))
    red = (r > 90) & (g < 70) & (b < 70)
    ys, xs = np.where(red)
    if len(ys) < 20:
        return None
    return int(np.median(xs)), int(ys.min()), int(ys.max())


def main():
    annots = sorted(ANNOT.glob("*_mask_refined.png"))
    annots = [p for p in annots if "_stem_height" not in p.name]
    tss, cxs, apex, base, raws, edges = [], [], [], [], [], []
    for p in annots:
        line = extract_red_line(cv2.imread(str(p)))
        if not line:
            continue
        ts = p.name.split("tl_", 1)[1].split("_mask_refined", 1)[0]
        raw_p = RAW_DIR / f"tl_{ts}.tif"
        mask_p = MASK_DIR / f"tl_{ts}_mask_refined.png"
        if not raw_p.exists() or not mask_p.exists():
            print("missing", ts); continue
        gray = v3.load_grayscale_working(raw_p, uint8=False)
        g8 = v3.auto_stretched_gray_u8(gray, saturated_fraction=0.0035)
        edge = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        tss.append(ts); cxs.append(line[0]); apex.append(line[1]); base.append(line[2])
        raws.append(g8); edges.append(edge)
        print("cached", ts, "shape", g8.shape)
    np.savez_compressed(CACHE, ts=np.array(tss), cx=np.array(cxs), apex=np.array(apex),
                        base=np.array(base), raw=np.array(raws), edge=np.array(edges))
    print(f"\nsaved {len(tss)} frames -> {CACHE}  (shape {raws[0].shape})")


if __name__ == "__main__":
    main()
