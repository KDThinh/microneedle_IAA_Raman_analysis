"""Measure the horizontal gap between the two parallel Sobel stem edge lines."""
import cv2, numpy as np
from pathlib import Path

MASKS_DIR = Path(
    r"G:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 4_1\DEV_1AB22C05B465\masks"
)
CORRIDOR_HALF_PX = 40
BOTTOM_FRAC = 0.15; PROJ_BAND = (0.20, 0.85); ANCHOR_MARGIN = 150; SMOOTH_K = 15

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
    bt = int(y0 + PROJ_BAND[0] * ph); bb = int(y0 + PROJ_BAND[1] * ph)
    band = mask[bt:bb+1, sx0:sx1+1] > 127
    if not band.any():
        return ca
    proj = col_max_run_lengths(band).astype(float)
    sm = np.convolve(proj, np.ones(SMOOTH_K) / SMOOTH_K, mode='same')
    return sx0 + int(np.argmax(sm))

masks = sorted(MASKS_DIR.glob("*_mask_refined.png"))
step  = max(1, len(masks) // 3)
for mp in masks[::step][:3]:
    mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
    ys, xs = np.where(mask > 127)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cx  = estimate_stem_cx(mask, y0, y1, x0, x1)
    cx0 = max(0, cx - CORRIDOR_HALF_PX)
    cx1 = min(mask.shape[1]-1, cx + CORRIDOR_HALF_PX)
    plant_h = y1 - y0

    ts = mp.stem.replace("tl_", "").replace("_mask_refined", "")
    print(f"\n=== {ts}  corridor [{cx0},{cx1}]  cx={cx}  plant_h={plant_h}px ===")
    print(f"{'Height%':>7}  {'y':>5}  {'gap_px':>7}  {'left_lineW':>11}  "
          f"{'right_lineW':>12}  {'total_white':>12}  note")

    gaps = []
    for frac in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]:
        y   = int(y0 + frac * plant_h)
        row = mask[y, cx0:cx1+1]
        wh  = np.where(row > 127)[0]
        if len(wh) < 2:
            print(f"{frac*100:>6.0f}%  {y:>5}  (no / single white pixel)")
            continue
        first_w, last_w = int(wh[0]), int(wh[-1])
        interior = row[first_w:last_w+1]

        # Find all dark runs inside the white region
        dark_runs = []
        run = 0; start = 0
        for i, v in enumerate(interior):
            if v <= 127:
                run += 1
            else:
                if run > 0:
                    dark_runs.append((run, first_w + start))
                    run = 0
                start = i + 1
        if run > 0:
            dark_runs.append((run, first_w + start))

        if not dark_runs:
            print(f"{frac*100:>6.0f}%  {y:>5}  (no gap — already merged/single blob)  "
                  f"white_span={last_w-first_w+1}px")
            continue

        # Largest dark run = the internal gap between the two Sobel lines
        gap_len, gap_abs_start = max(dark_runs)
        gap_abs_end = gap_abs_start + gap_len - 1

        left_whites  = wh[wh < gap_abs_start]
        right_whites = wh[wh > gap_abs_end]
        lw = int(left_whites[-1]  - left_whites[0]  + 1) if len(left_whites)  else 0
        rw = int(right_whites[-1] - right_whites[0] + 1) if len(right_whites) else 0

        gaps.append(gap_len)
        note = "  <<< too many dark runs" if len(dark_runs) > 2 else ""
        print(f"{frac*100:>6.0f}%  {y:>5}  gap={gap_len:>4}px  "
              f"left_line={lw:>4}px  right_line={rw:>4}px  "
              f"total_white={len(wh):>4}px{note}")

    if gaps:
        print(f"  --> gap summary: min={min(gaps)} max={max(gaps)} "
              f"median={int(np.median(gaps))} mean={np.mean(gaps):.1f}  "
              f"=> CLOSE_K should be >= {max(gaps)+2}px")
