"""Quick robustness check of the intensity-gate strategy across the timelapse."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from plant_contour_v3 import (
    gaussian_blur_binary_mask,
    largest_connected_component_mask,
    load_grayscale_working,
    refined_roi_and,
    sobel_find_edges,
    threshold_binary_mask,
    threshold_soft_uint8,
)

TL_DIR = Path(
    r"H:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 5_1\DEV_1AB22C05B465"
    r"\timelapse_2026-05-07_12-56-08"
)
OUT = HERE / "out3"
OUT.mkdir(parents=True, exist_ok=True)


def auto_stretch_u8(g, sat=0.0035):
    flat = g.ravel()
    n = flat.size
    lo = float(np.partition(flat, int(sat * n))[int(sat * n)])
    hi = float(np.partition(flat, int((1 - sat) * n))[int((1 - sat) * n)])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((g.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def refined_pipeline(gray):
    edges = sobel_find_edges(gray, blur_ksize=0, sobel_ksize=3, uint8=False)
    mask, _ = threshold_binary_mask(edges, "triangle", None)
    smeared, _ = gaussian_blur_binary_mask(mask, 40.0)
    mask_final, _ = threshold_soft_uint8(smeared, "otsu", None)
    mask_final = largest_connected_component_mask(mask_final)
    return refined_roi_and(mask, mask_final)


def otsu_inside_mask(gray_u8, region):
    vals = gray_u8[region > 0]
    if vals.size < 64:
        return 127
    hist = np.bincount(vals, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    idx = np.arange(256)
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * idx)
    mu_t = mu[-1]
    valid = (w0 > 1e-12) & (w1 > 1e-12)
    sb2 = np.zeros(256)
    sb2[valid] = (mu_t * w0[valid] - mu[valid]) ** 2 / (w0[valid] * w1[valid])
    return int(np.argmax(sb2))


def overlay(gray01, mask, color=(0, 0, 255)):
    rgb = cv2.cvtColor((gray01 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, contours, -1, color, 1)
    return rgb


paths = sorted(TL_DIR.glob("tl_*.tif"))
# 5 evenly spaced frames
idxs = np.linspace(0, len(paths) - 1, 5).round().astype(int)
sel = [paths[i] for i in idxs]
print(f"Selected: {[p.name for p in sel]}")

fig, axes = plt.subplots(len(sel), 3, figsize=(18, 5 * len(sel)))
for row, p in enumerate(sel):
    g16 = load_grayscale_working(p, uint8=False)
    g01 = auto_stretch_u8(g16)
    g_u8 = (g01 * 255).astype(np.uint8)
    refined = refined_pipeline(g16)
    t = otsu_inside_mask(g_u8, refined)
    margin = 25
    t_used = max(0, t - margin)
    gate = ((g_u8 >= t_used).astype(np.uint8)) * 255
    cleaned = cv2.bitwise_and(refined, gate)
    # Quick stats
    a_ref = int((refined > 0).sum())
    a_cln = int((cleaned > 0).sum())
    print(f"  {p.name}  T_otsu={t}, T_used={t_used}, refined={a_ref}, cleaned={a_cln}")
    axes[row, 0].imshow(g01, cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(p.name + "\n(orig auto)", fontsize=9)
    axes[row, 1].imshow(overlay(g01, refined))
    axes[row, 1].set_title(f"refined ({a_ref} px)", fontsize=9)
    axes[row, 2].imshow(overlay(g01, cleaned))
    axes[row, 2].set_title(
        f"refined AND (gray >= T_otsu-{margin})  "
        f"T_otsu={t}  →  T_used={t_used}  ({a_cln} px)",
        fontsize=9,
    )
    for ax in axes[row]:
        ax.axis("off")

plt.tight_layout()
fig.savefig(OUT / "robustness.png", dpi=110)
plt.close(fig)
print(f"Saved {OUT / 'robustness.png'}")
