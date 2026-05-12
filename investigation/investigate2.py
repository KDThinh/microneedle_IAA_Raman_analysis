"""
Investigate intensity / temporal strategies for line-artifact removal.

Premise: the artifact in this scene is the microneedle device — a static rod/clamp
that passes through the plant. The plant is bright; the device is dark.

Strategies tested:
  A. Single-frame intensity gating on the refined ROI
       - Otsu on gray *inside* the refined mask -> drop dark pixels
       - Adaptive (Gaussian) threshold on the full gray
       - Local mean threshold inside the refined mask
  B. Bright-anchor reconstruction: top-percentile gray pixels are 'plant anchors',
     then morphologically reconstruct under the refined mask
  C. Temporal background subtraction
       - Median across N evenly-spaced frames -> static background
       - |gray - median| >> noise floor -> 'moving plant'
       - Std-dev across frames -> high-std pixels are plant, low-std are device
  D. Combined: refined AND (plant_intensity_mask OR moving_plant_mask)
"""

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
TARGET = TL_DIR / "tl_2026-05-07_12-56-30.tif"

OUT = HERE / "out2"
OUT.mkdir(parents=True, exist_ok=True)


def _save(name: str, img: np.ndarray) -> None:
    p = OUT / name
    if not cv2.imwrite(str(p), img):
        raise OSError(p)


def _save_overlay(name: str, gray01: np.ndarray, mask: np.ndarray, color=(0, 0, 255)) -> None:
    rgb = cv2.cvtColor((gray01 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, contours, -1, color, 1)
    _save(name, rgb)


def auto_stretch_u8(g: np.ndarray, sat: float = 0.0035) -> np.ndarray:
    flat = g.ravel()
    n = flat.size
    lo = float(np.partition(flat, int(sat * n))[int(sat * n)])
    hi = float(np.partition(flat, int((1 - sat) * n))[int((1 - sat) * n)])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((g.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def refined_pipeline(gray: np.ndarray) -> np.ndarray:
    edges = sobel_find_edges(gray, blur_ksize=0, sobel_ksize=3, uint8=False)
    mask, _ = threshold_binary_mask(edges, "triangle", None)
    smeared, _ = gaussian_blur_binary_mask(mask, 40.0)
    mask_final, _ = threshold_soft_uint8(smeared, "otsu", None)
    mask_final = largest_connected_component_mask(mask_final)
    return refined_roi_and(mask, mask_final)


def gray_u8_full_stretch(g16: np.ndarray) -> np.ndarray:
    """Per-frame intensity to 0-255 with the same auto-stretch used for display."""
    return (auto_stretch_u8(g16) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
def otsu_inside_mask(gray_u8: np.ndarray, region: np.ndarray) -> int:
    """Otsu threshold on pixels inside `region` (uint8 mask 0/255)."""
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


def reconstruct_in_mask(anchor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Geodesic reconstruction: dilate anchor under mask until idempotent."""
    cur = cv2.bitwise_and(anchor, mask).copy()
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(500):
        nxt = cv2.dilate(cur, k)
        nxt = cv2.bitwise_and(nxt, mask)
        if np.array_equal(nxt, cur):
            return cur
        cur = nxt
    return cur


# ---------------------------------------------------------------------------
def temporal_stats(frames: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-pixel median, std, min, max across N frames (uint16 inputs ->
    median/std in float32).
    """
    print(f"  Reading {len(frames)} frames for temporal stats...")
    stack = []
    for p in frames:
        g = load_grayscale_working(p, uint8=False).astype(np.float32)
        stack.append(g)
    arr = np.stack(stack, axis=0)
    med = np.median(arr, axis=0)
    std = arr.std(axis=0)
    amin = arr.min(axis=0)
    amax = arr.max(axis=0)
    return med, std, amin, amax


def sample_frames(directory: Path, target: Path, count: int = 21) -> list[Path]:
    paths = sorted(directory.glob("tl_*.tif"))
    if not paths:
        raise FileNotFoundError(directory)
    if target not in paths:
        paths.append(target)
        paths = sorted(set(paths))
    if len(paths) <= count:
        return paths
    idxs = np.linspace(0, len(paths) - 1, count).round().astype(int)
    return [paths[i] for i in idxs]


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Target: {TARGET}")
    gray16 = load_grayscale_working(TARGET, uint8=False)
    g_u8 = gray_u8_full_stretch(gray16)
    g01 = g_u8.astype(np.float32) / 255.0

    refined = refined_pipeline(gray16)
    _save("00_refined.png", refined)
    _save_overlay("00b_refined_overlay.png", g01, refined)

    # ---- A. Intensity gating ------------------------------------------------
    print("\n[A. Intensity gating]")
    t_otsu_in = otsu_inside_mask(g_u8, refined)
    print(f"  Otsu inside refined mask: T={t_otsu_in}")
    bright_in = ((g_u8 >= t_otsu_in).astype(np.uint8)) * 255
    plant_intensity_mask = cv2.bitwise_and(refined, bright_in)
    _save("A1_intensity_clean.png", plant_intensity_mask)
    _save_overlay("A1b_intensity_overlay.png", g01, plant_intensity_mask)

    # Slightly relaxed: use a fixed margin below Otsu
    t_relaxed = max(0, t_otsu_in - 25)
    bright_relaxed = ((g_u8 >= t_relaxed).astype(np.uint8)) * 255
    plant_intensity_mask2 = cv2.bitwise_and(refined, bright_relaxed)
    _save("A2_intensity_relaxed.png", plant_intensity_mask2)
    _save_overlay("A2b_intensity_relaxed_overlay.png", g01, plant_intensity_mask2)

    # Adaptive (background subtraction style) on gray
    adap = cv2.adaptiveThreshold(
        g_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -8
    )
    adap_in_ref = cv2.bitwise_and(refined, adap)
    _save("A3_adaptive_in_refined.png", adap_in_ref)
    _save_overlay("A3b_adaptive_overlay.png", g01, adap_in_ref)

    # ---- B. Bright-anchor reconstruction ------------------------------------
    print("\n[B. Bright-anchor reconstruction]")
    # Anchor = very bright AND inside refined ROI (sure plant)
    # Use a high percentile of gray-inside-refined
    vals = g_u8[refined > 0]
    if vals.size > 0:
        anchor_t = int(np.percentile(vals, 75))
    else:
        anchor_t = 200
    print(f"  bright-anchor T (p75 inside refined) = {anchor_t}")
    anchors = ((g_u8 >= anchor_t).astype(np.uint8)) * 255
    anchors = cv2.bitwise_and(anchors, refined)
    # Reconstruct under (refined AND not-very-dark) so we don't propagate into rod pixels
    safe = cv2.bitwise_and(refined, bright_relaxed)
    recon = reconstruct_in_mask(anchors, safe)
    _save("B1_bright_anchors.png", anchors)
    _save("B2_bright_recon.png", recon)
    _save_overlay("B2b_bright_recon_overlay.png", g01, recon)

    # ---- C. Temporal background subtraction ---------------------------------
    print("\n[C. Temporal background subtraction]")
    samples = sample_frames(TL_DIR, TARGET, count=15)
    print(f"  Using {len(samples)} sample frames")
    med, std, amin, amax = temporal_stats(samples)

    # Save the temporal stats as previews
    med_u8 = (auto_stretch_u8(med.astype(np.uint16)) * 255).astype(np.uint8)
    std_u8 = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _save("C0_temporal_median.png", med_u8)
    _save("C1_temporal_std.png", std_u8)

    # Movement mask: where current frame is significantly different from median
    diff = np.abs(gray16.astype(np.float32) - med).astype(np.float32)
    diff_u8 = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _save("C2_abs_diff_from_median.png", diff_u8)

    # Threshold the diff to flag moving content
    # Robust scale: use median of diff over background-ish (low temporal std) area as noise
    noise_floor = float(np.median(std))
    print(f"  noise floor (median of temporal std) = {noise_floor:.1f}")
    diff_thresh = max(3.0 * noise_floor, 50.0)
    moving = ((diff > diff_thresh).astype(np.uint8)) * 255
    # Clean up small noise
    moving = cv2.morphologyEx(
        moving, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    _save("C3_moving_mask.png", moving)
    _save_overlay("C3b_moving_overlay.png", g01, moving)

    # Static mask: low temporal std -> 'fixed device + background' candidate
    static = ((std < noise_floor * 1.5).astype(np.uint8)) * 255
    _save("C4_static_mask.png", static)

    # Refined AND moving -> plant edges that are not part of static device
    refined_moving = cv2.bitwise_and(refined, moving)
    _save("C5_refined_AND_moving.png", refined_moving)
    _save_overlay("C5b_refined_AND_moving_overlay.png", g01, refined_moving)

    # Refined AND NOT static
    refined_minus_static = cv2.bitwise_and(refined, cv2.bitwise_not(static))
    _save("C6_refined_minus_static.png", refined_minus_static)
    _save_overlay("C6b_refined_minus_static_overlay.png", g01, refined_minus_static)

    # ---- D. Combined: intensity OR temporal ---------------------------------
    print("\n[D. Combined: intensity OR moving]")
    combined = cv2.bitwise_or(plant_intensity_mask, refined_moving)
    _save("D1_combined.png", combined)
    _save_overlay("D1b_combined_overlay.png", g01, combined)

    # The MOST conservative: refined AND (bright OR moving), so we recover plant
    # wherever EITHER strong prior agrees
    bright_or_moving = cv2.bitwise_or(bright_relaxed, moving)
    safe_clean = cv2.bitwise_and(refined, bright_or_moving)
    _save("D2_refined_AND_(bright_OR_moving).png", safe_clean)
    _save_overlay("D2b_safe_overlay.png", g01, safe_clean)

    # ---- Collage ------------------------------------------------------------
    cells = [
        ("orig (auto)", g01),
        ("refined", refined / 255.0),
        ("temporal median", med_u8 / 255.0),
        ("|gray - median|", diff_u8 / 255.0),
        ("A1 intensity (Otsu in mask)", plant_intensity_mask / 255.0),
        ("A2 intensity relaxed", plant_intensity_mask2 / 255.0),
        ("A3 adaptive in refined", adap_in_ref / 255.0),
        ("B bright-anchor recon", recon / 255.0),
        ("C5 refined AND moving", refined_moving / 255.0),
        ("C6 refined NOT static", refined_minus_static / 255.0),
        ("D1 intensity OR moving", combined / 255.0),
        ("D2 refined AND (bright OR moving)", safe_clean / 255.0),
    ]
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 14))
    for ax, (title, img) in zip(axes.flat, cells):
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.suptitle(TARGET.name, fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / "99_collage.png", dpi=130)
    plt.close(fig)
    print(f"  saved {OUT / '99_collage.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
