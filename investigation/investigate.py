"""
Investigate line-artifact removal strategies for one frame.

Loads:
  H:\\My Drive\\...\\tl_2026-05-07_12-56-30.tif

Stages tested:
  1) Visualize source + Sobel edges + the existing pipeline's refined mask
  2) Existing strategies (morph H/V opening, Probabilistic Hough on Canny)
  3) New candidate strategies, each operating on the refined mask:
       a. Directional / oriented opening (theta sweep) on refined mask
       b. Wide, thin, axis-aligned + diagonal openings combined
       c. Skeleton-based linear-component filter (length & straightness)
       d. Connected-component eccentricity / aspect ratio filter
       e. Radon-domain dominant-direction suppression (FFT-free option: cv2.warpAffine
          + 1-D opening per angle), keeping the best plant recovery
       f. LSD (Line Segment Detector via cv2.createLineSegmentDetector if available,
          else ximgproc); fallback to FastLineDetector or just skip
       g. Inpainting AFTER subtraction to repair plant pixels that touch lines
       h. Final: per-pixel decision rule — drop only if at least two strategies agree

Outputs: PNGs and a summary collage under .\\investigation\\out\\
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
    gray_u8_for_canny,
    hough_prob_line_mask_from_gray,
    largest_connected_component_mask,
    line_artifact_masks_and_cleaned,
    load_grayscale_working,
    refined_roi_and,
    sobel_find_edges,
    threshold_binary_mask,
    threshold_soft_uint8,
)

IMG_PATH = Path(
    r"H:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal"
    r"\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control"
    r"\Light_6to22\Temp_Hum_Variable\Run 5_1\DEV_1AB22C05B465"
    r"\timelapse_2026-05-07_12-56-08\tl_2026-05-07_12-56-30.tif"
)

OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)


def _save(name: str, img: np.ndarray) -> None:
    p = OUT / name
    if not cv2.imwrite(str(p), img):
        raise OSError(p)
    print(f"  wrote {p.name:50s} shape={img.shape} dtype={img.dtype}")


def _save_pair(name: str, gray01: np.ndarray, mask: np.ndarray) -> None:
    """Side-by-side: stretched gray with mask outline overlaid in red."""
    rgb = cv2.cvtColor((gray01 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(rgb, contours, -1, (0, 0, 255), 1)
    _save(name, rgb)


def auto_stretch_u8(g: np.ndarray, sat: float = 0.0035) -> np.ndarray:
    flat = g.ravel()
    n = flat.size
    lo = np.partition(flat, int(sat * n))[int(sat * n)]
    hi = np.partition(flat, int((1 - sat) * n))[int((1 - sat) * n)]
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((g.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return out


def get_refined_mask(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate the default pipeline up through the refined mask."""
    edges = sobel_find_edges(gray, blur_ksize=0, sobel_ksize=3, uint8=False)
    mask, _ = threshold_binary_mask(edges, "triangle", None)
    smeared, _ = gaussian_blur_binary_mask(mask, 40.0)
    mask_final, _ = threshold_soft_uint8(smeared, "otsu", None)
    mask_final = largest_connected_component_mask(mask_final)
    refined = refined_roi_and(mask, mask_final)
    return edges, mask, refined


# ---------------------------------------------------------------------------
# Strategy a: directional opening across angles
# ---------------------------------------------------------------------------
def directional_open_union(
    binary_u8: np.ndarray,
    length: int,
    angles_deg: list[int],
    thickness: int = 1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    For each angle, build a thin rectangular SE of given pixel length and 'thickness',
    rotated to that angle, and morphologically OPEN the binary. Returns the union of all
    opens (= 'long thin straight features at any of these angles'), plus per-angle masks.
    """
    h, w = binary_u8.shape
    parts: list[np.ndarray] = []
    union = np.zeros_like(binary_u8)
    for ang in angles_deg:
        L = max(3, int(length))
        T = max(1, int(thickness))
        base = np.zeros((L, L), dtype=np.uint8)
        cv2.line(
            base,
            (0, L // 2),
            (L - 1, L // 2),
            color=1,
            thickness=T,
        )
        M = cv2.getRotationMatrix2D((L / 2.0, L / 2.0), -float(ang), 1.0)
        rot = cv2.warpAffine(
            base, M, (L, L), flags=cv2.INTER_NEAREST, borderValue=0
        )
        rot = (rot > 0).astype(np.uint8)
        if rot.sum() < L * 0.5:
            continue
        opened = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, rot)
        parts.append(opened)
        union = cv2.bitwise_or(union, opened)
    return union, parts


# ---------------------------------------------------------------------------
# Strategy c: skeleton + straightness filter
# ---------------------------------------------------------------------------
def skeleton(mask: np.ndarray) -> np.ndarray:
    """Zhang–Suen-ish skeleton using OpenCV ximgproc if available, else iterative erosion."""
    try:
        import cv2.ximgproc as xip  # type: ignore[attr-defined]

        return xip.thinning(mask, thinningType=xip.THINNING_GUOHALL)
    except Exception:
        skel = np.zeros_like(mask)
        img = mask.copy()
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img, element)
            opened = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, opened)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                return skel


def straight_skeleton_lines(
    binary_u8: np.ndarray,
    min_branch_len: int = 60,
    straightness_min: float = 0.92,
    dilate: int = 1,
) -> tuple[np.ndarray, int]:
    """
    Skeletonize the binary, then for each 8-connected skeleton branch compute
    straightness = endpoint_distance / arc_length. Keep branches that are long enough
    AND straight enough -> rasterize them as the 'line skeleton mask', then dilate to
    match the original line thickness.
    """
    skel = skeleton(binary_u8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (skel > 0).astype(np.uint8), connectivity=8
    )
    line_skel = np.zeros_like(binary_u8)
    n_kept = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_branch_len:
            continue
        ys, xs = np.where(labels == i)
        if xs.size < 2:
            continue
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        # Endpoint distance: use farthest-pair approximation via PCA bounding
        d2 = ((pts[None, :, :] - pts[:, None, :]) ** 2).sum(-1)
        if d2.size == 0:
            continue
        max_d = float(np.sqrt(d2.max()))
        straightness = max_d / float(area)  # area ≈ arc length for 1-px skeleton
        if straightness >= straightness_min:
            line_skel[ys, xs] = 255
            n_kept += 1
    if dilate > 0:
        # Dilate so we can subtract the *original* thick line, not just the 1-px skeleton
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        line_skel = cv2.dilate(line_skel, k, iterations=int(dilate))
    return line_skel, n_kept


# ---------------------------------------------------------------------------
# Strategy d: eccentricity / aspect ratio filter on connected components
# ---------------------------------------------------------------------------
def linear_cc_filter(
    binary_u8: np.ndarray,
    min_len: int = 60,
    max_thickness: int = 6,
    min_aspect: float = 6.0,
    min_extent_fill: float = 0.0,
) -> tuple[np.ndarray, int]:
    """
    Mark connected components that are long and thin (i.e., line-like):
      - oriented bounding-box long side >= min_len
      - short side <= max_thickness
      - long/short >= min_aspect
    Returns a binary mask of those components (line candidates).
    """
    fg = (binary_u8 > 0).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = np.zeros_like(binary_u8)
    n_lines = 0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_len:
            continue
        ys, xs = np.where(labels == i)
        pts = np.stack([xs, ys], axis=1).astype(np.float32)
        rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h),angle)
        w_r, h_r = rect[1]
        long_s = float(max(w_r, h_r))
        short_s = float(min(w_r, h_r))
        if long_s < min_len:
            continue
        if short_s > max_thickness:
            continue
        if short_s < 1e-3:
            short_s = 1.0
        if long_s / short_s < min_aspect:
            continue
        if min_extent_fill > 0:
            fill = area / max(1.0, long_s * max(short_s, 1.0))
            if fill < min_extent_fill:
                continue
        out[labels == i] = 255
        n_lines += 1
    return out, n_lines


# ---------------------------------------------------------------------------
# Strategy e: LSD (line segment detector) on the original grayscale
# ---------------------------------------------------------------------------
def lsd_line_mask(
    gray_u8: np.ndarray,
    line_thickness: int = 3,
    min_length: int = 40,
) -> tuple[np.ndarray, int]:
    """Try a real Line Segment Detector. Fall back to None if unavailable."""
    out = np.zeros_like(gray_u8)
    n = 0
    fld = None
    try:
        import cv2.ximgproc as xip  # type: ignore[attr-defined]

        fld = xip.createFastLineDetector(
            length_threshold=int(min_length),
            distance_threshold=1.41,
            canny_th1=50,
            canny_th2=150,
            canny_aperture_size=3,
            do_merge=True,
        )
    except Exception:
        fld = None
    if fld is None:
        try:
            lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)  # type: ignore[attr-defined]
            lines, _, _, _ = lsd.detect(gray_u8)
            if lines is not None:
                for ln in lines:
                    x1, y1, x2, y2 = ln[0].astype(int)
                    if np.hypot(x2 - x1, y2 - y1) < min_length:
                        continue
                    cv2.line(out, (x1, y1), (x2, y2), 255, max(1, int(line_thickness)))
                    n += 1
        except Exception:
            pass
        return out, n
    lines = fld.detect(gray_u8)
    if lines is None:
        return out, 0
    for ln in lines:
        x1, y1, x2, y2 = ln[0].astype(int)
        cv2.line(out, (x1, y1), (x2, y2), 255, max(1, int(line_thickness)))
        n += 1
    return out, n


# ---------------------------------------------------------------------------
# Strategy g: edge-aware morphological reconstruction
# ---------------------------------------------------------------------------
def reconstruct_plant_by_marker(binary_u8: np.ndarray, line_mask: np.ndarray) -> np.ndarray:
    """
    'Marker' = binary minus a *fattened* line mask (sure-plant pixels, away from lines).
    Then reconstruct under binary (geodesic dilation until idempotent).
    This recovers thick plant regions that connect under the line mask, while
    discarding free-standing line debris.
    """
    fattened = cv2.dilate(line_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), 2)
    marker = cv2.bitwise_and(binary_u8, cv2.bitwise_not(fattened))
    # Geodesic reconstruction by iterative dilation, masked by `binary_u8`
    prev = np.zeros_like(marker)
    cur = marker.copy()
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(200):
        d = cv2.dilate(cur, k)
        d = cv2.bitwise_and(d, binary_u8)
        if np.array_equal(d, prev):
            break
        prev = cur
        cur = d
    return cur


# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Loading {IMG_PATH}")
    gray16 = load_grayscale_working(IMG_PATH, uint8=False)
    print(f"  dtype={gray16.dtype} shape={gray16.shape} "
          f"min={int(gray16.min())} max={int(gray16.max())}")

    g01 = auto_stretch_u8(gray16)
    g_u8 = (g01 * 255).astype(np.uint8)
    _save("00_gray_stretched.png", g_u8)

    edges, mask_edge, refined = get_refined_mask(gray16)
    _save("01_edge_mask.png", mask_edge)
    _save("02_refined_mask.png", refined)
    _save_pair("02b_refined_overlay.png", g01, refined)

    print("\n[Existing strategies]")
    _, _, morph_lines, morph_clean = line_artifact_masks_and_cleaned(
        refined, horiz_width=50, vert_height=50
    )
    _save("10_morph_lines.png", morph_lines)
    _save("11_morph_clean.png", morph_clean)
    _save_pair("11b_morph_clean_overlay.png", g01, morph_clean)

    g8 = gray_u8_for_canny(gray16)
    _canny, hough_lm, n_seg = hough_prob_line_mask_from_gray(
        g8,
        canny1=50,
        canny2=150,
        hough_thresh=40,
        min_line_len=60,
        max_gap=15,
        line_thickness=3,
        dilate_iter=1,
    )
    hough_clean = cv2.bitwise_and(refined, cv2.bitwise_not(hough_lm))
    _save("20_hough_lines.png", hough_lm)
    _save("21_hough_clean.png", hough_clean)
    _save_pair("21b_hough_clean_overlay.png", g01, hough_clean)
    print(f"  Hough segments: {n_seg}")

    merge_clean = cv2.bitwise_or(morph_clean, hough_clean)
    merge_lines = cv2.bitwise_and(morph_lines, hough_lm)
    _save("30_merge_clean.png", merge_clean)
    _save_pair("30b_merge_overlay.png", g01, merge_clean)
    _save("31_merge_lines_intersection.png", merge_lines)

    print("\n[Strategy a: directional opening on refined mask, theta sweep]")
    angles = list(range(0, 180, 10))
    dirL_lines, _ = directional_open_union(refined, length=60, angles_deg=angles, thickness=1)
    dirL_clean = cv2.bitwise_and(refined, cv2.bitwise_not(dirL_lines))
    _save("40_dirL_lines.png", dirL_lines)
    _save("41_dirL_clean.png", dirL_clean)
    _save_pair("41b_dirL_clean_overlay.png", g01, dirL_clean)

    print("\n[Strategy c: skeleton straightness filter]")
    skel_lines, n_kept = straight_skeleton_lines(
        refined, min_branch_len=60, straightness_min=0.92, dilate=2
    )
    skel_clean = cv2.bitwise_and(refined, cv2.bitwise_not(skel_lines))
    _save("50_skel_lines.png", skel_lines)
    _save("51_skel_clean.png", skel_clean)
    _save_pair("51b_skel_clean_overlay.png", g01, skel_clean)
    print(f"  straight branches kept: {n_kept}")

    print("\n[Strategy d: linear-CC eccentricity filter]")
    cc_lines, n_lin = linear_cc_filter(
        refined, min_len=60, max_thickness=6, min_aspect=6.0
    )
    cc_clean = cv2.bitwise_and(refined, cv2.bitwise_not(cc_lines))
    _save("60_cc_lines.png", cc_lines)
    _save("61_cc_clean.png", cc_clean)
    _save_pair("61b_cc_clean_overlay.png", g01, cc_clean)
    print(f"  line-like CCs: {n_lin}")

    print("\n[Strategy e: LSD/FLD line detector on grayscale]")
    lsd_lm, n_lsd = lsd_line_mask(g_u8, line_thickness=3, min_length=60)
    lsd_clean = cv2.bitwise_and(refined, cv2.bitwise_not(lsd_lm))
    _save("70_lsd_lines.png", lsd_lm)
    _save("71_lsd_clean.png", lsd_clean)
    _save_pair("71b_lsd_clean_overlay.png", g01, lsd_clean)
    print(f"  LSD segments: {n_lsd}")

    print("\n[Strategy g: morphological reconstruction with line marker]")
    # Combine line evidence: directional sweep | linear-CC | Hough
    line_evidence = cv2.bitwise_or(cv2.bitwise_or(dirL_lines, cc_lines), hough_lm)
    recon = reconstruct_plant_by_marker(refined, line_evidence)
    _save("80_line_evidence_union.png", line_evidence)
    _save("81_recon_clean.png", recon)
    _save_pair("81b_recon_overlay.png", g01, recon)

    print("\n[Strategy h: vote — drop only if >=2 detectors agree]")
    votes = (
        (morph_lines > 0).astype(np.uint8)
        + (hough_lm > 0).astype(np.uint8)
        + (dirL_lines > 0).astype(np.uint8)
        + (cc_lines > 0).astype(np.uint8)
        + (skel_lines > 0).astype(np.uint8)
        + (lsd_lm > 0).astype(np.uint8)
    )
    vote2 = ((votes >= 2).astype(np.uint8)) * 255
    vote3 = ((votes >= 3).astype(np.uint8)) * 255
    vote_clean2 = cv2.bitwise_and(refined, cv2.bitwise_not(vote2))
    vote_clean3 = cv2.bitwise_and(refined, cv2.bitwise_not(vote3))
    _save("90_vote2_lines.png", vote2)
    _save("91_vote2_clean.png", vote_clean2)
    _save_pair("91b_vote2_overlay.png", g01, vote_clean2)
    _save("92_vote3_lines.png", vote3)
    _save("93_vote3_clean.png", vote_clean3)
    _save_pair("93b_vote3_overlay.png", g01, vote_clean3)

    # Combine vote2 with reconstruction (keeps thick plant via reconstruction even when
    # voted-as-line touches it, but drops free-standing voted lines)
    vote2_then_recon = reconstruct_plant_by_marker(refined, vote2)
    _save("94_vote2_then_recon.png", vote2_then_recon)
    _save_pair("94b_vote2_then_recon_overlay.png", g01, vote2_then_recon)

    print("\n[Collage]")
    cells = [
        ("orig (auto)", g01),
        ("refined", refined / 255.0),
        ("morph", morph_clean / 255.0),
        ("hough", hough_clean / 255.0),
        ("merge OR", merge_clean / 255.0),
        ("dir-sweep", dirL_clean / 255.0),
        ("skel-straight", skel_clean / 255.0),
        ("linear-CC", cc_clean / 255.0),
        ("LSD/FLD", lsd_clean / 255.0),
        ("recon (evidence)", recon / 255.0),
        ("vote>=2", vote_clean2 / 255.0),
        ("vote>=2 + recon", vote2_then_recon / 255.0),
    ]
    rows, cols = 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 14))
    for ax, (title, img) in zip(axes.flat, cells):
        ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    for ax in list(axes.flat)[len(cells):]:
        ax.axis("off")
    plt.suptitle(IMG_PATH.name, fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT / "99_collage.png", dpi=130)
    plt.close(fig)
    print(f"  saved {OUT / '99_collage.png'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
