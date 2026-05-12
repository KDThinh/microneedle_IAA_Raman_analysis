"""
Sobel "Find Edges" (ImageJ / FIJI–style) on grayscale images.

ImageJ's *Process → Find Edges* uses a 3×3 Sobel operator; the result is the
gradient magnitude. This script computes |∇I| with ``cv2.Sobel`` + ``cv2.magnitude``,
optionally after Gaussian blur.

**Bit depth (default 16-bit):** inputs are kept / promoted to ``uint16`` for
processing and saved edge images (batch: ``*_edges.tif``). Pass ``--uint8`` for
the previous 8-bit normalize-to-255 path (batch: ``*_edges.png``).

- **Single frame** (`--image`): matplotlib preview **2×3**: original, edge binary, post-blur
  binary; refined AND; **line-artifact mask**; **refined minus that mask**. Use
  ``--line-artifact`` to pick **one** method per run (``morph`` = H/V opening, ``hough`` =
  Probabilistic Hough on ``Canny(gray)``, ``none`` = skip) so morph and Hough are not
  both computed (faster). Auto contrast on the original only (unless ``--preview-linear``).
- **Threshold** (on Sobel magnitude): ``triangle`` (default), ``otsu``, or ``fixed``
  (``--fixed-thresh`` in 0–255 for ``--uint8`` edges, else 0–65535).
- **Mask Gaussian blur** (``--mask-blur``, default **40**): **sigma in pixels**, in the
  same sense as ImageJ / Fiji *Process → Filters → Gaussian Blur* (not a small OpenCV
  kernel with auto sigma, which would be much weaker). Implemented as
  ``cv2.GaussianBlur(..., ksize=(0,0), sigmaX=sigma)`` so the kernel size follows sigma.
  Use **0** to disable. Then **post-threshold** (``--post-mask-thresh``, default **otsu**).
- **Post-blur binary** (``*_mask_final.png``): by default only the **largest**
  8-connected white region is kept (drop smaller slobs). Use
  ``--mask-keep-all-components`` to retain every foreground blob.
- **Batch** (`--frames-dir` + ``--edges-out-dir``): edge images; optional
  ``--mask-out-dir`` writes ``*_mask.png``, optional smeared/final when ``--mask-blur`` > 0,
  ``*_mask_refined.png``. With ``--line-artifact morph`` and H/V sizes > 0:
  ``*_mask_refined_lines.png`` / ``*_mask_refined_nolines.png``. With ``--line-artifact hough``:
  ``*_mask_refined_hough_lines.png`` / ``*_mask_refined_hough_nolines.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_grayscale_working(image_path: str | Path, *, uint8: bool) -> np.ndarray:
    """
    Load single-channel image. If ``uint8`` is False (default path), use
    ``uint16`` without squashing to 8 bits (8-bit files are scaled to full 16-bit range).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    if uint8:
        return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if img.dtype == np.uint16:
        return np.ascontiguousarray(img)
    if img.dtype == np.uint8:
        return (img.astype(np.uint32) * 257).astype(np.uint16)
    if img.dtype in (np.float32, np.float64):
        g = np.nan_to_num(img.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return cv2.normalize(g, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)

    g = img.astype(np.float64)
    return cv2.normalize(g, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)


def odd_kernel(k: int, minimum: int = 0) -> int:
    if k <= 0:
        return 0
    k = max(minimum if minimum >= 3 else 3, int(k))
    return k if k % 2 == 1 else k + 1


def sobel_find_edges(
    gray: np.ndarray,
    blur_ksize: int = 0,
    sobel_ksize: int = 3,
    *,
    uint8: bool,
) -> np.ndarray:
    """
    Gradient magnitude ~ ImageJ Find Edges (Sobel).

    ``gray`` must be ``uint8`` or ``uint16``. Output matches ``uint8`` flag:
    ``uint8`` or ``uint16`` (min–max of magnitude in that range).
    """
    sk = int(sobel_ksize)
    if sk not in (1, 3, 5, 7):
        raise ValueError(f"sobel_ksize must be 1, 3, 5, or 7; got {sobel_ksize}")
    if gray.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"Expected uint8 or uint16 gray image; got {gray.dtype}")

    work = gray
    bk = odd_kernel(blur_ksize, minimum=3)
    if blur_ksize > 0 and bk >= 3:
        work = cv2.GaussianBlur(gray, (bk, bk), 0)

    gx = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=sk)
    gy = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=sk)
    mag = cv2.magnitude(gx, gy)
    if uint8:
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.normalize(mag, None, 0, 65535, cv2.NORM_MINMAX).astype(np.uint16)


def list_frames(directory: Path, pattern: str) -> list[Path]:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {directory}")
    return paths


def auto_contrast_limits(
    img: np.ndarray,
    saturated_fraction: float = 0.0035,
) -> tuple[float, float]:
    """
    Display min/max in the spirit of ImageJ *Auto* / *Enhance Contrast* (default
    ``saturated_fraction=0.0035`` → ~0.35% of pixels clipped at each tail).

    For ``uint8`` / ``uint16``, uses the full-image histogram (fast, deterministic).
    """
    sat = float(np.clip(saturated_fraction, 0.0, 0.49))
    if img.size == 0:
        return 0.0, 1.0
    n = float(img.size)
    lo_n = sat * n
    hi_n = (1.0 - sat) * n

    if img.dtype == np.uint8:
        hist = np.bincount(img.ravel(), minlength=256).astype(np.int64)
    elif img.dtype == np.uint16:
        hist = np.bincount(img.ravel(), minlength=65536).astype(np.int64)
    else:
        flat = img.astype(np.float64, copy=False).ravel()
        vmin = float(np.percentile(flat, 100.0 * sat))
        vmax = float(np.percentile(flat, 100.0 * (1.0 - sat)))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax

    c = np.cumsum(hist)
    vmin = int(np.searchsorted(c, lo_n, side="right"))
    vmax = int(np.searchsorted(c, hi_n, side="right"))
    if vmax <= vmin:
        vmax = min(vmin + 1, hist.size - 1)
    return float(vmin), float(vmax)


def stretch_to_01(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Map ``[vmin, vmax]`` → ``[0, 1]`` for display (values outside range clip)."""
    if vmax <= vmin:
        vmax = vmin + 1.0
    out = (img.astype(np.float64) - vmin) / (vmax - vmin)
    return np.clip(out, 0.0, 1.0)


def preview_gray_edges_01(
    gray: np.ndarray,
    edges: np.ndarray,
    *,
    saturated_fraction: float,
    linear: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(gray_01, edges_01)`` for matplotlib (per-channel contrast)."""
    if linear:
        maxv = 255.0 if gray.dtype == np.uint8 else 65535.0
        g = np.clip(gray.astype(np.float64) / maxv, 0.0, 1.0)
        e = np.clip(edges.astype(np.float64) / maxv, 0.0, 1.0)
        return g, e
    lo_g, hi_g = auto_contrast_limits(gray, saturated_fraction)
    lo_e, hi_e = auto_contrast_limits(edges, saturated_fraction)
    return stretch_to_01(gray, lo_g, hi_g), stretch_to_01(edges, lo_e, hi_e)


def otsu_threshold_histogram(hist: np.ndarray) -> int:
    """Otsu optimal threshold from 1D histogram (non-negative counts)."""
    h = hist.astype(np.float64)
    total = h.sum()
    if total <= 0:
        return 0
    p = h / total
    n = len(h)
    idx = np.arange(n, dtype=np.float64)
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * idx)
    mu_t = mu[-1]
    valid = (w0 > 1e-12) & (w1 > 1e-12)
    num = (mu_t * w0 - mu) ** 2
    den = w0 * w1
    sigma_b2 = np.zeros_like(num)
    sigma_b2[valid] = num[valid] / den[valid]
    return int(np.argmax(sigma_b2))


def otsu_threshold_from_edges(edges: np.ndarray) -> int:
    if edges.dtype == np.uint8:
        hist = np.bincount(edges.ravel(), minlength=256)
    elif edges.dtype == np.uint16:
        hist = np.bincount(edges.ravel(), minlength=65536)
    else:
        raise TypeError(edges.dtype)
    return otsu_threshold_histogram(hist)


ThreshMode = Literal["triangle", "otsu", "fixed"]
LineArtifactMode = Literal["none", "morph", "hough"]


def threshold_binary_mask(
    edges: np.ndarray,
    mode: ThreshMode,
    fixed_thresh: int | None,
) -> tuple[np.ndarray, float]:
    """
    Binary mask (``uint8`` 0 / 255) from edge magnitude.

    - ``fixed``: threshold raw ``edges``; ``fixed_thresh`` in 0–255 (uint8 edges) or
      0–65535 (uint16 edges). Default used by caller if ``None``.
    - ``otsu``: Otsu on the full histogram of ``edges`` (native bit depth).
    - ``triangle``: OpenCV triangle on ``edges`` if ``uint8``; if ``uint16``, triangle
      is run on a min–max linear map to 8-bit (OpenCV requires 8-bit), and that 8-bit
      mask is returned.
    """
    if edges.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"Expected uint8 or uint16 edges; got {edges.dtype}")

    if mode == "fixed":
        if edges.dtype == np.uint8:
            t = int(fixed_thresh) if fixed_thresh is not None else 40
            t = int(np.clip(t, 0, 255))
        else:
            t = int(fixed_thresh) if fixed_thresh is not None else 10000
            t = int(np.clip(t, 0, 65535))
        binary = np.where(edges > t, 255, 0).astype(np.uint8)
        return binary, float(t)

    if mode == "otsu":
        if edges.dtype == np.uint8:
            ret, bin8 = cv2.threshold(
                edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return bin8.astype(np.uint8), float(ret)
        t = otsu_threshold_from_edges(edges)
        binary = np.where(edges > t, 255, 0).astype(np.uint8)
        return binary, float(t)

    # triangle
    if edges.dtype == np.uint8:
        ret, bin8 = cv2.threshold(
            edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
        )
        return bin8.astype(np.uint8), float(ret)

    emin = int(edges.min())
    emax = int(edges.max())
    if emax <= emin:
        return np.zeros(edges.shape, dtype=np.uint8), float(emin)
    work8 = np.clip(
        (edges.astype(np.float32) - emin) / (emax - emin) * 255.0,
        0.0,
        255.0,
    ).astype(np.uint8)
    ret, bin8 = cv2.threshold(
        work8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )
    return bin8.astype(np.uint8), float(ret)


def gaussian_blur_binary_mask(mask: np.ndarray, sigma: float) -> tuple[np.ndarray, float]:
    """
    Gaussian blur on a 0/255 ``uint8`` mask (soft ROI), **sigma in pixels**.

    Matches the usual ImageJ / Fiji *Gaussian Blur* **sigma** control (not a raw
    ``ksize`` with ``sigma=0`` in OpenCV, which auto-picks a small sigma ~6 for k≈41).

    Uses ``cv2.GaussianBlur(mask, (0, 0), sigma)`` so OpenCV chooses a large enough odd
    kernel from ``sigma``. Border handling is default OpenCV ``BORDER_DEFAULT`` (ImageJ
    may differ slightly at edges).

    Returns ``(blurred_uint8, sigma_applied)``. ``sigma <= 0`` disables blur (copy, 0.0).
    """
    if mask.dtype != np.uint8:
        raise TypeError(f"mask must be uint8; got {mask.dtype}")
    if sigma <= 0:
        return mask.copy(), 0.0
    s = float(sigma)
    blurred = cv2.GaussianBlur(mask, (0, 0), s)
    return blurred, s


def threshold_soft_uint8(
    soft: np.ndarray,
    mode: ThreshMode,
    fixed_thresh: int | None,
) -> tuple[np.ndarray, float]:
    """
    Binarize a soft ``uint8`` image (e.g. Gaussian-blurred mask) with Otsu, triangle,
    or fixed threshold (OpenCV auto methods require 8-bit).
    """
    if soft.dtype != np.uint8:
        raise TypeError(f"soft image must be uint8; got {soft.dtype}")
    if mode == "fixed":
        t = int(fixed_thresh) if fixed_thresh is not None else 127
        t = int(np.clip(t, 0, 255))
        bin8 = np.where(soft > t, 255, 0).astype(np.uint8)
        return bin8, float(t)
    if mode == "otsu":
        ret, bin8 = cv2.threshold(
            soft, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return bin8.astype(np.uint8), float(ret)
    ret, bin8 = cv2.threshold(
        soft, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE
    )
    return bin8.astype(np.uint8), float(ret)


def largest_connected_component_mask(mask_u8: np.ndarray) -> np.ndarray:
    """
    Keep only the largest 8-connected foreground region (pixels > 127); others set to 0.

    Returns ``uint8`` 0/255. Empty foreground yields an all-zero mask.
    """
    if mask_u8.dtype != np.uint8:
        raise TypeError(f"mask must be uint8; got {mask_u8.dtype}")
    fg = (mask_u8 > 127).astype(np.uint8)
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask_u8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(1 + int(np.argmax(areas)))
    out = np.zeros_like(mask_u8)
    out[labels == largest] = 255
    return out


def refined_roi_and(mask_edges: np.ndarray, mask_post_blur: np.ndarray) -> np.ndarray:
    """Bitwise AND of two ``uint8`` 0/255 binary masks (plant contour refinement)."""
    if mask_edges.shape != mask_post_blur.shape:
        raise ValueError("mask shapes must match for AND")
    if mask_edges.dtype != np.uint8 or mask_post_blur.dtype != np.uint8:
        raise TypeError("masks must be uint8")
    return cv2.bitwise_and(mask_edges, mask_post_blur)


def line_artifact_masks_and_cleaned(
    binary_u8: np.ndarray,
    horiz_width: int,
    vert_height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Isolate long horizontal / vertical bar-like foreground via morphological **opening**
    (ImageJ-style line isolation), then subtract their **union** from ``binary_u8``.

    Returns ``(lines_h, lines_v, lines_union, cleaned)`` as ``uint8`` 0/255.
    If ``horiz_width`` or ``vert_height`` is <= 0, that opening is skipped (zeros).
    """
    if binary_u8.dtype != np.uint8:
        raise TypeError(f"binary mask must be uint8; got {binary_u8.dtype}")
    h = int(horiz_width)
    v = int(vert_height)
    lines_h = np.zeros_like(binary_u8)
    lines_v = np.zeros_like(binary_u8)
    if h > 0:
        kw = max(1, h)
        kh = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, 1))
        lines_h = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, kh)
    if v > 0:
        khv = max(1, v)
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (1, khv))
        lines_v = cv2.morphologyEx(binary_u8, cv2.MORPH_OPEN, kv)
    lines_union = cv2.bitwise_or(lines_h, lines_v)
    cleaned = cv2.bitwise_and(binary_u8, cv2.bitwise_not(lines_union))
    return lines_h, lines_v, lines_union, cleaned


def gray_u8_for_canny(gray: np.ndarray) -> np.ndarray:
    """``uint8`` grayscale for ``cv2.Canny`` / Hough (normalize 16-bit if needed)."""
    if gray.dtype == np.uint8:
        return gray
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def hough_prob_line_mask_from_gray(
    gray8: np.ndarray,
    *,
    canny1: int,
    canny2: int,
    hough_thresh: int,
    min_line_len: int,
    max_gap: int,
    line_thickness: int,
    dilate_iter: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    ``Canny(gray8)`` then ``cv2.HoughLinesP``; draw detected segments on ``line_mask``.

    Returns ``(canny_edges_u8, line_mask_u8, n_segments)``. Subtract ``line_mask`` from a
    binary ROI to remove straight-ish features (also picks up some non-axis-aligned lines
    vs. morphological H/V openings).
    """
    c1 = int(np.clip(canny1, 1, 255))
    c2 = int(np.clip(canny2, 1, 255))
    if c2 <= c1:
        c2 = min(255, c1 + 1)
    canny = cv2.Canny(gray8, c1, c2)
    h, w = gray8.shape[:2]
    line_mask = np.zeros((h, w), dtype=np.uint8)
    ht = max(1, int(hough_thresh))
    mll = max(1, int(min_line_len))
    mg = max(0, int(max_gap))
    lines = cv2.HoughLinesP(
        canny,
        rho=1,
        theta=np.pi / 180.0,
        threshold=ht,
        minLineLength=mll,
        maxLineGap=mg,
    )
    n_seg = 0
    if lines is not None:
        n_seg = int(len(lines))
        th = max(1, int(line_thickness))
        for ln in lines:
            x1, y1, x2, y2 = (int(ln[0][0]), int(ln[0][1]), int(ln[0][2]), int(ln[0][3]))
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, th)
    di = int(np.clip(dilate_iter, 0, 20))
    if di > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        line_mask = cv2.dilate(line_mask, dk, iterations=di)
    return canny, line_mask, n_seg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sobel Find Edges (ImageJ-like): 16-bit by default; optional 8-bit."
    )
    p.add_argument("--image", type=Path, default=None, help="Single image path.")
    p.add_argument("--frames-dir", type=Path, default=None, help="Directory for batch mode.")
    p.add_argument("--bg-glob", default="*.tif", help="Glob under --frames-dir (batch).")
    p.add_argument(
        "--edges-out-dir",
        type=Path,
        default=None,
        help="Batch: write one edge image per input (mirrors subfolders).",
    )
    p.add_argument(
        "--save-edges",
        type=Path,
        default=None,
        help="Single-frame: save edge image to this path (extension/bit depth up to you).",
    )
    p.add_argument(
        "--uint8",
        action="store_true",
        help="Use 8-bit pipeline (normalize inputs and magnitude to 0–255). Default is 16-bit.",
    )
    p.add_argument(
        "--blur",
        type=int,
        default=0,
        help="Gaussian blur kernel size before Sobel (0 = skip). Use odd size, e.g. 3, 5.",
    )
    p.add_argument(
        "--sobel-ksize",
        type=int,
        default=3,
        choices=(1, 3, 5, 7),
        help="OpenCV Sobel aperture size (ImageJ ≈ 3).",
    )
    p.add_argument(
        "--thresh",
        choices=("triangle", "otsu", "fixed"),
        default="triangle",
        help="Threshold edge magnitude to a binary mask (default: triangle).",
    )
    p.add_argument(
        "--fixed-thresh",
        type=int,
        default=None,
        help="For --thresh fixed: intensity in 0–255 (--uint8) or 0–65535 (uint16 edges). "
        "Defaults: 40 (uint8) / 10000 (uint16) if omitted.",
    )
    p.add_argument(
        "--mask-out-dir",
        type=Path,
        default=None,
        help="Batch: *_mask.png, *_mask_refined.png, optional smeared/final; line outputs "
        "depend on --line-artifact (morph vs hough); mirrors subfolders.",
    )
    p.add_argument(
        "--save-mask",
        type=Path,
        default=None,
        help="Single-frame: save binary mask (PNG) to this path.",
    )
    p.add_argument(
        "--mask-blur",
        type=float,
        default=40.0,
        metavar="SIGMA",
        help="Gaussian **sigma** (pixels) on the thresholded binary mask, ImageJ-style "
        "(Process → Filters → Gaussian Blur). OpenCV uses ksize (0,0) derived from sigma. "
        "0 disables blur.",
    )
    p.add_argument(
        "--save-smeared-mask",
        type=Path,
        default=None,
        help="Single-frame: save blurred / smeared mask (PNG, uint8 0–255) to this path.",
    )
    p.add_argument(
        "--post-mask-thresh",
        choices=("triangle", "otsu", "fixed"),
        default="otsu",
        help="Second threshold on the Gaussian-blurred mask (uint8 0–255). Default: otsu.",
    )
    p.add_argument(
        "--post-mask-fixed-thresh",
        type=int,
        default=None,
        help="For --post-mask-thresh fixed: level 0–255 on blurred mask. Default 127 if omitted.",
    )
    p.add_argument(
        "--save-final-mask",
        type=Path,
        default=None,
        help="Single-frame: save binary mask after blur + post-threshold (PNG).",
    )
    p.add_argument(
        "--save-refined-mask",
        type=Path,
        default=None,
        help="Single-frame: save refined ROI = edge binary AND post-blur binary (PNG).",
    )
    p.add_argument(
        "--line-remove-horiz",
        type=int,
        default=50,
        metavar="W",
        help="For --line-artifact morph: horizontal opening W×1 on refined mask; 0 disables.",
    )
    p.add_argument(
        "--line-remove-vert",
        type=int,
        default=50,
        metavar="H",
        help="For --line-artifact morph: vertical opening 1×H on refined mask; 0 disables.",
    )
    p.add_argument(
        "--save-refined-nolines-mask",
        type=Path,
        default=None,
        help="Single-frame: save refined mask after line-artifact removal (morph or Hough; PNG).",
    )
    p.add_argument(
        "--line-artifact",
        choices=("none", "morph", "hough"),
        default="morph",
        help="Line cleanup on refined mask: morph H/V opening, Hough on Canny(gray), or none. "
        "Only one runs per invocation (not both). Default: morph.",
    )
    p.add_argument(
        "--hough-lines",
        action="store_true",
        help="Deprecated: same as --line-artifact hough (overrides --line-artifact if set).",
    )
    p.add_argument("--hough-canny1", type=int, default=50, help="Canny low threshold (Hough input).")
    p.add_argument("--hough-canny2", type=int, default=150, help="Canny high threshold (Hough input).")
    p.add_argument(
        "--hough-threshold",
        type=int,
        default=40,
        help="HoughLinesP accumulator threshold (higher = fewer lines).",
    )
    p.add_argument(
        "--hough-min-line-length",
        type=int,
        default=60,
        help="HoughLinesP min segment length (pixels).",
    )
    p.add_argument(
        "--hough-max-line-gap",
        type=int,
        default=15,
        help="HoughLinesP max gap between collinear points.",
    )
    p.add_argument(
        "--hough-line-thickness",
        type=int,
        default=3,
        help="Brush thickness when rasterizing Hough segments onto subtract mask.",
    )
    p.add_argument(
        "--hough-mask-dilate",
        type=int,
        default=1,
        help="3x3 elliptical dilations on the Hough line mask before subtract (0 = none).",
    )
    p.add_argument(
        "--mask-keep-all-components",
        action="store_true",
        help="Keep all blobs in the post-blur binary mask. Default: keep only the largest "
        "8-connected component (fencing ROI around the plant).",
    )
    p.add_argument(
        "--preview-saturated",
        type=float,
        default=0.0035,
        metavar="FRAC",
        help="Preview auto contrast: fraction of pixels saturated at each histogram tail "
        "(ImageJ default ~0.0035). Ignored with --preview-linear.",
    )
    p.add_argument(
        "--preview-linear",
        action="store_true",
        help="Preview: scale original gray linearly over full dtype range (no auto contrast).",
    )
    p.add_argument("--no-show", action="store_true", help="Single-frame: skip matplotlib window.")
    return p.parse_args()


def effective_line_artifact(ns: argparse.Namespace) -> LineArtifactMode:
    """``--hough-lines`` (deprecated) forces ``hough`` over ``--line-artifact``."""
    if bool(getattr(ns, "hough_lines", False)):
        return "hough"
    la = ns.line_artifact
    if la not in ("none", "morph", "hough"):
        raise ValueError(la)
    return la  # type: ignore[return-value]


def _batch_edges_suffix(uint8: bool) -> str:
    return "_edges.png" if uint8 else "_edges.tif"


def _fixed_thresh_resolved(
    thresh: ThreshMode, fixed_thresh: int | None, uint8: bool
) -> int | None:
    if thresh != "fixed":
        return None
    if fixed_thresh is not None:
        return fixed_thresh
    return 40 if uint8 else 10000


def _post_mask_fixed_resolved(
    post_mode: ThreshMode, fixed_thresh: int | None
) -> int | None:
    if post_mode != "fixed":
        return None
    if fixed_thresh is not None:
        return fixed_thresh
    return 127


def run_batch(
    frames_dir: Path,
    bg_glob: str,
    edges_out_dir: Path,
    mask_out_dir: Path | None,
    blur: int,
    sobel_ksize: int,
    uint8: bool,
    thresh: ThreshMode,
    fixed_thresh: int | None,
    mask_blur_sigma: float,
    post_mask_thresh: ThreshMode,
    post_mask_fixed_thresh: int | None,
    mask_keep_all_components: bool,
    line_remove_horiz: int,
    line_remove_vert: int,
    line_artifact: LineArtifactMode,
    hough_canny1: int,
    hough_canny2: int,
    hough_threshold: int,
    hough_min_line_length: int,
    hough_max_line_gap: int,
    hough_line_thickness: int,
    hough_mask_dilate: int,
) -> None:
    frames_dir = frames_dir.resolve()
    edges_out_dir = edges_out_dir.resolve()
    edges_out_dir.mkdir(parents=True, exist_ok=True)
    mask_root = mask_out_dir.resolve() if mask_out_dir else None
    if mask_root:
        mask_root.mkdir(parents=True, exist_ok=True)
    paths = list_frames(frames_dir, bg_glob)
    suffix = _batch_edges_suffix(uint8)
    fixed_res = _fixed_thresh_resolved(thresh, fixed_thresh, uint8)
    post_fixed_res = _post_mask_fixed_resolved(post_mask_thresh, post_mask_fixed_thresh)
    n = 0
    n_masks = 0
    n_smeared = 0
    n_final = 0
    n_refined = 0
    n_lines = 0
    n_nolines = 0
    n_hough_lines = 0
    n_hough_nolines = 0
    for p in paths:
        gray = load_grayscale_working(p, uint8=uint8)
        edges = sobel_find_edges(gray, blur_ksize=blur, sobel_ksize=sobel_ksize, uint8=uint8)
        try:
            rel = p.relative_to(frames_dir)
        except ValueError:
            rel = Path(p.name)
        dest = edges_out_dir / rel.with_name(rel.stem + suffix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(dest), edges):
            raise OSError(f"cv2.imwrite failed: {dest}")
        n += 1
        if mask_root is not None:
            mask, _ = threshold_binary_mask(edges, thresh, fixed_res)
            mdest = mask_root / rel.with_name(rel.stem + "_mask.png")
            mdest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mdest), mask):
                raise OSError(f"cv2.imwrite failed: {mdest}")
            n_masks += 1
            smeared, sig = gaussian_blur_binary_mask(mask, mask_blur_sigma)
            if sig > 0:
                sdest = mask_root / rel.with_name(rel.stem + "_mask_smeared.png")
                sdest.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(sdest), smeared):
                    raise OSError(f"cv2.imwrite failed: {sdest}")
                n_smeared += 1
                mask_final, _ = threshold_soft_uint8(
                    smeared, post_mask_thresh, post_fixed_res
                )
                if not mask_keep_all_components:
                    mask_final = largest_connected_component_mask(mask_final)
                fdest = mask_root / rel.with_name(rel.stem + "_mask_final.png")
                fdest.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(fdest), mask_final):
                    raise OSError(f"cv2.imwrite failed: {fdest}")
                n_final += 1
            else:
                mask_final = mask.copy()
            mask_refined = refined_roi_and(mask, mask_final)
            rdest = mask_root / rel.with_name(rel.stem + "_mask_refined.png")
            rdest.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(rdest), mask_refined):
                raise OSError(f"cv2.imwrite failed: {rdest}")
            n_refined += 1
            if line_artifact == "morph" and (
                line_remove_horiz > 0 or line_remove_vert > 0
            ):
                _lh, _lv, lines_u, mask_nolines = line_artifact_masks_and_cleaned(
                    mask_refined, line_remove_horiz, line_remove_vert
                )
                ldest = mask_root / rel.with_name(rel.stem + "_mask_refined_lines.png")
                ldest.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(ldest), lines_u):
                    raise OSError(f"cv2.imwrite failed: {ldest}")
                n_lines += 1
                ndest = mask_root / rel.with_name(rel.stem + "_mask_refined_nolines.png")
                ndest.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(ndest), mask_nolines):
                    raise OSError(f"cv2.imwrite failed: {ndest}")
                n_nolines += 1
            elif line_artifact == "hough":
                g8 = gray_u8_for_canny(gray)
                _canny, h_lm, _nseg = hough_prob_line_mask_from_gray(
                    g8,
                    canny1=hough_canny1,
                    canny2=hough_canny2,
                    hough_thresh=hough_threshold,
                    min_line_len=hough_min_line_length,
                    max_gap=hough_max_line_gap,
                    line_thickness=hough_line_thickness,
                    dilate_iter=hough_mask_dilate,
                )
                mh_clean = cv2.bitwise_and(mask_refined, cv2.bitwise_not(h_lm))
                hldest = mask_root / rel.with_name(rel.stem + "_mask_refined_hough_lines.png")
                hldest.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(hldest), h_lm):
                    raise OSError(f"cv2.imwrite failed: {hldest}")
                n_hough_lines += 1
                hndest = mask_root / rel.with_name(rel.stem + "_mask_refined_hough_nolines.png")
                hndest.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(hndest), mh_clean):
                    raise OSError(f"cv2.imwrite failed: {hndest}")
                n_hough_nolines += 1
    print(f"Wrote {n} edge images ({'uint8 PNG' if uint8 else 'uint16 TIFF'}) under {edges_out_dir}")
    if mask_root:
        print(f"Wrote {n_masks} binary masks under {mask_root}")
        print(f"Wrote {n_refined} refined masks (*_mask_refined.png) under {mask_root}")
        if line_artifact == "morph" and (line_remove_horiz > 0 or line_remove_vert > 0):
            print(f"Wrote {n_lines} morph line masks (*_mask_refined_lines.png) under {mask_root}")
            print(f"Wrote {n_nolines} morph-cleaned masks (*_mask_refined_nolines.png) under {mask_root}")
        elif line_artifact == "hough":
            print(
                f"Wrote {n_hough_lines} Hough line masks (*_mask_refined_hough_lines.png) "
                f"and {n_hough_nolines} Hough-cleaned masks (*_mask_refined_hough_nolines.png) "
                f"under {mask_root}"
            )
        if mask_blur_sigma > 0:
            print(f"Wrote {n_smeared} smeared masks (*_mask_smeared.png) under {mask_root}")
            print(f"Wrote {n_final} post-threshold masks (*_mask_final.png) under {mask_root}")


def main() -> None:
    args = parse_args()
    u8 = bool(args.uint8)

    if args.frames_dir is not None:
        if args.edges_out_dir is None:
            raise SystemExit("Batch mode requires --edges-out-dir")
        tm: ThreshMode = args.thresh  # type: ignore[assignment]
        ptm: ThreshMode = args.post_mask_thresh  # type: ignore[assignment]
        la: LineArtifactMode = effective_line_artifact(args)
        run_batch(
            args.frames_dir,
            args.bg_glob,
            args.edges_out_dir,
            args.mask_out_dir,
            blur=args.blur,
            sobel_ksize=args.sobel_ksize,
            uint8=u8,
            thresh=tm,
            fixed_thresh=args.fixed_thresh,
            mask_blur_sigma=args.mask_blur,
            post_mask_thresh=ptm,
            post_mask_fixed_thresh=args.post_mask_fixed_thresh,
            mask_keep_all_components=bool(args.mask_keep_all_components),
            line_remove_horiz=args.line_remove_horiz,
            line_remove_vert=args.line_remove_vert,
            line_artifact=la,
            hough_canny1=args.hough_canny1,
            hough_canny2=args.hough_canny2,
            hough_threshold=args.hough_threshold,
            hough_min_line_length=args.hough_min_line_length,
            hough_max_line_gap=args.hough_max_line_gap,
            hough_line_thickness=args.hough_line_thickness,
            hough_mask_dilate=args.hough_mask_dilate,
        )
        return

    if args.image is None:
        raise SystemExit("Pass --image (single-frame), or --frames-dir with --edges-out-dir (batch).")

    image = args.image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)

    tm: ThreshMode = args.thresh  # type: ignore[assignment]
    post_tm: ThreshMode = args.post_mask_thresh  # type: ignore[assignment]
    fixed_only = _fixed_thresh_resolved(tm, args.fixed_thresh, u8)
    post_fixed_only = _post_mask_fixed_resolved(post_tm, args.post_mask_fixed_thresh)

    gray = load_grayscale_working(image, uint8=u8)
    edges = sobel_find_edges(gray, blur_ksize=args.blur, sobel_ksize=args.sobel_ksize, uint8=u8)
    mask, t_used = threshold_binary_mask(edges, tm, fixed_only)
    smeared, mask_sigma = gaussian_blur_binary_mask(mask, args.mask_blur)
    if mask_sigma > 0:
        mask_final, t_post = threshold_soft_uint8(smeared, post_tm, post_fixed_only)
        if not args.mask_keep_all_components:
            mask_final = largest_connected_component_mask(mask_final)
    else:
        mask_final = mask.copy()
        t_post = None

    mask_refined = refined_roi_and(mask, mask_final)
    la: LineArtifactMode = effective_line_artifact(args)
    if bool(getattr(args, "hough_lines", False)):
        print("Note: --hough-lines is deprecated; use --line-artifact hough.")

    zline = np.zeros_like(mask_refined)
    n_hough_seg = 0
    if la == "morph":
        _lh, _lv, lines_union, mask_refined_nolines = line_artifact_masks_and_cleaned(
            mask_refined, args.line_remove_horiz, args.line_remove_vert
        )
    elif la == "hough":
        g8 = gray_u8_for_canny(gray)
        _canny, hough_line_mask, n_hough_seg = hough_prob_line_mask_from_gray(
            g8,
            canny1=args.hough_canny1,
            canny2=args.hough_canny2,
            hough_thresh=args.hough_threshold,
            min_line_len=args.hough_min_line_length,
            max_gap=args.hough_max_line_gap,
            line_thickness=args.hough_line_thickness,
            dilate_iter=args.hough_mask_dilate,
        )
        lines_union = hough_line_mask
        mask_refined_nolines = cv2.bitwise_and(
            mask_refined, cv2.bitwise_not(hough_line_mask)
        )
    else:
        lines_union = zline
        mask_refined_nolines = mask_refined.copy()

    if args.save_edges is not None:
        out = args.save_edges.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), edges):
            raise OSError(f"cv2.imwrite failed: {out}")
        print(f"Saved edges ({edges.dtype}) to {out}")

    if args.save_mask is not None:
        mout = args.save_mask.resolve()
        mout.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mout), mask):
            raise OSError(f"cv2.imwrite failed: {mout}")
        print(f"Saved binary mask to {mout}")

    if args.save_smeared_mask is not None:
        sout = args.save_smeared_mask.resolve()
        sout.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(sout), smeared):
            raise OSError(f"cv2.imwrite failed: {sout}")
        print(f"Saved smeared mask to {sout}")

    if args.save_final_mask is not None:
        fout = args.save_final_mask.resolve()
        fout.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(fout), mask_final):
            raise OSError(f"cv2.imwrite failed: {fout}")
        print(f"Saved final binary mask (after blur + post-threshold) to {fout}")

    if args.save_refined_mask is not None:
        rout = args.save_refined_mask.resolve()
        rout.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(rout), mask_refined):
            raise OSError(f"cv2.imwrite failed: {rout}")
        print(f"Saved refined ROI mask (edge AND post-blur) to {rout}")

    if args.save_refined_nolines_mask is not None:
        nlout = args.save_refined_nolines_mask.resolve()
        nlout.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(nlout), mask_refined_nolines):
            raise OSError(f"cv2.imwrite failed: {nlout}")
        print(f"Saved refined mask after line-artifact removal ({la}) to {nlout}")

    if tm == "fixed":
        print(f"Threshold (fixed): {t_used:.2f}")
    elif tm == "triangle" and edges.dtype == np.uint16:
        print(f"Threshold ({tm}, min–max 8-bit map): {t_used:.2f}")
    else:
        print(f"Threshold ({tm}): {t_used:.2f}")
    if mask_sigma > 0:
        print(f"Mask Gaussian blur: sigma={mask_sigma:g} (ImageJ-style; --mask-blur)")
        if t_post is not None:
            if post_tm == "fixed":
                print(f"Post-blur threshold (fixed): {t_post:.2f}")
            else:
                print(f"Post-blur threshold ({post_tm}): {t_post:.2f}")
    else:
        print("Post-blur threshold skipped (--mask-blur 0; final mask matches edge mask).")
    if mask_sigma > 0 and not args.mask_keep_all_components:
        print("Post-blur mask: kept largest 8-connected component only.")
    print("Refined ROI: edge binary AND post-blur binary (bitwise AND).")
    if la == "morph":
        if args.line_remove_horiz > 0 or args.line_remove_vert > 0:
            print(
                f"Line-artifact (--line-artifact morph): opening W×1={args.line_remove_horiz}, "
                f"1×H={args.line_remove_vert}; subtracted union from refined mask."
            )
        else:
            print(
                "Line-artifact morph: H/V sizes are 0 (--line-remove-horiz / --line-remove-vert); "
                "no subtraction."
            )
    elif la == "hough":
        print(
            f"Line-artifact (--line-artifact hough): Canny {args.hough_canny1}/{args.hough_canny2}, "
            f"Hough thresh={args.hough_threshold}, minLen={args.hough_min_line_length}, "
            f"maxGap={args.hough_max_line_gap} -> {n_hough_seg} segments; subtracted from refined."
        )
    else:
        print("Line-artifact removal: none (--line-artifact none).")

    if not args.no_show:
        g01, _ = preview_gray_edges_01(
            gray,
            edges,
            saturated_fraction=args.preview_saturated,
            linear=bool(args.preview_linear),
        )
        m01 = (mask.astype(np.float64) / 255.0).clip(0.0, 1.0)
        f01 = (mask_final.astype(np.float64) / 255.0).clip(0.0, 1.0)
        r01 = (mask_refined.astype(np.float64) / 255.0).clip(0.0, 1.0)
        lu01 = (lines_union.astype(np.float64) / 255.0).clip(0.0, 1.0)
        nl01 = (mask_refined_nolines.astype(np.float64) / 255.0).clip(0.0, 1.0)
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes[0, 0].imshow(g01, cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("Original")
        axes[0, 1].imshow(m01, cmap="gray", vmin=0, vmax=1)
        tlab = (
            f"T={t_used:.1f}"
            if tm != "triangle" or edges.dtype == np.uint8
            else f"T_8bit={t_used:.1f}"
        )
        axes[0, 1].set_title(f"Binary after Find Edges\n({tm}, {tlab})")
        axes[0, 2].imshow(f01, cmap="gray", vmin=0, vmax=1)
        if mask_sigma > 0 and t_post is not None:
            lf = (
                ""
                if args.mask_keep_all_components
                else "\n(largest component only)"
            )
            axes[0, 2].set_title(
                f"Binary after Gaussian blur{lf}\n"
                f"({post_tm}, sigma={mask_sigma:g}, T_post={t_post:.1f})"
            )
        else:
            axes[0, 2].set_title("Binary after blur\n(--mask-blur 0: same as center)")
        axes[1, 0].imshow(r01, cmap="gray", vmin=0, vmax=1)
        if mask_sigma > 0:
            axes[1, 0].set_title("Refined ROI\n(edge AND post-blur)")
        else:
            axes[1, 0].set_title("Refined ROI\n(AND; blur off → same as row1 col2)")
        axes[1, 1].imshow(lu01, cmap="gray", vmin=0, vmax=1)
        if la == "morph":
            if args.line_remove_horiz > 0 or args.line_remove_vert > 0:
                axes[1, 1].set_title(
                    "Morph line mask\n"
                    f"(open {args.line_remove_horiz}x1 | 1x{args.line_remove_vert})"
                )
            else:
                axes[1, 1].set_title("Morph line mask\n(sizes 0: empty)")
        elif la == "hough":
            axes[1, 1].set_title(f"Hough line mask\n({n_hough_seg} segments)")
        else:
            axes[1, 1].set_title("Line mask\n(--line-artifact none)")
        axes[1, 2].imshow(nl01, cmap="gray", vmin=0, vmax=1)
        if la == "morph":
            axes[1, 2].set_title("Refined minus morph\n(refined AND NOT morph mask)")
        elif la == "hough":
            axes[1, 2].set_title("Refined minus Hough\n(refined AND NOT Hough mask)")
        else:
            axes[1, 2].set_title("Refined (unchanged)\n(no line subtraction)")
        for ax in axes.flat:
            ax.axis("off")
        depth = "uint8" if u8 else "uint16"
        prev = "linear" if args.preview_linear else f"IJ auto ({args.preview_saturated:g} sat/side)"
        plt.suptitle(f"{image.name}  ({depth}, preview: {prev})")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
