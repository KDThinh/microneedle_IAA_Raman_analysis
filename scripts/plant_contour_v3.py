"""
Sobel "Find Edges" (ImageJ / FIJI–style) on grayscale images.

ImageJ's *Process → Find Edges* uses a 3×3 Sobel operator; the result is the
gradient magnitude. This script computes |∇I| with ``cv2.Sobel`` + ``cv2.magnitude``,
optionally after Gaussian blur.

**Bit depth (default 16-bit):** inputs are kept / promoted to ``uint16`` for
processing. Single-frame ``--save-edges`` can write the Sobel magnitude; batch mode does not
save standalone edge magnitude files (the preview figure includes the edge / mask panels).
Pass ``--uint8`` for the 8-bit normalize-to-255 path.

- **Single frame** (`--image`): matplotlib preview **3×3**: original, edge binary, post-blur
  binary; refined AND; **line-artifact mask**; **refined minus that mask**; then **growth
  post-process** (stabilized mask for metrics), **stem skeleton** on the lower bbox band
  (unless ``--growth-skip-stem``), and **axis-aligned + oriented bbox** on the preview gray.
  Use ``--line-artifact`` to pick **one** method per run (``morph`` = H/V opening, ``hough`` =
  Probabilistic Hough on ``Canny(gray)``, ``merge`` = run both and **OR** the two
  cleaned refined masks (keep plant if either method keeps it; removes only where **both**
  agree on line pixels), ``intensity`` = drop refined pixels whose original grayscale value
  is dark relative to plant-vs-device Otsu computed *inside* the refined mask (great when
  the artifact is a dark static device — e.g. microneedle rod/clamp — passing through a
  bright plant; uses ``--intensity-otsu-margin`` to relax the cut so the plant base / dark
  leaves are not eaten), ``none`` = skip). ``morph``, ``hough``, ``intensity``, and ``none``
  each use only that path (faster). Auto contrast on the original only (unless
  ``--preview-linear``).
- **Threshold** (on Sobel magnitude): ``triangle`` (default), ``otsu``, or ``fixed``
  (``--fixed-thresh`` in 0–255 for ``--uint8`` edges, else 0–65535).
- **Mask Gaussian blur** (``--mask-blur``, default **40**): **sigma in pixels**, in the
  same sense as ImageJ / Fiji *Process → Filters → Gaussian Blur* (not a small OpenCV
  kernel with auto sigma, which would be much weaker). Implemented as
  ``cv2.GaussianBlur(..., ksize=(0,0), sigmaX=sigma)`` so the kernel size follows sigma.
  Use **0** to disable. Then **post-threshold** (``--post-mask-thresh``, default **otsu**).
  With **otsu**, optional ``--post-mask-otsu-margin`` subtracts from that threshold (0–255) so
  dimmer interior regions stay above the cut when the plant would otherwise split. Batch can
  enable ``--batch-adaptive-post-otsu`` to raise that margin frame-to-frame when the largest
  connected component shrinks sharply vs. the previous frame (time-series guardrail).
- **Post-blur binary** (``*_mask_final.png``): by default only the **largest**
  8-connected white region is kept (drop smaller slobs). Use
  ``--mask-keep-all-components`` to retain every foreground blob.
- **Refined ROI** (edge AND post-blur): by default runs ImageJ-like **Remove Outliers**
  (``Process → Noise → Remove Outliers``): local **median** vs pixel, replace if deviation
  exceeds a threshold — defaults **radius 2.0**, **threshold 50**, **bright** outliers only
  (white specks). Neighbourhood uses OpenCV ``medianBlur`` with odd k ≈ ``2*ceil(radius)+1``
  (square window, close to ImageJ's circular disk). Disable with ``--refined-outlier-radius 0``.
- **Batch** (``--frames-dir`` … + ``--preview-out-dir`` + ``--mask-out-dir``): requires all three.
  Pass ``--frames-dir`` **once per root** (multiple directories); matches from ``--bg-glob`` are
  merged, **deduplicated** by resolved path, sorted globally, then ``--frames-step`` applies.
  With multiple roots, outputs are nested under ``<root_name>__<hash>/`` so identical relative paths
  from different folders do not collide.
  ``--preview-out-dir`` receives only ``*_preview.png`` (3×3 matplotlib summary per frame).
  ``--mask-out-dir`` receives only ``*_mask_refined.png`` (refined ROI after edge AND post-blur,
  including Remove Outliers when enabled). Intermediate masks (binary edge mask, smeared,
  line-artifact intermediates, etc.) are computed in memory for previews and growth metrics but
  are **not** written to disk. Optional ``--frames-step N`` (default 1) keeps every N-th file
  after sorting ``--bg-glob`` matches. ``--growth-save-stabilized`` writes
  ``*_mask_growth_stabilized.png`` under ``--preview-out-dir``. Use ``--growth-csv`` for CSV
  metrics (same batch requirements). Optional ``--batch-mask-area-csv`` logs per-frame largest-CC
  area on the post-Gaussian threshold (baseline vs after ``--batch-adaptive-post-otsu``).
- **Growth metrics** (optional): stabilize the **line-artifact-cleaned** mask (morphological
  close, optional hole-fill, optional largest foreground via **connected components** or
  ``findContours`` largest external contour filled), then record **axis-aligned bbox
  height/width**, **min-area rectangle (oriented) width/height/angle**, and a **stem-length
  proxy** = 8-connected **geodesic diameter** on a morphological skeleton restricted to the
  lower ``--growth-stem-frac`` of the bbox. Batch mode uses ``--growth-csv`` with the same
  ``--frames-dir`` / ``--preview-out-dir`` / ``--mask-out-dir`` setup; ``--growth-save-stabilized``
  in batch writes next to previews as above.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import deque
from pathlib import Path
from typing import Literal, Sequence

import cv2
import matplotlib.patches as mpatches
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


def collect_batch_frame_paths(
    frame_roots: Sequence[Path], pattern: str
) -> list[tuple[Path, Path]]:
    """
    All batch inputs under one or more roots.

    Returns ``(absolute_path, resolved_root)`` per file, sorted by path string.
    If the same file appears under two roots, only the **first** root’s entry is kept.
    """
    roots: list[Path] = []
    for d in frame_roots:
        r = Path(d).resolve()
        if not r.is_dir():
            raise NotADirectoryError(r)
        roots.append(r)
    roots = list(dict.fromkeys(roots))
    first_hit: dict[Path, Path] = {}
    for root in roots:
        for p in sorted(root.glob(pattern)):
            if not p.is_file():
                continue
            key = p.resolve()
            if key in first_hit:
                continue
            first_hit[key] = root
    if not first_hit:
        raise FileNotFoundError(
            f"No files matching {pattern!r} under {len(roots)} frame root(s): "
            + ", ".join(str(x) for x in roots)
        )
    out = [(path_abs, first_hit[path_abs]) for path_abs in first_hit]
    out.sort(key=lambda t: t[0].as_posix().lower())
    return out


def _batch_output_anchor(resolved_root: Path) -> Path:
    """
    Subfolder under batch output dirs when multiple ``--frames-dir`` roots are used,
    so two sources with the same relative path do not overwrite each other.
    """
    r = resolved_root.resolve()
    digest = hashlib.sha256(str(r).encode("utf-8")).hexdigest()[:10]
    tail = r.name or "root"
    tail = re.sub(r"[^\w.\-]", "_", tail).strip("._") or "root"
    tail = tail[:80]
    return Path(f"{tail}__{digest}")


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
LineArtifactMode = Literal["none", "morph", "hough", "merge", "intensity"]
RefinedOutlierWhich = Literal["bright", "dark", "both"]
GrowthLargestMode = Literal["components", "contours"]


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
    *,
    otsu_margin: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Binarize a soft ``uint8`` image (e.g. Gaussian-blurred mask) with Otsu, triangle,
    or fixed threshold (OpenCV auto methods require 8-bit).

    For ``mode == "otsu"``, ``otsu_margin`` is subtracted from the raw Otsu level (then
    clipped to 0–255) before binarizing: lower threshold keeps more of the blurred interior.
    """
    if soft.dtype != np.uint8:
        raise TypeError(f"soft image must be uint8; got {soft.dtype}")
    if mode == "fixed":
        t = int(fixed_thresh) if fixed_thresh is not None else 127
        t = int(np.clip(t, 0, 255))
        bin8 = np.where(soft > t, 255, 0).astype(np.uint8)
        return bin8, float(t)
    if mode == "otsu":
        ret, _bin = cv2.threshold(
            soft, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        t = int(np.clip(int(round(float(ret))) - int(otsu_margin), 0, 255))
        bin8 = np.where(soft > t, 255, 0).astype(np.uint8)
        return bin8, float(t)
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


def largest_external_contour_filled_mask(mask_u8: np.ndarray) -> np.ndarray:
    """
    Largest foreground blob by ``cv2.contourArea``, filled solid.

    Uses ``cv2.findContours(..., RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`` and
    ``cv2.drawContours(..., thickness=cv2.FILLED)``. Enclosed holes inside that outline
    become foreground (unlike keeping a single connected-component label, which preserves holes).

    Returns ``uint8`` 0/255. Empty foreground yields all-zero mask.
    """
    if mask_u8.dtype != np.uint8:
        raise TypeError(f"mask must be uint8; got {mask_u8.dtype}")
    fg255 = (mask_u8 > 127).astype(np.uint8) * 255
    contours, _h = cv2.findContours(fg255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask_u8)
    c = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask_u8)
    cv2.drawContours(out, [c], -1, 255, thickness=cv2.FILLED)
    return out


def refined_roi_and(mask_edges: np.ndarray, mask_post_blur: np.ndarray) -> np.ndarray:
    """Bitwise AND of two ``uint8`` 0/255 binary masks (plant contour refinement)."""
    if mask_edges.shape != mask_post_blur.shape:
        raise ValueError("mask shapes must match for AND")
    if mask_edges.dtype != np.uint8 or mask_post_blur.dtype != np.uint8:
        raise TypeError("masks must be uint8")
    return cv2.bitwise_and(mask_edges, mask_post_blur)


def refined_mask_remove_outliers(
    mask_u8: np.ndarray,
    *,
    radius: float,
    threshold: float,
    which: RefinedOutlierWhich = "bright",
) -> np.ndarray:
    """
    ImageJ *Process → Noise → Remove Outliers* on an 8-bit mask.

    For each pixel, compare to the **median** of a square neighborhood (OpenCV
    ``medianBlur`` with odd ``ksize`` ≈ ``2 * ceil(radius) + 1``, approximating ImageJ's
    circular radius). If the pixel deviates from that median by more than ``threshold``,
    replace it with the median (bright outliers, dark outliers, or both).

    ``radius <= 0`` skips processing (returns a copy).
    """
    if radius <= 0:
        return mask_u8.copy()
    if mask_u8.dtype != np.uint8:
        raise TypeError(f"mask must be uint8; got {mask_u8.dtype}")
    k = max(3, 2 * int(np.ceil(float(radius))) + 1)
    if k % 2 == 0:
        k += 1
    k = min(k, 255)
    med = cv2.medianBlur(mask_u8, k)
    out = mask_u8.copy()
    i16 = mask_u8.astype(np.int16)
    m16 = med.astype(np.int16)
    th = float(threshold)
    if which in ("bright", "both"):
        bright = (i16 - m16) > th
        out[bright] = med[bright]
    if which in ("dark", "both"):
        dark = (m16 - i16) > th
        out[dark] = med[dark]
    return out


def binary_fill_holes_u8(mask_u8: np.ndarray) -> np.ndarray:
    """
    Fill interior holes in a binary foreground mask (white = 255).

    Flood-fills from the image corner on a copy; holes are inverted regions not reached.
    """
    bin_in = (mask_u8 > 127).astype(np.uint8) * 255
    h, w = bin_in.shape[:2]
    flood = bin_in.copy()
    fill_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, fill_mask, (0, 0), 255)
    hole_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(bin_in, hole_inv)


def morph_close_binary_u8(
    mask_u8: np.ndarray, *, ksize: int, iterations: int
) -> np.ndarray:
    """Morphological closing (dilate then erode) with elliptical kernel."""
    if ksize <= 0 or iterations <= 0:
        return mask_u8.copy()
    k = odd_kernel(int(ksize), minimum=3)
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = (mask_u8 > 127).astype(np.uint8) * 255
    for _ in range(int(iterations)):
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, el)
    return out


def stabilize_plant_mask_for_measurement(
    nolines_u8: np.ndarray,
    *,
    close_ksize: int,
    close_iters: int,
    fill_holes: bool,
    keep_largest: bool,
    largest_mode: GrowthLargestMode = "components",
) -> np.ndarray:
    """
    Post-process the line-artifact-cleaned mask for stable bbox / stem metrics.

    Order: optional morphological close → optional hole fill → optional largest foreground.

    ``largest_mode='components'``: ``connectedComponentsWithStats`` (preserves holes).
    ``largest_mode='contours'``: largest ``RETR_EXTERNAL`` contour, filled via ``drawContours``.
    """
    m = (nolines_u8 > 127).astype(np.uint8) * 255
    if close_ksize > 0 and close_iters > 0:
        m = morph_close_binary_u8(m, ksize=close_ksize, iterations=close_iters)
    if fill_holes:
        m = binary_fill_holes_u8(m)
    if keep_largest:
        if largest_mode == "contours":
            m = largest_external_contour_filled_mask(m)
        else:
            m = largest_connected_component_mask(m)
    return m


def morphological_skeleton_u8(mask_u8: np.ndarray) -> np.ndarray:
    """Zhang-like iterative thinning (OpenCV morphology) → 1-px-wide skeleton, ``uint8`` 0/255."""
    img = (mask_u8 > 127).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break
    return skel


def _adjacency_from_skel_yx(coords: np.ndarray) -> dict[int, list[int]]:
    """8-neighbor grid graph: linear index -> neighbor indices."""
    ys = coords[:, 0].astype(np.int32)
    xs = coords[:, 1].astype(np.int32)
    pos = {(int(ys[i]), int(xs[i])): i for i in range(int(len(ys)))}
    n = int(len(ys))
    adj: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        y, x = int(ys[i]), int(xs[i])
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                j = pos.get((y + dy, x + dx))
                if j is not None:
                    adj[i].append(j)
    return adj


def _bfs_farthest(adj: dict[int, list[int]], start: int) -> tuple[int, dict[int, int]]:
    dist: dict[int, int] = {start: 0}
    q: deque[int] = deque([start])
    far = start
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
                if dist[v] > dist[far]:
                    far = v
    return far, dist


def skeleton_geodesic_diameter_px(skel_u8: np.ndarray) -> int:
    """
    Longest shortest-path length (in 8-connected pixel steps) between two skeleton pixels.

    Approximates the longest path along a tree-like skeleton (double BFS / approximate
    graph diameter). Returns ``0`` if fewer than two skeleton pixels.
    """
    coords = np.argwhere(skel_u8 > 127)
    if coords.shape[0] < 2:
        return 0
    adj = _adjacency_from_skel_yx(coords)
    start = 0
    u, _ = _bfs_farthest(adj, start)
    v, d = _bfs_farthest(adj, u)
    return int(d[v])


def growth_metrics_from_mask(
    stabilized_u8: np.ndarray,
    *,
    stem_lower_frac: float,
    skip_stem: bool,
) -> dict[str, float | int | bool]:
    """
    Measure foreground area, axis-aligned bbox, oriented bbox, and stem skeleton diameter.

    ``stem_lower_frac`` in ``(0, 1]``: keep only mask pixels in the bottom fraction of the
    **axis-aligned** bbox (``y >= y0 + (1-frac)*h``) before skeletonizing for the stem proxy.
    """
    fg = (stabilized_u8 > 127).astype(np.uint8)
    fg_px = int(fg.sum())
    fg255 = fg * 255
    if fg_px == 0:
        return {
            "fg_px": 0,
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_w": 0,
            "bbox_h": 0,
            "bbox_height_px": 0,
            "bbox_width_px": 0,
            "obb_w_px": 0.0,
            "obb_h_px": 0.0,
            "obb_angle_deg": 0.0,
            "stem_geodesic_diam_px": 0,
            "stem_skipped": bool(skip_stem),
        }

    contours, _h = cv2.findContours(fg255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "fg_px": fg_px,
            "bbox_x": 0,
            "bbox_y": 0,
            "bbox_w": 0,
            "bbox_h": 0,
            "bbox_height_px": 0,
            "bbox_width_px": 0,
            "obb_w_px": 0.0,
            "obb_h_px": 0.0,
            "obb_angle_deg": 0.0,
            "stem_geodesic_diam_px": 0,
            "stem_skipped": bool(skip_stem),
        }
    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    rect = cv2.minAreaRect(c)
    (cx, cy), (rw, rh), ang = rect
    obb_w = float(max(rw, rh))
    obb_h = float(min(rw, rh))
    stem_d = 0
    if not skip_stem and stem_lower_frac > 0:
        frac = float(np.clip(stem_lower_frac, 1e-6, 1.0))
        y_cut = int(y + (1.0 - frac) * bh)
        band = np.zeros_like(fg)
        band[y_cut : y + bh, x : x + bw] = 255
        stem_fg = cv2.bitwise_and(fg255, band)
        sk = morphological_skeleton_u8(stem_fg)
        stem_d = skeleton_geodesic_diameter_px(sk)

    return {
        "fg_px": fg_px,
        "bbox_x": int(x),
        "bbox_y": int(y),
        "bbox_w": int(bw),
        "bbox_h": int(bh),
        "bbox_height_px": int(bh),
        "bbox_width_px": int(bw),
        "obb_w_px": obb_w,
        "obb_h_px": obb_h,
        "obb_angle_deg": float(ang),
        "stem_geodesic_diam_px": int(stem_d),
        "stem_skipped": bool(skip_stem),
    }


def stem_skeleton_preview_mask(
    stabilized_u8: np.ndarray,
    *,
    stem_lower_frac: float,
    skip_stem: bool,
) -> np.ndarray:
    """
    Morphological skeleton of the lower ``stem_lower_frac`` of the axis-aligned bbox
    (same geometry as the stem-length proxy), for visualization only.
    """
    if skip_stem or stem_lower_frac <= 0:
        return np.zeros_like(stabilized_u8)
    fg255 = (stabilized_u8 > 127).astype(np.uint8) * 255
    if cv2.countNonZero(fg255) == 0:
        return np.zeros_like(stabilized_u8)
    contours, _h = cv2.findContours(fg255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(stabilized_u8)
    c = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(c)
    frac = float(np.clip(stem_lower_frac, 1e-6, 1.0))
    y_cut = int(y + (1.0 - frac) * bh)
    band = np.zeros_like(fg255)
    band[y_cut : y + bh, x : x + bw] = 255
    stem_fg = cv2.bitwise_and(fg255, band)
    return morphological_skeleton_u8(stem_fg)


def growth_csv_row(
    rel_str: str,
    stem_name: str,
    line_artifact: str,
    metrics: dict[str, float | int | bool],
) -> dict[str, str | int | float]:
    """Flatten metrics + file columns for CSV."""
    return {
        "rel_path": rel_str,
        "stem": stem_name,
        "line_artifact": line_artifact,
        "fg_px": int(metrics["fg_px"]),
        "bbox_x": int(metrics["bbox_x"]),
        "bbox_y": int(metrics["bbox_y"]),
        "bbox_w": int(metrics["bbox_w"]),
        "bbox_h": int(metrics["bbox_h"]),
        "bbox_height_px": int(metrics["bbox_height_px"]),
        "bbox_width_px": int(metrics["bbox_width_px"]),
        "obb_w_px": float(metrics["obb_w_px"]),
        "obb_h_px": float(metrics["obb_h_px"]),
        "obb_angle_deg": float(metrics["obb_angle_deg"]),
        "stem_geodesic_diam_px": int(metrics["stem_geodesic_diam_px"]),
        "stem_skipped": int(bool(metrics["stem_skipped"])),
    }


MASK_AREA_CSV_FIELDNAMES = [
    "index",
    "file_name",
    "original_binary_mask_area",
    "adaptive_binary_mask_area",
]

GROWTH_CSV_FIELDNAMES = [
    "rel_path",
    "stem",
    "line_artifact",
    "fg_px",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "bbox_height_px",
    "bbox_width_px",
    "obb_w_px",
    "obb_h_px",
    "obb_angle_deg",
    "stem_geodesic_diam_px",
    "stem_skipped",
]


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


def auto_stretched_gray_u8(
    gray: np.ndarray,
    *,
    saturated_fraction: float = 0.0035,
    linear: bool = False,
) -> np.ndarray:
    """
    ImageJ-style auto-contrast 8-bit version of ``gray`` for intensity-based decisions.

    Mirrors the preview transform: when ``linear`` is True, divide by full dtype range;
    otherwise compute a saturated min/max from the histogram and stretch to ``[0, 255]``.
    Use this so an intensity threshold is meaningful across frames with different exposure.
    """
    if linear:
        if gray.dtype == np.uint8:
            return gray.copy()
        maxv = 65535.0
        g01 = np.clip(gray.astype(np.float64) / maxv, 0.0, 1.0)
        return (g01 * 255.0).astype(np.uint8)
    lo, hi = auto_contrast_limits(gray, saturated_fraction)
    g01 = stretch_to_01(gray, lo, hi)
    return (g01 * 255.0).astype(np.uint8)


def _otsu_threshold_inside(values_u8: np.ndarray) -> int:
    """Otsu on a 1D uint8 sample (vectorized, returns int 0–255)."""
    if values_u8.size < 8:
        return 127
    hist = np.bincount(values_u8.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    idx = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu = np.cumsum(p * idx)
    mu_t = mu[-1]
    valid = (w0 > 1e-12) & (w1 > 1e-12)
    sb2 = np.zeros(256)
    sb2[valid] = (mu_t * w0[valid] - mu[valid]) ** 2 / (w0[valid] * w1[valid])
    return int(np.argmax(sb2))


def intensity_gate_inside_refined(
    gray: np.ndarray,
    refined_mask: np.ndarray,
    *,
    saturated_fraction: float = 0.0035,
    linear_preview: bool = False,
    otsu_margin: int = 25,
    fixed_thresh: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Plant-vs-device intensity gate (works well when artifacts are dark static structures
    passing through a bright plant, e.g. a microneedle rod / clamp).

    Steps:
      1. Stretch ``gray`` to 0–255 with the same auto-contrast as the preview.
      2. If ``fixed_thresh`` is given, use it; else Otsu on the gray pixels *inside*
         ``refined_mask``, then subtract ``otsu_margin`` to relax the cut (avoids eating
         the plant base and dark leaf undersides).
      3. ``gate = (gray_u8 >= T_used) * 255``; ``cleaned = refined AND gate``;
         ``dropped = refined AND NOT gate`` (the "line / artifact" pixels we remove).

    Returns ``(dropped_u8, cleaned_u8, T_used)``.
    """
    if refined_mask.dtype != np.uint8:
        raise TypeError(f"refined_mask must be uint8; got {refined_mask.dtype}")
    g_u8 = auto_stretched_gray_u8(
        gray, saturated_fraction=saturated_fraction, linear=linear_preview
    )
    inside = g_u8[refined_mask > 0]
    if fixed_thresh is not None:
        t_used = int(np.clip(int(fixed_thresh), 0, 255))
    else:
        t = _otsu_threshold_inside(inside) if inside.size else 127
        t_used = int(np.clip(t - int(otsu_margin), 0, 255))
    gate = (g_u8 >= t_used).astype(np.uint8) * 255
    cleaned = cv2.bitwise_and(refined_mask, gate)
    dropped = cv2.bitwise_and(refined_mask, cv2.bitwise_not(gate))
    return dropped, cleaned, t_used


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sobel Find Edges (ImageJ-like): 16-bit by default; optional 8-bit."
    )
    p.add_argument("--image", type=Path, default=None, help="Single image path.")
    p.add_argument(
        "--frames-dir",
        dest="frames_dirs",
        action="append",
        type=Path,
        default=None,
        metavar="DIR",
        help="Batch: input directory (repeat for multiple roots). "
        "Glob --bg-glob in each; list is merged, deduped, sorted, then --frames-step applies.",
    )
    p.add_argument("--bg-glob", default="*.tif", help="Glob under each --frames-dir (batch).")
    p.add_argument(
        "--frames-step",
        type=int,
        default=1,
        metavar="N",
        help="Batch: after sorting --bg-glob matches, process every N-th file only "
        "(1 = all; 2 = first, third, fifth, … in sorted order). Must be >= 1.",
    )
    p.add_argument(
        "--preview-out-dir",
        type=Path,
        default=None,
        help="Batch: directory for *_preview.png only (3×3 matplotlib summary per frame). "
        "Requires at least one --frames-dir and --mask-out-dir.",
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
        help="Batch: write only *_mask_refined.png per input (mirrors subfolders). "
        "Requires --preview-out-dir. Sobel / intermediate masks are not saved to disk.",
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
        "--post-mask-otsu-margin",
        type=int,
        default=0,
        metavar="M",
        help="When --post-mask-thresh otsu: subtract M from the Otsu threshold on the "
        "Gaussian-blurred mask (after cv2 Otsu, clipped 0–255). Helps when the plant interior "
        "is dim and the binary splits; try 5–25. Default 0. Ignored for triangle/fixed.",
    )
    p.add_argument(
        "--batch-adaptive-post-otsu",
        action="store_true",
        help="Batch only: if largest-CC area on the post-blur binary is below a fraction of "
        "the previous frame (see --adapt-post-otsu-trigger-ratio), increase post-Otsu margin "
        "in steps until area reaches --adapt-post-otsu-target-ratio of the previous size or "
        "margins cap out. Requires --post-mask-thresh otsu, --mask-blur > 0, and default "
        "largest-component post-blur behavior (not --mask-keep-all-components).",
    )
    p.add_argument(
        "--adapt-post-otsu-trigger-ratio",
        type=float,
        default=0.65,
        metavar="R",
        help="With --batch-adaptive-post-otsu: adapt only if largest-CC area < R × previous "
        "(e.g. 0.65 ≈ 65%%). Default 0.65.",
    )
    p.add_argument(
        "--adapt-post-otsu-target-ratio",
        type=float,
        default=0.90,
        metavar="R",
        help="With --batch-adaptive-post-otsu: stop raising margin when area ≥ R × previous. "
        "Default 0.90.",
    )
    p.add_argument(
        "--adapt-post-otsu-max-ratio",
        type=float,
        default=1.10,
        metavar="R",
        help="With --batch-adaptive-post-otsu: if area exceeds R × previous after a step, "
        "stop (avoid runaway foreground). Default 1.10.",
    )
    p.add_argument(
        "--adapt-post-otsu-margin-step",
        type=int,
        default=2,
        metavar="S",
        help="With --batch-adaptive-post-otsu: add S to margin each retry. Default 2.",
    )
    p.add_argument(
        "--adapt-post-otsu-max-extra-margin",
        type=int,
        default=60,
        metavar="M",
        help="With --batch-adaptive-post-otsu: margin will not exceed "
        "(--post-mask-otsu-margin + this value), capped at 255. Default 60.",
    )
    p.add_argument(
        "--batch-mask-area-csv",
        type=Path,
        default=None,
        help="Batch only: CSV with index, file_name (source path), "
        "original_binary_mask_area, adaptive_binary_mask_area. Areas are pixel counts of the "
        "largest 8-connected component on the **post-Gaussian** soft mask after thresholding "
        "(baseline uses --post-mask-otsu-margin only; adaptive column uses the mask after "
        "--batch-adaptive-post-otsu if that path changed the margin, else same as original). "
        "If --mask-blur 0, both columns use the edge binary mask. Appends if the file exists.",
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
        "--refined-outlier-radius",
        type=float,
        default=2.0,
        metavar="R",
        help="After refined = edge AND post-blur: ImageJ Remove Outliers (median neighborhood). "
        "OpenCV medianBlur uses odd k ≈ 2*ceil(R)+1 px (square, ~ImageJ disk). 0 disables. "
        "Default: 2.0.",
    )
    p.add_argument(
        "--refined-outlier-threshold",
        type=float,
        default=50.0,
        metavar="T",
        help="Remove Outliers: replace if |pixel - local median| > T (0–255 scale). Default: 50.",
    )
    p.add_argument(
        "--refined-outlier-which",
        choices=("bright", "dark", "both"),
        default="bright",
        help="Remove Outliers mode (ImageJ): bright = salt / white specks; dark = pepper; "
        "both. Default: bright.",
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
        help="Single-frame: save refined mask after line-artifact removal (morph, Hough, merge, or intensity; PNG).",
    )
    p.add_argument(
        "--line-artifact",
        choices=("none", "morph", "hough", "merge", "intensity"),
        default="morph",
        help="Line cleanup on refined mask: morph H/V opening, Hough on Canny(gray), merge = "
        "OR of morph-cleaned and Hough-cleaned refined (removes only where both agree on "
        "line pixels), intensity = drop refined pixels whose gray is dark relative to a "
        "plant-vs-device Otsu computed inside the refined mask (good for static dark "
        "artifacts on bright plants), or none. Default: morph.",
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
        "--intensity-otsu-margin",
        type=int,
        default=25,
        help="For --line-artifact intensity: subtract this from Otsu-inside-refined to "
        "produce the gate threshold (T_used = max(0, T_otsu - margin)). Larger = more "
        "plant kept and more device kept; smaller = stricter. Default: 25.",
    )
    p.add_argument(
        "--intensity-fixed-thresh",
        type=int,
        default=None,
        help="For --line-artifact intensity: override Otsu with a fixed 0–255 threshold on "
        "the auto-stretched gray. Pixels with gray < this are dropped from the refined mask.",
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
    p.add_argument(
        "--growth-csv",
        type=Path,
        default=None,
        help="Append per-frame growth metrics (CSV). Batch mode requires --frames-dir (one or more), "
        "--preview-out-dir, and --mask-out-dir. "
        "Columns: bbox height/width, OBB, stem geodesic diameter on stabilized mask.",
    )
    p.add_argument(
        "--growth-close-ksize",
        type=int,
        default=3,
        metavar="K",
        help="Stabilize mask: morphological close ellipse ksize (odd, ≥3). 0 skips close. Default: 3.",
    )
    p.add_argument(
        "--growth-close-iters",
        type=int,
        default=1,
        metavar="N",
        help="Stabilize mask: closing iterations. 0 skips close. Default: 1.",
    )
    p.add_argument(
        "--growth-fill-holes",
        action="store_true",
        help="Stabilize mask: fill interior holes (flood-fill from border) after closing.",
    )
    p.add_argument(
        "--growth-no-keep-largest",
        action="store_true",
        help="Stabilize mask: skip the largest-foreground step after close/fill "
        "(see --growth-largest-mode).",
    )
    p.add_argument(
        "--growth-largest-mode",
        choices=("components", "contours"),
        default="components",
        metavar="MODE",
        help="When keeping largest foreground after stabilize: components = "
        "connectedComponentsWithStats (same topology, preserves holes); contours = "
        "largest RETR_EXTERNAL contour via findContours, filled with drawContours "
        "(holes inside that outline become foreground). Ignored with "
        "--growth-no-keep-largest. Default: components.",
    )
    p.add_argument(
        "--growth-stem-frac",
        type=float,
        default=0.45,
        metavar="F",
        help="Stem proxy: use bottom F fraction of axis-aligned bbox for skeleton (0–1). "
        "Default: 0.45.",
    )
    p.add_argument(
        "--growth-skip-stem",
        action="store_true",
        help="Skip stem-length proxy (skeleton geodesic diameter).",
    )
    p.add_argument(
        "--growth-save-stabilized",
        action="store_true",
        help="Save stabilized measurement mask: batch → *_mask_growth_stabilized.png under "
        "--preview-out-dir; single-frame → <image_stem>_mask_growth_stabilized.png next to --image.",
    )
    p.add_argument(
        "--growth-print",
        action="store_true",
        help="Print one-line growth metrics to stdout (single-frame or batch).",
    )
    p.add_argument("--no-show", action="store_true", help="Single-frame: skip matplotlib window.")
    return p.parse_args()


def effective_line_artifact(ns: argparse.Namespace) -> LineArtifactMode:
    """``--hough-lines`` (deprecated) forces ``hough`` over ``--line-artifact``."""
    if bool(getattr(ns, "hough_lines", False)):
        return "hough"
    la = ns.line_artifact
    if la not in ("none", "morph", "hough", "merge", "intensity"):
        raise ValueError(la)
    return la  # type: ignore[return-value]


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


def plant_contour_preview_figure(
    *,
    title_name: str,
    gray: np.ndarray,
    edges: np.ndarray,
    mask: np.ndarray,
    mask_final: np.ndarray,
    mask_refined: np.ndarray,
    lines_union: np.ndarray,
    mask_refined_nolines: np.ndarray,
    t_used: float,
    t_post: float | None,
    mask_sigma: float,
    thresh_mode: ThreshMode,
    post_thresh_mode: ThreshMode,
    uint8_pipeline: bool,
    mask_keep_all_components: bool,
    line_artifact: LineArtifactMode,
    line_remove_horiz: int,
    line_remove_vert: int,
    n_hough_seg: int,
    int_t_used: int,
    growth_close_ksize: int,
    growth_close_iters: int,
    growth_fill_holes: bool,
    growth_keep_largest: bool,
    growth_largest_mode: GrowthLargestMode,
    growth_stem_frac: float,
    growth_skip_stem: bool,
    preview_saturated: float,
    preview_linear: bool,
) -> plt.Figure:
    """Build the 3×3 matplotlib figure used for interactive preview and batch ``*_preview.png``."""
    g01, _ = preview_gray_edges_01(
        gray,
        edges,
        saturated_fraction=preview_saturated,
        linear=preview_linear,
    )
    m01 = (mask.astype(np.float64) / 255.0).clip(0.0, 1.0)
    f01 = (mask_final.astype(np.float64) / 255.0).clip(0.0, 1.0)
    r01 = (mask_refined.astype(np.float64) / 255.0).clip(0.0, 1.0)
    lu01 = (lines_union.astype(np.float64) / 255.0).clip(0.0, 1.0)
    nl01 = (mask_refined_nolines.astype(np.float64) / 255.0).clip(0.0, 1.0)
    stab_pv = stabilize_plant_mask_for_measurement(
        mask_refined_nolines,
        close_ksize=growth_close_ksize,
        close_iters=growth_close_iters,
        fill_holes=growth_fill_holes,
        keep_largest=growth_keep_largest,
        largest_mode=growth_largest_mode,
    )
    stab01 = (stab_pv.astype(np.float64) / 255.0).clip(0.0, 1.0)
    met_pv = growth_metrics_from_mask(
        stab_pv,
        stem_lower_frac=growth_stem_frac,
        skip_stem=growth_skip_stem,
    )
    stem_sk = stem_skeleton_preview_mask(
        stab_pv,
        stem_lower_frac=growth_stem_frac,
        skip_stem=growth_skip_stem,
    )
    sk01 = (stem_sk.astype(np.float64) / 255.0).clip(0.0, 1.0)
    rch = np.clip(stab01 + 0.55 * sk01, 0.0, 1.0)
    gch = np.clip(stab01 - 0.12 * sk01, 0.0, 1.0)
    bch = np.clip(stab01 - 0.12 * sk01, 0.0, 1.0)
    sk_overlay = np.dstack([rch, gch, bch])

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    axes[0, 0].imshow(g01, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Original")
    axes[0, 1].imshow(m01, cmap="gray", vmin=0, vmax=1)
    tlab = (
        f"T={t_used:.1f}"
        if thresh_mode != "triangle" or edges.dtype == np.uint8
        else f"T_8bit={t_used:.1f}"
    )
    axes[0, 1].set_title(f"Binary after Find Edges\n({thresh_mode}, {tlab})")
    axes[0, 2].imshow(f01, cmap="gray", vmin=0, vmax=1)
    if mask_sigma > 0 and t_post is not None:
        lf = "" if mask_keep_all_components else "\n(largest component only)"
        axes[0, 2].set_title(
            f"Binary after Gaussian blur{lf}\n"
            f"({post_thresh_mode}, sigma={mask_sigma:g}, T_post={t_post:.1f})"
        )
    else:
        axes[0, 2].set_title("Binary after blur\n(--mask-blur 0: same as center)")
    axes[1, 0].imshow(r01, cmap="gray", vmin=0, vmax=1)
    if mask_sigma > 0:
        axes[1, 0].set_title("Refined ROI\n(edge AND post-blur)")
    else:
        axes[1, 0].set_title("Refined ROI\n(AND; blur off → same as row1 col2)")
    axes[1, 1].imshow(lu01, cmap="gray", vmin=0, vmax=1)
    if line_artifact == "morph":
        if line_remove_horiz > 0 or line_remove_vert > 0:
            axes[1, 1].set_title(
                "Morph line mask\n"
                f"(open {line_remove_horiz}x1 | 1x{line_remove_vert})"
            )
        else:
            axes[1, 1].set_title("Morph line mask\n(sizes 0: empty)")
    elif line_artifact == "hough":
        axes[1, 1].set_title(f"Hough line mask\n({n_hough_seg} segments)")
    elif line_artifact == "merge":
        axes[1, 1].set_title(
            "Merge: removed from refined\n(morph lines AND Hough lines)"
        )
    elif line_artifact == "intensity":
        axes[1, 1].set_title(
            f"Intensity-dropped pixels\n(gray < {int_t_used}/255 inside refined)"
        )
    else:
        axes[1, 1].set_title("Line mask\n(--line-artifact none)")
    axes[1, 2].imshow(nl01, cmap="gray", vmin=0, vmax=1)
    if line_artifact == "morph":
        axes[1, 2].set_title("Refined minus morph\n(refined AND NOT morph mask)")
    elif line_artifact == "hough":
        axes[1, 2].set_title("Refined minus Hough\n(refined AND NOT Hough mask)")
    elif line_artifact == "merge":
        axes[1, 2].set_title(
            "Merge result\n(morph-cleaned OR Hough-cleaned refined)"
        )
    elif line_artifact == "intensity":
        axes[1, 2].set_title(
            "Intensity-gated refined\n(refined AND gray >= T_used)"
        )
    else:
        axes[1, 2].set_title("Refined (unchanged)\n(no line subtraction)")

    axes[2, 0].imshow(stab01, cmap="gray", vmin=0, vmax=1)
    hole_lbl = " + fill holes" if growth_fill_holes else ""
    if not growth_keep_largest:
        lk_lbl = ""
    elif growth_largest_mode == "contours":
        lk_lbl = ", largest contour (filled)"
    else:
        lk_lbl = ", largest CC"
    axes[2, 0].set_title(
        "Stabilized (growth post-process)\n"
        f"close k={growth_close_ksize}, iters={growth_close_iters}"
        f"{hole_lbl}{lk_lbl}"
    )
    axes[2, 1].imshow(sk_overlay, vmin=0, vmax=1)
    if growth_skip_stem:
        axes[2, 1].set_title("Stem skeleton\n(--growth-skip-stem)")
    else:
        axes[2, 1].set_title(
            "Stem skeleton (red tint)\n"
            f"lower {growth_stem_frac:.0%} of axis bbox"
        )
    axes[2, 2].imshow(g01, cmap="gray", vmin=0, vmax=1)
    bx, by, bbw, bbh = (
        int(met_pv["bbox_x"]),
        int(met_pv["bbox_y"]),
        int(met_pv["bbox_w"]),
        int(met_pv["bbox_h"]),
    )
    if bbw > 0 and bbh > 0:
        axes[2, 2].add_patch(
            mpatches.Rectangle(
                (bx, by),
                bbw,
                bbh,
                linewidth=1.2,
                edgecolor="cyan",
                facecolor="none",
            )
        )
    fg255 = (stab_pv > 127).astype(np.uint8) * 255
    conts, _ = cv2.findContours(fg255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if conts:
        c0 = max(conts, key=cv2.contourArea)
        obox = cv2.boxPoints(cv2.minAreaRect(c0))
        axes[2, 2].add_patch(
            mpatches.Polygon(
                obox,
                closed=True,
                linewidth=1.0,
                edgecolor="orange",
                facecolor="none",
            )
        )
    axes[2, 2].set_title(
        "Bbox (cyan) + OBB (orange)\n"
        f"axis h={met_pv['bbox_height_px']} w={met_pv['bbox_width_px']} "
        f"stem_d={met_pv['stem_geodesic_diam_px']} fg={met_pv['fg_px']}"
    )

    for ax in axes.flat:
        ax.axis("off")
    depth = "uint8" if uint8_pipeline else "uint16"
    prev = (
        "linear"
        if preview_linear
        else f"IJ auto ({preview_saturated:g} sat/side)"
    )
    fig.suptitle(f"{title_name}  ({depth}, preview: {prev})")
    fig.tight_layout()
    return fig


def _post_blur_mask_and_largest_cc_area(
    smeared: np.ndarray,
    post_mask_thresh: ThreshMode,
    post_fixed_res: int | None,
    otsu_margin: int,
    mask_keep_all_components: bool,
) -> tuple[np.ndarray, float, int]:
    """
    Post-threshold the blurred soft mask; return ``mask_final``, ``t_post``, and the pixel
    count of the **largest** 8-connected foreground component of the raw threshold output
    (used for batch temporal area comparison).
    """
    raw, t_post = threshold_soft_uint8(
        smeared,
        post_mask_thresh,
        post_fixed_res,
        otsu_margin=otsu_margin,
    )
    lcc = largest_connected_component_mask(raw)
    largest_area = int((lcc > 127).sum())
    if mask_keep_all_components:
        mask_final = raw
    else:
        mask_final = lcc
    return mask_final, t_post, largest_area


def _adapt_post_otsu_margin_batch(
    smeared: np.ndarray,
    post_mask_thresh: ThreshMode,
    post_fixed_res: int | None,
    base_margin: int,
    mask_keep_all_components: bool,
    prev_largest_area: int | None,
    *,
    adapt_trigger_ratio: float,
    adapt_target_ratio: float,
    adapt_max_ratio: float,
    margin_step: int,
    max_extra_margin: int,
) -> tuple[np.ndarray, float, int, int, bool]:
    """
    If the current largest-CC area is far below the previous frame's, raise ``otsu_margin``
    in steps (lower soft threshold → larger foreground) until area recovers or caps hit.

    Returns ``(mask_final, t_post, largest_area, margin_used, did_adapt)``.
    """
    mf, t, a = _post_blur_mask_and_largest_cc_area(
        smeared,
        post_mask_thresh,
        post_fixed_res,
        base_margin,
        mask_keep_all_components,
    )
    margin = base_margin
    if post_mask_thresh != "otsu":
        return mf, t, a, margin, False
    if prev_largest_area is None or prev_largest_area <= 0:
        return mf, t, a, margin, False

    target_min = float(prev_largest_area) * adapt_target_ratio
    target_max = float(prev_largest_area) * adapt_max_ratio
    trigger_low = float(prev_largest_area) * adapt_trigger_ratio

    if a >= target_min:
        return mf, t, a, margin, False
    if a >= trigger_low:
        # Between trigger and target: optional future nudge; keep single-shot for now
        return mf, t, a, margin, False

    max_margin = min(255, base_margin + max_extra_margin)
    did_adapt = False
    while margin < max_margin:
        margin = min(max_margin, margin + margin_step)
        did_adapt = True
        mf, t, a = _post_blur_mask_and_largest_cc_area(
            smeared,
            post_mask_thresh,
            post_fixed_res,
            margin,
            mask_keep_all_components,
        )
        if a >= target_min:
            break
        if a > target_max:
            # Rare: huge overshoot; accept this frame as-is
            break

    return mf, t, a, margin, did_adapt


def run_batch(
    frames_dirs: Sequence[Path],
    bg_glob: str,
    frames_step: int,
    preview_out_dir: Path,
    mask_out_dir: Path,
    blur: int,
    sobel_ksize: int,
    uint8: bool,
    thresh: ThreshMode,
    fixed_thresh: int | None,
    mask_blur_sigma: float,
    post_mask_thresh: ThreshMode,
    post_mask_fixed_thresh: int | None,
    post_mask_otsu_margin: int,
    batch_adaptive_post_otsu: bool,
    adapt_post_otsu_trigger_ratio: float,
    adapt_post_otsu_target_ratio: float,
    adapt_post_otsu_max_ratio: float,
    adapt_post_otsu_margin_step: int,
    adapt_post_otsu_max_extra_margin: int,
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
    intensity_otsu_margin: int,
    intensity_fixed_thresh: int | None,
    preview_saturated: float,
    preview_linear: bool,
    refined_outlier_radius: float,
    refined_outlier_threshold: float,
    refined_outlier_which: RefinedOutlierWhich,
    growth_csv: Path | None,
    growth_close_ksize: int,
    growth_close_iters: int,
    growth_fill_holes: bool,
    growth_keep_largest: bool,
    growth_largest_mode: GrowthLargestMode,
    growth_stem_frac: float,
    growth_skip_stem: bool,
    growth_save_stabilized: bool,
    growth_print: bool,
    mask_area_csv: Path | None,
) -> None:
    if not frames_dirs:
        raise SystemExit("Batch mode requires at least one --frames-dir.")
    preview_out_dir = preview_out_dir.resolve()
    preview_out_dir.mkdir(parents=True, exist_ok=True)
    mask_root = mask_out_dir.resolve()
    mask_root.mkdir(parents=True, exist_ok=True)
    roots_uniq = list(dict.fromkeys(Path(d).resolve() for d in frames_dirs))
    n_roots = len(roots_uniq)
    multi_root = n_roots > 1
    root_anchor: dict[Path, Path] = {}
    if multi_root:
        for r in roots_uniq:
            root_anchor[r] = _batch_output_anchor(r)
    paths = collect_batch_frame_paths(frames_dirs, bg_glob)
    if n_roots > 1:
        print(
            f"Batch: {n_roots} frame root(s), {len(paths)} file(s) matched before --frames-step.",
            flush=True,
        )
    if frames_step < 1:
        raise SystemExit("Batch mode: --frames-step must be an integer >= 1")
    n_matched = len(paths)
    paths = paths[::frames_step]
    if frames_step > 1:
        print(
            f"Batch: --frames-step {frames_step} → processing {len(paths)} of {n_matched} "
            "matched frames (sorted order)."
        )
    fixed_res = _fixed_thresh_resolved(thresh, fixed_thresh, uint8)
    post_fixed_res = _post_mask_fixed_resolved(post_mask_thresh, post_mask_fixed_thresh)
    adapt_effective = (
        batch_adaptive_post_otsu
        and not mask_keep_all_components
        and post_mask_thresh == "otsu"
        and mask_blur_sigma > 0
    )
    if batch_adaptive_post_otsu:
        if post_mask_thresh != "otsu":
            raise SystemExit(
                "Batch: --batch-adaptive-post-otsu requires --post-mask-thresh otsu."
            )
        if mask_blur_sigma <= 0:
            print(
                "Batch: --batch-adaptive-post-otsu ignored (--mask-blur 0; no soft post-threshold).",
                flush=True,
            )
        if mask_keep_all_components:
            print(
                "Batch: --batch-adaptive-post-otsu ignored (--mask-keep-all-components; "
                "largest-CC area is undefined for comparison).",
                flush=True,
            )
    if (
        adapt_post_otsu_trigger_ratio <= 0
        or adapt_post_otsu_target_ratio <= 0
        or adapt_post_otsu_max_ratio <= 0
    ):
        raise SystemExit("Adapt post-Otsu ratios must be positive.")
    if adapt_post_otsu_trigger_ratio > adapt_post_otsu_target_ratio:
        raise SystemExit(
            "--adapt-post-otsu-trigger-ratio must be ≤ --adapt-post-otsu-target-ratio."
        )
    if adapt_post_otsu_target_ratio > adapt_post_otsu_max_ratio:
        raise SystemExit(
            "--adapt-post-otsu-target-ratio must be ≤ --adapt-post-otsu-max-ratio."
        )
    if adapt_post_otsu_margin_step < 1:
        raise SystemExit("--adapt-post-otsu-margin-step must be >= 1.")
    if adapt_post_otsu_max_extra_margin < 0:
        raise SystemExit("--adapt-post-otsu-max-extra-margin must be >= 0.")
    growth_f = None
    growth_w: csv.DictWriter | None = None
    if growth_csv is not None:
        gc = growth_csv.resolve()
        gc.parent.mkdir(parents=True, exist_ok=True)
        write_header = not gc.exists() or gc.stat().st_size == 0
        growth_f = gc.open("a", newline="", encoding="utf-8")
        growth_w = csv.DictWriter(growth_f, fieldnames=GROWTH_CSV_FIELDNAMES)
        if write_header:
            growth_w.writeheader()
    mask_area_f = None
    mask_area_w: csv.DictWriter | None = None
    if mask_area_csv is not None:
        mac = mask_area_csv.resolve()
        mac.parent.mkdir(parents=True, exist_ok=True)
        ma_new = not mac.exists() or mac.stat().st_size == 0
        mask_area_f = mac.open("a", newline="", encoding="utf-8")
        mask_area_w = csv.DictWriter(mask_area_f, fieldnames=MASK_AREA_CSV_FIELDNAMES)
        if ma_new:
            mask_area_w.writeheader()
    n_refined = 0
    n_previews = 0
    n_growth_stab = 0
    n_adaptive_post_otsu = 0
    n_total = len(paths)
    prev_largest_area: int | None = None
    for idx, (p, frame_root) in enumerate(paths, start=1):
        try:
            rel = p.relative_to(frame_root)
        except ValueError:
            rel = Path(p.name)
        logical_key = (frame_root / rel).as_posix()
        out_rel = rel if not multi_root else (root_anchor[frame_root] / rel)
        print(f"Batch [{idx}/{n_total}] {logical_key}", flush=True)
        gray = load_grayscale_working(p, uint8=uint8)
        edges = sobel_find_edges(gray, blur_ksize=blur, sobel_ksize=sobel_ksize, uint8=uint8)
        mask, t_used = threshold_binary_mask(edges, thresh, fixed_res)
        smeared, sig = gaussian_blur_binary_mask(mask, mask_blur_sigma)
        if sig > 0:
            mask_final, t_post, original_area = _post_blur_mask_and_largest_cc_area(
                smeared,
                post_mask_thresh,
                post_fixed_res,
                post_mask_otsu_margin,
                mask_keep_all_components,
            )
            adaptive_area = original_area
            if adapt_effective:
                mask_final, t_post, adaptive_area, margin_used, did_adapt = (
                    _adapt_post_otsu_margin_batch(
                        smeared,
                        post_mask_thresh,
                        post_fixed_res,
                        post_mask_otsu_margin,
                        mask_keep_all_components,
                        prev_largest_area,
                        adapt_trigger_ratio=adapt_post_otsu_trigger_ratio,
                        adapt_target_ratio=adapt_post_otsu_target_ratio,
                        adapt_max_ratio=adapt_post_otsu_max_ratio,
                        margin_step=adapt_post_otsu_margin_step,
                        max_extra_margin=adapt_post_otsu_max_extra_margin,
                    )
                )
                if did_adapt:
                    n_adaptive_post_otsu += 1
                    print(
                        f"  adaptive post-Otsu: margin {post_mask_otsu_margin}→{margin_used}, "
                        f"largest-CC area {adaptive_area} px "
                        f"(prev {prev_largest_area} px)",
                        flush=True,
                    )
            prev_largest_area = adaptive_area
        else:
            mask_final = mask.copy()
            t_post = None
            _lcc = largest_connected_component_mask(mask_final)
            original_area = adaptive_area = int((_lcc > 127).sum())
            prev_largest_area = adaptive_area
        if mask_area_w is not None:
            mask_area_w.writerow(
                {
                    "index": idx,
                    "file_name": logical_key.replace("\\", "/"),
                    "original_binary_mask_area": original_area,
                    "adaptive_binary_mask_area": adaptive_area,
                }
            )
        mask_refined = refined_roi_and(mask, mask_final)
        if refined_outlier_radius > 0:
            mask_refined = refined_mask_remove_outliers(
                mask_refined,
                radius=refined_outlier_radius,
                threshold=refined_outlier_threshold,
                which=refined_outlier_which,
            )
        rdest = mask_root / out_rel.with_name(out_rel.stem + "_mask_refined.png")
        rdest.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(rdest), mask_refined):
            raise OSError(f"cv2.imwrite failed: {rdest}")
        n_refined += 1
        mask_growth_src = mask_refined
        zline = np.zeros_like(mask_refined)
        n_hough_seg = 0
        int_t_used = 0
        lines_union = zline
        mask_refined_nolines = mask_refined.copy()
        if line_artifact == "morph" and (
            line_remove_horiz > 0 or line_remove_vert > 0
        ):
            _lh, _lv, lines_u, mask_nolines = line_artifact_masks_and_cleaned(
                mask_refined, line_remove_horiz, line_remove_vert
            )
            lines_union = lines_u
            mask_refined_nolines = mask_nolines
            mask_growth_src = mask_nolines
        elif line_artifact == "hough":
            g8 = gray_u8_for_canny(gray)
            _canny, h_lm, n_hough_seg = hough_prob_line_mask_from_gray(
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
            lines_union = h_lm
            mask_refined_nolines = mh_clean
            mask_growth_src = mh_clean
        elif line_artifact == "merge":
            _lh, _lv, morph_lines, morph_clean = line_artifact_masks_and_cleaned(
                mask_refined, line_remove_horiz, line_remove_vert
            )
            g8 = gray_u8_for_canny(gray)
            _canny, h_lm, n_hough_seg = hough_prob_line_mask_from_gray(
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
            merge_clean = cv2.bitwise_or(morph_clean, mh_clean)
            merge_removed = cv2.bitwise_and(
                mask_refined, cv2.bitwise_and(morph_lines, h_lm)
            )
            mask_growth_src = merge_clean
            lines_union = merge_removed
            mask_refined_nolines = merge_clean
        elif line_artifact == "intensity":
            int_dropped, int_clean, int_t_used = intensity_gate_inside_refined(
                gray,
                mask_refined,
                saturated_fraction=preview_saturated,
                linear_preview=preview_linear,
                otsu_margin=intensity_otsu_margin,
                fixed_thresh=intensity_fixed_thresh,
            )
            mask_growth_src = int_clean
            lines_union = int_dropped
            mask_refined_nolines = int_clean
        growth_need = (
            growth_w is not None or growth_save_stabilized or growth_print
        )
        if growth_need:
            stab = stabilize_plant_mask_for_measurement(
                mask_growth_src,
                close_ksize=growth_close_ksize,
                close_iters=growth_close_iters,
                fill_holes=growth_fill_holes,
                keep_largest=growth_keep_largest,
                largest_mode=growth_largest_mode,
            )
            met = growth_metrics_from_mask(
                stab,
                stem_lower_frac=growth_stem_frac,
                skip_stem=growth_skip_stem,
            )
            if growth_save_stabilized:
                gst = preview_out_dir / out_rel.with_name(
                    out_rel.stem + "_mask_growth_stabilized.png"
                )
                gst.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(gst), stab):
                    raise OSError(f"cv2.imwrite failed: {gst}")
                n_growth_stab += 1
            if growth_w is not None:
                row = growth_csv_row(
                    logical_key.replace("\\", "/"),
                    rel.stem,
                    str(line_artifact),
                    met,
                )
                growth_w.writerow(row)
            if growth_print:
                print(
                    f"Growth {rel.name}: bbox_h={met['bbox_height_px']} "
                    f"bbox_w={met['bbox_width_px']} obb_h={met['obb_h_px']:.1f} "
                    f"obb_w={met['obb_w_px']:.1f} stem_diam={met['stem_geodesic_diam_px']} "
                    f"fg_px={met['fg_px']}"
                )
        preview_path = preview_out_dir / out_rel.with_name(out_rel.stem + "_preview.png")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        fig = plant_contour_preview_figure(
            title_name=logical_key.replace("\\", "/"),
            gray=gray,
            edges=edges,
            mask=mask,
            mask_final=mask_final,
            mask_refined=mask_refined,
            lines_union=lines_union,
            mask_refined_nolines=mask_refined_nolines,
            t_used=t_used,
            t_post=t_post,
            mask_sigma=float(sig),
            thresh_mode=thresh,
            post_thresh_mode=post_mask_thresh,
            uint8_pipeline=uint8,
            mask_keep_all_components=mask_keep_all_components,
            line_artifact=line_artifact,
            line_remove_horiz=line_remove_horiz,
            line_remove_vert=line_remove_vert,
            n_hough_seg=n_hough_seg,
            int_t_used=int_t_used,
            growth_close_ksize=growth_close_ksize,
            growth_close_iters=growth_close_iters,
            growth_fill_holes=growth_fill_holes,
            growth_keep_largest=growth_keep_largest,
            growth_largest_mode=growth_largest_mode,
            growth_stem_frac=growth_stem_frac,
            growth_skip_stem=growth_skip_stem,
            preview_saturated=preview_saturated,
            preview_linear=preview_linear,
        )
        fig.savefig(str(preview_path), dpi=110, bbox_inches="tight")
        plt.close(fig)
        n_previews += 1
    if growth_f is not None:
        growth_f.close()
    if mask_area_f is not None:
        mask_area_f.close()
    if n_adaptive_post_otsu > 0:
        print(
            f"Batch adaptive post-Otsu: adjusted {n_adaptive_post_otsu} frame(s) "
            "(see per-frame lines)."
        )
    print(f"Wrote {n_refined} refined ROI masks (*_mask_refined.png) under {mask_root}")
    print(f"Wrote {n_previews} preview PNGs (*_preview.png) under {preview_out_dir}")
    if n_growth_stab > 0:
        print(
            f"Wrote {n_growth_stab} growth stabilized masks (*_mask_growth_stabilized.png) "
            f"under {preview_out_dir}"
        )
    if refined_outlier_radius > 0:
        _k = max(3, 2 * int(np.ceil(float(refined_outlier_radius))) + 1)
        if _k % 2 == 0:
            _k += 1
        _k = min(_k, 255)
        print(
            f"Each refined mask: Remove Outliers applied (radius={refined_outlier_radius:g}, "
            f"threshold={refined_outlier_threshold:g}, which={refined_outlier_which}, "
            f"medianBlur ksize={_k})."
        )
    if line_artifact == "merge" and line_remove_horiz <= 0 and line_remove_vert <= 0:
        print(
            "Note: merge with --line-remove-horiz / --line-remove-vert both 0: "
            "morph leg is empty; merge OR equals Hough-cleaned only per frame."
        )
    if growth_csv is not None:
        print(f"Growth metrics appended to {growth_csv.resolve()}")
    if mask_area_csv is not None:
        print(f"Post-blur largest-CC mask areas appended to {mask_area_csv.resolve()}")


def main() -> None:
    args = parse_args()
    u8 = bool(args.uint8)

    if args.frames_dirs is not None:
        if not args.frames_dirs:
            raise SystemExit("Batch mode requires at least one --frames-dir.")
        if args.preview_out_dir is None:
            raise SystemExit("Batch mode requires --preview-out-dir")
        if args.mask_out_dir is None:
            raise SystemExit("Batch mode requires --mask-out-dir")
        tm: ThreshMode = args.thresh  # type: ignore[assignment]
        ptm: ThreshMode = args.post_mask_thresh  # type: ignore[assignment]
        la: LineArtifactMode = effective_line_artifact(args)
        run_batch(
            args.frames_dirs,
            args.bg_glob,
            args.frames_step,
            args.preview_out_dir,
            args.mask_out_dir,
            blur=args.blur,
            sobel_ksize=args.sobel_ksize,
            uint8=u8,
            thresh=tm,
            fixed_thresh=args.fixed_thresh,
            mask_blur_sigma=args.mask_blur,
            post_mask_thresh=ptm,
            post_mask_fixed_thresh=args.post_mask_fixed_thresh,
            post_mask_otsu_margin=args.post_mask_otsu_margin,
            batch_adaptive_post_otsu=bool(args.batch_adaptive_post_otsu),
            adapt_post_otsu_trigger_ratio=args.adapt_post_otsu_trigger_ratio,
            adapt_post_otsu_target_ratio=args.adapt_post_otsu_target_ratio,
            adapt_post_otsu_max_ratio=args.adapt_post_otsu_max_ratio,
            adapt_post_otsu_margin_step=args.adapt_post_otsu_margin_step,
            adapt_post_otsu_max_extra_margin=args.adapt_post_otsu_max_extra_margin,
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
            intensity_otsu_margin=args.intensity_otsu_margin,
            intensity_fixed_thresh=args.intensity_fixed_thresh,
            preview_saturated=args.preview_saturated,
            preview_linear=bool(args.preview_linear),
            refined_outlier_radius=args.refined_outlier_radius,
            refined_outlier_threshold=args.refined_outlier_threshold,
            refined_outlier_which=args.refined_outlier_which,  # type: ignore[arg-type]
            growth_csv=args.growth_csv,
            growth_close_ksize=args.growth_close_ksize,
            growth_close_iters=args.growth_close_iters,
            growth_fill_holes=bool(args.growth_fill_holes),
            growth_keep_largest=not bool(args.growth_no_keep_largest),
            growth_largest_mode=args.growth_largest_mode,  # type: ignore[arg-type]
            growth_stem_frac=args.growth_stem_frac,
            growth_skip_stem=bool(args.growth_skip_stem),
            growth_save_stabilized=bool(args.growth_save_stabilized),
            growth_print=bool(args.growth_print),
            mask_area_csv=args.batch_mask_area_csv,
        )
        return

    if args.image is None:
        raise SystemExit(
            "Pass --image (single-frame), or at least one --frames-dir with --preview-out-dir and "
            "--mask-out-dir (batch)."
        )

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
        mask_final, t_post = threshold_soft_uint8(
            smeared,
            post_tm,
            post_fixed_only,
            otsu_margin=args.post_mask_otsu_margin,
        )
        if not args.mask_keep_all_components:
            mask_final = largest_connected_component_mask(mask_final)
    else:
        mask_final = mask.copy()
        t_post = None

    mask_refined = refined_roi_and(mask, mask_final)
    if args.refined_outlier_radius > 0:
        mask_refined = refined_mask_remove_outliers(
            mask_refined,
            radius=args.refined_outlier_radius,
            threshold=args.refined_outlier_threshold,
            which=args.refined_outlier_which,  # type: ignore[arg-type]
        )
    la: LineArtifactMode = effective_line_artifact(args)
    if bool(getattr(args, "hough_lines", False)):
        print("Note: --hough-lines is deprecated; use --line-artifact hough.")

    zline = np.zeros_like(mask_refined)
    n_hough_seg = 0
    int_t_used = 0
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
    elif la == "merge":
        if args.line_remove_horiz <= 0 and args.line_remove_vert <= 0:
            print(
                "Note: --line-artifact merge with --line-remove-horiz / --line-remove-vert "
                "both 0: morph leg is empty; OR-merge equals Hough-cleaned only."
            )
        _lh, _lv, morph_lines, morph_clean = line_artifact_masks_and_cleaned(
            mask_refined, args.line_remove_horiz, args.line_remove_vert
        )
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
        hough_clean = cv2.bitwise_and(mask_refined, cv2.bitwise_not(hough_line_mask))
        mask_refined_nolines = cv2.bitwise_or(morph_clean, hough_clean)
        lines_union = cv2.bitwise_and(
            mask_refined, cv2.bitwise_and(morph_lines, hough_line_mask)
        )
    elif la == "intensity":
        int_dropped, int_clean, int_t_used = intensity_gate_inside_refined(
            gray,
            mask_refined,
            saturated_fraction=args.preview_saturated,
            linear_preview=bool(args.preview_linear),
            otsu_margin=args.intensity_otsu_margin,
            fixed_thresh=args.intensity_fixed_thresh,
        )
        lines_union = int_dropped
        mask_refined_nolines = int_clean
    else:
        lines_union = zline
        mask_refined_nolines = mask_refined.copy()

    growth_active = (
        args.growth_csv is not None
        or args.growth_save_stabilized
        or args.growth_print
    )
    if growth_active:
        stab = stabilize_plant_mask_for_measurement(
            mask_refined_nolines,
            close_ksize=args.growth_close_ksize,
            close_iters=args.growth_close_iters,
            fill_holes=bool(args.growth_fill_holes),
            keep_largest=not bool(args.growth_no_keep_largest),
            largest_mode=args.growth_largest_mode,  # type: ignore[arg-type]
        )
        met = growth_metrics_from_mask(
            stab,
            stem_lower_frac=args.growth_stem_frac,
            skip_stem=bool(args.growth_skip_stem),
        )
        if args.growth_save_stabilized:
            gpath = image.parent / f"{image.stem}_mask_growth_stabilized.png"
            gpath.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(gpath), stab):
                raise OSError(f"cv2.imwrite failed: {gpath}")
            print(f"Saved growth stabilized mask to {gpath}")
        if args.growth_csv is not None:
            gcsv = args.growth_csv.resolve()
            gcsv.parent.mkdir(parents=True, exist_ok=True)
            is_new = not gcsv.exists() or gcsv.stat().st_size == 0
            with gcsv.open("a", newline="", encoding="utf-8") as gf:
                wr = csv.DictWriter(gf, fieldnames=GROWTH_CSV_FIELDNAMES)
                if is_new:
                    wr.writeheader()
                row = growth_csv_row(
                    image.name,
                    image.stem,
                    str(la),
                    met,
                )
                wr.writerow(row)
            print(f"Appended growth metrics to {gcsv}")
        if args.growth_print or args.growth_csv is not None:
            print(
                f"Growth ({image.name}): bbox_h={met['bbox_height_px']} "
                f"bbox_w={met['bbox_width_px']} obb_h={met['obb_h_px']:.1f} "
                f"obb_w={met['obb_w_px']:.1f} stem_diam={met['stem_geodesic_diam_px']} "
                f"fg_px={met['fg_px']}"
            )

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
            elif post_tm == "otsu" and args.post_mask_otsu_margin != 0:
                print(
                    f"Post-blur threshold (otsu, used after --post-mask-otsu-margin "
                    f"{args.post_mask_otsu_margin}): {t_post:.2f}"
                )
            else:
                print(f"Post-blur threshold ({post_tm}): {t_post:.2f}")
    else:
        print("Post-blur threshold skipped (--mask-blur 0; final mask matches edge mask).")
    if mask_sigma > 0 and not args.mask_keep_all_components:
        print("Post-blur mask: kept largest 8-connected component only.")
    print("Refined ROI: edge binary AND post-blur binary (bitwise AND).")
    if args.refined_outlier_radius > 0:
        _k = max(3, 2 * int(np.ceil(float(args.refined_outlier_radius))) + 1)
        if _k % 2 == 0:
            _k += 1
        _k = min(_k, 255)
        print(
            f"Refined Remove Outliers (ImageJ-style): radius={args.refined_outlier_radius:g} px, "
            f"threshold={args.refined_outlier_threshold:g}, which={args.refined_outlier_which} "
            f"(medianBlur ksize={_k})."
        )
    else:
        print("Refined Remove Outliers: skipped (--refined-outlier-radius 0).")
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
    elif la == "merge":
        print(
            "Line-artifact (--line-artifact merge): morph-cleaned OR Hough-cleaned refined "
            f"(Hough {n_hough_seg} segments); refined pixels removed only where morph line mask "
            "AND Hough line mask overlap."
        )
    elif la == "intensity":
        if args.intensity_fixed_thresh is not None:
            src = f"fixed --intensity-fixed-thresh={args.intensity_fixed_thresh}"
        else:
            src = (
                f"Otsu inside refined - margin "
                f"(--intensity-otsu-margin={args.intensity_otsu_margin})"
            )
        n_dropped = int((lines_union > 0).sum())
        n_kept = int((mask_refined_nolines > 0).sum())
        print(
            f"Line-artifact (--line-artifact intensity): gate T_used={int_t_used} on "
            f"auto-stretched 8-bit gray ({src}); dropped {n_dropped} px from refined, "
            f"kept {n_kept} px."
        )
    else:
        print("Line-artifact removal: none (--line-artifact none).")

    if not args.no_show:
        fig = plant_contour_preview_figure(
            title_name=image.name,
            gray=gray,
            edges=edges,
            mask=mask,
            mask_final=mask_final,
            mask_refined=mask_refined,
            lines_union=lines_union,
            mask_refined_nolines=mask_refined_nolines,
            t_used=t_used,
            t_post=t_post,
            mask_sigma=float(mask_sigma),
            thresh_mode=tm,
            post_thresh_mode=post_tm,
            uint8_pipeline=u8,
            mask_keep_all_components=bool(args.mask_keep_all_components),
            line_artifact=la,
            line_remove_horiz=args.line_remove_horiz,
            line_remove_vert=args.line_remove_vert,
            n_hough_seg=n_hough_seg,
            int_t_used=int_t_used,
            growth_close_ksize=args.growth_close_ksize,
            growth_close_iters=args.growth_close_iters,
            growth_fill_holes=bool(args.growth_fill_holes),
            growth_keep_largest=not bool(args.growth_no_keep_largest),
            growth_largest_mode=args.growth_largest_mode,  # type: ignore[arg-type]
            growth_stem_frac=args.growth_stem_frac,
            growth_skip_stem=bool(args.growth_skip_stem),
            preview_saturated=args.preview_saturated,
            preview_linear=bool(args.preview_linear),
        )
        plt.show()


if __name__ == "__main__":
    main()
