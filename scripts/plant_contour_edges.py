"""
Grayscale plant contour via gradient edges (ImageJ "Find Edges"-like).

Uses 3×3 Sobel magnitude (similar to ImageJ), optional blur, threshold, morphology,
then picks one contour with geometric filters aimed at tray horizontal lines and
upper/right probe rig.

Standalone script (no import from plant_contour_v2).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_grayscale_8u(image_path: str | Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def odd_kernel(k: int, minimum: int = 3) -> int:
    k = max(minimum, int(k))
    return k if k % 2 == 1 else k + 1


def sobel_magnitude_u8(gray_u8: np.ndarray) -> np.ndarray:
    """Gradient magnitude ~ ImageJ Find Edges (Sobel 3×3)."""
    gx = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def list_frames(directory: Path, pattern: str) -> list[Path]:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {directory}")
    return paths


def draw_contour_bgr(gray: np.ndarray, contour: np.ndarray | None) -> np.ndarray:
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if contour is not None:
        cv2.drawContours(bgr, [contour], -1, (0, 0, 255), 2)
    return bgr


def contour_rejected_horizontal_strip(
    cnt: np.ndarray,
    img_h: int,
    img_w: int,
    max_height_frac: float,
    min_width_frac: float,
) -> bool:
    """Long flat tray / shelf edge: very wide, very short bbox."""
    _x, _y, bw, bh = cv2.boundingRect(cnt)
    if bh <= 0 or bw <= 0:
        return False
    return (bw >= min_width_frac * img_w) and (bh <= max_height_frac * img_h)


def contour_rejected_probe_right_tall(
    cnt: np.ndarray,
    img_h: int,
    img_w: int,
    min_x_center_frac: float,
    max_aspect_w_over_h: float,
    min_height_frac: float,
) -> bool:
    """Tall narrow structure on the right (post / probe body)."""
    x, y, bw, bh = cv2.boundingRect(cnt)
    cx = x + bw / 2.0
    if cx < min_x_center_frac * img_w:
        return False
    if bh < min_height_frac * img_h:
        return False
    ar = bw / max(float(bh), 1.0)
    return ar <= max_aspect_w_over_h


def contour_rejected_top_region(
    cnt: np.ndarray,
    img_h: int,
    min_centroid_y_frac: float,
) -> bool:
    """Reject if centroid lies in the upper part of the frame (rig / probe)."""
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        return True
    cy = m["m01"] / m["m00"]
    return cy < min_centroid_y_frac * img_h


def largest_filtered_plant_contour(
    binary_closed: np.ndarray,
    img_h: int,
    img_w: int,
    min_area: float,
    min_bbox_cy_frac: float,
    max_bbox_width_frac: float,
    strip_max_h_frac: float,
    strip_min_w_frac: float,
    probe_right_min_xc_frac: float,
    probe_right_max_ar: float,
    probe_right_min_h_frac: float,
    probe_centroid_y_min_frac: float,
) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        binary_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    best = None
    best_area = 0.0
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cy_box = y + bh / 2.0
        if cy_box <= img_h * min_bbox_cy_frac:
            continue
        if bw >= img_w * max_bbox_width_frac:
            continue
        if contour_rejected_horizontal_strip(
            cnt, img_h, img_w, strip_max_h_frac, strip_min_w_frac
        ):
            continue
        if contour_rejected_probe_right_tall(
            cnt,
            img_h,
            img_w,
            probe_right_min_xc_frac,
            probe_right_max_ar,
            probe_right_min_h_frac,
        ):
            continue
        if contour_rejected_top_region(cnt, img_h, probe_centroid_y_min_frac):
            continue
        if area > best_area:
            best_area = area
            best = cnt
    return best


def segment_edges(
    gray_u8: np.ndarray,
    blur_ksize: int,
    thresh_mode: str,
    fixed_thresh: int | None,
    close_ksize: int,
    open_ksize: int,
    zero_top_frac: float,
    min_area: float,
    min_bbox_cy_frac: float,
    max_bbox_width_frac: float,
    strip_max_h_frac: float,
    strip_min_w_frac: float,
    probe_right_min_xc_frac: float,
    probe_right_max_ar: float,
    probe_right_min_h_frac: float,
    probe_centroid_y_min_frac: float,
) -> dict:
    bk = odd_kernel(blur_ksize)
    blurred = cv2.GaussianBlur(gray_u8, (bk, bk), 0)
    mag = sobel_magnitude_u8(blurred)
    if zero_top_frac > 0:
        cut = int(zero_top_frac * mag.shape[0])
        if cut > 0:
            mag = mag.copy()
            mag[:cut, :] = 0

    if thresh_mode == "otsu":
        _, binary = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif thresh_mode == "fixed":
        t = int(fixed_thresh) if fixed_thresh is not None else 40
        _, binary = cv2.threshold(mag, t, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError(thresh_mode)

    kc = odd_kernel(close_ksize)
    ko = odd_kernel(open_ksize)
    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)),
    )
    closed = cv2.morphologyEx(
        closed,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)),
    )

    h, w = gray_u8.shape[:2]
    contour = largest_filtered_plant_contour(
        closed,
        h,
        w,
        min_area,
        min_bbox_cy_frac,
        max_bbox_width_frac,
        strip_max_h_frac,
        strip_min_w_frac,
        probe_right_min_xc_frac,
        probe_right_max_ar,
        probe_right_min_h_frac,
        probe_centroid_y_min_frac,
    )
    return {
        "gray": gray_u8,
        "magnitude": mag,
        "binary": binary,
        "closed": closed,
        "contour": contour,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plant contour from Sobel edges (ImageJ-like) + rig/line filters."
    )
    p.add_argument("--image", type=Path, default=None)
    p.add_argument("--frames-dir", type=Path, default=None)
    p.add_argument("--bg-glob", default="*.tif")
    p.add_argument(
        "--batch-csv",
        type=Path,
        default=None,
        help="Per-frame metrics CSV (requires --frames-dir).",
    )
    p.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Save contour overlay PNGs (batch only).",
    )
    p.add_argument("--blur", type=int, default=3)
    p.add_argument(
        "--thresh",
        choices=("otsu", "fixed"),
        default="otsu",
    )
    p.add_argument("--fixed-thresh", type=int, default=40)
    p.add_argument("--close", type=int, default=7)
    p.add_argument("--open", type=int, default=3)
    p.add_argument(
        "--zero-top-frac",
        type=float,
        default=0.0,
        help="Zero edge magnitude in the top fraction of the image before threshold "
        "(e.g. 0.35 to drop upper rig).",
    )
    p.add_argument("--min-area", type=float, default=400.0)
    p.add_argument("--min-bbox-cy-frac", type=float, default=0.38)
    p.add_argument("--max-bbox-width-frac", type=float, default=0.82)
    p.add_argument(
        "--strip-max-h-frac",
        type=float,
        default=0.045,
        help="Reject bbox with height below this fraction of image if also very wide.",
    )
    p.add_argument("--strip-min-w-frac", type=float, default=0.42)
    p.add_argument(
        "--probe-right-min-xc-frac",
        type=float,
        default=0.52,
        help="BBox center-x must exceed this to apply tall-narrow-right rule.",
    )
    p.add_argument(
        "--probe-right-max-ar",
        type=float,
        default=0.5,
        help="Reject if bbox width/height <= this (tall narrow) on the right.",
    )
    p.add_argument("--probe-right-min-h-frac", type=float, default=0.18)
    p.add_argument(
        "--probe-centroid-y-min-frac",
        type=float,
        default=0.32,
        help="Reject contours whose centroid y is above this fraction of image height.",
    )
    p.add_argument("--no-show", action="store_true")
    p.add_argument(
        "--preview-last",
        action="store_true",
        help="After batch, show last frame pipeline in matplotlib.",
    )
    return p.parse_args()


def run_batch(
    frames_dir: Path,
    bg_glob: str,
    csv_path: Path,
    preview_dir: Path | None,
    seg_kwargs: dict,
) -> dict | None:
    frames_dir = frames_dir.resolve()
    paths = list_frames(frames_dir, bg_glob)
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    preview_root = preview_dir.resolve() if preview_dir else None
    if preview_root:
        preview_root.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        "rel_path",
        "filename",
        "width",
        "height",
        "largest_contour_area_px",
    )
    last_vis = None
    n_saved = 0
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames)
        w.writeheader()
        for p in paths:
            gray = load_grayscale_8u(p)
            out = segment_edges(gray, **seg_kwargs)
            cnt = out["contour"]
            h, wi = gray.shape[:2]
            carea = float(cv2.contourArea(cnt)) if cnt is not None else 0.0
            try:
                rel = str(p.relative_to(frames_dir))
            except ValueError:
                rel = p.name
            w.writerow(
                {
                    "rel_path": rel,
                    "filename": p.name,
                    "width": wi,
                    "height": h,
                    "largest_contour_area_px": f"{carea:.1f}",
                }
            )
            last_vis = {"out": out, "path": p}
            if preview_root:
                dest = preview_root / Path(rel).with_suffix(".png")
                dest.parent.mkdir(parents=True, exist_ok=True)
                overlay = draw_contour_bgr(gray, cnt)
                if not cv2.imwrite(str(dest), overlay):
                    raise OSError(f"cv2.imwrite failed: {dest}")
                n_saved += 1

    print(f"Wrote {len(paths)} rows to {csv_path}")
    if preview_root:
        print(f"Saved {n_saved} previews under {preview_root}")
    return last_vis


def main() -> None:
    args = parse_args()
    seg_kwargs = dict(
        blur_ksize=args.blur,
        thresh_mode=args.thresh,
        fixed_thresh=args.fixed_thresh,
        close_ksize=args.close,
        open_ksize=args.open,
        zero_top_frac=args.zero_top_frac,
        min_area=args.min_area,
        min_bbox_cy_frac=args.min_bbox_cy_frac,
        max_bbox_width_frac=args.max_bbox_width_frac,
        strip_max_h_frac=args.strip_max_h_frac,
        strip_min_w_frac=args.strip_min_w_frac,
        probe_right_min_xc_frac=args.probe_right_min_xc_frac,
        probe_right_max_ar=args.probe_right_max_ar,
        probe_right_min_h_frac=args.probe_right_min_h_frac,
        probe_centroid_y_min_frac=args.probe_centroid_y_min_frac,
    )

    if args.batch_csv:
        if not args.frames_dir:
            raise SystemExit("--batch-csv requires --frames-dir")
        last = run_batch(
            args.frames_dir,
            args.bg_glob,
            args.batch_csv,
            args.preview_dir,
            seg_kwargs,
        )
        if args.preview_last and last:
            o = last["out"]
            fig, axes = plt.subplots(1, 5, figsize=(22, 4))
            axes[0].imshow(o["gray"], cmap="gray")
            axes[0].set_title("Gray")
            axes[1].imshow(o["magnitude"], cmap="gray")
            axes[1].set_title("Sobel |grad|")
            axes[2].imshow(o["binary"], cmap="gray")
            axes[2].set_title("Thresh")
            axes[3].imshow(o["closed"], cmap="gray")
            axes[3].set_title("Morph")
            axes[4].imshow(
                cv2.cvtColor(draw_contour_bgr(o["gray"], o["contour"]), cv2.COLOR_BGR2RGB)
            )
            axes[4].set_title("Contour")
            for ax in axes:
                ax.axis("off")
            plt.suptitle(last["path"].name)
            plt.tight_layout()
            plt.show()
        return

    if not args.image:
        raise SystemExit("Pass --image, or --batch-csv with --frames-dir.")

    image = args.image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)

    gray = load_grayscale_8u(image)
    out = segment_edges(gray, **seg_kwargs)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    axes[0].imshow(out["gray"], cmap="gray")
    axes[0].set_title("Gray")
    axes[1].imshow(out["magnitude"], cmap="gray")
    axes[1].set_title("Sobel |grad|")
    axes[2].imshow(out["binary"], cmap="gray")
    axes[2].set_title("Thresh")
    axes[3].imshow(out["closed"], cmap="gray")
    axes[3].set_title("Morph")
    axes[4].imshow(
        cv2.cvtColor(draw_contour_bgr(out["gray"], out["contour"]), cv2.COLOR_BGR2RGB)
    )
    axes[4].set_title("Contour")
    for ax in axes:
        ax.axis("off")
    plt.suptitle(image.name)
    plt.tight_layout()
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
