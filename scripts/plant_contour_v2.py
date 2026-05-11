"""
Grayscale plant contour / mask — v2.

- Median background: sample frames from a timelapse folder, per-pixel median
  background, then |current − background| and threshold (Otsu on the diff).
- Otsu-only: single-frame Otsu on Gaussian-blurred intensity (baseline for
  comparison; often weaker when plant and soil are similar gray).
- Batch: ``--batch-csv`` + ``--frames-dir`` builds one median background, loops
  every matching frame, writes per-frame mask / contour area metrics to CSV.
  Optional ``--preview-dir`` saves one contour overlay PNG per frame (mirrors
  subfolders under ``--frames-dir`` when present).

Requires OpenCV, NumPy, Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def odd_kernel(k: int, minimum: int = 3) -> int:
    k = max(minimum, int(k))
    return k if k % 2 == 1 else k + 1


def load_grayscale_8u(image_path: str | Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def resize_max_dim(img: np.ndarray, max_dim: int | None) -> tuple[np.ndarray, float]:
    if max_dim is None:
        return img, 1.0
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return img, 1.0
    scale = max_dim / m
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, scale


def upsize_to_match(small: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    if small.shape[0] == h and small.shape[1] == w:
        return small
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def list_frames(directory: Path, pattern: str) -> list[Path]:
    paths = sorted(directory.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} under {directory}")
    return paths


def sample_evenly(paths: list[Path], max_count: int) -> list[Path]:
    if len(paths) <= max_count:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_count, dtype=int)
    return [paths[int(i)] for i in idx]


def compute_median_background(
    frame_paths: list[Path],
    max_frames: int,
    max_dim: int | None,
) -> np.ndarray:
    """Per-pixel median over sampled frames (each normalized to 8-bit)."""
    sampled = sample_evenly(frame_paths, max_frames)
    stack: list[np.ndarray] = []
    for p in sampled:
        g = load_grayscale_8u(p)
        g, _ = resize_max_dim(g, max_dim)
        stack.append(g.astype(np.float32))
    med = np.median(np.stack(stack, axis=0), axis=0)
    return np.clip(med, 0, 255).astype(np.uint8)


def morph_clean_mask(binary: np.ndarray, close_ksize: int, open_ksize: int) -> np.ndarray:
    """binary: 0/255 uint8."""
    out = binary.copy()
    if close_ksize >= 3:
        kc = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kc)
    if open_ksize >= 3:
        ko = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ko)
    return out


def median_background_u8(
    background_dir: Path,
    bg_glob: str,
    max_bg_frames: int,
    bg_max_dim: int | None,
    target_shape_hw: tuple[int, int],
) -> np.ndarray:
    """Median stack upsized to (H, W) of the target frames."""
    bg_paths = list_frames(background_dir, bg_glob)
    bg_small = compute_median_background(bg_paths, max_bg_frames, bg_max_dim)
    return upsize_to_match(bg_small, target_shape_hw)


def segment_current_median_bg(
    current: np.ndarray,
    background: np.ndarray,
    blur_ksize: int,
    close_ksize: int,
    open_ksize: int,
    min_area: float,
    min_cy_frac: float,
    max_width_frac: float,
) -> dict:
    """Same logic as segment_median_background but from in-memory images."""
    if current.shape != background.shape:
        raise ValueError(
            f"Shape mismatch current {current.shape} vs background {background.shape}"
        )
    bk = odd_kernel(blur_ksize)
    blurred = cv2.GaussianBlur(current, (bk, bk), 0)
    bg_blur = cv2.GaussianBlur(background, (bk, bk), 0)
    diff = cv2.absdiff(blurred, bg_blur)
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = morph_clean_mask(
        binary, odd_kernel(close_ksize), odd_kernel(open_ksize)
    )
    contour = largest_filtered_contour(
        binary, min_area, min_cy_frac, max_width_frac
    )
    return {
        "current": current,
        "background": background,
        "diff": diff,
        "mask": binary,
        "contour": contour,
    }


def largest_filtered_contour(
    mask_255: np.ndarray,
    min_area: float,
    min_cy_frac: float,
    max_width_frac: float,
) -> np.ndarray | None:
    contours, _ = cv2.findContours(
        mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h, w = mask_255.shape[:2]
    best = None
    best_area = 0.0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cy = y + bh / 2.0
        if cy <= h * min_cy_frac:
            continue
        if bw >= w * max_width_frac:
            continue
        if area > best_area:
            best_area = area
            best = cnt
    return best


def segment_median_background(
    current_path: Path,
    background_dir: Path,
    bg_glob: str,
    max_bg_frames: int,
    bg_max_dim: int | None,
    blur_ksize: int,
    close_ksize: int,
    open_ksize: int,
    min_area: float,
    min_cy_frac: float,
    max_width_frac: float,
) -> dict:
    current = load_grayscale_8u(current_path)
    h, w = current.shape[:2]
    background = median_background_u8(
        background_dir, bg_glob, max_bg_frames, bg_max_dim, (h, w)
    )
    return segment_current_median_bg(
        current,
        background,
        blur_ksize,
        close_ksize,
        open_ksize,
        min_area,
        min_cy_frac,
        max_width_frac,
    )


def segment_otsu_single(
    current_path: Path,
    blur_ksize: int,
    close_ksize: int,
    open_ksize: int,
    min_area: float,
    min_cy_frac: float,
    max_width_frac: float,
) -> dict:
    current = load_grayscale_8u(current_path)
    bk = odd_kernel(blur_ksize)
    blurred = cv2.GaussianBlur(current, (bk, bk), 0)
    _, otsu_raw = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Otsu does not know which class is "plant"; try both polarities and keep
    # the mask that yields the largest contour passing the geometric filters.
    best_mask = None
    best_contour = None
    best_area = 0.0
    for candidate in (otsu_raw, cv2.bitwise_not(otsu_raw)):
        cleaned = morph_clean_mask(
            candidate, odd_kernel(close_ksize), odd_kernel(open_ksize)
        )
        cnt = largest_filtered_contour(
            cleaned, min_area, min_cy_frac, max_width_frac
        )
        if cnt is None:
            continue
        a = cv2.contourArea(cnt)
        if a > best_area:
            best_area = a
            best_contour = cnt
            best_mask = cleaned
    if best_mask is None:
        best_mask = morph_clean_mask(
            otsu_raw, odd_kernel(close_ksize), odd_kernel(open_ksize)
        )
    return {
        "current": current,
        "mask": best_mask,
        "contour": best_contour,
    }


def draw_contour_bgr(gray: np.ndarray, contour: np.ndarray | None) -> np.ndarray:
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if contour is not None:
        cv2.drawContours(bgr, [contour], -1, (0, 0, 255), 2)
    return bgr


def batch_median_shoot_csv(
    frames_dir: Path,
    background_dir: Path | None,
    bg_glob: str,
    max_bg_frames: int,
    bg_max_dim: int | None,
    blur_ksize: int,
    close_ksize: int,
    open_ksize: int,
    min_area: float,
    min_cy_frac: float,
    max_width_frac: float,
    csv_path: Path,
    preview_dir: Path | None = None,
) -> dict | None:
    """
    One median background from background_dir (default: frames_dir), then every
    frame in frames_dir: thresholded mask → CSV columns for shoot-area proxies.
    If preview_dir is set, each frame is also written as a BGR PNG with contour drawn.
    """
    frames_dir = frames_dir.resolve()
    bg_dir = (background_dir or frames_dir).resolve()
    paths = list_frames(frames_dir, bg_glob)
    bg_paths = list_frames(bg_dir, bg_glob)
    bg_small = compute_median_background(bg_paths, max_bg_frames, bg_max_dim)

    fieldnames = (
        "rel_path",
        "filename",
        "width",
        "height",
        "mask_foreground_px",
        "largest_contour_area_px",
        "mask_fraction_of_image",
    )
    csv_path = csv_path.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    preview_root = preview_dir.resolve() if preview_dir is not None else None
    if preview_root is not None:
        preview_root.mkdir(parents=True, exist_ok=True)

    last_vis: dict | None = None
    n_saved = 0
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for p in paths:
            cur = load_grayscale_8u(p)
            h, w = cur.shape[:2]
            bg = upsize_to_match(bg_small, (h, w))
            out = segment_current_median_bg(
                cur,
                bg,
                blur_ksize,
                close_ksize,
                open_ksize,
                min_area,
                min_cy_frac,
                max_width_frac,
            )
            mask = out["mask"]
            cnt = out["contour"]
            mpx = int(np.count_nonzero(mask == 255))
            carea = float(cv2.contourArea(cnt)) if cnt is not None else 0.0
            denom = float(h * w)
            try:
                rel = str(p.relative_to(frames_dir))
            except ValueError:
                rel = p.name
            row = {
                "rel_path": rel,
                "filename": p.name,
                "width": w,
                "height": h,
                "mask_foreground_px": mpx,
                "largest_contour_area_px": f"{carea:.1f}",
                "mask_fraction_of_image": f"{(mpx / denom):.6f}",
            }
            writer.writerow(row)
            last_vis = {"out": out, "path": p}

            if preview_root is not None:
                overlay = draw_contour_bgr(cur, cnt)
                dest = preview_root / Path(rel).with_suffix(".png")
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(dest), overlay):
                    raise OSError(f"cv2.imwrite failed: {dest}")
                n_saved += 1

    print(f"Wrote {len(paths)} rows to {csv_path}")
    if preview_root is not None:
        print(f"Saved {n_saved} contour previews under {preview_root}")
    return last_vis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grayscale plant mask v2 (median bg + Otsu).")
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Single frame to visualize (not used with --batch-csv).",
    )
    p.add_argument(
        "--batch-csv",
        type=Path,
        default=None,
        help="If set, loop all frames in --frames-dir: one median background, then "
        "write per-frame shoot metrics to this CSV (optional --preview-dir for PNGs; "
        "no matplotlib unless --preview-last).",
    )
    p.add_argument(
        "--frames-dir",
        type=Path,
        default=None,
        help="Folder of images to process (required with --batch-csv).",
    )
    p.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="With --batch-csv: save each frame as a PNG with median-bg contour drawn "
        "(paths mirror rel_path under frames-dir).",
    )
    p.add_argument(
        "--background-dir",
        type=Path,
        default=None,
        help="Median stack folder (default: parent of --image, or --frames-dir in batch).",
    )
    p.add_argument(
        "--bg-glob",
        default="*.tif",
        help="Glob under background-dir (default: *.tif).",
    )
    p.add_argument(
        "--max-bg-frames",
        type=int,
        default=32,
        help="Max frames sampled evenly for median (default: 32).",
    )
    p.add_argument(
        "--bg-max-dim",
        type=int,
        default=1200,
        help="Max side length when building median stack (saves RAM); upscaled to frame size. 0=disable.",
    )
    p.add_argument(
        "--blur",
        type=int,
        default=7,
        help="Gaussian blur kernel size (odd, default: 7).",
    )
    p.add_argument(
        "--close",
        type=int,
        default=9,
        help="Morphological close kernel (odd, default: 9).",
    )
    p.add_argument(
        "--open",
        type=int,
        default=5,
        help="Morphological open kernel (odd, default: 5).",
    )
    p.add_argument("--min-area", type=float, default=500.0)
    p.add_argument(
        "--min-cy-frac",
        type=float,
        default=0.4,
        help="Ignore contours whose bbox center-y is above this fraction of image height.",
    )
    p.add_argument(
        "--max-width-frac",
        type=float,
        default=0.8,
        help="Ignore contours whose bbox width exceeds this fraction of image width.",
    )
    p.add_argument(
        "--method",
        choices=("median", "otsu", "both"),
        default="both",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (single-frame mode).",
    )
    p.add_argument(
        "--preview-last",
        action="store_true",
        help="With --batch-csv, also show matplotlib for the last frame (median row).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bg_max_dim = None if args.bg_max_dim == 0 else args.bg_max_dim
    bk = odd_kernel(args.blur)
    ck = odd_kernel(args.close)
    ok = odd_kernel(args.open)

    if args.batch_csv is not None:
        if args.frames_dir is None:
            raise SystemExit("--batch-csv requires --frames-dir (folder to loop).")
        last = batch_median_shoot_csv(
            args.frames_dir,
            args.background_dir,
            args.bg_glob,
            args.max_bg_frames,
            bg_max_dim,
            bk,
            ck,
            ok,
            args.min_area,
            args.min_cy_frac,
            args.max_width_frac,
            args.batch_csv,
            args.preview_dir,
        )
        if args.preview_last and last is not None:
            out = last["out"]
            p = last["path"]
            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
            axes[0].imshow(out["current"], cmap="gray")
            axes[0].set_title("Last frame")
            axes[1].imshow(out["background"], cmap="gray")
            axes[1].set_title("Median bg")
            axes[2].imshow(out["diff"], cmap="gray")
            axes[2].set_title("|diff|")
            axes[3].imshow(
                cv2.cvtColor(
                    draw_contour_bgr(out["current"], out["contour"]),
                    cv2.COLOR_BGR2RGB,
                )
            )
            axes[3].set_title("Contour")
            for ax in axes:
                ax.axis("off")
            plt.suptitle(p.name)
            plt.tight_layout()
            plt.show()
        return

    if args.image is None:
        raise SystemExit("Single-frame mode: pass --image, or use --batch-csv with --frames-dir.")

    image = args.image.resolve()
    if not image.is_file():
        raise FileNotFoundError(image)

    bg_dir = args.background_dir.resolve() if args.background_dir else image.parent

    if args.method in ("median", "both"):
        med = segment_median_background(
            image,
            bg_dir,
            args.bg_glob,
            args.max_bg_frames,
            bg_max_dim,
            bk,
            ck,
            ok,
            args.min_area,
            args.min_cy_frac,
            args.max_width_frac,
        )
    else:
        med = None

    if args.method in ("otsu", "both"):
        ots = segment_otsu_single(
            image,
            bk,
            ck,
            ok,
            args.min_area,
            args.min_cy_frac,
            args.max_width_frac,
        )
    else:
        ots = None

    if args.method == "both":
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        assert med is not None and ots is not None

        axes[0, 0].imshow(med["current"], cmap="gray")
        axes[0, 0].set_title("Current (8-bit)")
        axes[0, 1].imshow(med["background"], cmap="gray")
        axes[0, 1].set_title("Median background")
        axes[0, 2].imshow(med["diff"], cmap="gray")
        axes[0, 2].set_title("|current − bg| (blurred)")
        axes[0, 3].imshow(
            cv2.cvtColor(draw_contour_bgr(med["current"], med["contour"]), cv2.COLOR_BGR2RGB)
        )
        axes[0, 3].set_title("Median-bg contour")

        axes[1, 0].imshow(ots["current"], cmap="gray")
        axes[1, 0].set_title("Current (same)")
        axes[1, 1].imshow(ots["mask"], cmap="gray")
        axes[1, 1].set_title("Otsu mask (single frame)")
        axes[1, 2].axis("off")
        axes[1, 3].imshow(
            cv2.cvtColor(draw_contour_bgr(ots["current"], ots["contour"]), cv2.COLOR_BGR2RGB)
        )
        axes[1, 3].set_title("Otsu contour")

        for ax in axes.ravel():
            ax.axis("off")
        plt.suptitle(f"{image.name} — median vs Otsu")
        plt.tight_layout()
    elif args.method == "median":
        assert med is not None
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        axes[0].imshow(med["current"], cmap="gray")
        axes[0].set_title("Current")
        axes[1].imshow(med["background"], cmap="gray")
        axes[1].set_title("Median background")
        axes[2].imshow(med["diff"], cmap="gray")
        axes[2].set_title("|diff|")
        axes[3].imshow(
            cv2.cvtColor(draw_contour_bgr(med["current"], med["contour"]), cv2.COLOR_BGR2RGB)
        )
        axes[3].set_title("Contour")
        for ax in axes:
            ax.axis("off")
        plt.suptitle(f"{image.name} — median background")
        plt.tight_layout()
    else:
        assert ots is not None
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        axes[0].imshow(ots["current"], cmap="gray")
        axes[0].set_title("Current")
        axes[1].imshow(ots["mask"], cmap="gray")
        axes[1].set_title("Otsu mask")
        axes[2].imshow(
            cv2.cvtColor(draw_contour_bgr(ots["current"], ots["contour"]), cv2.COLOR_BGR2RGB)
        )
        axes[2].set_title("Contour")
        for ax in axes:
            ax.axis("off")
        plt.suptitle(f"{image.name} — Otsu only")
        plt.tight_layout()

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
