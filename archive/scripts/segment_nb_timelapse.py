#!/usr/bin/env python3
"""Segment Nicotiana benthamiana in grayscale VIS–NIR TIFF timelapses (Tier-1 + optional refinements)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from animate_plant_timelapse import collect_tiff_files, parse_timestamp_from_name  # noqa: E402

from plant_timelapse.segment_nb import SegmentConfig, run_segmentation_pipeline  # noqa: E402


def parse_crop(s: str) -> tuple[int, int, int, int]:
    parts = [int(x.strip()) for x in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be y0,y1,x0,x1")
    y0, y1, x0, x1 = parts
    if y1 <= y0 or x1 <= x0:
        raise argparse.ArgumentTypeError("crop requires y1>y0 and x1>x0")
    return y0, y1, x0, x1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plant segmentation for fixed-camera Alvium-style grayscale TIFF time series. "
            "Uses temporal median or MOG2 background + residual threshold, morphology, optional ECC "
            "alignment, temporal median smoothing, GrabCut, and optional SAM (see --sam-checkpoint)."
        ),
        epilog=(
            "Tier 0: use --crop to limit the pot region; --erode-shrink-px can drop thin probe stalks. "
            "Tier 3 (supervised): export masks with this script, correct labels in CVAT/LabelMe, then see "
            "scripts/train_plant_seg_unet.py for a training stub."
        ),
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        type=Path,
        required=True,
        help="Directories containing .tif/.tiff frames (same conventions as animate_plant_timelapse).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Writes masks/, plant_area.csv, and optional previews/.",
    )
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame (default: 1).")
    parser.add_argument("--max-frames", type=int, default=None, help="Cap number of frames after frame-step.")
    parser.add_argument(
        "--random-sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "After frame-step and max-frames, uniformly sample N frame indices without replacement "
            "(chronological order preserved). Default: use the full selection."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        metavar="INT",
        help="RNG seed for --random-sample (omit for a different random subset each run).",
    )
    parser.add_argument("--crop", type=parse_crop, default=None, help="Optional crop y0,y1,x0,x1 (inclusive-exclusive).")

    parser.add_argument(
        "--bg-mode",
        type=str,
        choices=("median", "mog2"),
        default="median",
        help="Background model: temporal median subsample or OpenCV MOG2 (default: median).",
    )
    parser.add_argument(
        "--bg-sample-count",
        type=int,
        default=48,
        help="Frames (indices spread across run) used to build median background (default: 48).",
    )
    parser.add_argument("--mog-train-frames", type=int, default=120, help="Frames to train MOG2 (default: 120).")
    parser.add_argument("--mog-history", type=int, default=500, help="MOG2 history length (default: 500).")
    parser.add_argument(
        "--mog-var-threshold",
        type=float,
        default=16.0,
        help="MOG2 variance threshold / sensitivity (default: 16).",
    )
    parser.add_argument(
        "--no-mog-shadows",
        action="store_true",
        help="Disable MOG2 shadow detection.",
    )
    parser.add_argument(
        "--mog-binary-threshold",
        type=int,
        default=200,
        help="Treat MOG2 foreground pixels with value >= this as FG (default: 200; use 127 if inverted).",
    )
    parser.add_argument(
        "--combine-mog-and-residual",
        action="store_true",
        help="When using MOG2, AND MOG FG with residual |I−B|; default is MOG mask only.",
    )

    parser.add_argument(
        "--residual-percentile",
        type=float,
        default=92.0,
        help="Percentile on |I−B| for residual binary map when background is defined (default: 92).",
    )
    parser.add_argument("--open-iterations", type=int, default=1, help="Morphological opening iterations (default: 1).")
    parser.add_argument("--close-iterations", type=int, default=2, help="Morphological closing iterations (default: 2).")
    parser.add_argument(
        "--min-area-px",
        type=int,
        default=500,
        help="Drop connected components smaller than this (crop coordinates; default: 500).",
    )
    parser.add_argument(
        "--erode-shrink-px",
        type=int,
        default=0,
        help="Extra binary erosion iterations to shrink masks (e.g. reduce attached probe halo; default: 0).",
    )
    parser.add_argument(
        "--keep-largest-only",
        action="store_true",
        help=(
            "After cleanup, keep a single foreground blob: the largest connected component "
            "(useful when one plant is in frame but rig/background produces extra patches)."
        ),
    )
    parser.add_argument(
        "--temporal-median-window",
        type=int,
        default=3,
        help="Odd window size for per-pixel temporal median on binary masks; 1 disables (default: 3).",
    )

    parser.add_argument(
        "--align-ecc",
        action="store_true",
        help="Align each cropped frame to the first processed frame via ECC before background subtraction.",
    )
    parser.add_argument(
        "--ecc-euclidean",
        action="store_true",
        help="Allow rotation+translation for ECC (default: translation-only).",
    )

    parser.add_argument(
        "--refine-grabcut",
        action="store_true",
        help="Tier-2b: GrabCut refinement seeded by the coarse Tier-1 mask.",
    )
    parser.add_argument("--grabcut-iters", type=int, default=5, help="GrabCut iterations hint (default: 5).")

    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help="Optional SAM ViT checkpoint path (.pth); requires segment_anything + torch.",
    )
    parser.add_argument(
        "--sam-model-type",
        type=str,
        default="vit_b",
        choices=("vit_b", "vit_l", "vit_h"),
        help="SAM model type matching checkpoint (default: vit_b).",
    )
    parser.add_argument(
        "--sam-device",
        type=str,
        default=None,
        help="Torch device override, e.g. cuda or cpu (default: auto).",
    )

    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Write semi-transparent overlays to previews/",
    )
    parser.add_argument(
        "--mask-format",
        choices=("png", "tif"),
        default="png",
        help="Mask file format under masks/ (default: png).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.frame_step < 1:
        raise SystemExit("--frame-step must be >= 1")
    if args.random_sample is not None and args.random_sample < 1:
        raise SystemExit("--random-sample must be >= 1")

    paths = collect_tiff_files([Path(p).expanduser().resolve() for p in args.input_dirs])
    timestamps = [parse_timestamp_from_name(p) for p in paths]

    cfg = SegmentConfig(
        output_dir=args.output_dir.expanduser().resolve(),
        frame_step=args.frame_step,
        max_frames=args.max_frames,
        crop=args.crop,
        bg_mode=args.bg_mode,  # type: ignore[arg-type]
        bg_sample_count=max(3, args.bg_sample_count),
        mog_train_frames=max(3, args.mog_train_frames),
        mog_history=max(10, args.mog_history),
        mog_var_threshold=args.mog_var_threshold,
        mog_detect_shadows=not args.no_mog_shadows,
        mog_binary_threshold=max(1, args.mog_binary_threshold),
        use_mog_fg=args.combine_mog_and_residual,
        residual_percentile=min(99.95, max(50.0, args.residual_percentile)),
        open_iterations=max(0, args.open_iterations),
        close_iterations=max(0, args.close_iterations),
        min_area_px=max(0, args.min_area_px),
        erode_shrink_px=max(0, args.erode_shrink_px),
        keep_largest_only=args.keep_largest_only,
        temporal_median_window=max(1, args.temporal_median_window),
        align_ecc=args.align_ecc,
        ecc_translation_only=not args.ecc_euclidean,
        refine_grabcut=args.refine_grabcut,
        grabcut_iters=max(1, args.grabcut_iters),
        sam_checkpoint=args.sam_checkpoint.expanduser().resolve() if args.sam_checkpoint else None,
        sam_model_type=args.sam_model_type,
        sam_device=args.sam_device,
        save_preview=args.save_preview,
        mask_format=args.mask_format,  # type: ignore[arg-type]
        random_sample_n=args.random_sample,
        random_sample_seed=args.random_seed,
    )

    csv_path = run_segmentation_pipeline(paths, timestamps, cfg)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
