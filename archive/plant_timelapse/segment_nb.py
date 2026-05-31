"""
Tier-1 segmentation: temporal background model + residual threshold + morphology +
optional ECC alignment, temporal median, GrabCut, and optional SAM refinement.

Assumes a mostly fixed camera so static background estimation is meaningful.
Probe hardware: tighten with morphological erosion or exclude via crop/heuristics using CLI knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from scipy import ndimage

try:
    import cv2  # type: ignore[import-untyped]
except ImportError as exc:  # pragma: no cover - runtime hint
    raise ImportError(
        "segment_nb requires OpenCV. Install with: pip install opencv-python-headless"
    ) from exc

BgMode = Literal["median", "mog2"]


def load_gray_f32(path: Path) -> np.ndarray:
    """Load single-channel grayscale as float32, shape (H, W)."""
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:
        r = arr[..., 0].astype(np.float32)
        g = arr[..., 1].astype(np.float32) if arr.shape[2] > 1 else r
        b = arr[..., 2].astype(np.float32) if arr.shape[2] > 2 else r
        gray = 0.299 * r + 0.587 * g + 0.114 * b
    else:
        gray = arr.astype(np.float32)
    return gray


def crop_or_none(arr: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return arr
    y0, y1, x0, x1 = crop
    return arr[y0:y1, x0:x1]


def uncrop_into(shape_hw: tuple[int, int], small: np.ndarray, crop: tuple[int, int, int, int] | None) -> np.ndarray:
    if crop is None:
        return small
    H, W = shape_hw
    out = np.zeros((H, W), dtype=small.dtype)
    y0, y1, x0, x1 = crop
    out[y0:y1, x0:x1] = small
    return out


def to_uint8_stretch(
    gray: np.ndarray,
    p_low: float = 2.0,
    p_high: float = 98.0,
) -> np.ndarray:
    v = gray[np.isfinite(gray)].ravel()
    if v.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    lo, hi = float(np.percentile(v, p_low)), float(np.percentile(v, p_high))
    if hi <= lo:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
    if hi <= lo:
        return np.zeros(gray.shape, dtype=np.uint8)
    clipped = np.clip((gray.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8)


def ecc_warp_matrix(
    ref_u8: np.ndarray,
    moving_u8: np.ndarray,
    motion_translation_only: bool,
) -> np.ndarray | None:
    """Return 2×3 warp (moving → ref frame) or None on failure."""
    warp_mode = cv2.MOTION_TRANSLATION if motion_translation_only else cv2.MOTION_EUCLIDEAN
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5)
    try:
        cv2.findTransformECC(
            ref_u8.astype(np.uint8),
            moving_u8.astype(np.uint8),
            warp_matrix,
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
    except cv2.error:
        return None
    return warp_matrix


def warp_with_matrix(image: np.ndarray, warp_matrix: np.ndarray | None, out_wh: tuple[int, int]) -> np.ndarray:
    if warp_matrix is None:
        return image
    w, h = out_wh
    is_float = np.issubdtype(image.dtype, np.floating)
    out = cv2.warpAffine(
        image.astype(np.float32),
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )
    if is_float:
        return out.astype(np.float32)
    return np.clip(np.round(out), 0, 255).astype(image.dtype)


def train_mog2_on_u8_frames(
    u8_frames: list[np.ndarray],
    history: int,
    var_threshold: float,
    detect_shadows: bool,
) -> tuple[cv2.BackgroundSubtractorMOG2, np.ndarray | None]:
    bs = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold, detectShadows=detect_shadows)
    for fr in u8_frames:
        bs.apply(fr)
    bg = bs.getBackgroundImage()
    bg_f32 = bg.astype(np.float32) if bg is not None else None
    return bs, bg_f32


def foreground_from_mog_and_residual(
    I_f32: np.ndarray,
    background_f32: np.ndarray | None,
    mog_bs: cv2.BackgroundSubtractorMOG2 | None,
    residual_percentile: float,
    mog_binary_threshold: int,
    combine_mog_and_residual: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (binary uint8 mask 0/255, residual magnitude float).
    MOG foreground is optional; residual uses |I - B| when background is supplied.
    """
    mog_bin: np.ndarray | None = None
    if mog_bs is not None:
        i_u8 = to_uint8_stretch(I_f32)
        fg = mog_bs.apply(i_u8, learningRate=0)
        mog_bin = np.where(fg >= mog_binary_threshold, 255, 0).astype(np.uint8)

    if background_f32 is not None:
        resid = np.abs(I_f32.astype(np.float32) - background_f32.astype(np.float32))
        vals = resid[np.isfinite(resid)].ravel()
        thr = float(np.percentile(vals, residual_percentile)) if vals.size else 1.0
        bin_residual = np.where(resid >= thr, 255, 0).astype(np.uint8)
    else:
        resid = np.abs(I_f32.astype(np.float32))
        bin_residual = np.zeros(I_f32.shape, dtype=np.uint8)

    if mog_bin is None:
        bin_out = bin_residual
    elif combine_mog_and_residual and background_f32 is not None:
        bin_out = mog_bin.copy()
        cv2.bitwise_and(bin_out, bin_residual, bin_out)
    else:
        bin_out = mog_bin
    return bin_out, resid


def morphology_and_cleanup(
    mask_u8: np.ndarray,
    open_iterations: int,
    close_iterations: int,
    min_area_px: int,
    erode_shrink_px: int,
) -> np.ndarray:
    """Binary mask cleanup; erosion helps drop thin probe appendages."""
    m = mask_u8.copy()
    if open_iterations > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=open_iterations)
    if close_iterations > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=close_iterations)
    if erode_shrink_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.erode(m, k, iterations=erode_shrink_px)
    if min_area_px > 0:
        num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
        out = np.zeros_like(m)
        for i in range(1, num):
            if stats[i, cv2.CC_STAT_AREA] >= min_area_px:
                out[labels == i] = 255
        m = out
    # fill small holes inside plant
    m_bool = ndimage.binary_fill_holes(m > 0)
    return (m_bool.astype(np.uint8) * 255)


def keep_largest_component_mask(mask_u8: np.ndarray) -> np.ndarray:
    """Keep single largest 8-connected foreground component; re-fill holes after selection."""
    m_bool = ndimage.binary_fill_holes(mask_u8 > 127)
    m = (m_bool.astype(np.uint8) * 255)
    if not np.any(m > 127):
        return m
    num, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return m
    best_i = max(range(1, num), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))
    out = np.zeros_like(m)
    out[labels == best_i] = 255
    m2 = ndimage.binary_fill_holes(out > 127)
    return (m2.astype(np.uint8) * 255)


def temporal_median_masks(masks: list[np.ndarray], window: int) -> list[np.ndarray]:
    """Odd window; sliding median filter per-pixel on binary stacks."""
    if window <= 1 or len(masks) < 2:
        return list(masks)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    stack = np.stack([m > 0 for m in masks], axis=0).astype(np.uint8)
    T = stack.shape[0]
    smoothed: list[np.ndarray] = []
    for t in range(T):
        a = max(0, t - pad)
        b = min(T, t + pad + 1)
        chunk = stack[a:b]
        med = (np.median(chunk, axis=0) >= 0.5).astype(np.uint8) * 255
        smoothed.append(med)
    return smoothed


def grabcut_refine(gray_crop: np.ndarray, coarse_binary: np.ndarray, iterations: int) -> np.ndarray:
    rgb = cv2.cvtColor(to_uint8_stretch(gray_crop), cv2.COLOR_GRAY2RGB)

    coarse = coarse_binary.astype(np.uint8)
    if not np.any(coarse > 127):
        return coarse

    gc = np.full(coarse.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_fg = cv2.erode(coarse, ker, iterations=3)
    sure_bg_patch = cv2.dilate(coarse, ker, iterations=6)
    sure_bg = 255 - sure_bg_patch

    gc[np.where(coarse > 0)] = cv2.GC_PR_FGD
    gc[sure_fg > 127] = cv2.GC_FGD
    gc[sure_bg > 127] = cv2.GC_BGD

    bg_m = np.zeros((1, 65), dtype=np.float64)
    fg_m = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(rgb, gc, rect=(-1, -1, rgb.shape[1], rgb.shape[0]), bgdModel=bg_m, fgdModel=fg_m, iterCount=max(3, iterations), mode=cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return coarse

    fg_like = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return morphology_and_cleanup(fg_like, open_iterations=0, close_iterations=1, min_area_px=0, erode_shrink_px=0)


def sam_refine_bbox(
    gray_crop_u8_rgb: np.ndarray,
    coarse_binary: np.ndarray,
    checkpoint_path: Path,
    model_type: str,
    device: str | None,
) -> np.ndarray:
    """SAM mask refinement using bbox from coarse mask."""
    try:
        from segment_anything import SamPredictor, sam_model_registry
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "SAM requires segment_anything and torch. "
            "See https://github.com/facebookresearch/segment-anything"
        ) from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    image = gray_crop_u8_rgb
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    predictor.set_image(image.astype(np.uint8))

    ys, xs = np.where(coarse_binary > 127)
    if ys.size == 0:
        return coarse_binary
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    inp_box = np.array([x0, y0, x1, y1], dtype=np.float32)

    masks_sam, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=inp_box, multimask_output=True)
    best = masks_sam[np.argmax(scores.astype(np.float32))]
    out = (best.astype(np.uint8) * 255)
    return out


@dataclass
class SegmentConfig:
    output_dir: Path
    frame_step: int
    max_frames: int | None
    crop: tuple[int, int, int, int] | None
    bg_mode: BgMode
    bg_sample_count: int
    mog_train_frames: int
    mog_history: int
    mog_var_threshold: float
    mog_detect_shadows: bool
    mog_binary_threshold: int
    use_mog_fg: bool
    residual_percentile: float
    open_iterations: int
    close_iterations: int
    min_area_px: int
    erode_shrink_px: int
    temporal_median_window: int
    align_ecc: bool
    ecc_translation_only: bool
    refine_grabcut: bool
    grabcut_iters: int
    sam_checkpoint: Path | None
    sam_model_type: str
    sam_device: str | None
    save_preview: bool
    mask_format: Literal["png", "tif"]
    keep_largest_only: bool
    random_sample_n: int | None
    random_sample_seed: int | None


def run_segmentation_pipeline(
    paths: list[Path],
    timestamps: list[datetime | None],
    cfg: SegmentConfig,
) -> Path:
    """
    Process ``paths`` in order; write masks and ``plant_area.csv`` under ``cfg.output_dir``.
    Returns path to CSV.
    """
    n = len(paths)
    if n == 0:
        raise ValueError("No input paths.")
    if len(timestamps) != n:
        raise ValueError("timestamps length must match paths.")

    selection = list(range(0, n, cfg.frame_step))
    if cfg.max_frames is not None:
        selection = selection[: cfg.max_frames]

    if cfg.random_sample_n is not None:
        if cfg.random_sample_n < 1:
            raise ValueError("random_sample_n must be >= 1 when set.")
        if len(selection) == 0:
            raise ValueError("No frames selected after frame-step / max-frames.")
        take = min(cfg.random_sample_n, len(selection))
        rng = np.random.default_rng(cfg.random_sample_seed)
        picks = rng.choice(len(selection), size=take, replace=False)
        selection = [selection[int(i)] for i in np.sort(picks)]

    out_masks = cfg.output_dir / "masks"
    out_prev = cfg.output_dir / "previews"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    if cfg.save_preview:
        out_prev.mkdir(parents=True, exist_ok=True)

    full_idx = selection
    ref_pidx = full_idx[0]
    H_full, W_full = load_gray_f32(paths[ref_pidx]).shape

    def load_gray_cropped(pid: int) -> np.ndarray:
        return crop_or_none(load_gray_f32(paths[pid]), cfg.crop)

    ref_gray_cropped = load_gray_cropped(ref_pidx)
    ref_shape = ref_gray_cropped.shape
    ref_u8_align = to_uint8_stretch(ref_gray_cropped) if cfg.align_ecc else None

    def align_if_needed(g_crop: np.ndarray) -> np.ndarray:
        g32 = g_crop.astype(np.float32)
        if not cfg.align_ecc or ref_u8_align is None:
            return g32
        mov_u8 = to_uint8_stretch(g32)
        mat = ecc_warp_matrix(ref_u8_align, mov_u8, cfg.ecc_translation_only)
        h_c, w_c = ref_gray_cropped.shape[:2]
        return warp_with_matrix(g32, mat, (w_c, h_c))

    bg_sample_positions = np.linspace(0, len(full_idx) - 1, num=min(cfg.bg_sample_count, len(full_idx)), dtype=int)
    bg_sample_frames = [full_idx[int(i)] for i in bg_sample_positions]

    mog_bs: cv2.BackgroundSubtractorMOG2 | None = None
    background: np.ndarray | None

    if cfg.bg_mode == "median":
        aligned_samples = [align_if_needed(load_gray_cropped(i)) for i in bg_sample_frames]
        background = np.median(np.stack(aligned_samples, axis=0), axis=0).astype(np.float32)
    elif cfg.bg_mode == "mog2":
        train_n = min(cfg.mog_train_frames, len(full_idx))
        train_pids = [full_idx[int(j)] for j in np.linspace(0, len(full_idx) - 1, num=train_n, dtype=int)]
        u8_train = [to_uint8_stretch(align_if_needed(load_gray_cropped(p))) for p in train_pids]
        mog_bs, bg_img = train_mog2_on_u8_frames(
            u8_train,
            history=cfg.mog_history,
            var_threshold=cfg.mog_var_threshold,
            detect_shadows=cfg.mog_detect_shadows,
        )
        background = bg_img
    else:  # pragma: no cover
        raise ValueError(cfg.bg_mode)

    if background is not None and background.shape != ref_shape:
        background = cv2.resize(background, (ref_shape[1], ref_shape[0]), interpolation=cv2.INTER_LINEAR)

    masks_crop: list[np.ndarray] = []
    for pidx in full_idx:
        I = align_if_needed(load_gray_cropped(pidx))

        diff_bin, _resid = foreground_from_mog_and_residual(
            I,
            background,
            mog_bs,
            residual_percentile=cfg.residual_percentile,
            mog_binary_threshold=cfg.mog_binary_threshold,
            combine_mog_and_residual=cfg.use_mog_fg,
        )

        cleaned = morphology_and_cleanup(
            diff_bin,
            open_iterations=cfg.open_iterations,
            close_iterations=cfg.close_iterations,
            min_area_px=cfg.min_area_px,
            erode_shrink_px=cfg.erode_shrink_px,
        )

        if cfg.refine_grabcut:
            cleaned = grabcut_refine(I.astype(np.float32), cleaned, iterations=cfg.grabcut_iters)

        if cfg.sam_checkpoint is not None:
            img_rgb = cv2.cvtColor(to_uint8_stretch(I.astype(np.float32)), cv2.COLOR_GRAY2RGB)
            cleaned = sam_refine_bbox(
                img_rgb,
                cleaned,
                cfg.sam_checkpoint,
                cfg.sam_model_type,
                cfg.sam_device,
            )

        if cfg.keep_largest_only:
            cleaned = keep_largest_component_mask(cleaned)

        masks_crop.append(cleaned)

    if cfg.temporal_median_window > 1:
        masks_crop = temporal_median_masks(masks_crop, cfg.temporal_median_window)

    rows: list[str] = []
    header = "timestamp_iso,path,stem,area_px,area_fraction\n"
    total_px = float(ref_shape[0] * ref_shape[1])

    for sel_i, pidx in enumerate(full_idx):
        m = masks_crop[sel_i]
        area = int(np.sum(m > 127))
        frac = area / total_px if total_px else 0.0
        ts = timestamps[pidx]
        ts_s = ts.isoformat(sep=" ") if ts is not None else ""
        path_s = str(paths[pidx]).replace("\\", "/")
        stem = paths[pidx].stem
        rows.append(f'"{ts_s}","{path_s}","{stem}",{area},{frac:.8f}\n')

        m_full = uncrop_into((H_full, W_full), m, cfg.crop)
        out_name = f"{stem}_mask.{cfg.mask_format}"
        out_path = out_masks / out_name
        cv2.imwrite(str(out_path), m_full)

        if cfg.save_preview:
            vis = to_uint8_stretch(load_gray_cropped(pidx))
            ov = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            red = np.zeros_like(ov)
            red[:, :, 2] = m
            ov = cv2.addWeighted(ov, 0.65, red, 0.35, 0)
            prev_path = out_prev / f"{stem}_preview.png"
            cv2.imwrite(str(prev_path), uncrop_into((H_full, W_full), ov, cfg.crop))

    csv_path = cfg.output_dir / "plant_area.csv"
    csv_path.write_text(header + "".join(rows), encoding="utf-8")
    return csv_path
