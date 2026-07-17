"""Turn DeepLabCut predictions into a stem-height curve.

DLC analyzes the CROPPED mp4, so predicted (x, y) are in cropped-frame pixels.
Stem height metrics from keypoint pairs:
  - height_px: vertical distance (base_y - meristem_y); default growth proxy.
  - height_eucl_px: Euclidean distance between base and meristem; useful if the
    stem tilts, but more sensitive to horizontal keypoint jitter.

Video frame i corresponds to the i-th time-sorted TIFF (same `step` used to build
the video), which lets us attach real timestamps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import ROI
from .frame_export import list_tiffs
from .frame_select import parse_timestamp


def load_dlc_predictions(h5_or_csv) -> pd.DataFrame:
    """Load a DLC output (.h5 preferred, .csv fallback) into a per-bodypart frame.

    Returns a DataFrame indexed by frame number with a MultiIndex column
    (bodypart, coord) where coord in {x, y, likelihood}.
    """
    path = Path(h5_or_csv)
    if path.suffix.lower() in (".h5", ".hdf5"):
        df = pd.read_hdf(path)
        # DLC columns are (scorer, bodypart, coord); drop the scorer level.
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 3:
            df.columns = df.columns.droplevel(0)
        return df
    # CSV: first 3 rows are scorer / bodyparts / coords headers.
    raw = pd.read_csv(path, header=[1, 2], index_col=0)
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    return raw


def _series(df: pd.DataFrame, bodypart: str, coord: str) -> pd.Series:
    return df[(bodypart, coord)].astype(float)


def savgol_smooth(series: pd.Series, window: int = 11, poly: int = 2) -> pd.Series:
    """Savitzky-Golay smoothing with safe window handling for short series."""
    from scipy.signal import savgol_filter

    n = len(series)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 3:
        return series.copy()
    if w % 2 == 0:
        w -= 1
    if poly >= w:
        poly = w - 1
    return pd.Series(savgol_filter(series.to_numpy(), w, poly), index=series.index)


def enforce_monotonic(series: pd.Series) -> pd.Series:
    """Growth prior: height should not decrease over time (running max)."""
    return series.cummax()


def _interp_low_conf(
    series: pd.Series,
    low_mask: pd.Series,
    interpolate_limit: Optional[int],
) -> pd.Series:
    """Blank low-confidence samples and interpolate short gaps."""
    cleaned = series.copy()
    if not low_mask.any():
        return cleaned
    cleaned.loc[low_mask] = np.nan
    interp_kwargs: dict = {"limit_direction": "both"}
    if interpolate_limit is not None:
        interp_kwargs["limit"] = interpolate_limit
    return cleaned.interpolate(**interp_kwargs)


def destep_series(series: pd.Series, jump_threshold: float) -> pd.Series:
    """Remove step discontinuities by shifting subsequent samples.

    When ``|y[i] - y[i-1]|`` exceeds ``jump_threshold``, apply a cumulative
    offset so the series continues from the pre-jump level (background-style
    step correction). NaNs are preserved and do not reset the offset.
    """
    if jump_threshold <= 0:
        return series.copy()

    y = series.to_numpy(dtype=float)
    out = y.copy()
    offset = 0.0
    for i in range(1, len(y)):
        if not np.isfinite(y[i]):
            out[i] = np.nan
            continue
        prev = out[i - 1]
        if not np.isfinite(prev):
            out[i] = y[i] + offset
            continue
        candidate = y[i] + offset
        step = candidate - prev
        if abs(step) > jump_threshold:
            offset -= step
            candidate = y[i] + offset
        out[i] = candidate
    return pd.Series(out, index=series.index)


def build_height_table(
    predictions,
    dataset_dir=None,
    step: int = 1,
    base: str = "base",
    meristem: str = "meristem",
    roi=ROI,
    likelihood_floor: float = 0.0,
    base_likelihood_floor: Optional[float] = None,
    meristem_likelihood_floor: Optional[float] = None,
    interpolate_limit: Optional[int] = None,
    destep_jump_px: Optional[float] = None,
    smooth_window: int = 11,
    monotonic: bool = False,
) -> pd.DataFrame:
    """Assemble a tidy per-frame table with raw + processed stem height.

    Columns: frame, [timestamp], raw base/meristem x/y + likelihoods,
    base_x/y_interp, meristem_x/y_interp, [base/meristem_y_refined],
    height_px_raw, height_px, height_eucl_px, height_valid, height_smooth,
    height_eucl_smooth, [height_monotonic].

    Per-bodypart likelihood floors default to ``likelihood_floor`` when unset.
    Low-confidence coordinates are interpolated separately per keypoint.
    Optional ``destep_jump_px`` removes leftover step artifacts in interpolated
    y coordinates by cumulatively shifting subsequent frames.
    """
    df = load_dlc_predictions(predictions)
    x0, y0 = (roi[0], roi[1]) if roi is not None else (0, 0)

    out = pd.DataFrame(index=df.index)
    out.index.name = "frame"
    for bp in (base, meristem):
        out[f"{bp}_x"] = _series(df, bp, "x") + x0
        out[f"{bp}_y"] = _series(df, bp, "y") + y0
        out[f"{bp}_likelihood"] = _series(df, bp, "likelihood")

    # Raw heights from DLC coordinates (before cleaning).
    out["height_px_raw"] = out["base_y"] - out["meristem_y"]
    out["height_eucl_px_raw"] = np.hypot(
        out["base_x"] - out["meristem_x"], out["base_y"] - out["meristem_y"]
    )

    base_floor = likelihood_floor if base_likelihood_floor is None else base_likelihood_floor
    mer_floor = likelihood_floor if meristem_likelihood_floor is None else meristem_likelihood_floor
    low_base = out[f"{base}_likelihood"] < base_floor if base_floor > 0 else pd.Series(False, index=out.index)
    low_mer = out[f"{meristem}_likelihood"] < mer_floor if mer_floor > 0 else pd.Series(False, index=out.index)
    out["height_valid"] = ~(low_base | low_mer)

    out["base_x_interp"] = _interp_low_conf(out["base_x"], low_base, interpolate_limit)
    out["base_y_interp"] = _interp_low_conf(out["base_y"], low_base, interpolate_limit)
    out["meristem_x_interp"] = _interp_low_conf(out["meristem_x"], low_mer, interpolate_limit)
    out["meristem_y_interp"] = _interp_low_conf(out["meristem_y"], low_mer, interpolate_limit)

    base_y_used = out["base_y_interp"]
    meristem_y_used = out["meristem_y_interp"]
    if destep_jump_px is not None and destep_jump_px > 0:
        out["base_y_refined"] = destep_series(out["base_y_interp"], destep_jump_px)
        out["meristem_y_refined"] = destep_series(out["meristem_y_interp"], destep_jump_px)
        base_y_used = out["base_y_refined"]
        meristem_y_used = out["meristem_y_refined"]

    out["height_px"] = base_y_used - meristem_y_used
    out["height_eucl_px"] = np.hypot(
        out["base_x_interp"] - out["meristem_x_interp"],
        base_y_used - meristem_y_used,
    )

    # Smooth using forward-filled gaps so Savitzky-Golay never sees NaN.
    smooth_input = out["height_px"].ffill().bfill()
    out["height_smooth"] = savgol_smooth(smooth_input, window=smooth_window)
    smooth_eucl = out["height_eucl_px"].ffill().bfill()
    out["height_eucl_smooth"] = savgol_smooth(smooth_eucl, window=smooth_window)
    if monotonic:
        out["height_monotonic"] = enforce_monotonic(out["height_smooth"])

    if dataset_dir is not None:
        tiffs = list_tiffs(dataset_dir)[::step]
        ts = [parse_timestamp(p) for p in tiffs]
        # align by position; DLC frame index is 0..n-1 in video order
        pos = out.index.to_numpy()
        out.insert(0, "timestamp", [ts[i] if 0 <= i < len(ts) else None for i in pos])

    return out.reset_index()


def build_height_table_from_labels(
    labeled_dir,
    dataset_dir,
    dataset: str,
    step: int = 1,
    base: str = "base",
    meristem: str = "meristem",
    smooth_window: int = 11,
    interpolate_gaps: bool = True,
    max_frame: Optional[int] = None,
    exclude_frames: Optional[set[int]] = None,
) -> pd.DataFrame:
    """Stem-height table from manual DLC labels (no model predictions).

    Uses ``frameXXX.png`` rows in ``CollectedData_*.csv`` aligned to video frame
    index ``XXX``. Other filenames (e.g. ``img*.png`` from kmeans) are ignored for
    the timeline but remain available for future training.

    Returns one row per video frame (optionally truncated with ``max_frame``).
    Unlabeled frames have NaN keypoints/heights. When ``interpolate_gaps`` is
    True, gaps are linearly interpolated and Savitzky–Golay smoothing is applied.
    """
    import re

    labeled_dir = Path(labeled_dir)
    matches = sorted(labeled_dir.glob("CollectedData_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No CollectedData_*.csv under {labeled_dir}")
    lab = pd.read_csv(matches[0], header=[0, 1, 2], index_col=[0, 1, 2])

    # Drop scorer level if present -> (bodypart, coord)
    if isinstance(lab.columns, pd.MultiIndex) and lab.columns.nlevels == 3:
        lab.columns = lab.columns.droplevel(0)

    tiffs = list_tiffs(dataset_dir)[::step]
    n = len(tiffs)
    if max_frame is not None:
        n = min(n, int(max_frame) + 1)
    exclude = exclude_frames or set()

    out = pd.DataFrame({"frame": np.arange(n)})
    out["timestamp"] = [parse_timestamp(p) for p in tiffs[:n]]
    for col in (
        f"{base}_x",
        f"{base}_y",
        f"{meristem}_x",
        f"{meristem}_y",
    ):
        out[col] = np.nan
    out["labeled"] = False

    frame_re = re.compile(r"^frame(\d+)\.png$", re.IGNORECASE)
    for idx in lab.index:
        fn = Path(str(idx[-1] if isinstance(idx, tuple) else idx)).name
        m = frame_re.match(fn)
        if not m:
            continue
        fi = int(m.group(1))
        if not 0 <= fi < n or fi in exclude:
            continue
        row = lab.loc[[idx]].iloc[0]
        try:
            bx = float(row[(base, "x")])
            by = float(row[(base, "y")])
            mx = float(row[(meristem, "x")])
            my = float(row[(meristem, "y")])
        except Exception:
            continue
        if not all(np.isfinite(v) for v in (bx, by, mx, my)):
            continue
        out.loc[fi, f"{base}_x"] = bx
        out.loc[fi, f"{base}_y"] = by
        out.loc[fi, f"{meristem}_x"] = mx
        out.loc[fi, f"{meristem}_y"] = my
        out.loc[fi, "labeled"] = True

    out["height_px"] = out[f"{base}_y"] - out[f"{meristem}_y"]
    out["height_eucl_px"] = np.hypot(
        out[f"{base}_x"] - out[f"{meristem}_x"],
        out[f"{base}_y"] - out[f"{meristem}_y"],
    )
    out["height_valid"] = out["labeled"]

    if interpolate_gaps:
        for col in (
            f"{base}_x",
            f"{base}_y",
            f"{meristem}_x",
            f"{meristem}_y",
            "height_px",
            "height_eucl_px",
        ):
            out[f"{col}_interp"] = out[col].interpolate(limit_direction="both")
        out["height_smooth"] = savgol_smooth(
            out["height_px_interp"].ffill().bfill(), window=smooth_window
        )
        out["height_eucl_smooth"] = savgol_smooth(
            out["height_eucl_px_interp"].ffill().bfill(), window=smooth_window
        )

    return out
