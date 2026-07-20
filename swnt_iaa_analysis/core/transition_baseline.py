"""Transition-ramp baseline correction (rate-of-change based, drift-free).

Lighting transitions add a fast, multi-scan step (a ramp) to the fluorescence
background; real biology varies gradually over a much larger timescale. This module
identifies the fast transition ramps by their rate of change, masks them, and
reconstructs a baseline as the slow component (rolling median over ~one light cycle).

Unlike additive step-stitching, the slow component cannot drift and cannot turn a
positive-definite quantity (intensity, peak area) negative: it is a median of the
data itself, so it is always bounded by the surrounding values.
"""

import numpy as np
import pandas as pd


def detect_transition_ramps(signal, slope_halfwindow=5, candidate_sigma=6.0,
                            min_displacement=250.0, min_rate_per_scan=30.0,
                            guard=2, level_window=6):
    """Detect lighting-transition ramps by rate of change.

    A ramp is a run of consecutive scans whose local slope exceeds the noise floor,
    kept only if the net level change across it is both large (``min_displacement``)
    and fast (``min_rate_per_scan`` per scan). The rate test — not an absolute
    duration cap — is what lets long early-experiment ramps still be recognised.

    Parameters
    ----------
    signal : array-like
        Reference channel (fluorescence). May contain NaN.
    slope_halfwindow : int
        Half-window k for the local slope: median(next k) - median(prev k).
    candidate_sigma : float
        Candidate threshold in robust MAD units of the slope distribution.
    min_displacement : float
        Minimum net level change (signal units) for a region to count.
    min_rate_per_scan : float
        Minimum sustained rate (|net| / duration, per scan) for a region to count.
    guard : int
        Scans skipped on each side of a ramp before measuring the before/after level.
    level_window : int
        Window (scans) for the robust before/after level medians.

    Returns
    -------
    list of (start, end)
        Inclusive index ranges (in the signal's own positional order) of detected ramps.
    """
    f = np.asarray(signal, dtype=float)
    n = len(f)
    k = int(slope_halfwindow)
    if n < 2 * k + 1:
        return []

    slope = np.full(n, np.nan)
    for i in range(k, n - k):
        slope[i] = np.median(f[i + 1:i + 1 + k]) - np.median(f[i - k:i])

    valid = ~np.isnan(slope)
    if valid.sum() == 0:
        return []
    sigma = np.median(np.abs(slope[valid] - np.median(slope[valid])))
    sigma = max(float(sigma), 1e-6)
    t_low = candidate_sigma * sigma

    cand = np.zeros(n, dtype=int)
    cand[valid & (slope > t_low)] = 1
    cand[valid & (slope < -t_low)] = -1

    # Merge contiguous same-sign candidates (allowing 1-scan gaps) into regions.
    regions = []
    i = 0
    while i < n:
        if cand[i] != 0:
            sign = cand[i]
            start = i
            gap = 0
            j = i
            while j + 1 < n and gap <= 1:
                if cand[j + 1] == sign:
                    j += 1
                    gap = 0
                elif cand[j + 1] == 0:
                    gap += 1
                    j += 1
                else:
                    break
            end = j - gap if gap > 0 else j
            regions.append((start, end))
            i = j + 1
        else:
            i += 1

    # Keep only fast + net-displacing ramps (rate, not absolute duration).
    kept = []
    for (a, b) in regions:
        before = f[max(0, a - guard - level_window):max(0, a - guard)]
        after = f[min(n, b + 1 + guard):min(n, b + 1 + guard + level_window)]
        if len(before) == 0 or len(after) == 0:
            continue
        net = np.median(after) - np.median(before)
        rate = abs(net) / (b - a + 1)
        if abs(net) > min_displacement and rate > min_rate_per_scan:
            kept.append((a, b))
    return kept


def fixed_stitch_baseline(signal, regions, drift_window_scans, guard=2, level_window=6,
                          mask_ramps=True):
    """Remove transition steps by stitching, then remove the accumulated drift.

    At each transition the (median-after - median-before) step is subtracted from all
    later scans, which flattens the lighting jump but makes the offset a cumulative
    staircase that ratchets (mornings outweigh evenings) and can sink the signal
    negative. The staircase is the sum of {local step corrections} + {slow accumulated
    drift}; we keep only the local corrections by subtracting the staircase's own slow
    component (its rolling median over ~one light cycle). This preserves the daily
    biology and removes the transition jumps without drifting below zero.

    Parameters
    ----------
    signal : array-like
        Channel to correct (already spike-cleaned), positional order matching ``regions``.
    regions : list of (start, end)
        Transition ramp ranges (inclusive), detected on the reference channel.
    drift_window_scans : int
        Window (scans) for estimating the accumulated drift (~one light cycle).
    guard : int
        Scans skipped on each side of a ramp before measuring the before/after level.
    level_window : int
        Window (scans) for the robust before/after level medians.
    mask_ramps : bool
        If True, the untrustworthy ramp scans are masked and linearly interpolated across.

    Returns
    -------
    np.ndarray
        Baseline-corrected signal: transition steps removed, drift removed, daily
        structure preserved.
    """
    y = np.asarray(signal, dtype=float).copy()
    n = len(y)

    offset = np.zeros(n)
    for (a, b) in regions:
        before = y[max(0, a - guard - level_window):max(0, a - guard)]
        after = y[min(n, b + 1 + guard):min(n, b + 1 + guard + level_window)]
        if len(before) == 0 or len(after) == 0:
            continue
        offset[b + 1:] += np.median(after) - np.median(before)

    drift_window_scans = int(max(3, drift_window_scans))
    min_periods = max(1, drift_window_scans // 3)
    drift = pd.Series(offset).rolling(drift_window_scans, center=True,
                                      min_periods=min_periods).median().to_numpy()
    corrected = y - (offset - drift)

    if mask_ramps and len(regions) > 0:
        for (a, b) in regions:
            corrected[a:b + 1] = np.nan
        corrected = pd.Series(corrected).interpolate(limit_direction="both").to_numpy()

    return corrected


def _region_levels(y, a, b, guard, level_window):
    """Robust level just before and just after a region, skipping the ramp itself."""
    n = len(y)
    before = y[max(0, a - guard - level_window):max(0, a - guard)]
    after = y[min(n, b + 1 + guard):min(n, b + 1 + guard + level_window)]
    before = before[np.isfinite(before)]
    after = after[np.isfinite(after)]
    if len(before) == 0 or len(after) == 0:
        return None, None
    return float(np.median(before)), float(np.median(after))


def _offset_staircase(y, regions, guard, level_window):
    """Cumulative step function: each region's (after - before) applied to all later scans."""
    offset = np.zeros(len(y))
    for (a, b) in regions:
        before, after = _region_levels(y, a, b, guard, level_window)
        if before is None:
            continue
        offset[b + 1:] += (after - before)
    return offset


def classify_regions(signal, regions, pairing_window_scans, oneoff_min_fold,
                     guard=2, level_window=6):
    """Split detected regions into recurring and one-off permanent steps.

    The test is *persistence*, not pairing: compare the level just before the step with the
    level one full cycle (``pairing_window_scans``) later.

    A lighting transition reverts within the cycle -- lights come on in the morning and go off
    again in the evening -- so a day later the level is back where it started. Its correction
    must be drift-removed, otherwise the offsets ratchet and the signal walks away from zero.

    A focus / sample-movement event does not revert; the shift is still there a cycle later.
    Drift-removing such a step cancels the correction entirely, so it is applied permanently.

    Pairing steps against each other by magnitude was tried first and proved fragile: morning
    and evening steps are often asymmetric, so a large morning ramp could fail to find a
    "comparable" partner and be misread as permanent, ratcheting the signal negative.

    Returns
    -------
    (recurring, oneoff) : two lists of (start, end) regions
    """
    y = np.asarray(signal, dtype=float)
    n = len(y)
    horizon = int(max(1, pairing_window_scans))

    recurring, oneoff = [], []
    for (a, b) in regions:
        before, _ = _region_levels(y, a, b, guard, level_window)
        # Level one full cycle after the step.
        start = min(n, b + horizon)
        later = y[start:min(n, start + level_window)]
        later = later[np.isfinite(later)]

        if before is None or before <= 0 or len(later) == 0:
            recurring.append((a, b))       # cannot judge -> treat as recurring (safe)
            continue

        persisted = abs(float(np.median(later)) / before - 1.0)
        if persisted >= oneoff_min_fold:
            oneoff.append((a, b))
        else:
            recurring.append((a, b))
    return recurring, oneoff


def stitch_classified(signal, recurring, oneoff, drift_window_scans,
                      guard=2, level_window=6, mask_ramps=False):
    """Remove recurring steps (drift-removed) and one-off steps (applied permanently).

    Recurring offsets have their own slow component subtracted, which keeps the daily
    corrections local and stops the cumulative ratchet. One-off offsets are applied as-is so
    a genuine permanent level shift is actually removed.
    """
    y = np.asarray(signal, dtype=float).copy()

    offset_recurring = _offset_staircase(y, recurring, guard, level_window)
    window = int(max(3, drift_window_scans))
    min_periods = max(1, window // 3)
    drift = pd.Series(offset_recurring).rolling(
        window, center=True, min_periods=min_periods).median().to_numpy()

    offset_oneoff = _offset_staircase(y, oneoff, guard, level_window)
    corrected = y - (offset_recurring - drift) - offset_oneoff

    if mask_ramps:
        for (a, b) in list(recurring) + list(oneoff):
            corrected[a:b + 1] = np.nan
        corrected = pd.Series(corrected).interpolate(limit_direction="both").to_numpy()
    return corrected


def stitch_channel_gated(signal, recurring, oneoff, drift_window_scans, gate_pct,
                         multiplicative=True, guard=2, level_window=6):
    """Correct a channel only at regions where *it* actually steps.

    The transition locations come from the fluorescence, which is the most sensitive channel,
    but the Raman peak areas do not step at ordinary lighting transitions — only at events that
    change the collection geometry (focus, sample movement). Applying a correction at every
    fluorescence transition would inject noise into an otherwise flat channel, so each region is
    kept only if this channel's own fold-change clears ``gate_pct``.

    ``multiplicative`` works in log space, which suits a positive-definite quantity such as a
    peak area: the correction scales rather than shifts, so the result cannot go negative.
    """
    y = np.asarray(signal, dtype=float)

    def significant(region):
        before, after = _region_levels(y, region[0], region[1], guard, level_window)
        if before is None or before <= 0:
            return False
        return abs(after / before - 1.0) >= gate_pct

    keep_recurring = [r for r in recurring if significant(r)]
    keep_oneoff = [r for r in oneoff if significant(r)]

    if not multiplicative:
        return stitch_classified(y, keep_recurring, keep_oneoff, drift_window_scans,
                                 guard=guard, level_window=level_window)

    positive = np.isfinite(y) & (y > 0)
    if not positive.any():
        return y.copy()
    y_log = np.full(len(y), np.nan)
    y_log[positive] = np.log(y[positive])
    corrected_log = stitch_classified(y_log, keep_recurring, keep_oneoff, drift_window_scans,
                                      guard=guard, level_window=level_window)
    out = np.exp(corrected_log)
    out[~np.isfinite(out)] = np.nan
    return out


def snap_event_index(signal, index, search_scans, slope_halfwindow):
    """Move a declared event index to the nearby scan carrying the largest local step.

    Declared times are approximate (read off a plot or a lab note), so the correction is
    anchored to the actual step within +/- ``search_scans`` rather than the literal timestamp.
    """
    y = np.asarray(signal, dtype=float)
    n = len(y)
    k = max(1, int(slope_halfwindow))
    lo = max(k, int(index) - int(search_scans))
    hi = min(n - k - 1, int(index) + int(search_scans))
    if hi <= lo:
        return int(index)
    best, best_slope = int(index), -np.inf
    for i in range(lo, hi + 1):
        before = y[i - k:i]
        after = y[i + 1:i + 1 + k]
        before = before[np.isfinite(before)]
        after = after[np.isfinite(after)]
        if len(before) == 0 or len(after) == 0:
            continue
        slope = abs(float(np.median(after)) - float(np.median(before)))
        if slope > best_slope:
            best, best_slope = i, slope
    return best


def apply_declared_steps(signal, indices, guard=2, level_window=6, multiplicative=False):
    """Permanently remove a level step at each declared event index.

    Declared events -- a refocus, the sample being knocked, a sensor re-seat -- shift the level
    permanently. Unlike the daily lighting transitions their correction must NOT be drift-removed,
    since subtracting the offset's slow component would cancel a permanent step entirely.

    Automatic recurring-vs-one-off classification was attempted and abandoned (see
    ``classify_regions``); declaring these rare events explicitly is reliable and cannot ratchet.

    ``multiplicative`` works in log space, which suits positive-definite peak areas: the
    correction scales rather than shifts, so the result cannot go negative.
    """
    y = np.asarray(signal, dtype=float).copy()
    indices = [int(i) for i in indices]
    if not indices:
        return y

    if multiplicative:
        positive = np.isfinite(y) & (y > 0)
        if not positive.any():
            return y
        work = np.full(len(y), np.nan)
        work[positive] = np.log(y[positive])
    else:
        work = y

    for idx in sorted(indices):
        if idx <= 0 or idx >= len(work):
            continue
        before, after = _region_levels(work, idx, idx, guard, level_window)
        if before is None:
            continue
        work[idx:] -= (after - before)

    if multiplicative:
        out = np.exp(work)
        out[~np.isfinite(out)] = np.nan
        return out
    return work


def slow_component_baseline(signal, regions, window_scans, min_periods=None):
    """Mask transition ramps, then return the slow component (centered rolling median).

    Parameters
    ----------
    signal : array-like
        Channel to correct (positional order matching ``regions``).
    regions : list of (start, end)
        Ramp ranges to mask (inclusive), typically detected on the reference channel.
    window_scans : int
        Rolling-median window in scans (~one light cycle).
    min_periods : int, optional
        Minimum non-NaN points required in a window; defaults to window_scans // 3.

    Returns
    -------
    np.ndarray
        Baseline-corrected signal, drift-free and >= 0 wherever the raw data is.
        Masked ramp scans are NaN.
    """
    y = np.asarray(signal, dtype=float).copy()
    for (a, b) in regions:
        y[a:b + 1] = np.nan

    window_scans = int(max(3, window_scans))
    if min_periods is None:
        min_periods = max(1, window_scans // 3)
    slow = pd.Series(y).rolling(window_scans, center=True, min_periods=min_periods).median()
    return slow.to_numpy()
