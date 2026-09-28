# Archived code

Superseded or unused scripts kept for reference. Not maintained.

The tables below under `scripts/` and `plant_timelapse/` list files kept here as
actual files. **Deleted (not kept here)** lists code removed outright — recover it
from git history when needed.

## `scripts/`

| File | Was | Replaced by |
|------|-----|-------------|
| `plant_contour.py` | Canny-edge prototype | `plant_timelapse/plant_contour_v3.py` |
| `plant_contour_v2.py` | Median-background contour masks | `plant_timelapse/plant_contour_v3.py` |
| `segment_nb_timelapse.py` | CLI for temporal-background segmentation | Removed (approach not used) |
| `train_plant_seg_unet.py` | U-Net training stub | — |
| `generate_tuning_sample_tiffs.py` | Synthetic data for `segment_nb` tuning | — |

## `plant_timelapse/`

| File | Was |
|------|-----|
| `segment_nb.py` | Temporal median/MOG2 background segmentation library |
| `README_segment_nb.md` | Docs for `segment_nb_timelapse.py` |
| `tuning_sample_README.md` | Synthetic tuning stack docs |
| `__init___segment_nb.py` | Lazy import wrapper for `segment_nb` |

## Deleted (not kept here)

Removed from the tree; still in git history. Both were exploration scripts whose
logic was merged into `scripts/stem_analysis_pipeline.py`, and nothing imported
either one.

| File | Was | Replaced by | Recover with |
|------|-----|-------------|--------------|
| `_tmp_freq_fill.py` | Sliding-window temporal frequency voting to repair gaps in binary stem masks | `scripts/stem_analysis_pipeline.py` (step 1) | `git show 4ca39cb:_tmp_freq_fill.py` |
| `_tmp_graph_filter_demo.py` | Comparison figure for skeleton graph-level branch-length filtering | `scripts/stem_analysis_pipeline.py` (step 4) | `git show 4ca39cb:_tmp_graph_filter_demo.py` |

Earlier siblings from the same exploration (`_tmp_gap_measure.py`,
`_tmp_raw_skeleton_demo.py`, `_tmp_skeleton_pruning_demo.py`,
`_tmp_stem_diagnostic.py`, `_tmp_stem_width_profile*.py`) were deleted in commit
`6338040` and are recoverable the same way, e.g.
`git show 6338040~1:_tmp_stem_diagnostic.py`.

## Relocated out of the repository root

Moved, not archived — these are current and maintained. Logged here because the
paths in older notes and commit messages no longer resolve.

| Was | Now | Status |
|-----|-----|--------|
| `stem_analysis_pipeline.py` | `scripts/stem_analysis_pipeline.py` | Current — full skeleton pipeline (slow; produces `masks_freq_filled/`) |
| `stem_height_analysis_v2.py` | `scripts/stem_height_analysis_v2.py` | Current — use this for stem-height measurement |
| `stem_height_analysis.py` | `scripts/stem_height_analysis.py` | Superseded for *measurement* by `_v2`, but still the only source of the interactive annotation tool (`--annotate`) and `--px-per-mm` calibration, which `_v2` cannot do |
