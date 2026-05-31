# Archived code

Superseded or unused scripts kept for reference. Not maintained.

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
