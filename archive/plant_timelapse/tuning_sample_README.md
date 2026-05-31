# Synthetic tuning stack (full frame, day + night)

This folder is meant to hold **generated** grayscale TIFFs for testing
`scripts/segment_nb_timelapse.py` **without cropping**, while the synthetic “plant”
still **grows upward** over time (similar constraint to a real timelapse).

The repository `.gitignore` ignores `*.tif`, so these files usually stay **local**
after you generate them.

## Generate

From the **repo root** (`microneedle_IAA_Raman_analysis`):

```bash
python scripts/generate_tuning_sample_tiffs.py --output-dir plant_timelapse/tuning_sample
```

Options (see `python scripts/generate_tuning_sample_tiffs.py -h`):

- `--frames` — number of images (default 72)
- `--width` / `--height` — resolution
- `--seed` — reproducible random day/night mix and noise
- `--night-prob` — how often “night” exposure/noise pattern appears
- `--start-time` / `--minutes-step` — filename timestamps (30 min default)

## Run segmentation on the sample

```bash
python scripts/segment_nb_timelapse.py \
  --input-dirs plant_timelapse/tuning_sample \
  --output-dir plant_timelapse/tuning_sample/segmentation_out \
  --random-sample 24 \
  --random-seed 42 \
  --keep-largest-only \
  --align-ecc \
  --save-preview
```

Tune `--residual-percentile`, `--open-iterations`, `--close-iterations`, and MOG settings
against the previews, then apply the same settings to your Alvium stacks.
