# Plant timelapse segmentation

Segment **Nicotiana benthamiana** (or similar static-scene subjects) in **grayscale VIS–NIR** TIFF time series from a **fixed** camera. The core idea is a **temporal background model** (median or MOG2), a **residual** mask on `|I − B|`, morphological cleanup, and optional **temporal smoothing** on the binary masks. Optional steps: **ECC alignment**, **GrabCut**, and **SAM** (Segment Anything) refinement.

Core implementation: [`segment_nb.py`](segment_nb.py).  
Command-line entry point (run from the **repository root**): [`../scripts/segment_nb_timelapse.py`](../scripts/segment_nb_timelapse.py).

---

## Synthetic tuning data (full frame, day + night)

To tune parameters **without cropping**—with a canopy that moves **upward** over time—generate a semi-random grayscale stack:

```bash
python scripts/generate_tuning_sample_tiffs.py --output-dir plant_timelapse/tuning_sample
```

See [`tuning_sample/README.md`](tuning_sample/README.md) for segmentation command examples.

---

## Install

From the repo root:

```bash
pip install -e ".[plant_timelapse]"
```

That pulls **OpenCV** (headless) and **Pillow** in addition to the base package. **SciPy** is already a main dependency (used for hole filling).

Optional (Tier 2 SAM only):

- `torch` and the [Segment Anything](https://github.com/facebookresearch/segment-anything) package, plus a matching checkpoint (e.g. ViT-B).

---

## Quick start

Point at the folder that **directly contains** your `.tif` / `.tiff` frames (same discovery rules as `scripts/animate_plant_timelapse.py`: sorted by timestamp parsed from the filename when possible).

```bash
python scripts/segment_nb_timelapse.py ^
  --input-dirs "path\to\timelapse_folder" ^
  --output-dir "path\to\segmentation_output"
```

### Recommended first pass (real runs)

1. **Crop** to the pot / plant region so the background model is mostly static (soil, pot rim, etc.).
2. If the rig drifts slightly between frames, turn on **ECC alignment** to the first processed frame.
3. Enable **previews** while tuning, then turn them off for large jobs.

Example (Windows paths; adjust crop to your image size—format is `y0,y1,x0,x1`, half-open intervals):

```bash
python scripts/segment_nb_timelapse.py ^
  --input-dirs "G:\My Drive\...\timelapse_folder" ^
  --output-dir "outputs\nb_run1_seg" ^
  --crop 80,920,120,1380 ^
  --align-ecc ^
  --temporal-median-window 3 ^
  --save-preview
```

Use `--frame-step 2` or `--max-frames 200` for faster trials.

To **tune on a random subset** of the run (after frame-step / max-frames), use `--random-sample N`. The same `N` indices are drawn without replacement, then sorted in time. Pass `--random-seed INT` to reproduce the subset; omit it for a new random draw each time. With very small `N`, the temporal median / median background have less data.

---

## Outputs

Under `--output-dir`:

| Path | Description |
|------|-------------|
| `masks/` | One mask per input frame: `{original_stem}_mask.png` (or `.tif` with `--mask-format tif`). Foreground **255**, background **0**. Full-frame size; cropped regions are pasted back onto a black canvas. |
| `plant_area.csv` | Columns: ISO timestamp (from filename when parseable), file path, stem, plant **area in pixels** (within crop only), **area fraction** of the crop rectangle. |
| `previews/` | Optional RGB overlays (`--save-preview`): grayscale + semi-transparent red mask. |

Filenames should include parseable timestamps (see `animate_plant_timelapse.parse_timestamp_from_name`). If parsing fails, the CSV timestamp field is left empty.

---

## Background modes

### Median (`--bg-mode median`, default)

Builds **B** from a **spread subsample** of frames (`--bg-sample-count`, default 48). Robust when the plant occupies a minority of pixels over time. With `--align-ecc`, both the sample frames and every processed frame are aligned to the **first** frame in the run before subtraction.

Tune **`--residual-percentile`** (default 92): higher → stricter (smaller FG); lower → more permissive.

### MOG2 (`--bg-mode mog2`)

Uses OpenCV’s **MOG2** background subtractor trained on `--mog-train-frames` frames spread across the run. By default the mask is **MOG foreground only**. Pass **`--combine-mog-and-residual`** to **AND** MOG with a residual threshold on `|I − B|` (useful when MOG alone is noisy).

Tune `--mog-var-threshold`, `--mog-binary-threshold`; use `--no-mog-shadows` if shadow labels confuse the mask.

---

## Morphology and probe handling

| Flag | Role |
|------|------|
| `--open-iterations`, `--close-iterations` | Remove speckles / close gaps |
| `--min-area-px` | Drop small connected components (areas are in **crop** coordinates if you use `--crop`) |
| `--erode-shrink-px` | Extra **erosion** passes to shave thin structures (e.g. probe halo)—use if you want the plant mask **without** narrow attachments |

### Single plant, extra patches on the rig (`--keep-largest-only`)

If masks look **almost right** around the plant but **scattered** blobs also appear on mounts, cables, or dark background texture (typical when `|I − B|` is similar for leaves and hardware), add **`--keep-largest-only`**. After each frame is fully processed, only the **largest connected component** is kept. This assumes the plant is the biggest foreground object in the (optionally cropped) image; if your rig or pot reads as a **larger** area than the canopy, **tighten `--crop`** first so the apparatus is excluded.

---

## Temporal smoothing

`--temporal-median-window` (default **3**): per-pixel **median** over a sliding window of binary masks. Reduces single-frame flicker. Use **1** for no temporal filtering (the pipeline still enforces a minimum of 1 internally; very small windows behave like no smoothing—see `temporal_median_masks` in `segment_nb.py`).

---

## Optional refinements

### GrabCut (`--refine-grabcut`)

Runs OpenCV **GrabCut** seeded by the Tier-1 mask. Slower but can clean soft boundaries. Adjust `--grabcut-iters`.

### SAM (`--sam-checkpoint`)

Loads a **Segment Anything** checkpoint; uses a **bounding box** derived from the coarse mask, then keeps the best-scoring SAM mask. Requires compatible `--sam-model-type` (`vit_b` / `vit_l` / `vit_h`). Use `--sam-device cuda` or `cpu` if needed.

---

## Programmatic use

```python
from pathlib import Path
from plant_timelapse.segment_nb import SegmentConfig, run_segmentation_pipeline

# Build paths and timestamps the same way as the CLI, or any ordered list.
paths = [...]
timestamps = [...]  # same length; use None if unknown

cfg = SegmentConfig(
    output_dir=Path("out_seg"),
    frame_step=1,
    max_frames=None,
    crop=(80, 920, 120, 1380),  # or None
    bg_mode="median",
    keep_largest_only=False,
    random_sample_n=None,
    random_sample_seed=None,
    # ... set remaining fields; mirror defaults in scripts/segment_nb_timelapse.py
)
csv_path = run_segmentation_pipeline(paths, timestamps, cfg)
```

Importing the package root (`import plant_timelapse`) does **not** load OpenCV until you access `run_segmentation_pipeline` or import `plant_timelapse.segment_nb`.

---

## Supervised training (Tier 3)

After exporting masks, you can correct a subset in **CVAT** / **LabelMe** and train a small **U-Net** (or similar) on single-channel input. A minimal recipe and optional `torch` check live in [`../scripts/train_plant_seg_unet.py`](../scripts/train_plant_seg_unet.py).

---

## When results are incomplete or noisy

1. **Crop** to the pot / lower canopy so the median background is not dominated by tall black hardware.
2. **`--align-ecc`** if the frame shifts slightly between captures.
3. **Residual**: try `--residual-percentile` a few points lower (more plant, more background risk) or higher (stricter plant, more holes).
4. **Morphology**: try **`--open-iterations 0`** if you are **losing** leafy detail; increase **`--close-iterations`** to fill small holes inside the plant.
5. **`--keep-largest-only`** to drop false-positive patches when you know there is **one** main plant blob.
6. **`--refine-grabcut`** (or SAM) if edges stay ragged after the above.

---

## Limitations

- Works best when the **camera and scene** are **stable**; large motion or lighting that changes the background texture will break background assumptions.
- **Single-channel** data: soil, labels, and hardware can still match plant intensities; combine **crop**, **erosion**, and higher tiers (GrabCut / SAM / supervised model) when needed.
- **Windows**: duplicate paths from mixed `*.tif` / `*.TIF` globs are deduplicated in `collect_tiff_files` (see `animate_plant_timelapse.py`).

For questions about filename patterns or animated exports of the same TIFF folders, see [`../scripts/animate_plant_timelapse.py`](../scripts/animate_plant_timelapse.py).
