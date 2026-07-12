# Handoff — Plant stem-height analysis (NIR timelapse)

Purpose of this file: let a **new Cursor chat on another machine** continue the work without the
original chat history. Read this top-to-bottom, then continue from "Next steps".

Last updated: 2026-07-08

---

## Goal

Segment a *Nicotiana benthamiana* plant from a fixed-camera NIR timelapse and measure plant
morphology over time — primarily **stem height = distance from the soil/base line to the apical
meristem** (top of the visible stem where the newest leaves emerge). Data are 12-bit grayscale
`.tif` frames (2592x1944), ~300–325 frames per timelapse. Currently **2 datasets** (~300 images
each); more are being collected. Same plant species, same imaging rig.

## Where things stand

### Segmentation — DONE and working
Notebook: `plant_timelapse/notebooks/plant_segmentation_v2.ipynb` (run with the
`microneedle (plant mask)` kernel; env setup via `scripts/setup_plant_mask_notebook.*`).

Pipeline stages:
- **Stage 0** — index frames, parse timestamps, `FRAME_STEP` subsampling, tuning-frame preview.
- **Stage 1** — fixed-rig calibration: builds a plant-free `background` (low percentile over time,
  shown as QC only) and a `hardware_mask` (thin bright horizontal rail removed by SHAPE, not by
  "bright+static", so the plant base is never deleted).
- **Stage 2** — per-frame segmentation. Key design decisions that were hard-won:
  - Scale frames to 8-bit using the **absolute** sensor range (0–4095) via `raw_u8`, NOT per-frame
    min/max — so a threshold means the same thing day and night.
  - Threshold is **background-relative hysteresis**: a HIGH level `mean + K_SIGMA*sigma` (confident
    plant "seed") and a LOW level `mean + K_SIGMA_LOW*sigma`; grow the seed into connected
    low-threshold pixels via `skimage.morphology.reconstruction`. This traces the real dim
    stem/petioles and reunites the plant WITHOUT inventing pixels (no petiole webbing, no solid
    column to the pot). `ROI` crops to the plant column; drop everything below `BASE_LINE_Y`.
  - Params (Stage 2 cell): `ROI=(250,700,1750,2560)`, `BASE_LINE_Y=2450`, `K_SIGMA=3.0`,
    `K_SIGMA_LOW=2.0`, `CLOSE_KSIZE=15` (keep small — large closes web the petioles),
    `MIN_PLANT_AREA=150`, `BRIDGE_KSIZE=161`.
- **Stage 3** — height metrics: Height A (vertical apex→base) and Height C (skeleton path).
- **Stage 4** — batch → masks/overlays, `heights.csv`, growth curve, jump-flagging.

Result: masks are clean and connected across day/night; **canopy height** (topmost plant pixel →
base line) is smooth and monotonic (~453→835 px over the series). This is a robust growth proxy.

### Stem-to-meristem measurement — NOT solved classically
Approaches tried and why they failed (do NOT repeat these):
- Vertical morphological close to bridge the stem → paints a solid column down to the bright pot.
- Isotropic close → webs the triangle between petioles.
- Skeleton geodesic base→topmost point → apex lands on a leaf tip, path detours through leaves.
- Column/centerline tracer up the stem → **heights bounce 0–790 px frame-to-frame** (unreliable);
  cannot even measure occlusion frequency because the tracer itself fails.

Root cause: distinguishing stem from petiole/leaf by mask geometry is ambiguous, and the apical
meristem is **intermittently occluded by leaves and the probe post**.

## Decision / chosen direction: ML keypoint detection

Agreed plan is to detect two keypoints per frame — **base** and **apical meristem** — with a small
learned model (pose/keypoint style). Rationale: a learned model predicts the meristem from overall
plant appearance/context, including sensible estimates under partial occlusion, and — critically —
**generalizes across samples of the same species on the same rig**, so labeling is a one-time cost
reused across all datasets. Keep classical **canopy height + leaf area** (and optionally SAM 2
zero-shot masks) as robust auxiliary metrics.

### Pilot plan (2 datasets, ~300 imgs each)
1. Label **~40–60 frames per dataset** (base + meristem), spread across the full growth range and
   both day and night (labeling adjacent frames is wasted — they're near-identical). ~80–120
   labeled frames total.
2. Train a keypoint model. Recommended tooling: **DeepLabCut or SLEAP** (purpose-built for
   fixed-camera keypoint tracking with small labeled sets, built-in labeling GUI + temporal
   smoothing). Lightweight custom heatmap model in the existing `.venv` is the fallback.
3. Validate **leave-one-plant-out** (train on dataset A, test on B, and vice versa) to gauge
   cross-sample generalization. With only 2 plants this is a pilot signal, not a guarantee — add a
   few labeled frames per new plant as more data arrives.
4. Post-process predictions per timelapse with temporal smoothing + a monotonic-growth prior;
   flag low-confidence/occluded frames.

## RESOLVED decisions (2026-07-08)
- **GPU**: confirmed **NVIDIA GeForce RTX 3070, 8 GB VRAM** (driver 596.08, CUDA 13.2 capable) via
  `nvidia-smi`. CUDA training is a go on native Windows (PyTorch engine — no WSL needed).
- **Tooling**: **DeepLabCut 3.x, PyTorch engine** (default). Supports Python 3.10–3.12; DLC pins
  `numpy<2` + `matplotlib<3.9`, so it lives in a **dedicated env** (`.venv-dlc`), separate from the
  numpy-2.x plant-mask `.venv`.
- **Model input**: **raw grayscale** (12-bit → 8-bit via the *absolute* 0–4095 scaling, i.e.
  `raw_u8`), **NOT** the binary/edge mask. The mask discards the intensity/texture/context cues the
  model needs to infer the meristem under occlusion, and re-couples us to segmentation errors.
- **Field of view**: a **single fixed crop = the plant-column `ROI (250,700,1750,2560)`**, identical
  every frame. Not the full frame (wastes resolution) and NOT a per-frame mask bbox (unstable scale,
  can crop out an occluded meristem). Height = base_y − meristem_y is a vertical pixel distance, so
  the crop offset cancels.
- **Retraining policy**: on each new dataset, run inference first and check DLC confidence + curve
  smoothness. If it degrades, **fine-tune** with ~10–20 top-up labels (don't retrain from scratch).
  Full retrain only if the rig/optics or species changes. Keep the crop + scaling FROZEN so old
  labels/weights stay valid.
- **Where the model is developed**: CLI-driven, not notebook-driven. Long training + the labeling
  GUI run from the terminal (robust to kernel disconnects); a thin notebook is QC-only.

## Project structure (new)
- `plant_timelapse/keypoints/` — version-controlled module + `typer` CLI:
  - `config.py` (SENSOR_MAX, ROI, KEYPOINTS, dataset paths, DLC project layout),
  - `frame_export.py` (absolute 8-bit + ROI crop → cropped mp4 / PNG),
  - `frame_select.py` (timestamp parsing, day/night, even sampling),
  - `postprocess.py` (DLC predictions → smoothed stem-height CSV),
  - `cli.py` / `__main__.py` (orchestration).
- `plant_timelapse/notebooks/keypoint_dlc_pipeline.ipynb` — QC only (kernel: `microneedle (dlc keypoints)`).
- `plant_timelapse/dlc/` — GENERATED, gitignored: cropped videos, DLC project, weights, height CSVs.
- `scripts/setup_keypoints_dlc.ps1` + `requirements/keypoints_dlc.txt` — one-time env setup.

## CLI workflow (run from repo root, in the DLC env)
```
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints gpu-check
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints build-videos      # cropped 8-bit mp4 per dataset
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints create-project    # bodyparts = base, meristem
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints extract           # kmeans picks diverse frames
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints label             # GUI: click base + meristem
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints check-labels
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints train
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints evaluate
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints analyze
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints postprocess --monotonic
```

## Next steps (start here)
1. Finish env install (`scripts/setup_keypoints_dlc.ps1`) and confirm `gpu-check` shows CUDA True.
2. `build-videos` → `create-project` → `extract` → **label base + meristem** (~50 frames/dataset,
   spread across growth + day/night, include occluded ones).
3. `train` → `evaluate`; run **leave-one-plant-out** (train run4 → test run5 and vice versa).
4. `analyze` → `postprocess`; QC in the notebook.
5. Wire the predictor / height CSV into Stage 3 of the segmentation pipeline, keeping canopy height
   as the fallback metric.

## Data locations
- Dataset 1 (run4): `G:\My Drive\...\Run 4\DEV_1AB22C05B465\timelapse_2026-05-07_12-56-08\*.tif`
  — **325 frames**, May 6–13 (~7 days), 2592×1944 uint16, range 61–4095.
- Dataset 2 (run5): `G:\My Drive\...\Run 5\DEV_1AB22C05B465\timelapse_2026-05-16_16-16-30\*.tif`
  — **995 frames**, May 16–Jun 6 (~21 days), 2592×1944 uint16, range 61–4095.
  (Full paths are hard-coded in `plant_timelapse/keypoints/config.py::DATASETS`.)
- Segmentation outputs: `plant_timelapse/outputs/<timelapse_name>/`.
- Keypoint outputs: `plant_timelapse/dlc/` (gitignored).

## How this conversation was continued
This file was written so the work survives moving between machines. On the new machine, open a
Cursor chat and say: "Read plant_timelapse/HANDOFF_stem_height.md and continue from Next steps."
