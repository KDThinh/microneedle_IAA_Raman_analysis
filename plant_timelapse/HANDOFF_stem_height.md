# Handoff — Plant stem-height analysis (NIR timelapse)

Purpose of this file: let a **new Cursor chat on another machine** continue the work without the
original chat history. Read this top-to-bottom, then continue from "Next steps".

Last updated: 2026-07-07

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

## OPEN QUESTIONS to answer at home
- [ ] **GPU**: user thinks the home laptop has an **NVIDIA GPU** — confirm (run `nvidia-smi`, or
      check Device Manager → Display adapters). This decides DeepLabCut/SLEAP config (CUDA vs CPU).
- [ ] **Tooling choice**: DeepLabCut/SLEAP (recommended) vs lightweight custom model.

## Next steps (start here in the new chat)
1. Confirm GPU (`nvidia-smi`) and paste the result.
2. Pick tooling (default: DeepLabCut).
3. Set up the labeling project and select the frames to label from the 2 datasets.
4. Label base + meristem, train, run leave-one-plant-out validation.
5. Wire the trained keypoint predictor into Stage 3 of `plant_segmentation_v2.ipynb`, keeping
   canopy height as the fallback metric.

## Data locations
- Timelapse frames (example dataset):
  `H:\My Drive\...\Run 4\DEV_1AB22C05B465\timelapse_2026-05-07_12-56-08\*.tif`
  (set `TIMELAPSE_DIR` in Stage 0 of the notebook; second dataset path TBD).
- Outputs: `plant_timelapse/outputs/<timelapse_name>/`.

## How this conversation was continued
This file was written so the work survives moving between machines. On the new machine, open a
Cursor chat and say: "Read plant_timelapse/HANDOFF_stem_height.md and continue from Next steps."
