# Handoff — Run 5 stem-height optimization (DLC keypoints)

Last updated: 2026-07-13

## Goal

Get reliable **stem height = base_y − meristem_y** for run 5 (NIR timelapse, ~995 frames, ~21 days) using the DeepLabCut keypoint pipeline (`base` + `meristem`), matching the quality already achieved for run 4.

## Context / starting point

- Pipeline module: `plant_timelapse/keypoints/` (CLI-driven).
- Imaging: full frame (`ROI = None`), absolute 8-bit scaling 0–4095.
- Datasets in `keypoints/config.py`: `run4` (~325 frames, ~7 days), `run5` (~995 frames, ~21 days).
- Run 4 worked via: kmeans labels + **targeted top-up labels** + postprocess (likelihood floors → interpolate → destep → smooth).
- Run 5 is harder: longer series, plant bolts, meristem travels ~5× more vertically; apical tip often near top of frame.

## What was done for run 5 (chronological)

### 1. Initial state

- ~50 kmeans-extracted labels for run 5 (`labeled-data/run5/`, 49 filled; `img360` empty).
- Combined train with run 4; analyze + postprocess produced a poor run 5 curve.
- Diagnosis: model saturates on tall plants (meristem stuck ~y1850), base drifts off soil, height goes negative; destep can fake a smooth wrong curve.

### 2. First top-up labeling (2026-07-12)

- Exported additional frames via `add-label-frames run5 --frames ...` (even spacing through mid/late growth + worst prediction failures).
- Labeled in napari (`label --dataset run5`); napari hang fixed with `python -m napari --reset` when needed.
- Then: `check-labels` → `train` → `evaluate` → `analyze --dataset run5`.

### 3. Snapshot / postprocess issues discovered

- DLC “best” snapshot was **epoch 40** (`snapshot_best-40`): very low confidence (base/mer medians ~0.15 / 0.09).
- `postprocess` auto-picks last alphabetical `*filtered.h5`, which preferred `best-40` over older `best-120` → all frames blanked at floor 0.6 → Savitzky–Golay crash (`array must not contain infs or NaNs`).
- Workaround: move stale preds aside so postprocess picks the intended file.
- Explicit late-snapshot analyze on **shuffle1** `snapshot_index=4` (epoch 200) still not usable (~0.4% `height_valid` at floors 0.6; impossible heights).

### 4. Second top-up labeling (2026-07-12 evening)

- Exported **52 more frames** into `labeled-data/run5` (failure-focused, mid/late heavy).
- Labeled in napari; `check-labels` reported ~59 run4 + ~154 run5 label images.
- Run5 labels after this round: **~146 filled / 154 rows**.

### 5. Critical bug: `train` kept using shuffle1

- Each `create_training_dataset` / `train` cycle creates a **new shuffle** (shuffle1 → … → **shuffle4**).
- New labels landed in **shuffle4** documentation (~191 frames in training doc).
- Default CLI `train` / `analyze` use **shuffle=1**, whose `snapshot-100…200.pt` were still dated **2026-07-08**.
- Only `snapshot-best-040.pt` on shuffle1 was updated by a later train — late-epoch analyze with shuffle1 therefore re-inferred the **old** model (predictions matched prior aside file almost exactly).

### 6. Train + analyze shuffle4 (2026-07-12 night → 2026-07-13)

- Trained explicitly: `deeplabcut.train_network(cfg, shuffle=4, epochs=200)`.
- Shuffle4 snapshots (all fresh):

| index | file |
|------:|------|
| 0 | snapshot-100.pt |
| 1 | snapshot-125.pt |
| 2 | snapshot-150.pt |
| 3 | snapshot-175.pt |
| **4** | **snapshot-200.pt** |
| 5 / −1 | snapshot-best-050.pt |

- Analyzed run5 with `shuffle=4, snapshot_index=4` → `...shuffle4_snapshot_200.h5`.
- Postprocess at floors **0.6** crashed again (meristem ≥0.6 = **0%** → all NaN).
- Postprocess at floors **0.3** succeeded.

### 7. Shuffle4 snapshot comparison (CSV QC)

| Metric | shuffle4 snap-200 (floors 0.3) | shuffle4 snap-150 (floors 0.3) |
|--------|--------------------------------|--------------------------------|
| `height_valid` | **23.5%** | **14.6%** |
| Meristem ≥ 0.6 | **0%** | **0%** |
| Meristem ≥ 0.3 | ~40% | ~24% |
| Negative heights | ~19% | ~29% |
| Height median | ~330 px | ~6.5 px |
| Smooth first→last | ~732 → ~1395 (rises) | ~906 → ~46 (**falls**) |

**Verdict:** snap-150 is **worse** than snap-200. Snap-200 is the better shuffle4 checkpoint so far, but still **not usable** as a final growth curve (meristem confidence too low; many negative heights; tall-phase base often off soil).

## Decisions / lessons learned

1. **Always pass `shuffle=`** to train/analyze after re-creating the training dataset — or you evaluate the wrong weights.
2. **Never trust `snapshotindex: -1` / “best”** without checking — best-40 / best-050 have been weak.
3. **`postprocess` file picker is fragile** — aside stale `run5DLC_*` preds; confirm the printed `.h5` name.
4. **If analyze finishes instantly**, DLC skipped because `*_full.pickle` exists — move aside and re-run.
5. **Floors 0.6 are too strict** for current run5 models — use ~0.3 (or none) for QC; do not trust a destepped smooth curve when valid fraction is low.
6. **Snapshot-hopping is secondary** to more/better labels once late epochs are already weak.
7. Notebook QC: set `DATASET = 'run5'` and re-run stem-height cells; confirm plot title says run5.

## Current artifacts

| Item | Location / note |
|------|------------------|
| Labels | `plant_timelapse/dlc/plant_meristem-ryank-2026-07-08/labeled-data/run5/` (~146 filled) |
| Best shuffle4 weights so far | `.../trainset95shuffle4/train/snapshot-200.pt` |
| Active preds (last postprocess) | `videos_prepared/run5DLC_...shuffle4_snapshot_150.h5` (worse; for comparison only) |
| Prefer for reference | re-postprocess from aside / re-analyze **shuffle4 snapshot_200** |
| Aside folders | `_aside_best40/`, `_aside_old_preds/`, `_aside_old_snapshot200/`, `_aside_before_round2/`, `_aside_snapshot_tests/`, `_aside_pre_shuffle4/` |
| Height CSV | `plant_timelapse/dlc/stem_height_run5.csv` (currently from **snap-150**; poor) |
| Run 4 reference | `stem_height_run4.csv` (good) |

## Next steps

1. Optional: try **shuffle4 snapshot-175** (`snapshot_index=3`) once — last snapshot check before more labeling.
2. If 175 is also weak: **stop snapshot-hopping**; do another **meristem-focused label round** (frames ~600–995; tip near top of frame) + keep base on soil mid/late.
3. After labeling: `check-labels` → train with **`shuffle=` set to the newest shuffle** (or recreate dataset carefully) → analyze with explicit late snapshot (not −1).
4. Aside old preds → postprocess with floors **0.3** for QC → notebook `DATASET = 'run5'`.
5. Longer-term: CLI should expose `--shuffle` and `--snapshot-index`; consider run5-heavy fine-tune / larger training crop so base+meristem fit together on tall plants.

## Useful commands

```powershell
# Top-up frames then label
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints add-label-frames run5 --frames <list>
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints label --dataset run5
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints check-labels

# Train newest shuffle (example: shuffle 4 — bump if a new shuffle was created)
.venv-dlc\Scripts\python.exe -c @"
import deeplabcut
from plant_timelapse.keypoints import config as C
cfg = C.POINTER_FILE.read_text(encoding='utf-8').strip()
deeplabcut.train_network(cfg, shuffle=4, epochs=200)
"@

# List snapshot indices for a shuffle
.venv-dlc\Scripts\python.exe -c @"
from pathlib import Path
from deeplabcut.pose_estimation_pytorch.apis.utils import get_model_snapshots
from deeplabcut.pose_estimation_pytorch.task import Task
folder = Path(r'plant_timelapse/dlc/plant_meristem-ryank-2026-07-08/dlc-models-pytorch/iteration-0/plant_meristemJul8-trainset95shuffle4/train')
for i, s in enumerate(get_model_snapshots('all', folder, Task.BOTTOM_UP)):
    print(f'{i}: {s.path.name}')
"@

# Analyze specific shuffle + snapshot (move aside old run5DLC_* first if re-running)
.venv-dlc\Scripts\python.exe -c @"
import deeplabcut
from plant_timelapse.keypoints import config as C
cfg = C.POINTER_FILE.read_text(encoding='utf-8').strip()
videos = [str(C.video_path('run5'))]
deeplabcut.analyze_videos(cfg, videos, shuffle=4, save_as_csv=True, snapshot_index=4)
"@

# Postprocess for QC (0.3 floors — 0.6 often blanks everything on run5)
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints postprocess --dataset run5 `
  --base-likelihood-floor 0.3 --meristem-likelihood-floor 0.3 `
  --interpolate-limit 3 --destep-jump-px 15
```

## Related docs

- `plant_timelapse/HANDOFF_stem_height.md` — original stem-height / DLC project handoff (broader than this run5 optimization log).
