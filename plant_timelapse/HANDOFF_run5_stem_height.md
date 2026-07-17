# Handoff — Run 5 stem-height optimization (DLC keypoints)

Last updated: 2026-07-13 (evening)

## Goal

Get reliable stem height for run 5 (NIR timelapse, ~995 frames, ~21 days) using DeepLabCut keypoints (`base` + `meristem`).

**Preferred metric when plant tilts:** Euclidean distance  
`height_eucl = hypot(base_x − meristem_x, base_y − meristem_y)`  
(`height_eucl_px` / `height_eucl_smooth` in the CSV). Vertical `base_y − meristem_y` underestimates leaning stems and can go negative when keypoints are wrong.

## Context / starting point

- Pipeline: `plant_timelapse/keypoints/` (CLI-driven).
- Imaging: full frame (`ROI = None`), absolute 8-bit scaling 0–4095.
- Datasets: `run4` (~325 frames, ~7 days), `run5` (~995 frames, ~21 days).
- Run 4 is good (labels + top-ups + postprocess).
- Run 5 is harder: longer series, bolting, large meristem travel, tilt late in growth.

## What was done (chronological)

### 1–3. Early rounds + snapshot traps

- Started with ~50 kmeans labels; model failed on tall plants (meristem stuck mid-frame, base off soil).
- First top-up labeling; discovered `postprocess` can pick the wrong `.h5` alphabetically; floors 0.6 can blank all frames → Savitzky–Golay NaN crash.
- Never trust `snapshotindex: -1` / “best” without checking.

### 4–5. Shuffle bug

- Each `create_training_dataset` creates a **new shuffle**. Default CLI train/analyze uses **shuffle=1** (old weights).
- Must train/analyze with the **newest shuffle** explicitly.

### 6. Shuffle4 (labels after second top-up)

- Trained shuffle4 to epoch 200.
- Snap-200 better than snap-150, but still not usable (meristem ≥0.6 ≈ 0%; many negative vertical heights).
- Postprocess floors **0.3** required for QC (0.6 blanks everything).

### 7. Third top-up + shuffle5 (2026-07-13)

- Another ~51 meristem-focused labels; run5 filled labels grew to ~193 then ~205 rows after labeling.
- Trained **shuffle=5**, epochs 200. Snapshots: 100/125/150/175/200 + `best-090`.
- Analyzed `shuffle=5, snapshot_index=4` (snapshot-200) → `...shuffle5_snapshot_200.h5`.
- Postprocess floors 0.3 → `stem_height_run5.csv`.

### 8. Shuffle5 notebook QC (confirmed run5)

Stem-height plot + keypoint contribution plots showed:

| Check | Result |
|-------|--------|
| Vertical height | Often **negative**; smooth collapses late |
| Euclidean height | Mostly positive; late rise more plausible, but still huge jumps |
| Base/meristem confidence | Low (base ~0.2–0.4, meristem ~0.2–0.6) |
| Raw `base_y` jumps | mean **246** / max **2113** px |
| Raw `meristem_y` jumps | mean **230** / max **1717** px |
| `height_valid` | ~23% at floors 0.3 |
| Suspicious frames | **376** flagged — do **not** label all |

**Verdict:** shuffle5 snap-200 still **not usable**. Destep makes refined curves look smooth but invents continuity. Prefer Euclidean for QC; fix keypoints (especially **base on soil**) before trusting any curve.

### 9. Shuffle6 + snap-175 (2026-07-13 evening)

- After another curated top-up (~53 frames) → train **shuffle=6**; eval favored mid/late epochs over final 200.
- Analyzed `shuffle=6, snapshot_index=3` (**snapshot-175**).
- CSV-level: **first biologically plausible run5 curve** — no negative vertical heights; smooth ~275→1728 (vertical) / ~1933 (euclidean); early base_ok ~98%; meristem y 1985→346.
- Notebook QC still shows **staircase jumps**, Euclidean spikes, low confidence (~0.1–0.5), gappy refined series; **303** frames flagged.
- **Verdict:** clear progress, not shippable yet. One more **targeted** top-up on jump clusters (esp. ~700–870 mid drop, ~920–950 late jump) is worthwhile; do **not** label all 303/380 neighbors.

## Decisions / lessons learned

1. Always pass **`shuffle=`** to train/analyze after recreating the training dataset.
2. Never trust **best / −1** without checking metrics; prefer snapshot by eval (e.g. 175 over 200 when 200 degrades).
3. Aside stale `run5DLC_*` preds; confirm printed `.h5` name.
4. Instant analyze = DLC skip (existing pickle) — move aside and re-run.
5. Floors **0.6** too strict for current run5 models → use **0.3** for QC.
6. Do **not** label hundreds of suspicious neighbors; thin to ~40–60 targeted frames.
7. For tilted plants, report **`height_eucl_*`**; vertical is secondary.
8. Destep/smooth are for mild noise only — not a substitute for good labels.
9. Labeling still helps when the trend is right but jumps remain — focus on failure clusters.

## Current artifacts

| Item | Note |
|------|------|
| Labels | `labeled-data/run5/` (~258 rows before next top-up export) |
| Latest weights | `...trainset95shuffle6/train/snapshot-175.pt` (preferred) |
| Active preds | `videos_prepared/run5DLC_...shuffle6_snapshot_175.h5` |
| Height CSV | `stem_height_run5.csv` (shuffle6/175; improved but jumpy) |
| Aside folders | `_aside_*` under `videos_prepared/` |
| Run 4 reference | `stem_height_run4.csv` (good) |

## Next steps (in progress)

1. **Curated label round (~54 frames)** on staircase/jump clusters — not the full 303.
2. `label --dataset run5` → `check-labels`
3. `create_training_dataset` + `train_network(shuffle=7)` (expect **7**)
4. Analyze preferred late snapshot by eval (not blindly −1) → postprocess floors 0.3 → QC **`height_eucl_smooth`**
5. Longer-term: CLI `--shuffle` / `--snapshot-index`; optional larger training crop

## Useful commands

```powershell
# Label
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints add-label-frames run5 --frames <list>
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints label --dataset run5
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints check-labels

# Train newest shuffle (example: 7 after next create_training_dataset)
.venv-dlc\Scripts\python.exe -c @"
import deeplabcut
from plant_timelapse.keypoints import config as C
cfg = C.POINTER_FILE.read_text(encoding='utf-8').strip()
deeplabcut.create_training_dataset(cfg, net_type=C.DEFAULT_NET_TYPE)
deeplabcut.train_network(cfg, shuffle=7, epochs=200)
"@

# Analyze (adjust shuffle + snapshot_index after listing snapshots)
.venv-dlc\Scripts\python.exe -c @"
import deeplabcut
from plant_timelapse.keypoints import config as C
cfg = C.POINTER_FILE.read_text(encoding='utf-8').strip()
videos = [str(C.video_path('run5'))]
deeplabcut.analyze_videos(cfg, videos, shuffle=7, save_as_csv=True, snapshot_index=3)
"@

# Postprocess QC
.venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints postprocess --dataset run5 `
  --base-likelihood-floor 0.3 --meristem-likelihood-floor 0.3 `
  --interpolate-limit 3 --destep-jump-px 15
```

## Related docs

- `plant_timelapse/HANDOFF_stem_height.md` — broader stem-height / DLC project handoff.
