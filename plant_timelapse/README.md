# plant_contour_v3 CLI

Sobel **Find Edges** segmentation (ImageJ/Fiji-style) for grayscale plant timelapse TIFFs: edge mask, refined ROI, line-artifact cleanup, optional growth metrics.

Run from the **repository root**:

```text
python plant_timelapse/plant_contour_v3.py …
```

Equivalent: `python -m plant_timelapse.plant_contour_v3 …`  
(`scripts/plant_contour_v3.py` is a thin launcher to the same module.)

## Install

```bash
pip install -e ".[plant_timelapse]"
```

Needs OpenCV, matplotlib, and numpy.

## Single frame

Preview (3×3 figure: original, masks, growth panels):

```powershell
python plant_timelapse/plant_contour_v3.py --image path/to/frame.tif
```

Useful options:

```powershell
python plant_timelapse/plant_contour_v3.py --image frame.tif --no-show --save-mask out/mask.png
python plant_timelapse/plant_contour_v3.py --image frame.tif --line-artifact intensity
python plant_timelapse/plant_contour_v3.py --image frame.tif --uint8
```

## Batch timelapse

All three dirs are required. One line (works in any shell):

```powershell
python plant_timelapse/plant_contour_v3.py --frames-dir "path\to\timelapse_folder" --preview-out-dir "path\to\previews" --mask-out-dir "path\to\masks" --bg-glob "*.tif"
```

Or split across lines in **PowerShell** (Cursor’s default terminal on Windows) with a **backtick** `` ` `` at the end of each line:

```powershell
python plant_timelapse/plant_contour_v3.py `
  --frames-dir "path\to\timelapse_folder" `
  --preview-out-dir "path\to\previews" `
  --mask-out-dir "path\to\masks" `
  --bg-glob "*.tif"
```

Optional batch outputs (single line):

```powershell
python plant_timelapse/plant_contour_v3.py --frames-dir "path\to\timelapse" --preview-out-dir "path\to\previews" --mask-out-dir "path\to\masks" --frames-step 2 --line-artifact intensity --growth-csv "path\to\growth.csv" --growth-save-stabilized
```

| Output | Flag / path |
|--------|-------------|
| `*_preview.png` | `--preview-out-dir` |
| `*_mask_refined.png` | `--mask-out-dir` |
| `plant_contour_v3_batch.log` | `--preview-out-dir` and `--mask-out-dir` (identical copy in each) |
| `*_mask_growth_stabilized.png` | `--growth-save-stabilized` (under preview dir) |
| Growth metrics CSV | `--growth-csv` |

Repeat `--frames-dir` to merge multiple timelapse roots (paths deduped and sorted globally).

**Line continuation:** PowerShell `` ` `` · cmd.exe `^` · bash `\`

## Flags worth tuning

| Flag | Default | Notes |
|------|---------|-------|
| `--line-artifact` | (see script) | `intensity` for dark microneedle/rig through bright plant; `morph`, `hough`, `merge`, or `none` |
| `--mask-blur` | `40` | Gaussian sigma on binary mask before re-threshold; `0` disables |
| `--post-mask-thresh` | `otsu` | Threshold after blur; try `--batch-adaptive-post-otsu` in batch if the plant splits |
| `--thresh` | `triangle` | Edge magnitude threshold: `triangle`, `otsu`, or `fixed` |
| `--uint8` | off | 8-bit pipeline; default keeps 16-bit |

Full options: `python plant_timelapse/plant_contour_v3.py -h`  
Pipeline details: module docstring at the top of [`plant_contour_v3.py`](plant_contour_v3.py).
