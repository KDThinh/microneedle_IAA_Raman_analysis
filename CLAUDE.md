# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A Python package (`swnt_iaa_analysis`) for analyzing Raman spectroscopy data from
SWNT-IAA (Single-Walled Carbon Nanotube / Indole-3-Acetic Acid) nanosensors used
to monitor auxin (IAA) levels in plants via microneedle-delivered sensors.

The pipeline takes raw time-series Raman spectra (TSV) and produces:
- Fluorescence-to-G-band and fluorescence-to-Raman-peak ratios over time
- Baseline-corrected, spike-corrected signal time-series
- FFT/frequency-domain analysis (e.g. diurnal periodicity in IAA signal)
- Plots of spectra, time-series, signal corrections, and FFT results

## Repo layout

- `swnt_iaa_analysis/` — the installable package
  - `core/` — data loading, baseline correction, preprocessing (`loader.py`, `baseline.py`, `preprocessing.py`, `utils.py`)
  - `analysis/` — scientific calculations (`peaks.py`, `ratios.py`, `fourier.py`)
  - `io/` — config loading and result export (`config.py`, `exporter.py`)
  - `visualization/` — plotting (`plotting.py`)
  - `pipeline.py` — orchestrates the full analysis flow (~1000 lines, the core of the package)
  - `cli.py` — Typer-based CLI entry point
- `config.yaml` — profile-based config with inheritance (base `processing_default` + per-experiment overrides); lives at repo root alongside `setup.py`
- `plant_timelapse/` — separate image-processing module (plant contour tracking), not part of the main Raman pipeline
- `scripts/` — standalone utility scripts (animation, ratio plotting, etc.)
- `archive/` — old/obsolete code versions; do not build on this
- `_tmp_*.py`, `stem_analysis_pipeline.py` at repo root — ad-hoc experiment scripts, not part of the package proper
- `README.md` — comprehensive (650+ lines); read it for the detailed step-by-step pipeline algorithm before changing pipeline logic

## Setup & running

```bash
python -m pip install -e .              # from repo root (where setup.py and config.yaml live)
swnt_iaa_analysis analyze <profile>      # run full pipeline for one profile
swnt_iaa_analysis list-profiles          # list profiles defined in config.yaml
swnt_iaa_analysis batch-process ...      # process multiple profiles
```

If the console script isn't on PATH: `python -m swnt_iaa_analysis.cli analyze <profile>`.

Input data: TSV with `Scan Number`, `Seconds`, and wavenumber columns (cm⁻¹), referenced via paths in `config.yaml` profiles. Outputs (CSV + PNG plots) are written to a timestamped `results_v4_YYYYMMDD_HHMMSS/` directory next to the input file.

## Conventions & gotchas

- **V4 is the active algorithm; V3 raises `NotImplementedError`.** Don't try to "fix" V3 — it's intentionally disabled. New work should target/extend V4 in `pipeline.py`.
- **Python 3.8 compatibility is required.** Use `Optional[X]` / `Union[X, Y]` from `typing`, not the `X | Y` syntax.
- **No automated test suite and no lint config exist.** There's nothing to run for CI-style checks; validate changes by running the CLI against a real profile and inspecting outputs/plots.
- **`archive/` and `_tmp_*` files are not maintained code** — don't refactor them or treat them as references for current conventions.
- Logging uses the standard library `logging` module (module-level loggers), not print statements.
- Config profiles use inheritance — when adding/editing a profile in `config.yaml`, check what it inherits from `processing_default` rather than duplicating settings.
- Code organization (`core`/`analysis`/`io`/`visualization`) intentionally mirrors a sibling repo (`microneedle_nir_imaging_analysis`) for consistency across the lab's analysis packages — keep that parallel structure in mind if reorganizing.

## Domain context (for interpreting the science correctly)

- **G-band**: a Raman peak (~1600 cm⁻¹) characteristic of carbon nanotubes, used as a stable reference signal.
- **Raman peak ~850 cm⁻¹**: a secondary reference peak used for an alternative ratio.
- **Fluorescence**: measured as area-under-curve at higher wavenumbers (≥1250 cm⁻¹), isolated by subtracting the G-band area; its ratio to the reference peaks is the actual biological signal of interest (correlates with IAA concentration).
- **Lieberfit baseline correction**: iterative polynomial fitting to remove fluorescence background drift from spectra.
- **Hampel filter / MAD-based jump detection**: used for time-series signal corrections (spike removal and baseline-shift correction), producing `_BaselineCorrected` columns alongside the originals (both are kept for QC comparison).
- **FFT/diurnal analysis**: looks for periodic (e.g. day/night) patterns in the corrected fluorescence ratio time-series.
