# Changelog

Notable changes to the `swnt_iaa_analysis` package.

For installation and command usage, see [README.md](README.md) — it is the single
source of truth for how to run the pipeline. This file records *what changed*, not
how to get started.

The package version in `setup.py` has been `1.0.0` throughout and the repository
carries no release tags, so entries below are grouped by date rather than by
released version.

## 2026-09-28

### Changed
- Repository root now contains only packaging and documentation files; the one
  remaining root `.py` file is `setup.py`. The standalone stem-analysis scripts
  moved to `scripts/`:
  - `stem_analysis_pipeline.py`
  - `stem_height_analysis.py`
  - `stem_height_analysis_v2.py`
- `stem_height_analysis_v2.py` now resolves the repository root as
  `Path(__file__).resolve().parents[1]` so its `plant_timelapse` import keeps
  working from `scripts/`.

### Removed
- `_tmp_freq_fill.py` and `_tmp_graph_filter_demo.py`. Both were exploration
  scripts whose logic had already been merged into `stem_analysis_pipeline.py`,
  and nothing imported either one. See [archive/README.md](archive/README.md) for
  the `git show` refs to recover them.

## 2026-07-20

### Added
- `core/transition_baseline.py` — transition-ramp baseline correction driven by
  rate of change. Lighting transitions add a fast multi-scan ramp to the
  fluorescence background while real biology varies gradually; this module masks
  the fast ramps and reconstructs the baseline as the slow component (rolling
  median over roughly one light cycle). Unlike additive step-stitching it cannot
  drift, and cannot drive a positive-definite quantity negative.
- `scripts/condition_report.py` — one figure per experimental condition.

### Changed
- Timeseries baseline correction redesigned around the above.

## 2026-04-28

### Added
- `reprocess` CLI command, for re-running analysis over previously processed
  output without repeating the per-scan spectral work.

## Initial packaging

### Fixed
- **Python 3.8 compatibility in `io/config.py`** — `List[str] | None` replaced
  with `Optional[List[str]]`. PEP 604 `|` unions require Python 3.10, while
  `setup.py` declares `python_requires=">=3.8"`.
- **`_process_v3` in `pipeline.py`** now raises `NotImplementedError` instead of
  silently falling through to the V4 code path. Selecting `algorithm='v3'` and
  quietly getting V4 results was a correctness hazard; V3 remains unimplemented
  in the refactored package.
