# V4 Refactoring Checklist

This checklist tracks the migration of functions from `main_test_v3.py` to `src/pipeline/` modules for the new `main_test_v4.py`.

---

## Phase 1: Utilities → `src/pipeline/utils.py` ✅ COMPLETE

- [x] Move `parse_light_transition_config(config)`
- [x] Move `parse_shade_transition_config(config)`
- [x] Move `parse_treatment_events_config(config)`
- [x] Move `remove_spikes_hampel(...)`
- [x] Update imports in `utils.py` if needed (added `from scipy.signal import medfilt`)
- [ ] Test utility functions independently

---

## Phase 2: Core Processing → `src/pipeline/processing.py`

- [ ] Move `lorentzian(x, amplitude, center, width, offset)`
- [ ] Move `calculate_peak_area_analytical(amplitude, width, peak_type='lorentzian')`
- [ ] Move `find_peak_lorentzian(...)`
- [ ] Move `process_scan_v2(...)` → rename to `process_scan_v3(...)` or `process_scan_v4(...)`
- [ ] Move `aggregate_summaries_to_dataframe_v3(...)` → rename to `aggregate_summaries_to_dataframe_v4(...)`
- [ ] Update imports in `processing.py`
- [ ] Test processing functions independently

---

## Phase 3: Post-Processing → `src/pipeline/post_processing.py`

- [ ] Move `apply_baseline_correction_v3(...)` → rename to `apply_baseline_correction_v4(...)`
- [ ] Move `calculate_ratios(df)`
- [ ] Move `apply_als_and_gaussian_v3(...)` → rename to `apply_als_and_gaussian_v4(...)`
- [ ] Move `compute_fourier_transform_v3(...)` → rename to `compute_fourier_transform_v4(...)`
- [ ] Move `compute_diurnal_average_v3(...)` → rename to `compute_diurnal_average_v4(...)`
- [ ] Update imports in `post_processing.py`
- [ ] Test post-processing functions independently

---

## Phase 4: Plotting → `src/pipeline/plotting.py`

- [ ] Move `plot_normalized_baseline_correction_comparison(...)`
- [ ] Move `plot_normalized_smoothing_comparison(...)`
- [ ] Move `plot_representative_raman_spectrum(...)`
- [ ] Move `plot_ratio_timeseries(...)`
- [ ] Update imports in `plotting.py`
- [ ] Test plotting functions independently

---

## Phase 5: Create main_test_v4.py

- [ ] Copy `src/cli/main_test_v3.py` to `src/cli/main_test_v4.py`
- [ ] Remove all function definitions (now in pipeline modules)
- [ ] Update imports to use pipeline modules:
  - [ ] `from pipeline.utils import parse_light_transition_config, parse_shade_transition_config, parse_treatment_events_config, remove_spikes_hampel`
  - [ ] `from pipeline.processing import process_scan_v4, aggregate_summaries_to_dataframe_v4, lorentzian, calculate_peak_area_analytical, find_peak_lorentzian`
  - [ ] `from pipeline.post_processing import apply_baseline_correction_v4, calculate_ratios, apply_als_and_gaussian_v4, compute_fourier_transform_v4, compute_diurnal_average_v4`
  - [ ] `from pipeline.plotting import plot_normalized_baseline_correction_comparison, plot_normalized_smoothing_comparison, plot_representative_raman_spectrum, plot_ratio_timeseries`
- [ ] Update function calls to use new names (v4 instead of v3)
- [ ] Update docstrings to reflect v4
- [ ] Keep `build_parser()` and `main()` in CLI module

---

## Phase 6: Create Root-Level Wrapper

- [ ] Create `main_test_v4.py` at root level
- [ ] Add wrapper: `from src.cli.main_test_v4 import main`
- [ ] Update docstring with v4 information

---

## Phase 7: Testing

- [ ] Test all moved functions work independently
- [ ] Test `main_test_v4.py` with sample data
- [ ] Compare outputs between v3 and v4 (should be identical)
- [ ] Test all command-line arguments
- [ ] Test batch processing
- [ ] Test single scan processing
- [ ] Verify all plots are generated correctly
- [ ] Verify all CSV outputs are correct

---

## Phase 8: Documentation

- [ ] Update `README.md` to mention v4
- [ ] Create `docs/V3_VS_V4.md` comparison document
- [ ] Update `docs/ARCHITECTURE.md` with v4 information
- [ ] Add migration guide if needed

---

## Notes

- Keep `main_test_v3.py` unchanged throughout this process
- All function renames should use `_v4` suffix for clarity
- Test thoroughly at each phase before proceeding
- Consider backward compatibility if other code depends on v3 functions

