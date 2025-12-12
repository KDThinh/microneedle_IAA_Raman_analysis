# Refactoring Plan: Creating main_test_v4.py with Refactored Code

This document outlines the plan to create a new `main_test_v4.py` with refactored code, where functions from `src/cli/main_test_v3.py` are moved to `src/pipeline/` modules for better code organization and reusability.

**Note**: `main_test_v3.py` will remain unchanged to preserve stability. The refactoring will be implemented in the new `v4` version.

---

## Function Categories

### ✅ **Should Move to `src/pipeline/`**

#### 1. **Config Parsing Functions** → `src/pipeline/utils.py` or `src/pipeline/config_loader.py`

These are utility functions for parsing configuration:

- `parse_light_transition_config(config)` - Parse light transition config
- `parse_shade_transition_config(config)` - Parse shade transition config  
- `parse_treatment_events_config(config)` - Parse treatment events config

**Reason**: Reusable configuration parsing utilities that could be used by other CLI modules.

**Target**: `src/pipeline/utils.py` (add to existing utils module)

---

#### 2. **Peak Fitting Functions** → `src/pipeline/processing.py`

Core spectral analysis functions:

- `lorentzian(x, amplitude, center, width, offset)` - Lorentzian peak function
- `calculate_peak_area_analytical(amplitude, width, peak_type='lorentzian')` - Calculate peak area
- `find_peak_lorentzian(...)` - Find and fit peaks using Lorentzian

**Reason**: Core spectral processing functions that are part of the processing pipeline. Similar functions may already exist in `processing.py`.

**Target**: `src/pipeline/processing.py` (add to existing processing module)

---

#### 3. **Core Processing Functions** → `src/pipeline/processing.py`

Main data processing functions:

- `process_scan_v2(...)` - Process single scan (v3 workflow)
  - **Note**: Consider renaming to `process_scan_v3()` for clarity
- `aggregate_summaries_to_dataframe_v3(...)` - Aggregate scan summaries

**Reason**: These are core processing functions that should be reusable. `process_scan_v2` is actually the v3 processing workflow.

**Target**: `src/pipeline/processing.py` (add new functions or extend existing ones)

---

#### 4. **Signal Processing Utilities** → `src/pipeline/utils.py`

Signal processing helper functions:

- `remove_spikes_hampel(...)` - Remove spikes using Hampel filter

**Reason**: General-purpose signal processing utility that could be reused.

**Target**: `src/pipeline/utils.py` (add to existing utils module)

---

#### 5. **Baseline Correction** → `src/pipeline/post_processing.py`

Baseline correction functions:

- `apply_baseline_correction_v3(...)` - Apply baseline correction to normalized data

**Reason**: Post-processing function that fits with existing `apply_als_and_gaussian` in `post_processing.py`.

**Target**: `src/pipeline/post_processing.py` (add to existing post-processing module)

---

#### 6. **Ratio Calculation** → `src/pipeline/post_processing.py`

Data transformation functions:

- `calculate_ratios(df)` - Calculate fluorescence ratios

**Reason**: Post-processing data transformation that fits with other post-processing functions.

**Target**: `src/pipeline/post_processing.py` (add to existing post-processing module)

---

#### 7. **Plotting Functions** → `src/pipeline/plotting.py`

Visualization functions:

- `plot_normalized_baseline_correction_comparison(...)` - Plot before/after baseline correction
- `plot_normalized_smoothing_comparison(...)` - Plot before/after smoothing
- `plot_representative_raman_spectrum(...)` - Plot representative spectrum
- `plot_ratio_timeseries(...)` - Plot ratio timeseries

**Reason**: Visualization functions that belong with other plotting functions in `plotting.py`.

**Target**: `src/pipeline/plotting.py` (add to existing plotting module)

---

#### 8. **Post-Processing Functions** → `src/pipeline/post_processing.py`

Advanced post-processing:

- `apply_als_and_gaussian_v3(...)` - ALS baseline correction for v3 (ratios)
  - **Note**: Similar to existing `apply_als_and_gaussian` but for v3 workflow
- `compute_fourier_transform_v3(...)` - FFT analysis for v3
  - **Note**: Similar to existing `compute_fourier_transform` but for v3 workflow
- `compute_diurnal_average_v3(...)` - Diurnal averaging for v3
  - **Note**: Similar to existing `compute_diurnal_average` but for v3 workflow

**Reason**: These are v3-specific versions of existing post-processing functions. Could either:
- Replace existing functions (if v3 becomes standard)
- Keep both versions (if both are needed)
- Refactor to be more generic

**Target**: `src/pipeline/post_processing.py` (add to existing post-processing module)

---

### ❌ **Should Stay in `src/cli/`**

#### CLI-Specific Functions

- `build_parser()` - Argument parser setup
- `main()` - Main CLI entry point

**Reason**: These are CLI-specific and should remain in the CLI module.

---

## Implementation Plan for v4

### Step 1: Move Functions to src/pipeline/

1. **Phase 1: Utilities** (Low risk, high reuse)
   - Move config parsing functions → `src/pipeline/utils.py`
   - Move `remove_spikes_hampel` → `src/pipeline/utils.py`

2. **Phase 2: Core Processing** (Medium risk, high impact)
   - Move peak fitting functions → `src/pipeline/processing.py`
   - Move `process_scan_v2` → `src/pipeline/processing.py` (rename to `process_scan_v3`)
   - Move `aggregate_summaries_to_dataframe_v3` → `src/pipeline/processing.py`

3. **Phase 3: Post-Processing** (Medium risk)
   - Move `apply_baseline_correction_v3` → `src/pipeline/post_processing.py`
   - Move `calculate_ratios` → `src/pipeline/post_processing.py`
   - Move v3 post-processing functions → `src/pipeline/post_processing.py`

4. **Phase 4: Plotting** (Low risk)
   - Move all plotting functions → `src/pipeline/plotting.py`

### Step 2: Create main_test_v4.py

1. Copy `src/cli/main_test_v3.py` to `src/cli/main_test_v4.py`
2. Update imports to use functions from `src/pipeline/` modules
3. Remove function definitions (they're now in pipeline modules)
4. Keep only CLI-specific code (`build_parser()`, `main()`)
5. Update docstrings and comments to reflect v4

### Step 3: Create Root-Level Wrapper

Create `main_test_v4.py` at root level:
```python
from src.cli.main_test_v4 import main

if __name__ == "__main__":
    main()
```

---

## Implementation Strategy for v4

### Approach: Clean Refactoring
1. **Move functions to pipeline modules** (as-is initially)
2. **Create `main_test_v4.py`** that imports from pipeline modules
3. **Keep `main_test_v3.py` unchanged** for stability
4. **Test v4 thoroughly** before considering it as replacement

### Benefits of This Approach
- ✅ `main_test_v3.py` remains stable and working
- ✅ Can compare v3 vs v4 side-by-side
- ✅ Gradual migration path
- ✅ Cleaner architecture in v4
- ✅ Functions become reusable for other modules

### Future Considerations
- Once v4 is proven stable, could consider:
  - Making v4 the default
  - Deprecating v3
  - Or keeping both for different use cases

---

## Benefits of Refactoring

1. **Code Reusability**: Functions can be used by other CLI modules
2. **Better Organization**: Clear separation of concerns
3. **Easier Testing**: Functions can be tested independently
4. **Maintainability**: Changes to processing logic in one place
5. **Consistency**: Follows the same pattern as existing codebase

---

## Potential Issues

1. **Naming Conflicts**: Some v3 functions have similar names to existing functions
   - Solution: Use `_v3` suffix or rename appropriately

2. **Dependencies**: Functions may have dependencies on each other
   - Solution: Move related functions together, update imports

3. **Breaking Changes**: Moving functions could break existing code
   - Solution: Update all imports, test thoroughly

4. **Circular Imports**: Moving functions might create import cycles
   - Solution: Careful module organization, use relative imports where needed

---

## Summary

**Total Functions to Move: 20**
- Config parsing: 3 functions
- Peak fitting: 3 functions
- Core processing: 2 functions
- Signal processing: 1 function
- Baseline correction: 1 function
- Ratio calculation: 1 function
- Plotting: 4 functions
- Post-processing: 5 functions

**Functions to Keep in CLI: 2**
- `build_parser()`
- `main()`

This refactoring will significantly improve code organization and make the codebase more maintainable.

