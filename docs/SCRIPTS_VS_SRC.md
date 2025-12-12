# Differences Between `scripts/` and `src/`

This document explains the organizational difference between the `scripts/` and `src/` directories.

---

## Quick Summary

| Directory | Purpose | Usage | Structure |
|-----------|---------|-------|-----------|
| **`src/`** | **Core library code** | Imported as modules | Organized into packages (`pipeline/`, `cli/`, `analysis/`) |
| **`scripts/`** | **Standalone utility scripts** | Run directly or imported for specific functions | Collection of utility/testing scripts |

---

## `src/` - Core Library Code

**Purpose**: Reusable modules that form the core functionality of the pipeline.

**Structure**:
```
src/
├── pipeline/          # Core processing modules
│   ├── processing.py  # Spectral processing functions
│   ├── ingestion.py   # Data loading
│   ├── plotting.py    # Visualization functions
│   ├── post_processing.py  # ALS, FFT, diurnal analysis
│   └── utils.py       # Utility functions
├── cli/               # CLI implementations
│   └── main*.py       # Main entry point implementations
└── analysis/          # Analysis utilities
    └── fourier_aggregated.py
```

**Characteristics**:
- ✅ **Imported as modules**: `from pipeline.processing import process_raman_data`
- ✅ **Reusable functions**: Used by main scripts and other modules
- ✅ **Well-organized packages**: Clear module structure
- ✅ **Core functionality**: Essential processing logic
- ✅ **Part of the pipeline**: Required for the system to work

**Example Usage**:
```python
from pipeline.config_loader import load_profile_config
from pipeline.ingestion import load_raman_dataset
from pipeline.processing import process_all_raman_data
```

---

## `scripts/` - Standalone Utility Scripts

**Purpose**: Utility scripts, testing tools, and helper scripts that support the pipeline.

**Structure**:
```
scripts/
├── generate_profiles.py           # Auto-generate config profiles
├── optimize_als_parameters.py    # Parameter optimization
├── fix_baseline_shifts_v2.py      # Baseline correction utilities
├── test_*.py                      # Testing scripts
├── analysis/                      # Analysis comparison scripts
│   ├── analyze_fluorescence_jumps.py
│   └── create_overlay_plots.py
└── test_outputs/                  # Test output files
```

**Characteristics**:
- ✅ **Run directly**: `python scripts/generate_profiles.py`
- ✅ **Standalone tools**: Can be used independently
- ✅ **Testing/development**: Many are for testing or optimization
- ✅ **Sometimes imported**: Some functions imported by main scripts
- ✅ **Supporting tools**: Help with setup, testing, or analysis

**Example Usage**:
```python
# Run directly
python scripts/generate_profiles.py --overwrite

# Or imported for specific functions
from scripts.fix_baseline_shifts_v2 import correct_baseline_shifts
```

---

## Key Differences

### 1. **Import Pattern**

**`src/`** - Imported as packages:
```python
from pipeline.processing import process_raman_data
from pipeline.utils import lieberfit
```

**`scripts/`** - Imported as modules or run directly:
```python
# Direct execution
python scripts/generate_profiles.py

# Or imported
from scripts.fix_baseline_shifts_v2 import correct_baseline_shifts
```

### 2. **Dependencies**

**`src/`**:
- Core modules depend on each other
- Part of the main pipeline
- Required for the system to function

**`scripts/`**:
- Often depend on `src/` modules
- Optional tools
- Can be run independently

### 3. **Organization**

**`src/`**:
- Well-organized into packages (`pipeline/`, `cli/`, `analysis/`)
- Clear module boundaries
- Follows Python package conventions

**`scripts/`**:
- Collection of utility scripts
- Less formal structure
- Organized by function (testing, optimization, analysis)

### 4. **Purpose**

**`src/`**:
- Core business logic
- Data processing algorithms
- Reusable functions

**`scripts/`**:
- Development tools
- Testing utilities
- One-off analysis scripts
- Configuration helpers

---

## Examples from the Codebase

### `src/pipeline/processing.py`
```python
def process_raman_data(raman_df, scan_number, config):
    """Core processing function - used by main pipeline"""
    # Processing logic...
```

**Used by**: `main_test_v3.py`, `src/cli/main.py`, etc.

### `scripts/generate_profiles.py`
```python
"""
Auto-generate pipeline profiles for each Raw data file.
Run directly: python scripts/generate_profiles.py
"""
```

**Used by**: Run directly when setting up new experiments

### `scripts/fix_baseline_shifts_v2.py`
```python
def correct_baseline_shifts(...):
    """Baseline correction - imported by main_test_v3.py"""
```

**Used by**: Imported by `main_test_v3.py` when needed

---

## When to Add Code Where

### Add to `src/` when:
- ✅ Creating reusable processing functions
- ✅ Adding core pipeline functionality
- ✅ Building modules that will be imported
- ✅ Creating CLI implementations
- ✅ Adding analysis utilities that are part of the core system

### Add to `scripts/` when:
- ✅ Creating one-off analysis scripts
- ✅ Building testing/development tools
- ✅ Creating configuration helpers
- ✅ Making optimization utilities
- ✅ Writing comparison/validation scripts

---

## Summary

- **`src/`** = **Library code** (imported, reusable, core functionality)
- **`scripts/`** = **Utility scripts** (run directly, supporting tools, testing)

Think of it as:
- **`src/`** = The engine (core functionality)
- **`scripts/`** = The tools (utilities and helpers)

Both are important, but serve different roles in the project structure.

