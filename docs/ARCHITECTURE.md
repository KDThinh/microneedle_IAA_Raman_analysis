# Project Architecture

This document explains the architecture of the SWNT-IAA Analysis Pipeline and how different components relate to each other.

---

## Architecture Overview

The project has evolved through different versions with different architectural patterns:

### Old Architecture (v1/v2 - Archived)

**Pattern**: Separation of entry points and CLI implementations

```
Root Level (Entry Points)
├── main.py                    # Thin wrapper
├── main_wo_append.py          # Thin wrapper
└── main_in_vitro.py           # Thin wrapper

src/cli/ (CLI Implementations)
├── main.py                    # Full CLI implementation
├── main_wo_append.py          # Full CLI implementation
└── main_in_vitro.py           # Full CLI implementation

src/pipeline/ (Core Modules)
├── processing.py
├── ingestion.py
└── ...
```

**How it worked:**
- Root-level files were thin wrappers: `from src.cli.main import main`
- `src/cli/` contained the actual CLI implementations
- `src/pipeline/` contained reusable processing modules
- Both `src/cli/` and root wrappers imported from `src/pipeline/`

**Example:**
```python
# archive/main.py (root level)
from src.cli.main import main

if __name__ == "__main__":
    main()
```

```python
# src/cli/main.py (implementation)
from pipeline.processing import process_all_raman_data
from pipeline.ingestion import load_raman_dataset
# ... full implementation
```

---

### New Architecture (v3 - Current)

**Pattern**: Thin wrapper with CLI implementation in src/cli/

```
Root Level (Entry Points)
├── main_test_v3.py            # Thin wrapper
├── main_test_v4.py            # Thin wrapper (refactored version)
└── batch_process_in_planta_profiles.py  # Batch processing wrapper

src/cli/ (CLI Implementations)
├── main_test_v3.py            # Full CLI implementation (2663 lines)
├── main_test_v4.py            # Refactored CLI implementation (modular)
└── batch_process_in_planta_profiles.py  # Batch processing implementation

src/pipeline/ (Core Modules)
├── processing.py
├── ingestion.py
└── ...
```

**How it works:**
- Root-level `main_test_v3.py` is a thin wrapper: `from src.cli.main_test_v3 import main`
- `src/cli/main_test_v3.py` contains the full CLI implementation
- Directly imports from `src/pipeline/` modules
- Follows the same pattern as the old architecture but for v3

**Example:**
```python
# main_test_v3.py (root level - thin wrapper)
from src.cli.main_test_v3 import main

if __name__ == "__main__":
    main()
```

```python
# src/cli/main_test_v3.py (full implementation)
from pipeline.config_loader import load_profile_config
from pipeline.ingestion import load_raman_dataset
from pipeline.utils import add_day_night_shading, create_dir_if_needed, lieberfit
from scripts.fix_baseline_shifts_v2 import correct_baseline_shifts

# ... all processing logic ...
def main():
    # Full implementation here
    pass
```

---

## Key Differences

| Aspect | Old Architecture (v1/v2) | New Architecture (v3) |
|--------|-------------------------|----------------------|
| **Entry Points** | Thin wrappers in root | Self-contained scripts |
| **CLI Logic** | In `src/cli/` | Inline in main script |
| **Uses `src/cli/`** | ✅ Yes | ❌ No |
| **Uses `src/pipeline/`** | ✅ Yes | ✅ Yes |
| **Code Reuse** | High (shared CLI modules) | Lower (self-contained) |
| **File Size** | Small wrappers + large CLI files | Very large single file (2663 lines) |
| **Flexibility** | Easy to create variants | Harder to create variants |

---

## Current Usage

### Active Components

**`main_test_v3.py`** (Root level - Current):
- ✅ Thin wrapper that imports from `src/cli/main_test_v3`
- Status: **Active**

**`src/cli/main_test_v3.py`** (Implementation - Current):
- ✅ Uses: `src/pipeline/*`
- ✅ Uses: `scripts/*`
- ✅ Contains full CLI implementation
- Status: **Active**

**`batch_process_in_planta_profiles.py`** (Root level - Current):
- ✅ Thin wrapper that imports from `src/cli/batch_process_in_planta_profiles`
- Status: **Active**

**`src/cli/batch_process_in_planta_profiles.py`** (Implementation - Current):
- ✅ Batch processes all in_planta profiles
- ✅ Supports both v3 and v4 (default: v4)
- ✅ Uses: `main_test_v3.py` or `main_test_v4.py`
- Status: **Active**

### Legacy Components

**`src/cli/main*.py`** (Other CLI files):
- Used by: Archived root-level main files
- Status: **Legacy** (kept for reference, old architecture)

**Archived main files** (`archive/main*.py`):
- Used by: Nothing (archived)
- Status: **Archived**

---

## Why the Change?

The architectural change from v1/v2 to v3 likely occurred because:

1. **Simplification**: Self-contained scripts are easier to understand and modify
2. **Rapid Development**: v3 was developed as a test/experimental version
3. **Different Workflow**: v3 focuses on normalized data with ratios, different from v1/v2
4. **Less Abstraction**: Direct implementation allows faster iteration

---

## Implications

### For Development

- **Adding new features to v3**: Modify `main_test_v3.py` directly
- **Creating variants**: Copy `main_test_v3.py` and modify (not ideal, but current pattern)
- **Reusing code**: Extract common functions to `src/pipeline/` or `scripts/`

### For Maintenance

- **`src/cli/` is legacy**: Not used by current active code
- **Could be archived**: Consider moving `src/cli/` to `archive/` if not needed
- **Documentation**: Should clarify that v3 doesn't use `src/cli/`

---

## Recommendations

1. **Document the architecture**: Make it clear that v3 doesn't use `src/cli/`
2. **Consider refactoring**: If v3 becomes the standard, consider:
   - Moving `src/cli/` to `archive/` if truly obsolete
   - Or refactoring v3 to use a cleaner architecture
3. **Code organization**: Consider breaking up `main_test_v3.py` (2663 lines is very large)

---

## Summary

- **`main_test_v3.py`** (root) = Thin wrapper, imports from `src/cli/main_test_v3`
- **`src/cli/main_test_v3.py`** = Full CLI implementation, uses `src/pipeline/` and `scripts/`
- **`src/cli/main*.py`** (others) = Legacy CLI implementations for old architecture
- **`src/pipeline/`** = Core reusable modules, used by all architectures
- **Archived files** = Old architecture with separation of concerns

The current active code (`main_test_v3.py`) now follows the same architectural pattern as the old codebase, with separation between entry points and CLI implementations.

