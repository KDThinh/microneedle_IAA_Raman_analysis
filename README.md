# SWNT-IAA Analysis Pipeline v2

Raman/fluorescence spectroscopy analysis pipeline for SWNT-based nanosensor data for monitoring indole-3-acetic acid (IAA) levels in plants.

## Quick Start

```bash
# Process data with a profile
python main_test_v3.py --profile your_profile_name

# Batch process all profiles
python batch_process_in_planta_profiles.py
```

## Documentation

All documentation is located in the [`docs/`](docs/) folder:

- **[docs/PROCESSING_GUIDE.md](docs/PROCESSING_GUIDE.md)** - User guide (start here!)
- **[docs/DATA_PROCESSING_STEPS.md](docs/DATA_PROCESSING_STEPS.md)** - Complete workflow documentation
- **[docs/PROCESSING_ALGORITHM_ANALYSIS.md](docs/PROCESSING_ALGORITHM_ANALYSIS.md)** - Algorithm evaluation
- **[docs/DOCUMENTATION_COMPARISON.md](docs/DOCUMENTATION_COMPARISON.md)** - Documentation overview

See [docs/README.md](docs/README.md) for a complete documentation index.

## Project Structure

```
swnt_iaa_analysis_v2/
├── main_test_v3.py          # Main entry point (current version)
├── batch_process_in_planta_profiles.py  # Batch processing script for in_planta profiles
├── config/
│   └── pipeline.yml         # Configuration profiles
├── src/
│   ├── pipeline/           # Core processing modules
│   ├── cli/                # CLI implementations
│   └── analysis/           # Analysis utilities
├── scripts/                # Utility scripts
├── docs/                   # Documentation
└── archive/                # Archived old code
```

## Configuration

Edit `config/pipeline.yml` to create or modify experiment profiles. Each profile defines:
- Data source paths
- Processing parameters
- Metadata (plant type, treatment, light cycle, etc.)
- Output settings

See [docs/PROCESSING_GUIDE.md](docs/PROCESSING_GUIDE.md) for details on creating profiles.

## Requirements

- Python 3.7+
- numpy, pandas, matplotlib, scipy
- PyYAML (for configuration)
- tqdm (for progress bars)

Install dependencies:
```bash
pip install numpy pandas matplotlib scipy pyyaml tqdm
```

