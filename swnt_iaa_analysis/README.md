# SWNT IAA Raman Analysis

Professional Python package for analyzing SWNT IAA Raman spectroscopy data.

## Installation

Install the package in development mode:

```bash
cd swnt_iaa_analysis
pip install -e .
```

This will install the package and make it available as `swnt-iaa-analysis` command.

## Usage

### Command Line Interface

#### Analyze a single profile:

```bash
swnt-iaa-analysis analyze bokchoy_control_6to22_Temp_Hum_Variable_Run1
```

#### List available profiles:

```bash
swnt-iaa-analysis list-profiles
```

#### Batch process multiple profiles:

```bash
swnt-iaa-analysis batch-process --experiment "in planta"
```

### Python API

```python
from swnt_iaa_analysis.pipeline import RamanPipeline

pipeline = RamanPipeline(
    config_path="config.yaml",
    profile_name="bokchoy_control_6to22_Temp_Hum_Variable_Run1",
    output_dir="./results",
    algorithm="v4"
)

results = pipeline.run()
```

## Package Structure

```
swnt_iaa_analysis/
├── setup.py              # Package setup
├── config.yaml           # Default configuration
├── swnt_iaa_analysis/    # Source code
│   ├── __init__.py
│   ├── core/             # Core processing
│   │   ├── loader.py     # Data loading
│   │   ├── preprocessing.py  # Spectral preprocessing
│   │   ├── baseline.py   # Baseline correction
│   │   └── utils.py      # Utility functions
│   ├── analysis/         # Scientific analysis
│   │   ├── peaks.py      # Peak fitting
│   │   ├── ratios.py     # Ratio calculations
│   │   └── fourier.py    # Fourier transforms
│   ├── visualization/    # Plotting
│   │   └── plotting.py   # Plotting functions
│   ├── io/               # I/O operations
│   │   ├── config.py     # Configuration management
│   │   └── exporter.py  # Data export
│   ├── pipeline.py       # Main pipeline class
│   └── cli.py            # Command-line interface
└── tests/                # Unit tests
```

## Configuration

The package uses YAML configuration files with profile-based inheritance. See `config.yaml` for examples.

## Development

This package follows the same architecture as `microneedle_nir_imaging_analysis` for consistency and maintainability.

