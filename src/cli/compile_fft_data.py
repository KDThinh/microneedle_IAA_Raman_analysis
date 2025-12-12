"""
CLI script to compile FFT (Fourier Transform) CSV files from multiple profiles.

This script finds the latest FFT CSV files (fft_full_gband and fft_full_raman) for each profile
and combines them into a single compiled CSV/Parquet file, filtering frequency from 0 to 0.5.

Usage:
    python compile_fft_data.py --output compiled_fft.csv
    python compile_fft_data.py --fft-types gband --compression parquet
    python compile_fft_data.py --frequency-max 0.3 --output compiled_fft_0.3.csv
"""

import argparse
import re
import sys
from pathlib import Path
from datetime import datetime

# Get the script directory and project root
SCRIPT_DIR = Path(__file__).resolve().parent  # src/cli/
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up from src/cli/ to project root
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yml"

# Import from pipeline
PROJECT_SRC_DIR = SCRIPT_DIR.parent  # src/
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from pipeline.fft_compiler import compile_fft_data
from pipeline.batch_summary_compiler import _get_default_output_path


def _get_default_fft_output_path(config_path: str, version: str = 'v4') -> Path:
    """
    Get the default output path for compiled FFT data.
    
    Returns path to: IAA Nanosensor Experiment/Result Summary/compiled_fft_data_{version}_{timestamp}.csv
    """
    default_path = _get_default_output_path(config_path, version=version)
    # Replace batch_summaries with fft_data in the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = default_path.parent / f"compiled_fft_data_{version}_{timestamp}.csv"
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Compile FFT (Fourier Transform) CSV files from multiple profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile all in_planta profiles (default) to compiled_fft.csv
  python compile_fft_data.py --output compiled_fft.csv
  
  # Compile with Parquet format (highly compressed, recommended for large files)
  python compile_fft_data.py --output compiled_fft.parquet --compression parquet
  
  # Compile only gband FFT data
  python compile_fft_data.py --fft-types gband --output compiled_fft_gband.csv
  
  # Compile with custom frequency range
  python compile_fft_data.py --frequency-min 0.0 --frequency-max 0.3 --output compiled_fft_0.3.csv
  
  # Compile specific profiles
  python compile_fft_data.py --profiles "profile1,profile2" --output compiled_fft.csv
  
  # Compile in_vitro profiles
  python compile_fft_data.py --experiment-type "in vitro" --output compiled_fft_in_vitro.csv
        """
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for compiled FFT file (default: IAA Nanosensor Experiment/Result Summary/compiled_fft_data_{version}_{timestamp}.csv)'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        default=str(CONFIG_PATH),
        help='Path to pipeline.yml config file'
    )
    
    parser.add_argument(
        '--profiles',
        type=str,
        default=None,
        help='Comma-separated list of specific profiles to compile (default: all matching experiment-type)'
    )
    
    parser.add_argument(
        '--experiment-type',
        type=str,
        default='in planta',
        help='Filter profiles by experiment type (default: "in planta"). Ignored if --profiles is provided.'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        choices=['v3', 'v4'],
        default='v4',
        help='Version to look for: v3 or v4 (default: v4)'
    )
    
    parser.add_argument(
        '--fft-types',
        type=str,
        default='gband,raman',
        help='Comma-separated list of FFT types to compile: gband,raman (default: both)'
    )
    
    parser.add_argument(
        '--frequency-min',
        type=float,
        default=0.0,
        help='Minimum frequency to include (cycles/hour, default: 0.0)'
    )
    
    parser.add_argument(
        '--frequency-max',
        type=float,
        default=0.5,
        help='Maximum frequency to include (cycles/hour, default: 0.5)'
    )
    
    parser.add_argument(
        '--no-profile-column',
        dest='add_profile_column',
        action='store_false',
        help='Do not add a "Profile" column to identify source profile'
    )
    
    parser.add_argument(
        '--no-metadata',
        dest='add_metadata_columns',
        action='store_false',
        help='Do not add metadata columns (plant_type, treatment, light_cycle, etc.)'
    )
    
    parser.add_argument(
        '--compression',
        type=str,
        choices=['gzip', 'bz2', 'xz', 'zip', 'parquet'],
        default=None,
        help='Compression format: gzip/bz2/xz/zip for CSV, or parquet for Parquet format (highly compressed, recommended)'
    )
    
    parser.add_argument(
        '--no-optimize',
        dest='optimize_dtypes',
        action='store_false',
        help='Do not optimize data types (keeps original precision, larger file size)'
    )
    
    parser.add_argument(
        '--float-precision',
        type=int,
        default=6,
        help='Number of decimal places for float columns when optimizing (default: 6)'
    )
    
    args = parser.parse_args()
    
    # Parse profiles list if provided
    profiles = None
    if args.profiles:
        profiles = [p.strip() for p in args.profiles.split(',') if p.strip()]
    
    # Parse FFT types
    fft_types = [t.strip() for t in args.fft_types.split(',') if t.strip()]
    if not fft_types:
        fft_types = ['gband', 'raman']  # Default to both
    
    # Determine experiment_type
    experiment_type = args.experiment_type if not profiles else None
    
    # Determine output path (use default if not provided)
    output_path = args.output
    if output_path is None:
        output_path = str(_get_default_fft_output_path(args.config_file, version=args.version))
        print(f"\nNo output path specified. Using default: {output_path}")
    else:
        # Add timestamp to output filename if it doesn't already have one
        output_path_obj = Path(output_path)
        # Check if filename already contains a timestamp pattern (YYYYMMDD_HHMMSS)
        if not re.search(r'\d{8}_\d{6}', output_path_obj.stem):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Insert timestamp before file extension
            new_stem = f"{output_path_obj.stem}_{timestamp}"
            output_path = str(output_path_obj.parent / f"{new_stem}{output_path_obj.suffix}")
            print(f"\nAdded timestamp to output filename: {output_path}")
    
    try:
        # Compile FFT data
        compiled_df = compile_fft_data(
            config_path=args.config_file,
            output_path=output_path,
            profiles=profiles,
            experiment_type=experiment_type,
            version=args.version,
            fft_types=fft_types,
            frequency_min=args.frequency_min,
            frequency_max=args.frequency_max,
            add_profile_column=args.add_profile_column,
            add_metadata_columns=args.add_metadata_columns,
            optimize_dtypes=args.optimize_dtypes,
            float_precision=args.float_precision,
            compression=args.compression,
        )
        
        if not args.output:
            print("\n" + "="*80)
            print("Compiled DataFrame (first 5 rows):")
            print("="*80)
            print(compiled_df.head())
            print(f"\nTotal rows: {len(compiled_df)}")
            print("\nNote: Use --output to save the compiled FFT file")
        
        print("\n✓ Compilation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

