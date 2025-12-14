"""
CLI script to compile batch summary CSV files from multiple profiles.

This script finds the latest batch_summary CSV for each profile and combines them
into a single compiled CSV file.

Usage:
    python compile_batch_summaries.py --output compiled_summaries.csv
    python compile_batch_summaries.py --profiles "profile1,profile2" --version v3
    python compile_batch_summaries.py --experiment-type "in vitro" --output compiled_in_vitro.csv
"""

import argparse
import sys
from pathlib import Path

# Get the script directory and project root
SCRIPT_DIR = Path(__file__).resolve().parent  # src/cli/
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up from src/cli/ to project root
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yml"

# Import from pipeline
PROJECT_SRC_DIR = SCRIPT_DIR.parent  # src/
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

from pipeline.batch_summary_compiler import compile_batch_summaries, _get_default_output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compile batch summary CSV files from multiple profiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile all in_planta profiles (default) to compiled_summaries.csv
  python compile_batch_summaries.py --output compiled_summaries.csv
  
  # Compile with Parquet format (highly compressed, recommended for large files)
  python compile_batch_summaries.py --output compiled.parquet --compression parquet
  
  # Compile with gzip compression
  python compile_batch_summaries.py --output compiled.csv.gz --compression gzip
  
  # Compile specific profiles
  python compile_batch_summaries.py --profiles "profile1,profile2" --output compiled.csv
  
  # Compile in_vitro profiles
  python compile_batch_summaries.py --experiment-type "in vitro" --output compiled_in_vitro.csv
  
  # Compile v3 summaries instead of v4
  python compile_batch_summaries.py --version v3 --output compiled_v3.csv
  
  # Compile without saving (just print summary)
  python compile_batch_summaries.py
        """
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for compiled CSV file (default: IAA Nanosensor Experiment/Result Summary/compiled_batch_summaries_{version}_{timestamp}.csv)'
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
    
    # Determine experiment_type
    experiment_type = args.experiment_type if not profiles else None
    
    # Determine output path (use default if not provided)
    output_path = args.output
    if output_path is None:
        output_path = str(_get_default_output_path(args.config_file, version=args.version))
        print(f"\nNo output path specified. Using default: {output_path}")
    
    try:
        # Compile batch summaries
        compiled_df = compile_batch_summaries(
            config_path=args.config_file,
            output_path=output_path,
            profiles=profiles,
            experiment_type=experiment_type,
            version=args.version,
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
            print("\nNote: Use --output to save the compiled CSV file")
        
        print("\n✓ Compilation completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

