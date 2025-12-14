"""
Entry point for compiling batch summary CSV files from multiple profiles.

This is a thin wrapper that imports the main implementation from src/cli/compile_batch_summaries.py.

Usage:
    python compile_batch_summaries.py --output compiled_summaries.csv
    python compile_batch_summaries.py --profiles "profile1,profile2" --version v3
    python compile_batch_summaries.py --experiment-type "in vitro" --output compiled_in_vitro.csv
"""
from src.cli.compile_batch_summaries import main


if __name__ == "__main__":
    main()

