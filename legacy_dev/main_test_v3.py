"""
Entry point for processing v3 - continuation of v2 focusing on normalized data with ratios.

This is a thin wrapper that imports the main implementation from src/cli/main_test_v3.py.

Usage examples:
    # Batch processing (default skips first 500 scans)
    python main_test_v3.py --batch
    
    # Batch with baseline correction parameters
    python main_test_v3.py --batch --baseline-threshold 7.5 --baseline-window 15 --correct-smoothed
    
    # Batch with custom scan range
    python main_test_v3.py --batch --scans "100,200,300" --max-scans 10
"""
from src.cli.main_test_v3 import main


if __name__ == "__main__":
    main()
