"""
Entry point for batch processing all in_planta profiles.

This is a thin wrapper that imports the main implementation from src/cli/batch_process_in_planta_profiles.py.

Usage:
    python batch_process_in_planta_profiles.py --list            # Show all in planta profiles
    python batch_process_in_planta_profiles.py                   # Process all profiles (v4)
    python batch_process_in_planta_profiles.py --max-scans 100   # Limit scans per profile
    python batch_process_in_planta_profiles.py --skip-failed     # Continue even if a profile fails
    python batch_process_in_planta_profiles.py --version v3      # Use v3 instead of v4
"""
from src.cli.batch_process_in_planta_profiles import main


if __name__ == "__main__":
    main()

