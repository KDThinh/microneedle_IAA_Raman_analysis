"""
Non-interactive test runner for bok_choy_control_6to22_run1 profile.
Automatically uses default datetime range (full dataset).
"""
import sys
from pathlib import Path
from unittest.mock import patch
from io import StringIO

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cli.main import main

def test_run_bok_choy():
    """Run main.py with bok_choy_control_6to22_run1 profile, using default datetime range."""
    
    # Mock input to provide default values (press Enter twice)
    with patch('sys.argv', ['main.py', '--profile', 'bok_choy_control_6to22_run1']):
        with patch('builtins.input', side_effect=['', '']):  # Two empty inputs for start and end
            try:
                print("=" * 60)
                print("Running test with bok_choy_control_6to22_run1 profile")
                print("Using default datetime range (full dataset)")
                print("=" * 60)
                main()
                print("\n" + "=" * 60)
                print("Test run completed successfully!")
                print("=" * 60)
            except Exception as e:
                print(f"\nError during test run: {e}")
                import traceback
                traceback.print_exc()
                return 1
    return 0

if __name__ == "__main__":
    exit(test_run_bok_choy())

