"""
Automated batch processing script for all in_planta profiles.

This script runs main_test_v4.py (or main_test_v3.py) with --batch flag for all in_planta profiles
found in the pipeline.yml configuration file.

Usage:
    python batch_process_in_planta_profiles.py --list            # Show all in planta profiles
    python batch_process_in_planta_profiles.py                   # Process all profiles
    python batch_process_in_planta_profiles.py --max-scans 100   # Limit scans per profile
    python batch_process_in_planta_profiles.py --skip-failed     # Continue even if a profile fails
    python batch_process_in_planta_profiles.py --version v3      # Use v3 instead of v4
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Get the script directory and project root
SCRIPT_DIR = Path(__file__).resolve().parent  # src/cli/
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # Go up from src/cli/ to project root
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yml"

# Import from pipeline
import sys as sys_module
PROJECT_SRC_DIR = SCRIPT_DIR.parent  # src/
if str(PROJECT_SRC_DIR) not in sys_module.path:
    sys_module.path.insert(0, str(PROJECT_SRC_DIR))

from pipeline.config_loader import get_profiles_by_experiment


def run_profile_processing(profile_name, version='v4', max_scans=None, skip_scans=500, additional_args=None):
    """
    Run main_test_v4.py (or main_test_v3.py) for a specific profile.
    
    Parameters:
    -----------
    profile_name : str
        Name of the profile to process
    version : str
        Version to use ('v3' or 'v4', default: 'v4')
    max_scans : int, optional
        Maximum number of scans to process
    skip_scans : int, optional
        Number of scans to skip from beginning (default: 500)
    additional_args : list, optional
        Additional command-line arguments to pass
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Processing profile: {profile_name} (using {version})")
    print(f"{'='*80}")
    
    # Determine which main script to use
    if version == 'v3':
        main_script = PROJECT_ROOT / "main_test_v3.py"
    else:  # v4 (default)
        main_script = PROJECT_ROOT / "main_test_v4.py"
    
    # Build command
    cmd = [sys.executable, str(main_script), '--profile', profile_name, '--batch']
    
    if skip_scans > 0:
        cmd.extend(['--skip-scans', str(skip_scans)])
    
    if max_scans:
        cmd.extend(['--max-scans', str(max_scans)])
    
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        print(f"\n✓ Successfully completed: {profile_name}")
        print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to process: {profile_name}")
        print(f"Error code: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error processing {profile_name}: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch process all in_planta profiles using main_test_v4.py (or v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show all in planta profiles without processing
  python batch_process_in_planta_profiles.py --list
  
  # Process all profiles with default settings (v4)
  python batch_process_in_planta_profiles.py
  
  # Process all profiles with max 100 scans each
  python batch_process_in_planta_profiles.py --max-scans 100
  
  # Process all profiles, continue even if one fails
  python batch_process_in_planta_profiles.py --skip-failed
  
  # Process specific profiles only
  python batch_process_in_planta_profiles.py --profiles "bokchoy_control_6to22_Temp_Hum_Variable_Run1,Nb_Control_6to22_Temp_Hum_Variable_Run1"
  
  # Use v3 instead of v4
  python batch_process_in_planta_profiles.py --version v3
        """
    )
    
    parser.add_argument(
        '--version',
        type=str,
        choices=['v3', 'v4'],
        default='v4',
        help='Version to use: v3 or v4 (default: v4)'
    )
    
    parser.add_argument(
        '--max-scans',
        type=int,
        default=None,
        help='Maximum number of scans to process per profile (default: all scans)'
    )
    
    parser.add_argument(
        '--skip-scans',
        type=int,
        default=500,
        help='Number of scans to skip from beginning (default: 500)'
    )
    
    parser.add_argument(
        '--skip-failed',
        action='store_true',
        help='Continue processing other profiles even if one fails'
    )
    
    parser.add_argument(
        '--profiles',
        type=str,
        default=None,
        help='Comma-separated list of specific profiles to process (default: all in_planta profiles)'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        default=str(CONFIG_PATH),
        help='Path to pipeline.yml config file'
    )
    
    parser.add_argument(
        '--additional-args',
        type=str,
        default=None,
        help='Additional arguments to pass to main_test_v4.py (e.g., "--baseline-threshold 7.5 --baseline-window 15")'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='Show all in planta profiles and exit without processing'
    )
    
    args = parser.parse_args()
    
    # Get profiles to process
    if args.profiles:
        profiles_to_process = [p.strip() for p in args.profiles.split(',') if p.strip()]
    else:
        profiles_to_process = get_profiles_by_experiment(str(args.config_file), 'in planta')
    
    if not profiles_to_process:
        print("No in planta profiles found!")
        return
    
    # If --list flag is set, just show profiles and exit
    if args.list:
        print(f"\n{'='*80}")
        print(f"IN PLANTA PROFILES ({len(profiles_to_process)} total)")
        print(f"{'='*80}\n")
        for i, profile in enumerate(profiles_to_process, 1):
            print(f"  {i:2d}. {profile}")
        print(f"\n{'='*80}\n")
        return
    
    # Continue with processing
    if args.profiles:
        print(f"Processing specified profiles: {profiles_to_process}")
    else:
        print(f"Found {len(profiles_to_process)} in_planta profiles to process")
    
    print(f"\nProfiles to process:")
    for i, profile in enumerate(profiles_to_process, 1):
        print(f"  {i}. {profile}")
    
    # Parse additional arguments
    additional_args = []
    if args.additional_args:
        additional_args = args.additional_args.split()
    
    # Process each profile
    print(f"\n{'='*80}")
    print(f"Starting batch processing of {len(profiles_to_process)} profiles (using {args.version})")
    print(f"{'='*80}")
    
    start_time = datetime.now()
    results = {}
    
    for i, profile_name in enumerate(profiles_to_process, 1):
        print(f"\n[{i}/{len(profiles_to_process)}] Processing: {profile_name}")
        
        success = run_profile_processing(
            profile_name,
            version=args.version,
            max_scans=args.max_scans,
            skip_scans=args.skip_scans,
            additional_args=additional_args
        )
        
        results[profile_name] = success
        
        if not success and not args.skip_failed:
            print(f"\n✗ Stopping batch processing due to failure in {profile_name}")
            print("Use --skip-failed to continue processing other profiles even if one fails.")
            break
    
    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total profiles: {len(profiles_to_process)}")
    print(f"Successful: {sum(results.values())}")
    print(f"Failed: {len(results) - sum(results.values())}")
    print(f"Total duration: {duration}")
    print(f"\nResults:")
    for profile, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {status}: {profile}")
    
    # Exit with error code if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()

