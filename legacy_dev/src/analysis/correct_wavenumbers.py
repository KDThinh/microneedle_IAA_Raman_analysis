"""
Script to correct wavenumbers in Raman data files when excitation wavelength was incorrectly recorded.

Usage:
    python correct_wavenumbers.py <input_file> <output_file> --recorded 840 --actual 830
"""

import sys
import numpy as np
import argparse
from pathlib import Path

# Add src to path so we can import the utility function
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from pipeline.utils import correct_wavenumber_for_excitation


def read_raman_data_file(file_path):
    """
    Read Raman data file where first row contains wavenumbers and subsequent rows contain scans.
    
    Returns:
    --------
    wavenumbers : ndarray
        Array of wavenumber values from first row
    data_lines : list
        List of lines containing scan data (date, time, seconds, intensities)
    header_line : str
        The first line containing wavenumbers
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # First line contains wavenumbers
    header_line = lines[0].strip()
    wavenumbers = np.array([float(x) for x in header_line.split('\t')])
    
    # Remaining lines contain scan data
    data_lines = lines[1:]
    
    return wavenumbers, data_lines, header_line


def write_corrected_file(output_path, corrected_wavenumbers, data_lines):
    """
    Write corrected data to output file.
    """
    with open(output_path, 'w') as f:
        # Write corrected wavenumbers header
        f.write('\t'.join([f'{w:.6f}' for w in corrected_wavenumbers]) + '\n')
        # Write remaining data lines unchanged
        f.writelines(data_lines)


def main():
    parser = argparse.ArgumentParser(
        description='Correct wavenumbers in Raman data file for incorrect excitation wavelength recording'
    )
    parser.add_argument('input_file', type=str, help='Path to input Raman data file')
    parser.add_argument('output_file', type=str, help='Path to output corrected Raman data file')
    parser.add_argument('--recorded', type=float, default=840, 
                        help='Excitation wavelength (nm) that was incorrectly recorded (default: 840)')
    parser.add_argument('--actual', type=float, default=830,
                        help='Actual excitation wavelength (nm) that was used (default: 830)')
    
    args = parser.parse_args()
    
    # Read input file
    print(f"Reading data from: {args.input_file}")
    wavenumbers, data_lines, header_line = read_raman_data_file(args.input_file)
    print(f"Found {len(wavenumbers)} wavenumber points")
    print(f"Found {len(data_lines)} scan lines")
    print(f"Wavenumber range: {wavenumbers.min():.2f} - {wavenumbers.max():.2f} cm^-1")
    
    # Correct wavenumbers
    print(f"\nCorrecting wavenumbers from {args.recorded} nm to {args.actual} nm excitation...")
    corrected_wavenumbers = correct_wavenumber_for_excitation(
        wavenumbers, 
        recorded_excitation_nm=args.recorded, 
        actual_excitation_nm=args.actual
    )
    print(f"Corrected wavenumber range: {corrected_wavenumbers.min():.2f} - {corrected_wavenumbers.max():.2f} cm^-1")
    
    # Calculate differences
    differences = corrected_wavenumbers - wavenumbers
    print(f"\nWavenumber correction statistics:")
    print(f"  Mean difference: {differences.mean():.4f} cm^-1")
    print(f"  Std deviation: {differences.std():.4f} cm^-1")
    print(f"  Min difference: {differences.min():.4f} cm^-1")
    print(f"  Max difference: {differences.max():.4f} cm^-1")
    
    # Write output file
    print(f"\nWriting corrected data to: {args.output_file}")
    write_corrected_file(args.output_file, corrected_wavenumbers, data_lines)
    print("Done!")


if __name__ == '__main__':
    main()

