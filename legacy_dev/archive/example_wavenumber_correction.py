"""
Example script demonstrating wavenumber correction for incorrect excitation wavelength.

This shows how to correct wavenumbers when the real excitation wavelength was 830 nm
but was recorded as 840 nm.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from pipeline.utils import (
    correct_wavenumber_for_excitation,
    raman_wavenumber_to_emission_nm,
    emission_nm_to_raman_wavenumber
)

# Example: Single scan correction
print("=" * 70)
print("Example: Correcting wavenumbers from 840 nm to 830 nm excitation")
print("=" * 70)

# Sample wavenumber values (in cm^-1) that were recorded assuming 840 nm excitation
recorded_wavenumbers = np.array([
    98.74, 102.03, 105.31, 108.60, 111.88, 115.16, 118.44, 121.71, 124.99, 128.26,
    131.53, 134.79, 138.06, 141.32, 144.58, 147.83, 151.09, 154.34, 157.59, 160.83
])

print(f"\nRecorded wavenumbers (assuming 840 nm excitation):")
print(f"  Range: {recorded_wavenumbers.min():.2f} - {recorded_wavenumbers.max():.2f} cm^-1")

# Correct to 830 nm excitation
corrected_wavenumbers = correct_wavenumber_for_excitation(
    recorded_wavenumbers,
    recorded_excitation_nm=840,
    actual_excitation_nm=830
)

print(f"\nCorrected wavenumbers (for 830 nm excitation):")
print(f"  Range: {corrected_wavenumbers.min():.2f} - {corrected_wavenumbers.max():.2f} cm^-1")

# Show differences
differences = corrected_wavenumbers - recorded_wavenumbers
print(f"\nCorrection differences:")
print(f"  Mean: {differences.mean():.4f} cm^-1")
print(f"  Std:  {differences.std():.4f} cm^-1")
print(f"  Min:  {differences.min():.4f} cm^-1")
print(f"  Max:  {differences.max():.4f} cm^-1")

# Verify by checking that emission wavelengths match
print(f"\nVerification:")
print(f"  Converting both to emission wavelengths to verify they match...")
emission_from_recorded = raman_wavenumber_to_emission_nm(recorded_wavenumbers, excitation_nm=840)
emission_from_corrected = raman_wavenumber_to_emission_nm(corrected_wavenumbers, excitation_nm=830)

emission_diff = emission_from_corrected - emission_from_recorded
print(f"  Emission wavelength differences (should be ~0):")
print(f"    Max difference: {np.abs(emission_diff).max():.6f} nm")
print(f"    Mean difference: {emission_diff.mean():.6e} nm")

if np.abs(emission_diff).max() < 1e-6:
    print("  [OK] Verification passed! Emission wavelengths match.")
else:
    print("  [WARNING] Emission wavelengths don't match exactly.")

# Show a few specific examples
print(f"\n" + "=" * 70)
print("Detailed example for first few wavenumbers:")
print("=" * 70)
print(f"{'Recorded (cm^-1)':>15} {'Corrected (cm^-1)':>18} {'Difference (cm^-1)':>20} {'Emission (nm)':>15}")
print("-" * 70)
for i in range(min(5, len(recorded_wavenumbers))):
    w_recorded = recorded_wavenumbers[i]
    w_corrected = corrected_wavenumbers[i]
    diff = differences[i]
    emission = emission_from_corrected[i]
    print(f"{w_recorded:>15.4f} {w_corrected:>18.4f} {diff:>20.4f} {emission:>15.2f}")

print(f"\n" + "=" * 70)
print("Usage in your pipeline:")
print("=" * 70)
print("""
To correct a data file:
    python correct_wavenumbers.py input_file.txt output_file.txt --recorded 840 --actual 830

To use in code:
    from pipeline.utils import correct_wavenumber_for_excitation
    
    corrected_wavenumbers = correct_wavenumber_for_excitation(
        wavenumbers, 
        recorded_excitation_nm=840, 
        actual_excitation_nm=830
    )
""")

