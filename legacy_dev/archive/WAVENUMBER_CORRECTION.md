# Wavenumber Correction for Incorrect Excitation Wavelength

## Problem

When Raman spectroscopy data is collected, wavenumbers are calculated based on the excitation wavelength. If the excitation wavelength was incorrectly recorded (e.g., recorded as 840 nm when it was actually 830 nm), the wavenumbers in the data file will be incorrect and need to be corrected.

## Solution

The correction algorithm works by:

1. **Preserving the actual emission wavelength**: The physical emission wavelength of the Raman scattered light is invariant - it doesn't change based on what excitation wavelength was recorded.

2. **Two-step conversion**:
   - Convert the recorded wavenumber (based on incorrect excitation) to the actual emission wavelength
   - Convert that emission wavelength back to wavenumber using the correct excitation wavelength

## Mathematical Derivation

The relationship between wavenumber (cm⁻¹) and emission wavelength (nm) is:

```
emission_nm = excitation_nm / (1 - (wavenumber_cm * excitation_nm / 1e7))
```

The inverse relationship is:

```
wavenumber_cm = 1e7 * (emission_nm - excitation_nm) / (emission_nm * excitation_nm)
```

Given:
- `w_recorded`: Wavenumber calculated assuming `excitation_recorded` nm
- `excitation_recorded`: The excitation wavelength that was incorrectly recorded (e.g., 840 nm)
- `excitation_actual`: The actual excitation wavelength (e.g., 830 nm)

We want to find `w_corrected`: Wavenumber for the actual excitation wavelength.

**Step 1**: Convert recorded wavenumber to emission wavelength:
```
emission_nm = excitation_recorded / (1 - (w_recorded * excitation_recorded / 1e7))
```

**Step 2**: Convert emission wavelength to corrected wavenumber:
```
w_corrected = 1e7 * (emission_nm - excitation_actual) / (emission_nm * excitation_actual)
```

## Usage

### Command Line Tool

Correct a data file:

**On Windows PowerShell:**
```powershell
cd Script\swnt_iaa_analysis_v2
python correct_wavenumbers.py "path\to\input_file.txt" "output_file.txt" --recorded 840 --actual 830
```

**On Linux/Mac (bash):**
```bash
cd Script/swnt_iaa_analysis_v2
python correct_wavenumbers.py path/to/input_file.txt output_file.txt --recorded 840 --actual 830
```

**Example for your file:**
```powershell
cd Script\swnt_iaa_analysis_v2
python correct_wavenumbers.py "..\..\IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_Constant\Temp_Hum_Constant\Run 2\Raw data\run 1.txt" "run 1_corrected.txt" --recorded 840 --actual 830
```

Options:
- `input_file`: Path to input Raman data file (first row contains wavenumbers)
- `output_file`: Path to output corrected file
- `--recorded`: The excitation wavelength (nm) that was incorrectly recorded (default: 840)
- `--actual`: The actual excitation wavelength (nm) that was used (default: 830)

### Python API

In your code:

```python
from pipeline.utils import correct_wavenumber_for_excitation
import numpy as np

# Example: Correct wavenumbers from 840 nm to 830 nm
wavenumbers_recorded = np.array([100, 200, 300])  # cm^-1
wavenumbers_corrected = correct_wavenumber_for_excitation(
    wavenumbers_recorded,
    recorded_excitation_nm=840,
    actual_excitation_nm=830
)
```

### Example

Run the example script to see how it works:

```bash
python example_wavenumber_correction.py
```

This will:
- Show correction of sample wavenumbers
- Display statistics about the correction
- Verify that emission wavelengths match
- Show detailed examples

## Example: Correcting Your File

For the file at:
```
IAA Nanosensor Experiment\In planta\Nb\Treatment_Control\Light_Constant\Temp_Hum_Constant\Run 2\Raw data\run 1.txt
```

Run:

```bash
python correct_wavenumbers.py \
    "IAA Nanosensor Experiment/In planta/Nb/Treatment_Control/Light_Constant/Temp_Hum_Constant/Run 2/Raw data/run 1.txt" \
    "IAA Nanosensor Experiment/In planta/Nb/Treatment_Control/Light_Constant/Temp_Hum_Constant/Run 2/Raw data/run 1_corrected.txt" \
    --recorded 840 \
    --actual 830
```

This will create a corrected file with wavenumbers properly calculated for 830 nm excitation.

## Notes

- The correction preserves the actual emission wavelength (physical property)
- Only the first row (wavenumber header) is modified; scan data remains unchanged
- Typical corrections are on the order of a few cm⁻¹ for a 10 nm excitation difference
- The correction is more significant for higher wavenumber shifts (Stokes shifts)

## Function Reference

### `correct_wavenumber_for_excitation(wavenumber_cm, recorded_excitation_nm, actual_excitation_nm)`

Correct Raman wavenumber when the excitation wavelength was incorrectly recorded.

**Parameters:**
- `wavenumber_cm` (array-like): Wavenumbers (cm⁻¹) calculated assuming `recorded_excitation_nm` excitation
- `recorded_excitation_nm` (float): The excitation wavelength (nm) that was incorrectly recorded/used
- `actual_excitation_nm` (float): The actual excitation wavelength (nm) that was used

**Returns:**
- `corrected_wavenumber_cm` (array): Corrected wavenumbers (cm⁻¹) for the actual excitation wavelength

