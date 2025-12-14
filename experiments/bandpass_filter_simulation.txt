import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def raman_to_emission_wavelength(wavenumbers_cm: np.ndarray, excitation_nm: float) -> np.ndarray:
    """
    Convert Raman shift (cm^-1) to emission wavelength (nm) for a given excitation wavelength.

    Formula:
        λ_emission = 10^7 / (10^7/λ_exc - Raman_shift)
    """
    return 1e7 / (1e7 / excitation_nm - wavenumbers_cm)


def lorentzian_transmission(wl: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Lorentzian bandpass filter transmission profile."""
    gamma = fwhm / 2.0
    return 1.0 / (1.0 + ((wl - center) / gamma) ** 2)


def gaussian_transmission(wl: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Gaussian bandpass filter transmission profile."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-((wl - center) ** 2) / (2.0 * sigma ** 2))


def parse_filter(arg: str) -> tuple[float, float]:
    """
    Parse a single --filter argument of the form "center,fwhm"
    into (center_nm, fwhm_nm).
    """
    try:
        center_str, fwhm_str = arg.split(",")
        center = float(center_str)
        fwhm = float(fwhm_str)
    except Exception as exc:  # noqa: BLE001
        raise argparse.ArgumentTypeError(
            f"Invalid --filter '{arg}'. Expected 'center,fwhm', e.g. '980,3'."
        ) from exc

    if fwhm <= 0:
        raise argparse.ArgumentTypeError("FWHM must be > 0.")

    return center, fwhm


def build_output_dir(user_output_dir: str | None, filters: list[tuple[float, float]] | None = None) -> Path:
    """
    Build the output directory path.

    If user_output_dir is None, use:
        H:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\Bandpass filter simulation\output_{filter_info}_{timestamp}
    """
    if user_output_dir:
        out_dir = Path(user_output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = Path(r"H:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal")
        
        # Build filter info string for folder name
        if filters:
            filter_info = "_".join(f"{int(c)}nm-{fwhm:.1f}nm" for c, fwhm in filters)
            folder_name = f"output_{filter_info}_{timestamp}"
        else:
            folder_name = f"output_{timestamp}"
        
        out_dir = base_path / "Bandpass filter simulation" / folder_name

    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def parse_datetime_columns(dates: pd.Series, times: pd.Series) -> pd.DatetimeIndex:
    """
    Parse datetime from date and time columns (compatible with raw_data.txt format).
    """
    formats = (
        "%m-%d-%Y %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m-%d-%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
    )
    combined = dates.astype(str).str.replace("/", "-", regex=False) + " " + times.astype(str)
    for fmt in formats:
        try:
            parsed = pd.to_datetime(combined, format=fmt, errors="raise")
            return pd.DatetimeIndex(parsed)
        except (ValueError, TypeError):
            continue
    # Fallback: try without format specification
    try:
        parsed = pd.to_datetime(combined, errors="coerce")
        if parsed.isna().any():
            raise ValueError("Some dates could not be parsed")
        return pd.DatetimeIndex(parsed)
    except Exception:
        raise ValueError(
            "Unable to parse datetime columns. Please verify the date/time format."
        )


def calculate_filtered_area(
    filtered_spectra_2d: np.ndarray, wavelength_nm: np.ndarray
) -> np.ndarray:
    """
    Calculate the integrated area (filtered intensity) for each scan.
    
    Uses trapezoidal integration over wavelength.
    """
    areas = np.zeros(filtered_spectra_2d.shape[0])
    for i in range(filtered_spectra_2d.shape[0]):
        areas[i] = np.trapz(filtered_spectra_2d[i, :], x=wavelength_nm)
    return areas


def plot_filtered_area_vs_time(
    datetimes: pd.DatetimeIndex,
    filtered_areas: np.ndarray,
    filters: list[tuple[float, float]],
    profile: str,
    show: bool = True,
    save_path: Path | None = None,
) -> None:
    """
    Plot filtered area (integrated intensity) vs datetime for each scan.
    Annotated with bandpass filter information.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(datetimes, filtered_areas, "b-", linewidth=1.5, marker="o", markersize=3)
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Filtered Area (integrated intensity)", fontsize=12)

    # Build filter description for title and annotation
    filter_desc = " + ".join(f"{c:.1f}±{fwhm/2:.1f} nm" for c, fwhm in filters)
    plt.title(
        f"Filtered Area vs Time\nBandpass Filter(s): {filter_desc} [{profile}]",
        fontsize=14,
        fontweight="bold",
    )

    # Add text annotation with filter details
    filter_text = "Filters:\n"
    for i, (c, fwhm) in enumerate(filters, start=1):
        filter_text += f"  {i}. {c:.2f} nm, FWHM = {fwhm:.2f} nm\n"
    filter_text += f"Profile: {profile}"

    plt.text(
        0.02,
        0.98,
        filter_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def apply_bandpass_filters(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    filters: list[tuple[float, float]],
    profile: str = "lorentzian",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply one or more bandpass filters to the spectrum.

    filters: list of (center_nm, fwhm_nm)
    profile: 'lorentzian' or 'gaussian'

    Returns:
        total_transmission, filtered_intensity
    """
    if profile == "lorentzian":
        tx_func = lorentzian_transmission
    elif profile == "gaussian":
        tx_func = gaussian_transmission
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown profile '{profile}'.")

    total_transmission = np.ones_like(wavelength_nm, dtype=float)

    for center, fwhm in filters:
        tx = tx_func(wavelength_nm, center=center, fwhm=fwhm)
        # Normalize each filter to 0.95 peak transmission (realistic)
        if tx.max() != 0:
            tx = tx / tx.max() * 0.95
        total_transmission *= tx

    filtered_intensity = intensity * total_transmission

    return total_transmission, filtered_intensity


def plot_results(
    wavelength_nm: np.ndarray,
    intensity_first_scan: np.ndarray,
    filtered_spectra_per_filter: list[np.ndarray],
    transmissions_per_filter: list[np.ndarray],
    filters: list[tuple[float, float]],
    profile: str,
    show: bool = True,
    save_path: Path | None = None,
) -> None:
    """
    Generate an overlaid plot for a representative scan (first scan):
      - Original spectrum
      - Each filter applied independently to the original (overlaid)
      - Transmission curves for each filter shown for reference
    """
    plt.figure(figsize=(10, 7))

    # Plot original spectrum
    plt.plot(
        wavelength_nm,
        intensity_first_scan,
        "b-",
        linewidth=2,
        alpha=0.8,
        label="Original spectrum (first scan)",
    )

    # Colors for different filters
    filter_colors = ["r", "g", "m", "c", "orange", "brown", "pink"]
    
    # Plot each independently filtered spectrum
    max_filtered = intensity_first_scan.max()
    for i, (filtered_scan, transmission, (center, fwhm)) in enumerate(
        zip(filtered_spectra_per_filter, transmissions_per_filter, filters)
    ):
        color = filter_colors[i % len(filter_colors)]
        plt.plot(
            wavelength_nm,
            filtered_scan,
            f"{color}-",
            linewidth=1.5,
            alpha=0.8,
            label=f"Filter {i+1}: {center:.1f} nm, FWHM={fwhm:.1f} nm",
        )
        
        # Scale and plot transmission curve for this filter
        if np.any(transmission > 0):
            trans_scaled = transmission / transmission.max() * intensity_first_scan.max() * 0.7
            plt.plot(
                wavelength_nm,
                trans_scaled,
                f"{color}--",
                alpha=0.4,
                linewidth=1.5,
                label=f"  → Transmission {i+1} (scaled)",
            )
        
        max_filtered = max(max_filtered, filtered_scan.max())

    plt.xlim(wavelength_nm.min(), wavelength_nm.max())
    plt.ylim(0, max(intensity_first_scan.max(), max_filtered, 1e-12) * 1.1)

    filter_desc = " | ".join(f"{c:.1f}±{fwhm/2:.1f} nm" for c, fwhm in filters)
    plt.xlabel("Emission wavelength (nm)", fontsize=12)
    plt.ylabel("Intensity (a.u.)", fontsize=12)
    plt.title(
        f"Original vs Independently Filtered Spectra (First Scan)\n"
        f"Filters Applied Separately: {filter_desc} [{profile}]",
        fontsize=13,
        fontweight="bold",
    )
    plt.legend(fontsize=9, loc="best", ncol=1)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def run(
    input_file: Path,
    excitation_nm: float,
    filters: list[tuple[float, float]],
    profile: str,
    output_dir: Path,
    no_plot: bool,
) -> None:
    # ============== 1. Load data ==============
    # Use sep=None to automatically infer delimiter (comma, tab, space, etc.),
    # so this works for both CSV and raw TXT data.
    df = pd.read_csv(input_file, sep=None, engine="python", header=None)

    # Heuristic 1: raw_data-style TXT (date, time, seconds, spectra...)
    is_raw_style = input_file.suffix.lower() == ".txt" and df.shape[1] >= 4 and df.shape[0] >= 2

    if is_raw_style:
        # First row, columns 3+ are wavenumbers
        wavenumbers = df.iloc[0, 3:].astype(float).to_numpy()
        # Remaining rows, columns 3+ are spectra for each scan
        spectra_2d = df.iloc[1:, 3:].astype(float).to_numpy()  # shape: (n_scans, n_wavenumbers)
        # Parse datetime from columns 0 (date) and 1 (time)
        dates = df.iloc[1:, 0]
        times = df.iloc[1:, 1]
        datetimes = parse_datetime_columns(dates, times)
    else:
        # Treat as "simple" spectrum file with wavenumber + intensity columns
        # Try to interpret header if present
        df_with_header = pd.read_csv(input_file, sep=None, engine="python")

        if "wavenumber" in df_with_header.columns:
            wavenumbers = df_with_header["wavenumber"].to_numpy(dtype=float)
        elif "raman_shift" in df_with_header.columns:
            wavenumbers = df_with_header["raman_shift"].to_numpy(dtype=float)
        else:
            # Fall back: assume first column is wavenumber
            if df_with_header.shape[1] < 2:
                raise ValueError(
                    "Input file must contain at least two columns (wavenumber and intensity)."
                )
            wavenumbers = df_with_header.iloc[:, 0].to_numpy(dtype=float)

        # Intensity column
        for col in ("intensity", "Intensity", "counts", "Counts"):
            if col in df_with_header.columns:
                intensity = df_with_header[col].to_numpy(dtype=float)
                break
        else:
            intensity = df_with_header.iloc[:, 1].to_numpy(dtype=float)

        spectra_2d = intensity.reshape(1, -1)  # shape: (1, n_wavenumbers)
        # For simple spectrum files, no datetime information
        datetimes = None

    n_scans, n_wavenumbers = spectra_2d.shape

    # ============== 2. Convert Raman shift → wavelength (nm) ==============
    wavelength_nm = raman_to_emission_wavelength(wavenumbers, excitation_nm=excitation_nm)

    # ============== 3. Apply each filter independently to the original spectrum ==============
    # Each filter is applied separately to the original spectrum
    filtered_spectra_per_filter = []  # List of (n_scans, n_wavenumbers) arrays
    transmissions_per_filter = []  # List of transmission curves
    
    if profile == "lorentzian":
        tx_func = lorentzian_transmission
    elif profile == "gaussian":
        tx_func = gaussian_transmission
    else:
        raise ValueError(f"Unknown profile '{profile}'.")
    
    for center, fwhm in filters:
        # Compute transmission for this filter
        tx = tx_func(wavelength_nm, center=center, fwhm=fwhm)
        # Normalize to 0.95 peak transmission (realistic)
        if tx.max() != 0:
            tx = tx / tx.max() * 0.95
        transmissions_per_filter.append(tx)
        
        # Apply this filter independently to all scans (original × filter_transmission)
        filtered_2d = spectra_2d * tx  # broadcasting over scans
        filtered_spectra_per_filter.append(filtered_2d)

    # ============== 4. Calculate area under curve for original and each filter ==============
    # Calculate original area
    original_areas = calculate_filtered_area(spectra_2d, wavelength_nm)
    
    # Calculate filtered area for each filter independently
    filtered_areas_per_filter = []
    for filtered_2d in filtered_spectra_per_filter:
        areas = calculate_filtered_area(filtered_2d, wavelength_nm)
        filtered_areas_per_filter.append(areas)

    # ============== 5. Save filtered data (CSV with scan numbers and areas only) ==============
    base_name = input_file.stem
    filter_tag = "_".join(f"{int(c)}nm-{fwhm:.1f}nmFWHM" for c, fwhm in filters)
    
    # Create DataFrame with scan numbers and areas only
    scan_numbers = np.arange(1, n_scans + 1)
    
    # Start with scan numbers and original area
    area_df = pd.DataFrame({
        "Scan Number": scan_numbers,
        "Original_Area": original_areas,
    })
    
    # Add filtered area for each filter
    for i, (areas, (center, fwhm)) in enumerate(zip(filtered_areas_per_filter, filters)):
        col_name = f"Filter{i+1}_{int(center)}nm_FWHM{fwhm:.1f}nm_Area"
        area_df[col_name] = areas
    
    # Save CSV with areas only
    csv_out = output_dir / f"{base_name}_filtered_areas_{filter_tag}.csv"
    area_df.to_csv(csv_out, index=False)

    # ============== 6. Report results for each filter ==============
    print(f"Input file        : {input_file}")
    print(f"Output CSV        : {csv_out}")
    print(f"Output directory  : {output_dir}")
    print(f"Excitation (nm)   : {excitation_nm:.2f}")
    print(f"Filter profile    : {profile}")
    print("Filters applied independently:")
    for i, ((center, fwhm), filtered_2d, areas) in enumerate(zip(filters, filtered_spectra_per_filter, filtered_areas_per_filter), start=1):
        first_scan_filtered = filtered_2d[0, :]
        peak_idx = int(np.argmax(first_scan_filtered))
        peak_wavelength = wavelength_nm[peak_idx]
        peak_intensity = first_scan_filtered[peak_idx]
        
        print(f"\n  Filter {i}: center = {center:.2f} nm, FWHM = {fwhm:.2f} nm")
        print(f"    Peak (scan 1)  : {peak_wavelength:.2f} nm, Intensity = {peak_intensity:.4g}")
        print(f"    Area (scan 1)  : {areas[0]:.4g}")
    print(f"\nTotal scans       : {n_scans}")
    print(f"CSV contains: Scan Number, Original_Area, and Area for each filter.")

    # ============== 7. Plot (representative first scan with all filters overlaid) ==============
    png_out = output_dir / f"{base_name}_filtered_{filter_tag}.png"
    # Extract first scan filtered spectra for plotting
    first_scan_filtered_per_filter = [filtered_2d[0, :] for filtered_2d in filtered_spectra_per_filter]
    
    plot_results(
        wavelength_nm=wavelength_nm,
        intensity_first_scan=spectra_2d[0, :],
        filtered_spectra_per_filter=first_scan_filtered_per_filter,
        transmissions_per_filter=transmissions_per_filter,
        filters=filters,
        profile=profile,
        show=not no_plot,
        save_path=png_out,
    )

    # ============== 8. Plot filtered area vs datetime for each filter (if datetime available) ==============
    if datetimes is not None and len(datetimes) == n_scans:
        for i, (filtered_areas, (center, fwhm)) in enumerate(zip(filtered_areas_per_filter, filters)):
            area_plot_out = output_dir / f"{base_name}_filtered_area_vs_time_F{i+1}_{int(center)}nm-{fwhm:.1f}nmFWHM.png"
            plot_filtered_area_vs_time(
                datetimes=datetimes,
                filtered_areas=filtered_areas,
                filters=[(center, fwhm)],  # Single filter for this plot
                profile=profile,
                show=not no_plot,
                save_path=area_plot_out,
            )
            print(f"Filtered area plot (Filter {i+1}): {area_plot_out}")
    else:
        print("Note: Datetime information not available, skipping filtered area vs time plots.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the effect of one or more bandpass filters on a Raman spectrum.\n"
            "The script converts Raman shift (cm^-1) to emission wavelength (nm), "
            "applies the specified bandpass filter(s), and saves the filtered spectrum.\n\n"
            "Filters are specified with --filter center_nm,fwhm_nm and can be repeated.\n"
            "Example: --filter 980,3 --filter 1000,5"
        )
    )

    parser.add_argument(
        "input_path",
        nargs="?",
        type=str,
        default="raw_data.txt",
        help=(
            "Path to input Raman spectrum file (TXT/CSV). "
            "Default: 'raw_data.txt' in the current working directory."
        ),
    )
    parser.add_argument(
        "--excitation-nm",
        type=float,
        default=830.0,
        help="Excitation wavelength in nm (default: 830.0).",
    )
    parser.add_argument(
        "--filter",
        type=parse_filter,
        action="append",
        required=True,
        help="Bandpass filter specification 'center_nm,fwhm_nm'. "
        "Can be passed multiple times to simulate multiple filters.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["lorentzian", "gaussian"],
        default="lorentzian",
        help="Filter transmission profile (default: lorentzian).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save results. "
            "Default: 'IAA Nanosensor Experiment/Bandpass filter simulation' "
            "under the current working directory."
        ),
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display the plot interactively (still saved as PNG).",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_file = Path(args.input_path).resolve()
    if not input_file.is_file():
        raise SystemExit(f"Input file not found: {input_file}")

    output_dir = build_output_dir(args.output_dir, filters=args.filter)

    run(
        input_file=input_file,
        excitation_nm=args.excitation_nm,
        filters=args.filter,
        profile=args.profile,
        output_dir=output_dir,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()


