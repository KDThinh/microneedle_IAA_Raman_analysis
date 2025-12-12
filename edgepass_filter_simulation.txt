"""
Edge-Pass Filter Simulation Script

This script simulates the effect of long-pass and/or short-pass filters on Raman spectra.

HOW TO MODIFY FILTER PARAMETERS:
================================

1. Edit existing filters in the FILTER_SPECS dictionary below:
   - cut_on_nm (long-pass): wavelength where filter starts transmitting (nm)
   - cut_off_nm (short-pass): wavelength where filter stops transmitting (nm)
   - min_transmission: minimum transmission in passband (0.0 to 1.0, e.g., 0.90 = 90%)
   - transition_width: width of transition region (nm, default 5.0)
   - transmission_region: (min, max) wavelength range for passband (informational only)

2. Add new filters by adding entries to FILTER_SPECS:
   Example:
   "LP3": {
       "type": "longpass",
       "cut_on_nm": 1000.0,
       "transmission_region": (1015.0, 2150.0),
       "min_transmission": 0.95,
       "transition_width": 20.0,
       "description": "Long pass Filter 3: Cut-On 1000 nm (T>95%)",
   }

3. Parameters explained:
   - transition_width: Controls how sharp the transition is. Smaller = sharper transition.
   - min_transmission: Peak transmission in the passband (realistic filters aren't 100%).
   - cut_on_nm: For long-pass, wavelengths > cut_on_nm are transmitted.
   - cut_off_nm: For short-pass, wavelengths < cut_off_nm are transmitted.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============== Filter Definitions ==============
# To modify filter parameters, edit the values below:
#   - cut_on_nm / cut_off_nm: wavelength where filter transitions (nm)
#   - min_transmission: minimum transmission in passband (0.0 to 1.0)
#   - transition_width: width of transition region (nm, default 5.0)
#   - transmission_region: (min, max) wavelength range for passband (nm, informational only)
FILTER_SPECS = {
    "LP1": {
        "type": "longpass",
        "cut_on_nm": 975.0,
        "transmission_region": (990.0, 2150.0),
        "min_transmission": 0.90,
        "transition_width": 5.0,  # Width of transition region in nm
        "description": "Long pass Filter 1: Cut-On 975 nm (T>90%), transmission 990-2150 nm",
    },
    "LP2": {
        "type": "longpass",
        "cut_on_nm": 925.0,
        "transmission_region": (939.0, 2150.0),
        "min_transmission": 0.90,
        "transition_width": 5.0,
        "description": "Long pass Filter 2: Cut-On 925 nm (T>90%), transmission 939-2150 nm",
    },
    "LP3": {
        "type": "longpass",
        "cut_on_nm": 950.0,
        "transmission_region": (962.0, 2150.0),
        "min_transmission": 0.90,
        "transition_width": 5.0,
        "description": "Long pass Filter 3: Cut-On 950 nm (T>90%), transmission 962-2150 nm",
    },
    "SP1": {
        "type": "shortpass",
        "cut_off_nm": 925.0,
        "transmission_region": (400.0, 911.0),
        "min_transmission": 0.90,
        "transition_width": 5.0,
        "description": "Short pass Filter 1: Cut-Off 925 nm (T>90%), transmission 400-911 nm",
    },

    "SP2": {
        "type": "shortpass",
        "cut_off_nm": 950.0,
        "transmission_region": (575.0, 930.0),
        "min_transmission": 0.90,
        "transition_width": 5.0,
        "description": "Short pass Filter 2: Cut-Off 950 nm (T>90%), transmission 575-930 nm",
    },
     "SP3": {
        "type": "shortpass",
        "cut_off_nm": 975.0,
        "transmission_region": (400.0, 960.0),
        "min_transmission": 0.90,
        "transition_width": 5.0,
        "description": "Short pass Filter 3: Cut-Off 975 nm (T>90%), transmission 400-960 nm",
    },
}


def raman_to_emission_wavelength(wavenumbers_cm: np.ndarray, excitation_nm: float) -> np.ndarray:
    """
    Convert Raman shift (cm^-1) to emission wavelength (nm) for a given excitation wavelength.

    Formula:
        λ_emission = 10^7 / (10^7/λ_exc - Raman_shift)
    """
    return 1e7 / (1e7 / excitation_nm - wavenumbers_cm)


def longpass_transmission(
    wl: np.ndarray, cut_on_nm: float, transition_width: float = 5.0, min_transmission: float = 0.90
) -> np.ndarray:
    """
    Long-pass filter transmission profile.
    
    Uses a sigmoid transition around the cut-on wavelength.
    - Wavelengths BELOW cut_on_nm: near-zero transmission (blocked)
    - Wavelengths ABOVE cut_on_nm: min_transmission (passed)
    
    transition_width: width of transition region (nm)
    """
    # Sigmoid function: 0 below cut-on, 1 above cut-on
    # Using transition_width/4.0 gives a smooth transition over ~transition_width nm
    transition = 1.0 / (1.0 + np.exp(-(wl - cut_on_nm) / (transition_width / 4.0)))
    
    # Scale: near-zero (blocked) below cut-on, min_transmission (passed) above cut-on
    # Add small baseline to avoid exactly zero (more realistic)
    baseline = 0.01  # 1% minimum transmission in blocked region (realistic)
    transmission = baseline * (1.0 - transition) + min_transmission * transition
    return transmission


def shortpass_transmission(
    wl: np.ndarray, cut_off_nm: float, transition_width: float = 5.0, min_transmission: float = 0.90
) -> np.ndarray:
    """
    Short-pass filter transmission profile.
    
    Uses a sigmoid transition around the cut-off wavelength.
    - Wavelengths BELOW cut_off_nm: min_transmission (passed)
    - Wavelengths ABOVE cut_off_nm: near-zero transmission (blocked)
    
    transition_width: width of transition region (nm)
    """
    # Sigmoid function (inverted): 1 below cut-off, 0 above cut-off
    transition = 1.0 / (1.0 + np.exp((wl - cut_off_nm) / (transition_width / 4.0)))
    
    # Scale: min_transmission (passed) below cut-off, near-zero (blocked) above cut-off
    baseline = 0.01  # 1% minimum transmission in blocked region (realistic)
    transmission = min_transmission * transition + baseline * (1.0 - transition)
    return transmission


def get_filter_transmission(wl: np.ndarray, filter_name: str) -> np.ndarray:
    """
    Get transmission curve for a named filter (LP1, LP2, SP1, etc.).
    """
    if filter_name not in FILTER_SPECS:
        raise ValueError(f"Unknown filter: {filter_name}. Available: {list(FILTER_SPECS.keys())}")

    spec = FILTER_SPECS[filter_name]

    # Get transition_width from spec, default to 15.0 if not specified
    transition_width = spec.get("transition_width", 15.0)

    if spec["type"] == "longpass":
        return longpass_transmission(
            wl,
            cut_on_nm=spec["cut_on_nm"],
            min_transmission=spec["min_transmission"],
            transition_width=transition_width,
        )
    elif spec["type"] == "shortpass":
        return shortpass_transmission(
            wl,
            cut_off_nm=spec["cut_off_nm"],
            min_transmission=spec["min_transmission"],
            transition_width=transition_width,
        )
    else:
        raise ValueError(f"Unknown filter type: {spec['type']}")


def parse_filter(arg: str) -> list[str]:
    """
    Parse a single --filter argument.
    
    Can be:
    - Single filter: "LP1" or "SP1"
    - Combined filters (applied consecutively): "LP2,SP1"
    
    Returns list of filter names.
    """
    filters = [f.strip().upper() for f in arg.split(",")]
    
    # Validate filter names
    for f in filters:
        if f not in FILTER_SPECS:
            available = ", ".join(FILTER_SPECS.keys())
            raise argparse.ArgumentTypeError(
                f"Unknown filter '{f}' in '{arg}'. Available filters: {available}"
            )
    
    return filters


def build_output_dir(user_output_dir: str | None, filter_configs: list[list[str]] | None = None) -> Path:
    """
    Build the output directory path.

    If user_output_dir is None, use:
        H:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal\Edgepass filter simulation\output_{filter_info}_{timestamp}
    """
    if user_output_dir:
        out_dir = Path(user_output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = Path(r"H:\My Drive\Work\DiSTAP\Research\Auxin IAA\IAA-MN longitudinal")

        # Build filter info string for folder name
        if filter_configs:
            filter_parts = []
            for config in filter_configs:
                filter_parts.append("_".join(config))
            filter_info = "_".join(filter_parts)
            folder_name = f"output_{filter_info}_{timestamp}"
        else:
            folder_name = f"output_{timestamp}"

        out_dir = base_path / "Edgepass filter simulation" / folder_name

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


def apply_edgepass_filters(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    filter_names: list[str],
) -> tuple[np.ndarray, str]:
    """
    Apply one or more edge-pass filters consecutively to the spectrum.

    If multiple filters are provided, they are applied in series (consecutively).

    Returns:
        total_transmission, filter_description
    """
    total_transmission = np.ones_like(wavelength_nm, dtype=float)

    filter_descriptions = []
    for filter_name in filter_names:
        tx = get_filter_transmission(wavelength_nm, filter_name)
        total_transmission *= tx
        filter_descriptions.append(FILTER_SPECS[filter_name]["description"])

    filter_description = " → ".join(filter_names) + " (consecutive)" if len(filter_names) > 1 else filter_names[0]

    return total_transmission, filter_description


def plot_results(
    wavelength_nm: np.ndarray,
    intensity_first_scan: np.ndarray,
    filtered_spectra_per_filter: list[np.ndarray],
    transmissions_per_filter: list[np.ndarray],
    filter_configs: list[list[str]],
    show: bool = True,
    save_path: Path | None = None,
) -> None:
    """
    Generate a stacked plot for a representative scan (first scan):
      - Top panel: Original spectrum
      - Separate panels below: Each filtered scan in its own panel
      - All panels share the same x-axis scale
    """
    # Set consistent x-axis limits
    x_min = wavelength_nm.min()
    x_max = wavelength_nm.max()
    
    # Colors for different filter configurations
    filter_colors = ["r", "g", "m", "c", "orange", "brown", "pink"]
    
    # Number of panels: 1 for original + 1 for each filter configuration
    n_panels = 1 + len(filter_configs)
    
    # Create figure with stacked subplots (original + one per filter config)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 + 3 * n_panels), sharex=True)

    # Handle case when there's only one subplot (single filter config)
    if n_panels == 1:
        axes = [axes]
    else:
        axes = list(axes)

    # ============== Top panel: Original spectrum ==============
    ax_original = axes[0]
    ax_original.plot(
        wavelength_nm,
        intensity_first_scan,
        "b-",
        linewidth=2,
        alpha=0.8,
        label="Original spectrum (first scan)",
    )
    ax_original.set_xlim(x_min, x_max)
    ax_original.set_ylim(0, intensity_first_scan.max() * 1.1)
    ax_original.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax_original.set_title("Original Spectrum (First Scan)", fontsize=13, fontweight="bold")
    ax_original.legend(fontsize=10, loc="best")
    ax_original.grid(True, alpha=0.3)

    # ============== Separate panels for each filtered scan ==============
    for i, (filtered_scan, transmission, filter_names) in enumerate(
        zip(filtered_spectra_per_filter, transmissions_per_filter, filter_configs)
    ):
        ax = axes[i + 1]  # +1 because first panel is original
        color = filter_colors[i % len(filter_colors)]
        filter_label = " → ".join(filter_names) if len(filter_names) > 1 else filter_names[0]
        
        # Plot filtered spectrum
        ax.plot(
            wavelength_nm,
            filtered_scan,
            f"{color}-",
            linewidth=1.5,
            alpha=0.8,
            label=f"Filtered: {filter_label}",
        )

        # Scale and plot transmission curve for this filter configuration
        if np.any(transmission > 0):
            trans_scaled = transmission / transmission.max() * intensity_first_scan.max() * 0.7
            ax.plot(
                wavelength_nm,
                trans_scaled,
                f"{color}--",
                alpha=0.4,
                linewidth=1.5,
                label="Transmission (scaled)",
            )

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, max(filtered_scan.max(), 1e-12) * 1.1)
        ax.set_ylabel("Intensity (a.u.)", fontsize=12)
        ax.set_title(
            f"Filtered Spectrum - Config {i+1}: {filter_label}",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    # Set x-axis label only on the bottom panel
    axes[-1].set_xlabel("Emission wavelength (nm)", fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def plot_overlaid_results(
    wavelength_nm: np.ndarray,
    intensity_first_scan: np.ndarray,
    filtered_spectra_per_filter: list[np.ndarray],
    transmissions_per_filter: list[np.ndarray],
    filter_configs: list[list[str]],
    show: bool = True,
    save_path: Path | None = None,
) -> None:
    """
    Generate an overlaid plot for a representative scan (first scan):
      - Original spectrum and all filtered spectra overlaid on the same panel
      - Transmission curves shown for reference
    """
    # Set consistent x-axis limits
    x_min = wavelength_nm.min()
    x_max = wavelength_nm.max()
    
    # Colors for different filter configurations
    filter_colors = ["r", "g", "m", "c", "orange", "brown", "pink"]
    
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

    # Plot each independently filtered spectrum
    max_filtered = intensity_first_scan.max()
    for i, (filtered_scan, transmission, filter_names) in enumerate(
        zip(filtered_spectra_per_filter, transmissions_per_filter, filter_configs)
    ):
        color = filter_colors[i % len(filter_colors)]
        filter_label = " → ".join(filter_names) if len(filter_names) > 1 else filter_names[0]
        
        # Plot filtered spectrum
        plt.plot(
            wavelength_nm,
            filtered_scan,
            f"{color}-",
            linewidth=1.5,
            alpha=0.8,
            label=f"Filter Config {i+1}: {filter_label}",
        )

        # Scale and plot transmission curve for this filter configuration
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

    plt.xlim(x_min, x_max)
    plt.ylim(0, max(intensity_first_scan.max(), max_filtered, 1e-12) * 1.1)

    filter_desc = " | ".join(" → ".join(fn) if len(fn) > 1 else fn[0] for fn in filter_configs)
    plt.xlabel("Emission wavelength (nm)", fontsize=12)
    plt.ylabel("Intensity (a.u.)", fontsize=12)
    plt.title(
        f"Original vs Filtered Spectra Overlaid (First Scan)\n"
        f"Filter Configurations: {filter_desc}",
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


def plot_filtered_area_vs_time(
    datetimes: pd.DatetimeIndex,
    filtered_areas: np.ndarray,
    filter_config: list[str],
    show: bool = True,
    save_path: Path | None = None,
) -> None:
    """
    Plot filtered area (integrated intensity) vs datetime for each scan.
    Annotated with edge-pass filter information.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(datetimes, filtered_areas, "b-", linewidth=1.5, marker="o", markersize=3)
    plt.xlabel("Datetime", fontsize=12)
    plt.ylabel("Filtered Area (integrated intensity)", fontsize=12)

    # Build filter description
    filter_label = " → ".join(filter_config) if len(filter_config) > 1 else filter_config[0]
    filter_desc = " + ".join(FILTER_SPECS[f]["description"] for f in filter_config)

    plt.title(
        f"Filtered Area vs Time\nFilter Configuration: {filter_label}",
        fontsize=14,
        fontweight="bold",
    )

    # Add text annotation with filter details
    filter_text = "Filter Configuration:\n"
    for i, fname in enumerate(filter_config, start=1):
        spec = FILTER_SPECS[fname]
        filter_text += f"  {i}. {spec['description']}\n"

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


def run(
    input_file: Path,
    excitation_nm: float,
    filter_configs: list[list[str]],
    output_dir: Path,
    no_plot: bool,
) -> None:
    # ============== 1. Load data ==============
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
        df_with_header = pd.read_csv(input_file, sep=None, engine="python")

        if "wavenumber" in df_with_header.columns:
            wavenumbers = df_with_header["wavenumber"].to_numpy(dtype=float)
        elif "raman_shift" in df_with_header.columns:
            wavenumbers = df_with_header["raman_shift"].to_numpy(dtype=float)
        else:
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
        datetimes = None

    n_scans, n_wavenumbers = spectra_2d.shape

    # ============== 2. Convert Raman shift → wavelength (nm) ==============
    wavelength_nm = raman_to_emission_wavelength(wavenumbers, excitation_nm=excitation_nm)

    # ============== 3. Apply each filter configuration independently to the original spectrum ==============
    filtered_spectra_per_config = []
    transmissions_per_config = []
    filter_descriptions = []

    for filter_names in filter_configs:
        # Apply filters consecutively if multiple filters in this configuration
        transmission, description = apply_edgepass_filters(
            wavelength_nm=wavelength_nm,
            intensity=np.ones_like(wavelength_nm),
            filter_names=filter_names,
        )
        transmissions_per_config.append(transmission)
        filter_descriptions.append(description)

        # Apply this filter configuration independently to all scans
        filtered_2d = spectra_2d * transmission  # broadcasting over scans
        filtered_spectra_per_config.append(filtered_2d)

    # ============== 4. Calculate area under curve for original and each filter configuration ==============
    # Calculate original area
    original_areas = calculate_filtered_area(spectra_2d, wavelength_nm)

    # Calculate filtered area for each filter configuration independently
    filtered_areas_per_config = []
    for filtered_2d in filtered_spectra_per_config:
        areas = calculate_filtered_area(filtered_2d, wavelength_nm)
        filtered_areas_per_config.append(areas)

    # ============== 5. Save filtered data (CSV with scan numbers and areas only) ==============
    base_name = input_file.stem
    filter_tag = "_".join("_".join(fn) for fn in filter_configs)

    # Create DataFrame with scan numbers and areas only
    scan_numbers = np.arange(1, n_scans + 1)

    # Start with scan numbers and original area
    area_df = pd.DataFrame(
        {
            "Scan Number": scan_numbers,
            "Original_Area": original_areas,
        }
    )

    # Add filtered area for each filter configuration
    for i, (areas, filter_names) in enumerate(zip(filtered_areas_per_config, filter_configs)):
        filter_label = "_".join(filter_names)
        col_name = f"FilterConfig{i+1}_{filter_label}_Area"
        area_df[col_name] = areas

    # Save CSV with areas only
    csv_out = output_dir / f"{base_name}_filtered_areas_{filter_tag}.csv"
    area_df.to_csv(csv_out, index=False)

    # ============== 6. Report results for each filter configuration ==============
    print(f"Input file        : {input_file}")
    print(f"Output CSV        : {csv_out}")
    print(f"Output directory  : {output_dir}")
    print(f"Excitation (nm)   : {excitation_nm:.2f}")
    print("Filter configurations applied independently:")
    for i, (filter_names, filtered_2d, areas) in enumerate(
        zip(filter_configs, filtered_spectra_per_config, filtered_areas_per_config), start=1
    ):
        first_scan_filtered = filtered_2d[0, :]
        peak_idx = int(np.argmax(first_scan_filtered))
        peak_wavelength = wavelength_nm[peak_idx]
        peak_intensity = first_scan_filtered[peak_idx]

        filter_label = " → ".join(filter_names) if len(filter_names) > 1 else filter_names[0]
        print(f"\n  Config {i}: {filter_label}")
        for fname in filter_names:
            print(f"    {FILTER_SPECS[fname]['description']}")
        print(f"    Peak (scan 1)  : {peak_wavelength:.2f} nm, Intensity = {peak_intensity:.4g}")
        print(f"    Area (scan 1)  : {areas[0]:.4g}")
    print(f"\nTotal scans       : {n_scans}")
    print(f"CSV contains: Scan Number, Original_Area, and Area for each filter configuration.")

    # ============== 7. Plot (stacked plot with separate panels) ==============
    png_out_stacked = output_dir / f"{base_name}_filtered_stacked_{filter_tag}.png"
    # Extract first scan filtered spectra for plotting
    first_scan_filtered_per_config = [filtered_2d[0, :] for filtered_2d in filtered_spectra_per_config]

    plot_results(
        wavelength_nm=wavelength_nm,
        intensity_first_scan=spectra_2d[0, :],
        filtered_spectra_per_filter=first_scan_filtered_per_config,
        transmissions_per_filter=transmissions_per_config,
        filter_configs=filter_configs,
        show=not no_plot,
        save_path=png_out_stacked,
    )

    # ============== 7b. Plot (overlaid plot with all spectra on same panel) ==============
    png_out_overlaid = output_dir / f"{base_name}_filtered_overlaid_{filter_tag}.png"
    
    plot_overlaid_results(
        wavelength_nm=wavelength_nm,
        intensity_first_scan=spectra_2d[0, :],
        filtered_spectra_per_filter=first_scan_filtered_per_config,
        transmissions_per_filter=transmissions_per_config,
        filter_configs=filter_configs,
        show=not no_plot,
        save_path=png_out_overlaid,
    )
    
    print(f"\nPlots generated:")
    print(f"  Stacked plot  : {png_out_stacked}")
    print(f"  Overlaid plot : {png_out_overlaid}")

    # ============== 8. Plot filtered area vs datetime for each filter configuration (if datetime available) ==============
    if datetimes is not None and len(datetimes) == n_scans:
        for i, (filtered_areas, filter_names) in enumerate(zip(filtered_areas_per_config, filter_configs)):
            filter_label = "_".join(filter_names)
            area_plot_out = (
                output_dir / f"{base_name}_filtered_area_vs_time_Config{i+1}_{filter_label}.png"
            )
            plot_filtered_area_vs_time(
                datetimes=datetimes,
                filtered_areas=filtered_areas,
                filter_config=filter_names,
                show=not no_plot,
                save_path=area_plot_out,
            )
            print(f"Filtered area plot (Config {i+1}): {area_plot_out}")
    else:
        print("Note: Datetime information not available, skipping filtered area vs time plots.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the effect of one or more edge-pass filters (long-pass and/or short-pass) "
            "on a Raman spectrum.\n"
            "The script converts Raman shift (cm^-1) to emission wavelength (nm), "
            "applies the specified filter(s), and saves the filtered spectrum.\n\n"
            "Filters are specified with --filter and can be:\n"
            "  - Single filter: --filter LP1\n"
            "  - Combined filters (applied consecutively): --filter LP2,SP1\n"
            "Multiple --filter arguments are applied independently to the original spectrum.\n\n"
            "Available filters:\n"
            "  LP1: Long pass Filter 1 (Cut-On 975 nm, T>90%, transmission 990-2150 nm)\n"
            "  LP2: Long pass Filter 2 (Cut-On 925 nm, T>90%, transmission 939-2150 nm)\n"
            "  SP1: Short pass Filter 1 (Cut-Off 925 nm, T>90%, transmission 400-911 nm)\n\n"
            "Example: --filter LP1 --filter LP2,SP1"
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
        help="Edge-pass filter specification. Can be a single filter (e.g., 'LP1') "
        "or combined filters applied consecutively (e.g., 'LP2,SP1'). "
        "Can be passed multiple times to simulate multiple independent filter configurations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory to save results. "
            "Default: 'Edgepass filter simulation/output_{filter_info}_{timestamp}' "
            "under the base path."
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

    output_dir = build_output_dir(args.output_dir, filter_configs=args.filter)

    run(
        input_file=input_file,
        excitation_nm=args.excitation_nm,
        filter_configs=args.filter,
        output_dir=output_dir,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()

