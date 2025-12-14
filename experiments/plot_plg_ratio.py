"""
Plot PL/G Ratio from Filtered Areas CSV

This script processes filtered areas CSV files and calculates the PL/G ratio,
where PL is the second-to-last numeric column and G is the last numeric column.
PL/G ratio = PL / G = (second-to-last column) / (last column).

Usage:
    python plot_plg_ratio.py path/to/filtered_areas.csv
    python plot_plg_ratio.py path/to/filtered_areas.csv --x-axis datetime
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_plg_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate PL/G ratio from the last two filtered area columns.
    
    PL = second-to-last column (numeric)
    G = last column (numeric)
    PL/G ratio = PL / G = (second-to-last) / (last)
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with filtered areas. Last two numeric columns should be filtered areas.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added 'PL', 'G', and 'PL_G_Ratio' columns
    """
    # Get the last two numeric columns (excluding Scan Number if it exists)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        raise ValueError(
            f"Expected at least 2 numeric columns for PL/G calculation. "
            f"Found {len(numeric_cols)} numeric columns."
        )
    
    # Use the last two numeric columns as filtered areas
    PL_col = numeric_cols[-2]  # Second-to-last column
    G_col = numeric_cols[-1]   # Last column
    
    print(f"Using columns for PL/G calculation:")
    print(f"  PL (second-to-last): {PL_col}")
    print(f"  G (last): {G_col}")
    
    # Get the two area values
    PL = df[PL_col].values  # Second-to-last column
    G = df[G_col].values    # Last column
    
    # Calculate PL/G ratio (avoid division by zero)
    PL_G_ratio = np.where(G > 0, PL / G, np.nan)
    
    # Create result DataFrame
    result_df = df.copy()
    result_df['PL'] = PL
    result_df['G'] = G
    result_df['PL_G_Ratio'] = PL_G_ratio
    
    # Add column names for reference
    result_df.attrs['PL_column'] = PL_col
    result_df.attrs['G_column'] = G_col
    
    return result_df


def plot_plg_ratio(
    df: pd.DataFrame,
    x_column: str = "Scan Number",
    output_path: Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot PL/G ratio vs scan number or datetime.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'PL_G_Ratio' column
    x_column : str
        Column to use for x-axis ('Scan Number' or 'Datetime')
    output_path : Path, optional
        Path to save the plot
    show : bool
        Whether to display the plot interactively
    """
    if 'PL_G_Ratio' not in df.columns:
        raise ValueError("DataFrame must contain 'PL_G_Ratio' column.")
    
    plt.figure(figsize=(6, 6))
    
    # Get x-axis data
    if x_column == "Scan Number":
        x_data = df['Scan Number'].values if 'Scan Number' in df.columns else np.arange(1, len(df) + 1)
        x_label = "Scan Number"
    elif x_column == "datetime":
        if 'Datetime' in df.columns:
            x_data = pd.to_datetime(df['Datetime'])
            x_label = "Datetime"
        else:
            print("Warning: 'Datetime' column not found. Using Scan Number instead.")
            x_data = df['Scan Number'].values if 'Scan Number' in df.columns else np.arange(1, len(df) + 1)
            x_label = "Scan Number"
    else:
        if x_column in df.columns:
            x_data = df[x_column].values
            x_label = x_column
        else:
            print(f"Warning: '{x_column}' column not found. Using Scan Number instead.")
            x_data = df['Scan Number'].values if 'Scan Number' in df.columns else np.arange(1, len(df) + 1)
            x_label = "Scan Number"
    
    # Plot PL/G ratio
    plt.plot(x_data, df['PL_G_Ratio'].values, 'b-', linewidth=1.5, marker='o', markersize=3, label='PL/G Ratio')
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("PL/G Ratio", fontsize=12)
    plt.legend(fontsize=10, loc="best")
    plt.grid(True, alpha=0.3)
    
    # Format x-axis if datetime
    if x_label == "Datetime":
        plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate and plot PL/G ratio from filtered areas CSV file.\n"
            "PL is the second-to-last numeric column, G is the last numeric column.\n"
            "PL/G ratio = PL / G = (second-to-last column) / (last column).\n"
            "The plot is saved in the same folder as the input CSV file."
        )
    )
    
    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to input CSV file with filtered areas (must have at least 2 filtered area columns).",
    )
    parser.add_argument(
        "--x-axis",
        type=str,
        choices=["Scan Number", "datetime"],
        default="Scan Number",
        help="Column to use for x-axis: 'Scan Number' or 'datetime' (default: 'Scan Number').",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not display the plot interactively (still saved as PNG).",
    )
    
    args = parser.parse_args()
    
    # Read input CSV
    input_path = Path(args.input_csv).resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input CSV file not found: {input_path}")
    
    print(f"Reading CSV file: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"CSV shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Calculate PL/G ratio
    df_with_ratio = calculate_plg_ratio(df)
    
    # Print statistics
    ratio_values = df_with_ratio['PL_G_Ratio'].dropna()
    print(f"\nPL/G Ratio Statistics:")
    print(f"  Mean: {ratio_values.mean():.4f}")
    print(f"  Median: {ratio_values.median():.4f}")
    print(f"  Min: {ratio_values.min():.4f}")
    print(f"  Max: {ratio_values.max():.4f}")
    print(f"  Std: {ratio_values.std():.4f}")
    
    # Determine output path (same folder as input)
    output_dir = input_path.parent
    output_stem = input_path.stem
    output_path = output_dir / f"{output_stem}_PLG_ratio.png"
    
    # Plot PL/G ratio
    plot_plg_ratio(
        df=df_with_ratio,
        x_column=args.x_axis,
        output_path=output_path,
        show=not args.no_plot,
    )
    
    # Optionally save CSV with PL/G ratio
    output_csv = output_dir / f"{output_stem}_with_PLG_ratio.csv"
    df_with_ratio.to_csv(output_csv, index=False)
    print(f"\nCSV with PL/G ratio saved to: {output_csv}")


if __name__ == "__main__":
    main()

