"""
Entry point for compiling FFT (Fourier Transform) data across profiles.

This is a thin wrapper that imports the main implementation from src/cli/compile_fft_data.py.

Usage examples:
    # Compile all in_planta profiles (default frequency 0-0.5)
    python compile_fft_data.py --output compiled_fft.csv

    # Compile with Parquet format (recommended for large files)
    python compile_fft_data.py --compression parquet

    # Compile only gband FFT data
    python compile_fft_data.py --fft-types gband --output compiled_fft_gband.csv

    # Compile with custom frequency range
    python compile_fft_data.py --frequency-max 0.3 --output compiled_fft_0.3.csv
"""
from src.cli.compile_fft_data import main


if __name__ == "__main__":
    main()

