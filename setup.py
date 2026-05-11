"""Setup script for swnt_iaa_analysis package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding='utf-8') if readme_file.exists() else ""

setup(
    name="swnt_iaa_analysis",
    version="1.0.0",
    author="",
    author_email="",
    description="SWNT IAA Raman Analysis Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    package_dir={'swnt_iaa_analysis': 'swnt_iaa_analysis'},
    packages=['swnt_iaa_analysis'] + 
              [f'swnt_iaa_analysis.{pkg}' for pkg in ['analysis', 'core', 'io', 'visualization']],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "typer>=0.4.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "plant_timelapse": ["opencv-python-headless>=4.5", "Pillow>=8.0"],
    },
    entry_points={
        "console_scripts": [
            "swnt_iaa_analysis=swnt_iaa_analysis.cli:app",
        ],
    },
)

