"""
Helper utilities to execute the experimental Lieberfit workflow on multiple scans.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from pipeline.config_loader import load_profile_config
from pipeline.ingestion import load_raman_dataset
from pipeline.utils import create_dir_if_needed

from .test_single_scan_new_method import (
    DEFAULT_TEST_PROFILE,
    PROJECT_ROOT,
    test_new_method,
)


def parse_scan_list(scan_numbers: Optional[Iterable[int]], available: Sequence[int]) -> List[int]:
    """Validate and sanitize requested scan numbers."""
    if not scan_numbers:
        return list(available)
    
    available_set = set(available)
    cleaned = []
    for scan in scan_numbers:
        if scan in available_set:
            cleaned.append(scan)
        else:
            print(f"[Batch] Warning: scan {scan} not found in dataset and will be skipped.")
    return cleaned


def run_new_method_batch(
    scan_numbers: Optional[Iterable[int]] = None,
    max_scans: Optional[int] = None,
    optimize: bool = False,
    custom_order: Optional[int] = None,
    custom_tot_iter: Optional[int] = None,
    config: Optional[dict] = None,
    profile_name: str = DEFAULT_TEST_PROFILE,
    raman_df=None,
    output_dir: Optional[str] = None,
):
    """
    Execute the new processing method for multiple scans.

    Parameters
    ----------
    scan_numbers : iterable of int, optional
        Specific scan numbers to process. If None, processes all available scans.
    max_scans : int, optional
        Limit the number of scans (after filtering) that will be processed.
    optimize : bool
        Whether to run the Lieberfit optimization routine.
    custom_order : int, optional
        Override for polynomial order.
    custom_tot_iter : int, optional
        Override for total iterations.
    config : dict, optional
        Pipeline configuration dictionary. If None, loads default profile.
    profile_name : str
        Profile name to use if configuration is loaded internally.
    raman_df : pandas.DataFrame, optional
        Raman dataset. If None, loads dataset according to the configuration.
    output_dir : str, optional
        Directory to store generated figures/results. Defaults to a timestamped folder.
    """
    if config is None:
        config_path = PROJECT_ROOT / "config" / "pipeline.yml"
        config = load_profile_config(str(config_path), profile_name)
    
    if raman_df is None:
        dataset = load_raman_dataset(config)
        raman_df = dataset.spectra
    else:
        dataset = None
    
    available_scans = sorted(raman_df["Scan Number"].unique())
    if not available_scans:
        print("[Batch] No scans available to process.")
        return []
    
    scan_list = parse_scan_list(scan_numbers, available_scans)
    if max_scans is not None:
        scan_list = scan_list[:max_scans]
    
    if not scan_list:
        print("[Batch] No scans left to process after filtering.")
        return []
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "scripts" / f"new_method_batch_{timestamp}"
    output_dir = Path(output_dir)
    create_dir_if_needed(str(output_dir))
    
    summaries = []
    for scan in scan_list:
        print(f"\n[Batch] Processing scan {scan} ({scan_list.index(scan)+1}/{len(scan_list)})")
        summary = test_new_method(
            scan_number=scan,
            optimize_params=optimize,
            custom_order=custom_order,
            custom_tot_iter=custom_tot_iter,
            raman_df=raman_df,
            config=config,
            output_dir=output_dir,
            profile_name=profile_name,
        )
        summaries.append(summary)
    
    print(f"\n[Batch] Completed new-method processing for {len(scan_list)} scans. Outputs: {output_dir}")
    return summaries


__all__ = ["run_new_method_batch"]

