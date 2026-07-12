"""Keypoint (base + apical-meristem) detection pipeline for the NIR plant timelapse.

This package wraps a DeepLabCut (PyTorch engine) workflow with project-specific glue:
frame export using the SAME conventions as the segmentation pipeline (absolute
0-4095 -> 8-bit scaling + fixed plant-column ROI crop), frame selection, and
post-processing of predictions into a stem-height curve.

Heavy DeepLabCut steps are driven from the CLI (`python -m plant_timelapse.keypoints ...`),
which is the robust way to run long training / launch the labeling GUI. The optional
notebook is only for QC visualization.
"""

from . import config, frame_export, frame_select, postprocess

__all__ = ["config", "frame_export", "frame_select", "postprocess"]
