#!/usr/bin/env python3
"""
Tier-3 stub: supervised plant segmentation fine-tuning.

This repository does not pin a segmentation training stack (torch /
segmentation_models_pytorch weights are large). Use this file as an integration
recipe after you export masks via scripts/segment_nb_timelapse.py and refine a
fraction of frames in CVAT or LabelMe.

Suggested workflow:

1. Run segment_nb_timelapse.py to get coarse masks under output_dir/masks/.
2. Correct ~50–200 representative frames’ PNG masks (plant=255, bg=0) and split
   into train/ val/ folders with matching grayscale TIFF filenames.
3. Install torch + torchvision + a segmentation library you prefer (e.g.
   segmentation-models-pytorch) in a dedicated environment.
4. Implement a Dataset that loads float32 grayscale (H,W) or 3×duplicate channels
   and returns (image, mask) tensors; train U-Net / DeepLabv3+ with BCE+Dice or
   focal loss; export ONNX or torchscript for batch inference on the full run.

The following is a minimal dependency check only.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check-imports",
        action="store_true",
        help="Try importing torch and exit 0 if available.",
    )
    args = parser.parse_args()
    if not args.check_imports:
        print(__doc__)
        return
    try:
        import torch  # noqa: F401

        print("torch OK:", torch.__version__)
    except ImportError as exc:
        raise SystemExit("Install torch in your training environment first.") from exc


if __name__ == "__main__":
    main()
