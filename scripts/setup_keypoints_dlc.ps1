# One-time setup for the DeepLabCut keypoint pipeline (plant base + apical-meristem).
# Creates a DEDICATED env (.venv-dlc) separate from .venv, because DeepLabCut pins
# numpy<2 / matplotlib<3.9 which conflict with the plant-mask env.
#
# Run from repo root:  .\scripts\setup_keypoints_dlc.ps1
#
# Requires Python 3.10-3.12 on PATH (DeepLabCut 3.x supports 3.10-3.12; 3.13/3.14 are too new).
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

# CUDA wheel tag for PyTorch. RTX 3070 (Ampere) works with cu121/cu124/cu126/cu128.
$CudaTag = "cu126"

$VenvDir = Join-Path $Root ".venv-dlc"
$Py = Join-Path $VenvDir "Scripts\python.exe"

if (-not (Test-Path $Py)) {
    Write-Host "Creating .venv-dlc (Python 3.12) ..."
    # 'python' on PATH must be 3.10-3.12. Verify before creating.
    $ver = (python -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
    if ($ver -notin @("3.10", "3.11", "3.12")) {
        throw "Default 'python' is $ver; DeepLabCut needs 3.10-3.12. Point 'python' at a 3.12 interpreter and retry."
    }
    python -m venv .venv-dlc
}

Write-Host "Upgrading pip ..."
& $Py -m pip install --upgrade pip

# Order matters (see DLC install docs + issue #3219):
Write-Host "Installing pandas[hdf5,performance]<3 first (avoids resolver conflicts) ..."
& $Py -m pip install "pandas[hdf5,performance]<3"

Write-Host "Installing PyTorch + torchvision ($CudaTag) BEFORE DeepLabCut (for GPU support) ..."
& $Py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$CudaTag"

Write-Host "Verifying CUDA is visible to PyTorch ..."
& $Py -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'))"

Write-Host "Installing DeepLabCut (PyTorch engine) + GUI ..."
& $Py -m pip install "deeplabcut[gui]"

Write-Host "Installing CLI extras (typer, tifffile, ipykernel, ...) ..."
& $Py -m pip install -r requirements\keypoints_dlc.txt

Write-Host "Registering Jupyter kernel 'microneedle-dlc' ..."
& $Py -m ipykernel install --user --name microneedle-dlc --display-name "microneedle (dlc keypoints)"

Write-Host ""
Write-Host "Done. Sanity check:"
& $Py -c "import deeplabcut, torch; print('deeplabcut', deeplabcut.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
Write-Host ""
Write-Host "Next:"
Write-Host "  .venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints gpu-check"
Write-Host "  .venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints build-videos"
Write-Host "  .venv-dlc\Scripts\python.exe -m plant_timelapse.keypoints create-project"
