# One-time setup for plant_timelapse/notebooks/plant_mask_pipeline.ipynb
# Run from repo root:  .\scripts\setup_plant_mask_notebook.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "Creating .venv ..."
    python -m venv .venv
}

Write-Host "Upgrading pip ..."
& $Py -m pip install --upgrade pip

Write-Host "Installing project + OpenCV (plant_timelapse extra) ..."
& $Py -m pip install -e ".[plant_timelapse]"

Write-Host "Installing PlantCV stack (plantcv installed without deps; opencv-python-headless is enough) ..."
& $Py -m pip install plantcv --no-deps
& $Py -m pip install -r requirements\plant_mask_notebook.txt

Write-Host "Registering Jupyter kernel 'microneedle-plant-mask' ..."
& $Py -m ipykernel install --user --name microneedle-plant-mask --display-name "microneedle (plant mask)"

Write-Host ""
Write-Host "Done."
Write-Host "  1. Open plant_timelapse/notebooks/plant_mask_pipeline.ipynb"
Write-Host "  2. Select kernel: microneedle (plant mask)"
Write-Host "  3. Run the imports cell"
Write-Host ""
& $Py -c "import cv2; from plantcv import plantcv as pcv; print('opencv', cv2.__version__, '| plantcv', pcv.__version__)"
