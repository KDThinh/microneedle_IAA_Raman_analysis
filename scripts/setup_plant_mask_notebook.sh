#!/usr/bin/env bash
# One-time setup for plant_timelapse/notebooks/plant_mask_pipeline.ipynb
# Run from repo root:  bash scripts/setup_plant_mask_notebook.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif [[ -f "$ROOT/.venv/Scripts/python.exe" ]]; then
  PY="$ROOT/.venv/Scripts/python.exe"
else
  echo "Creating .venv ..."
  python3 -m venv .venv
  if [[ -f "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
  else
    PY="$ROOT/.venv/Scripts/python.exe"
  fi
fi

echo "Upgrading pip ..."
"$PY" -m pip install --upgrade pip

echo "Installing project + OpenCV (plant_timelapse extra) ..."
"$PY" -m pip install -e ".[plant_timelapse]"

echo "Installing PlantCV stack ..."
"$PY" -m pip install plantcv --no-deps
"$PY" -m pip install -r requirements/plant_mask_notebook.txt

echo "Registering Jupyter kernel 'microneedle-plant-mask' ..."
"$PY" -m ipykernel install --user --name microneedle-plant-mask --display-name "microneedle (plant mask)"

echo ""
echo "Done."
echo "  1. Open plant_timelapse/notebooks/plant_mask_pipeline.ipynb"
echo "  2. Select kernel: microneedle (plant mask)"
echo "  3. Run the imports cell"
echo ""
"$PY" -c "import cv2; from plantcv import plantcv as pcv; print('opencv', cv2.__version__, '| plantcv', pcv.__version__)"
