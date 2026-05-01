#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/setup_python_venv.sh
#   bash scripts/setup_python_venv.sh .venv
#
# This script creates a local virtual environment and installs dependencies
# with NumPy/SciPy pins that avoid common ABI mismatch issues.

VENV_DIR="${1:-.venv}"

echo "==> FHE-S4 Python environment bootstrap"
echo "==> Target venv: ${VENV_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is not installed or not on PATH."
  exit 1
fi

echo "==> Python found: $(python3 --version)"
echo "==> Creating virtual environment..."
python3 -m venv "${VENV_DIR}"

echo "==> Activating venv..."
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "==> Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "==> Installing compatible base dependencies..."
# Pin NumPy/SciPy for broad compatibility with scientific wheels and avoid
# NumPy 2.x + older SciPy ABI conflicts.
python -m pip install "numpy<2" "scipy<1.11" torch matplotlib

if [[ -f "requirements.txt" ]]; then
  echo "==> Installing repo requirements.txt..."
  # Keep pinned NumPy/SciPy in control by installing requirements after base.
  python -m pip install -r requirements.txt || true
fi

echo "==> Running dependency sanity checks..."
python - <<'PY'
import numpy
import scipy
import torch
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("torch:", torch.__version__)
print("Sanity import check: OK")
PY

echo
echo "Setup complete."
echo "To use this environment in a new shell:"
echo "  source ${VENV_DIR}/bin/activate"
echo
echo "Suggested first run:"
echo "  python eval_approx_backbone.py"

