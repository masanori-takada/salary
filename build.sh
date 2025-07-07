#!/usr/bin/env bash
# exit on error
set -o errexit

echo "=== Build Script Started ==="
echo "Python version check:"
python --version
python -c "import sys; print(f'Python {sys.version}')"

echo "=== Installing dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Verifying installations ==="
pip list | grep -E "(Flask|pandas|scikit-learn|lightgbm|numpy)"

echo "=== Build completed successfully! ===" 