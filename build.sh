#!/usr/bin/env bash
# exit on error
set -o errexit

# Force Python 3.10
echo "Python version check:"
python --version

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "Build completed successfully!" 