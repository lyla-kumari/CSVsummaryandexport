#!/usr/bin/env zsh
set -euo pipefail

# Package the export_app folder for distribution.
# Creates a virtual env, installs requirements, locks installed packages, and creates a zip package.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

echo "Creating virtual environment .venv..."
python3 -m venv .venv
source .venv/bin/activate

echo "Upgrading pip and installing requirements..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "Freezing installed packages to requirements.lock.txt..."
pip freeze > requirements.lock.txt

echo "Deactivating virtual environment..."
deactivate

# Create package zip (exclude venv and locks if desired)
PACKAGE_NAME="export_app_package.zip"
echo "Creating package $PACKAGE_NAME..."
zip -r ../$PACKAGE_NAME . -x ".venv/*" "*.pyc" "__pycache__/*"

echo "Package created at ../$PACKAGE_NAME"

# End
