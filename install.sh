#!/usr/bin/env bash
# Simple installation script for inference-economics
set -euo pipefail

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python3 is required but not found" >&2
  exit 1
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo "Dependencies installed successfully."
