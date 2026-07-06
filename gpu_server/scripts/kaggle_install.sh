#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/kaggle/working/CallRobot}"

cd "${PROJECT_DIR}"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-kaggle.txt
