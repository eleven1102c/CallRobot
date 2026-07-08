#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR:-${PROJECT_DIR}/third_party/CosyVoice}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models}"
COSYVOICE_MODEL="${COSYVOICE_MODEL:-${MODEL_DIR}/CosyVoice-300M-SFT}"

export PYTHONPATH="${PROJECT_DIR}:${COSYVOICE_REPO_DIR}:${COSYVOICE_REPO_DIR}/third_party/Matcha-TTS:${PYTHONPATH:-}"
export COSYVOICE_REPO_DIR
export COSYVOICE_MODEL

cd "${PROJECT_DIR}"
python3 gpu_server/scripts/check_cosyvoice.py "$@"
