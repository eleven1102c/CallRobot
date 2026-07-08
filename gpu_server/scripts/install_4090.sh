#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR:-${PROJECT_DIR}/third_party/CosyVoice}"

cd "${PROJECT_DIR}"

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r requirements-gpu.txt --no-dependencies

if [ ! -d "${COSYVOICE_REPO_DIR}/cosyvoice" ]; then
  mkdir -p "$(dirname "${COSYVOICE_REPO_DIR}")"
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "${COSYVOICE_REPO_DIR}"
fi

cat << 'EOF' >> "${COSYVOICE_REPO_DIR}/requirements.txt"
ruamel.yaml==0.17.28
onnxruntime
whisper
more_itertools
typeguard==4.0.1
EOF

python3 -m pip install -r "${COSYVOICE_REPO_DIR}/requirements.txt" --no-dependencies

python3 - <<'PY'
import fastapi, funasr, transformers, vllm
print("core deps ok")
PY

PYTHONPATH="${COSYVOICE_REPO_DIR}:${COSYVOICE_REPO_DIR}/third_party/Matcha-TTS:${PYTHONPATH:-}" \
python3 - <<'PY'
from cosyvoice.cli.cosyvoice import CosyVoice
print("cosyvoice import ok")
PY
