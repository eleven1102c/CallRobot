#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PORT="${GPU_SERVER_PORT:-9000}"
HOST="${GPU_SERVER_HOST:-0.0.0.0}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_DIR}/models}"
COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR:-${PROJECT_DIR}/third_party/CosyVoice}"

export PYTHONPATH="${PROJECT_DIR}:${COSYVOICE_REPO_DIR}:${COSYVOICE_REPO_DIR}/third_party/Matcha-TTS:${PYTHONPATH:-}"
export MODEL_DIR="${MODEL_DIR}"
export GPU_SERVER_HOST="${HOST}"
export GPU_SERVER_PORT="${PORT}"

export QWEN_MODEL="${QWEN_MODEL:-${MODEL_DIR}/Qwen2.5-7B-Instruct}"
export QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.82}"
export QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-8192}"
export QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-512}"
export QWEN_DTYPE="${QWEN_DTYPE:-float16}"
export QWEN_QUANTIZATION="${QWEN_QUANTIZATION:-}"
export QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-2}"
export QWEN_MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-4096}"
export QWEN_CPU_OFFLOAD_GB="${QWEN_CPU_OFFLOAD_GB:-0}"
export QWEN_SWAP_SPACE="${QWEN_SWAP_SPACE:-4}"
export QWEN_ENFORCE_EAGER="${QWEN_ENFORCE_EAGER:-false}"

export FUNASR_MODEL="${FUNASR_MODEL:-${MODEL_DIR}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online}"
export FUNASR_USE_VAD="${FUNASR_USE_VAD:-false}"
export FUNASR_VAD_MODEL="${FUNASR_VAD_MODEL:-${MODEL_DIR}/speech_fsmn_vad_zh-cn-16k-common-pytorch}"
export FUNASR_PUNC_MODEL="${FUNASR_PUNC_MODEL:-${MODEL_DIR}/punc_ct-transformer_zh-cn-common-vocab272727-pytorch}"

export COSYVOICE_MODEL="${COSYVOICE_MODEL:-${MODEL_DIR}/CosyVoice-300M-SFT}"
export COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR}"
export COSYVOICE_SPK="${COSYVOICE_SPK:-中文女}"
export TTS_FLUSH_CHARS="${TTS_FLUSH_CHARS:-8}"

cd "${PROJECT_DIR}"
python3 -m uvicorn gpu_server.app.main:app --host "${HOST}" --port "${PORT}"
