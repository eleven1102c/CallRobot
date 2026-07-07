#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/kaggle/working/CallRobot}"
PORT="${GPU_SERVER_PORT:-8081}"
HOST="${GPU_SERVER_HOST:-0.0.0.0}"
MODEL_DIR="${MODEL_DIR:-/kaggle/working/models}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export MODEL_DIR="${MODEL_DIR}"
export GPU_SERVER_HOST="${HOST}"
export GPU_SERVER_PORT="${PORT}"

export QWEN_MODEL="${QWEN_MODEL:-${MODEL_DIR}/Qwen2.5-7B-Instruct}"
export QWEN_GPU_MEMORY_UTILIZATION="${QWEN_GPU_MEMORY_UTILIZATION:-0.72}"
export QWEN_MAX_MODEL_LEN="${QWEN_MAX_MODEL_LEN:-4096}"
export QWEN_MAX_TOKENS="${QWEN_MAX_TOKENS:-256}"
export QWEN_DTYPE="${QWEN_DTYPE:-float16}"
export QWEN_MAX_NUM_SEQS="${QWEN_MAX_NUM_SEQS:-1}"
export QWEN_MAX_NUM_BATCHED_TOKENS="${QWEN_MAX_NUM_BATCHED_TOKENS:-2048}"
export QWEN_CPU_OFFLOAD_GB="${QWEN_CPU_OFFLOAD_GB:-0}"
export QWEN_SWAP_SPACE="${QWEN_SWAP_SPACE:-2}"
export QWEN_ENFORCE_EAGER="${QWEN_ENFORCE_EAGER:-true}"
export FUNASR_MODEL="${FUNASR_MODEL:-${MODEL_DIR}/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online}"
export FUNASR_VAD_MODEL="${FUNASR_VAD_MODEL:-${MODEL_DIR}/speech_fsmn_vad_zh-cn-16k-common-pytorch}"
export FUNASR_PUNC_MODEL="${FUNASR_PUNC_MODEL:-${MODEL_DIR}/punc_ct-transformer_zh-cn-common-vocab272727-pytorch}"
export COSYVOICE_MODEL="${COSYVOICE_MODEL:-${MODEL_DIR}/CosyVoice-300M-SFT}"
export COSYVOICE_REPO_DIR="${COSYVOICE_REPO_DIR:-/kaggle/working/CosyVoice}"

cd "${PROJECT_DIR}"
python -m uvicorn gpu_server.app.main:app --host "${HOST}" --port "${PORT}"
