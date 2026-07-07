#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/mac_client:${PYTHONPATH:-}"

python3 -m callrobot_mac.mic_check "$@"
