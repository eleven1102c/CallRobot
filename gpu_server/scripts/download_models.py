from __future__ import annotations

import os
from pathlib import Path

from modelscope import snapshot_download


MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models")).resolve()

# MODELS = {
#     "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
#     "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
#     "speech_fsmn_vad_zh-cn-16k-common-pytorch": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
#     "punc_ct-transformer_zh-cn-common-vocab272727-pytorch": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
#     "CosyVoice-300M-SFT": "iic/CosyVoice-300M-SFT",
# }

MODELS = {
    "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    "speech_fsmn_vad_zh-cn-16k-common-pytorch": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    "punc_ct-transformer_zh-cn-common-vocab272727-pytorch": "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
    "CosyVoice-300M-SFT": "iic/CosyVoice-300M-SFT",
    "CosyVoice2-0.5B": "iic/CosyVoice2-0.5B",
}


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for local_name, repo_id in MODELS.items():
        target = MODEL_DIR / local_name
        print(f"Downloading {repo_id} -> {target}")
        snapshot_download(repo_id, local_dir=str(target))


if __name__ == "__main__":
    main()
