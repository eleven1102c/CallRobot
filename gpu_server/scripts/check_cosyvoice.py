from __future__ import annotations

import argparse
import io
import os
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np


def add_cosyvoice_paths(repo_dir: Path) -> None:
    paths = [repo_dir, repo_dir / "third_party" / "Matcha-TTS"]
    for path in paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def check_pkg_resources() -> None:
    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CosyVoice depends on pkg_resources, which is provided by setuptools. "
            "Install it in the active environment with: python3 -m pip install -U setuptools"
        ) from exc


def detect_model_type(model_dir: Path) -> str:
    if (model_dir / "cosyvoice2.yaml").exists():
        return "cosyvoice2"
    if (model_dir / "cosyvoice.yaml").exists():
        return "cosyvoice1"
    raise FileNotFoundError(
        f"CosyVoice yaml not found in {model_dir}. "
        "Expected cosyvoice.yaml or cosyvoice2.yaml."
    )


def wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = np.asarray(audio).reshape(-1)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


def extract_audio(item: dict[str, Any]) -> np.ndarray:
    audio = item.get("tts_speech")
    if audio is None:
        raise RuntimeError(f"CosyVoice output does not contain tts_speech. keys={list(item.keys())}")
    if hasattr(audio, "detach"):
        return audio.detach().cpu().numpy().reshape(-1)
    return np.asarray(audio).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CosyVoice import, model load, and optional inference")
    parser.add_argument("--repo-dir", default=os.environ.get("COSYVOICE_REPO_DIR", "third_party/CosyVoice"))
    parser.add_argument("--model-dir", default=os.environ.get("COSYVOICE_MODEL", "models/CosyVoice-300M-SFT"))
    parser.add_argument("--spk", default=os.environ.get("COSYVOICE_SPK", "中文女"))
    parser.add_argument("--text", default="你好，这是一次语音合成环境测试。")
    parser.add_argument("--infer", action="store_true", help="Run a short TTS inference after loading")
    parser.add_argument("--out", default="/tmp/cosyvoice_check.wav")
    parser.add_argument("--sample-rate", type=int, default=22050)
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()

    print(f"[check] python={sys.executable}")
    print(f"[check] repo_dir={repo_dir}")
    print(f"[check] model_dir={model_dir}")
    print(f"[check] spk={args.spk}")

    if not repo_dir.exists():
        raise FileNotFoundError(f"CosyVoice repo_dir does not exist: {repo_dir}")
    if not (repo_dir / "cosyvoice" / "cli" / "cosyvoice.py").exists():
        raise FileNotFoundError(f"Official CosyVoice source not found under: {repo_dir}")
    if not model_dir.exists():
        raise FileNotFoundError(f"CosyVoice model_dir does not exist: {model_dir}")

    add_cosyvoice_paths(repo_dir)

    print("[check] checking pkg_resources from setuptools ...")
    check_pkg_resources()

    print("[check] importing cosyvoice.cli.cosyvoice ...")
    from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

    model_type = detect_model_type(model_dir)
    print(f"[check] model_type={model_type}")

    print("[check] loading model ...")
    model = CosyVoice2(str(model_dir)) if model_type == "cosyvoice2" else CosyVoice(str(model_dir))
    print("[check] model loaded")

    if not args.infer:
        print("[check] inference skipped. Pass --infer to synthesize a short wav.")
        return

    print("[check] running inference_sft ...")
    chunks: list[np.ndarray] = []
    for item in model.inference_sft(args.text, args.spk, stream=True):
        chunks.append(extract_audio(item))

    if not chunks:
        raise RuntimeError("CosyVoice inference produced no audio chunks")

    audio = np.concatenate(chunks)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(wav_bytes(audio, args.sample_rate))
    print(f"[check] wrote {out} samples={audio.shape[0]}")


if __name__ == "__main__":
    main()
