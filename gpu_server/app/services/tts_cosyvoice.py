from __future__ import annotations

import asyncio
import io
import sys
import threading
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from gpu_server.app.config import Settings


class StreamingTTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self.model_type: str | None = None

    def load(self) -> None:
        if self.settings.cosyvoice_repo_dir:
            repo_dir = Path(self.settings.cosyvoice_repo_dir).expanduser().resolve()
            extra_paths = [repo_dir, repo_dir / "third_party" / "Matcha-TTS"]
            for path in extra_paths:
                if path.exists() and str(path) not in sys.path:
                    sys.path.insert(0, str(path))

        from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2

        model_dir = Path(self.settings.cosyvoice_model).expanduser().resolve()
        if (model_dir / "cosyvoice2.yaml").exists():
            self.model = CosyVoice2(str(model_dir))
            self.model_type = "cosyvoice2"
        elif (model_dir / "cosyvoice.yaml").exists():
            self.model = CosyVoice(str(model_dir))
            self.model_type = "cosyvoice1"
        else:
            raise FileNotFoundError(
                f"CosyVoice model config not found in {model_dir}. "
                "Expected cosyvoice.yaml for CosyVoice 1 or cosyvoice2.yaml for CosyVoice 2."
            )

    def _prompt_speech_16k(self) -> Any:
        if not self.settings.cosyvoice_prompt_wav:
            raise ValueError("COSYVOICE_PROMPT_WAV is required for zero_shot/instruct2 CosyVoice mode")
        wav_path = Path(self.settings.cosyvoice_prompt_wav).expanduser().resolve()
        if not wav_path.exists():
            raise FileNotFoundError(f"CosyVoice prompt wav not found: {wav_path}")
        audio, sample_rate = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            audio = self._resample(audio, sample_rate, 16000)
        return torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate or audio.size == 0:
            return audio.astype(np.float32)
        duration = audio.size / float(src_rate)
        dst_size = max(1, int(round(duration * dst_rate)))
        src_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
        dst_x = np.linspace(0.0, duration, num=dst_size, endpoint=False)
        return np.interp(dst_x, src_x, audio).astype(np.float32)

    def _inference_generator(self, text: str):
        if self.model is None:
            raise RuntimeError("StreamingTTS is not loaded")

        mode = self.settings.cosyvoice_mode.lower()
        if mode == "sft":
            if not hasattr(self.model, "inference_sft"):
                raise RuntimeError(f"{self.model_type} does not support inference_sft")
            return self.model.inference_sft(text, self.settings.cosyvoice_spk, stream=True)

        if mode == "zero_shot":
            if not hasattr(self.model, "inference_zero_shot"):
                raise RuntimeError(f"{self.model_type} does not support inference_zero_shot")
            if not self.settings.cosyvoice_prompt_text:
                raise ValueError("COSYVOICE_PROMPT_TEXT is required for zero_shot CosyVoice mode")
            return self.model.inference_zero_shot(
                text,
                self.settings.cosyvoice_prompt_text,
                self._prompt_speech_16k(),
                stream=True,
            )

        if mode == "instruct2":
            if not hasattr(self.model, "inference_instruct2"):
                raise RuntimeError(f"{self.model_type} does not support inference_instruct2")
            return self.model.inference_instruct2(
                text,
                self.settings.cosyvoice_instruct_text or "",
                self._prompt_speech_16k(),
                stream=True,
            )

        raise ValueError("COSYVOICE_MODE must be one of: sft, zero_shot, instruct2")

    @staticmethod
    def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        pcm = np.clip(audio, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    async def stream(self, text: str, cancel_event: asyncio.Event) -> AsyncIterator[bytes]:
        if self.model is None:
            raise RuntimeError("StreamingTTS is not loaded")
        if not text.strip():
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=8)

        def _infer() -> None:
            try:
                generator = self._inference_generator(text)
                for item in generator:
                    if cancel_event.is_set():
                        break
                    audio = item.get("tts_speech")
                    if audio is None:
                        continue
                    if hasattr(audio, "detach"):
                        audio_np = audio.detach().cpu().numpy().reshape(-1)
                    else:
                        audio_np = np.asarray(audio).reshape(-1)
                    chunk = self._wav_bytes(audio_np, self.settings.tts_sample_rate)
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

        thread = threading.Thread(target=_infer, daemon=True)
        thread.start()

        while True:
            if cancel_event.is_set():
                break
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk
