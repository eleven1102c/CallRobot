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

from gpu_server.app.config import Settings


class StreamingTTS:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None

    def load(self) -> None:
        if self.settings.cosyvoice_repo_dir:
            repo_dir = Path(self.settings.cosyvoice_repo_dir).expanduser().resolve()
            extra_paths = [repo_dir, repo_dir / "third_party" / "Matcha-TTS"]
            for path in extra_paths:
                if path.exists() and str(path) not in sys.path:
                    sys.path.insert(0, str(path))

        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2

            self.model = CosyVoice2(self.settings.cosyvoice_model)
        except ImportError:
            from cosyvoice.cli.cosyvoice import CosyVoice

            self.model = CosyVoice(self.settings.cosyvoice_model)

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
                generator = self.model.inference_sft(text, self.settings.cosyvoice_spk, stream=True)
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
