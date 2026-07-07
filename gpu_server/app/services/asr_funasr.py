from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from gpu_server.app.config import Settings


class StreamingASR:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: Any | None = None
        self._caches: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    def load(self) -> None:
        from funasr import AutoModel

        kwargs: dict[str, Any] = {
            "model": self.settings.funasr_model,
            "disable_update": True,
        }
        if self.settings.funasr_use_vad and self.settings.funasr_vad_model:
            kwargs["vad_model"] = self.settings.funasr_vad_model
        if self.settings.funasr_punc_model:
            kwargs["punc_model"] = self.settings.funasr_punc_model
        self.model = AutoModel(**kwargs)

    async def transcribe_chunk(self, session_id: str, pcm16: bytes, is_final: bool = False) -> str:
        if self.model is None:
            raise RuntimeError("StreamingASR is not loaded")

        wav = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        async with self._lock:
            cache = self._caches.setdefault(session_id, {})

        def _infer() -> str:
            result = self.model.generate(
                input=wav,
                cache=cache,
                is_final=is_final,
                chunk_size=[int(x) for x in self.settings.funasr_chunk_size.split(",")],
                encoder_chunk_look_back=self.settings.funasr_encoder_chunk_look_back,
                decoder_chunk_look_back=self.settings.funasr_decoder_chunk_look_back,
            )
            if not result:
                return ""
            item = result[0] if isinstance(result, list) else result
            return str(item.get("text", "")).strip()

        return await asyncio.to_thread(_infer)

    async def reset(self, session_id: str) -> None:
        async with self._lock:
            self._caches.pop(session_id, None)
