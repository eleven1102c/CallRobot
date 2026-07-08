from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WebRTCApmResult:
    pcm16: bytes
    enabled: bool
    available: bool
    error: str | None = None


class WebRTCApmController:
    """Optional WebRTC Audio Processing wrapper.

    Python wrappers around WebRTC APM do not expose a single stable API. This
    class supports the common `webrtc_audio_processing` style methods and
    cleanly reports unavailable status so callers can fall back to simpler
    echo/noise gates.
    """

    def __init__(
        self,
        enabled: bool,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        channels: int = 1,
        farend_buffer_ms: int = 1200,
    ) -> None:
        self.enabled = enabled
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.channels = channels
        self.frame_samples = sample_rate * frame_ms // 1000
        self.frame_bytes = self.frame_samples * channels * 2
        self.available = False
        self.error: str | None = None
        self._apm: Any | None = None
        self._farend_frames: deque[bytes] = deque(maxlen=max(1, farend_buffer_ms // frame_ms))

        if enabled:
            self._init_apm()

    def _init_apm(self) -> None:
        try:
            module = __import__("webrtc_audio_processing")
            apm_cls = getattr(module, "AudioProcessingModule", None) or getattr(module, "AudioProcessing", None)
            if apm_cls is None:
                raise ImportError("AudioProcessingModule not found in webrtc_audio_processing")

            self._apm = self._construct_apm(apm_cls)
            self._configure_apm()
            self.available = True
        except Exception as exc:
            self.available = False
            self.error = repr(exc)

    @staticmethod
    def _construct_apm(apm_cls: Any) -> Any:
        for kwargs in (
            {"enable_ns": True, "enable_agc": True, "enable_aec": True},
            {"enable_ns": True, "agc_type": 1},
            {},
        ):
            try:
                return apm_cls(**kwargs)
            except TypeError:
                continue
        return apm_cls()

    def _configure_apm(self) -> None:
        assert self._apm is not None
        for name in ("set_stream_format", "set_reverse_stream_format"):
            method = getattr(self._apm, name, None)
            if method is None:
                continue
            try:
                method(self.sample_rate, self.channels)
            except TypeError:
                method(self.sample_rate, self.channels, self.sample_rate, self.channels)

        self._call_if_exists("set_echo_cancellation", True)
        self._call_if_exists("set_noise_suppression", True)
        self._call_if_exists("set_gain_control", True)
        self._call_if_exists("set_high_pass_filter", True)

    def _call_if_exists(self, name: str, *args: Any) -> None:
        if self._apm is None:
            return
        method = getattr(self._apm, name, None)
        if method is None:
            return
        try:
            method(*args)
        except Exception:
            return

    def feed_farend_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        if not self.enabled:
            return
        mono = np.asarray(audio, dtype=np.float32)
        if mono.ndim > 1:
            mono = mono.mean(axis=1)
        mono = self._resample(mono.reshape(-1), sample_rate, self.sample_rate)
        pcm16 = np.clip(mono, -1.0, 1.0)
        pcm16 = (pcm16 * 32767.0).astype(np.int16).tobytes()
        for offset in range(0, len(pcm16) - self.frame_bytes + 1, self.frame_bytes):
            self._farend_frames.append(pcm16[offset: offset + self.frame_bytes])

    def process_mic_frame(self, pcm16: bytes) -> WebRTCApmResult:
        if not self.enabled:
            return WebRTCApmResult(pcm16=pcm16, enabled=False, available=False)
        if not self.available or self._apm is None:
            return WebRTCApmResult(pcm16=pcm16, enabled=True, available=False, error=self.error)
        if len(pcm16) != self.frame_bytes:
            return WebRTCApmResult(pcm16=pcm16, enabled=True, available=True, error="unexpected_frame_size")

        try:
            farend = self._farend_frames.popleft() if self._farend_frames else b"\x00" * self.frame_bytes
            self._process_reverse(farend)
            processed = self._process_stream(pcm16)
            return WebRTCApmResult(pcm16=processed, enabled=True, available=True)
        except Exception as exc:
            self.available = False
            self.error = repr(exc)
            return WebRTCApmResult(pcm16=pcm16, enabled=True, available=False, error=self.error)

    def _process_reverse(self, pcm16: bytes) -> None:
        assert self._apm is not None
        for name in ("process_reverse_stream", "process_reverse"):
            method = getattr(self._apm, name, None)
            if method is not None:
                method(pcm16)
                return

    def _process_stream(self, pcm16: bytes) -> bytes:
        assert self._apm is not None
        for name in ("process_stream", "process"):
            method = getattr(self._apm, name, None)
            if method is not None:
                out = method(pcm16)
                return bytes(out) if out is not None else pcm16
        return pcm16

    @staticmethod
    def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate or audio.size == 0:
            return audio
        duration = audio.size / float(src_rate)
        dst_size = max(1, int(round(duration * dst_rate)))
        src_x = np.linspace(0.0, duration, num=audio.size, endpoint=False)
        dst_x = np.linspace(0.0, duration, num=dst_size, endpoint=False)
        return np.interp(dst_x, src_x, audio).astype(np.float32)
