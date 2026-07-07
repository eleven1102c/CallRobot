from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import webrtcvad


@dataclass(frozen=True)
class VadResult:
    should_send: bool
    utterance_ended: bool
    pcm16: bytes | None = None
    is_speech: bool = False


class FastVadEndpoint:
    """Small WebRTC VAD endpointing layer for 16 kHz PCM16 frames."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        aggressiveness: int = 2,
        speech_start_ms: int = 80,
        speech_end_ms: int = 700,
        preroll_ms: int = 240,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = sample_rate * frame_ms // 1000 * 2
        self.vad = webrtcvad.Vad(aggressiveness)
        self.speech_start_frames = max(1, speech_start_ms // frame_ms)
        self.speech_end_frames = max(1, speech_end_ms // frame_ms)
        self.preroll_frames = max(0, preroll_ms // frame_ms)
        self._preroll: deque[bytes] = deque(maxlen=self.preroll_frames)
        self._triggered = False
        self._speech_run = 0
        self._silence_run = 0

    def process(self, frame: bytes) -> list[VadResult]:
        if len(frame) != self.frame_bytes:
            return []

        is_speech = self.vad.is_speech(frame, self.sample_rate)
        results: list[VadResult] = []

        if not self._triggered:
            self._preroll.append(frame)
            if is_speech:
                self._speech_run += 1
            else:
                self._speech_run = 0

            if self._speech_run >= self.speech_start_frames:
                self._triggered = True
                self._silence_run = 0
                for preroll_frame in self._preroll:
                    results.append(VadResult(should_send=True, utterance_ended=False, pcm16=preroll_frame, is_speech=True))
                self._preroll.clear()
            return results

        results.append(VadResult(should_send=True, utterance_ended=False, pcm16=frame, is_speech=is_speech))
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1

        if self._silence_run >= self.speech_end_frames:
            self._triggered = False
            self._speech_run = 0
            self._silence_run = 0
            self._preroll.clear()
            results.append(VadResult(should_send=False, utterance_ended=True))

        return results

    def reset(self) -> None:
        self._preroll.clear()
        self._triggered = False
        self._speech_run = 0
        self._silence_run = 0
