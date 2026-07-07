from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil

import webrtcvad


@dataclass(frozen=True)
class VadResult:
    should_send: bool
    utterance_ended: bool
    pcm16: bytes | None = None
    is_speech: bool = False
    user_speaking: bool = False
    user_speaking_started: bool = False
    user_speaking_stopped: bool = False


class FastVadEndpoint:
    """Small WebRTC VAD endpointing layer for 16 kHz PCM16 frames."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_ms: int = 20,
        aggressiveness: int = 2,
        speech_start_ms: int = 120,
        speech_end_ms: int = 500,
        preroll_ms: int = 240,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_bytes = sample_rate * frame_ms // 1000 * 2
        self.vad = webrtcvad.Vad(aggressiveness)
        self.speech_start_frames = max(1, ceil(speech_start_ms / frame_ms))
        self.speech_end_frames = max(1, ceil(speech_end_ms / frame_ms))
        self.preroll_frames = max(0, ceil(preroll_ms / frame_ms))
        self._preroll: deque[bytes] = deque(maxlen=self.preroll_frames)
        self.user_speaking = False
        self._speech_run = 0
        self._silence_run = 0

    def process(self, frame: bytes) -> list[VadResult]:
        if len(frame) != self.frame_bytes:
            return []

        is_speech = self.vad.is_speech(frame, self.sample_rate)
        results: list[VadResult] = []

        if not self.user_speaking:
            self._preroll.append(frame)
            if is_speech:
                self._speech_run += 1
            else:
                self._speech_run = 0

            if self._speech_run >= self.speech_start_frames:
                self.user_speaking = True
                self._silence_run = 0
                preroll_frames = list(self._preroll)
                for index, preroll_frame in enumerate(preroll_frames):
                    results.append(
                        VadResult(
                            should_send=True,
                            utterance_ended=False,
                            pcm16=preroll_frame,
                            is_speech=True,
                            user_speaking=True,
                            user_speaking_started=index == 0,
                        )
                    )
                self._preroll.clear()
            return results

        results.append(
            VadResult(
                should_send=True,
                utterance_ended=False,
                pcm16=frame,
                is_speech=is_speech,
                user_speaking=True,
            )
        )
        if is_speech:
            self._silence_run = 0
        else:
            self._silence_run += 1

        if self._silence_run >= self.speech_end_frames:
            self.user_speaking = False
            self._speech_run = 0
            self._silence_run = 0
            self._preroll.clear()
            results.append(
                VadResult(
                    should_send=False,
                    utterance_ended=True,
                    is_speech=False,
                    user_speaking=False,
                    user_speaking_stopped=True,
                )
            )

        return results

    def reset(self) -> None:
        self._preroll.clear()
        self.user_speaking = False
        self._speech_run = 0
        self._silence_run = 0
