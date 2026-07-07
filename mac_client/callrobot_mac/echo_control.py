from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EchoControlResult:
    pcm16: bytes
    rms: float
    peak: int
    noise_floor: float
    suppressed: bool
    playback_active: bool


class EchoController:
    """Lightweight local echo/noise control before VAD.

    This is not a full adaptive-filter AEC. It is a pragmatic barge-in gate:
    when local TTS playback is active, low-energy microphone frames are treated
    as echo/noise and zeroed before VAD. Strong near-end speech still passes.
    """

    def __init__(
        self,
        enabled: bool = True,
        noise_suppress: bool = True,
        min_rms: float = 220.0,
        echo_rms: float = 900.0,
        noise_margin: float = 3.0,
        noise_alpha: float = 0.98,
    ) -> None:
        self.enabled = enabled
        self.noise_suppress = noise_suppress
        self.min_rms = min_rms
        self.echo_rms = echo_rms
        self.noise_margin = noise_margin
        self.noise_alpha = noise_alpha
        self.noise_floor = min_rms

    def process(self, pcm16: bytes, playback_active: bool, user_speaking: bool) -> EchoControlResult:
        samples = np.frombuffer(pcm16, dtype=np.int16)
        if samples.size == 0:
            return EchoControlResult(pcm16, 0.0, 0, self.noise_floor, False, playback_active)

        float_samples = samples.astype(np.float32)
        rms = float(np.sqrt(np.mean(float_samples * float_samples)))
        peak = int(np.max(np.abs(samples)))

        if not playback_active and not user_speaking:
            self.noise_floor = self.noise_alpha * self.noise_floor + (1.0 - self.noise_alpha) * max(rms, 1.0)

        if not self.enabled:
            return EchoControlResult(pcm16, rms, peak, self.noise_floor, False, playback_active)

        threshold = max(self.min_rms, self.noise_floor * self.noise_margin)
        if playback_active:
            threshold = max(threshold, self.echo_rms)

        suppressed = False
        if self.noise_suppress and not user_speaking and rms < threshold:
            suppressed = True
            pcm16 = b"\x00" * len(pcm16)

        return EchoControlResult(pcm16, rms, peak, self.noise_floor, suppressed, playback_active)
