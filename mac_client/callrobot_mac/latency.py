from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic


@dataclass
class TurnLatency:
    turn_id: int
    marks: dict[str, float] = field(default_factory=dict)

    def mark(self, name: str) -> None:
        self.marks.setdefault(name, monotonic())

    def ms_between(self, start: str, end: str) -> float | None:
        if start not in self.marks or end not in self.marks:
            return None
        return (self.marks[end] - self.marks[start]) * 1000.0


class LatencyTracker:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._next_turn_id = 1
        self.current: TurnLatency | None = None

    def start_turn(self) -> None:
        if not self.enabled:
            return
        self.current = TurnLatency(turn_id=self._next_turn_id)
        self._next_turn_id += 1
        self.mark("speech_start")

    def mark(self, name: str) -> None:
        if not self.enabled:
            return
        if self.current is None:
            self.start_turn()
        assert self.current is not None
        self.current.mark(name)

    def report(self, reason: str = "summary") -> None:
        if not self.enabled or self.current is None:
            return
        turn = self.current
        pairs = [
            ("speech", "speech_start", "speech_stop"),
            ("endpoint_to_send", "speech_stop", "end_utterance_sent"),
            ("asr_final", "end_utterance_sent", "asr_final"),
            ("first_token", "asr_final", "first_llm_token"),
            ("server_tts_flush", "asr_final", "server_first_tts_flush"),
            ("server_tts_audio", "server_first_tts_flush", "server_first_tts_audio"),
            ("first_tts", "asr_final", "first_tts_audio"),
            ("play_start", "first_tts_audio", "first_playback_start"),
            ("e2e_to_audio", "speech_stop", "first_playback_start"),
            ("e2e_to_token", "speech_stop", "first_llm_token"),
            ("bot_total", "asr_final", "bot_final"),
        ]
        parts: list[str] = []
        for label, start, end in pairs:
            value = turn.ms_between(start, end)
            if value is not None:
                parts.append(f"{label}={value:.0f}ms")
        marks = ",".join(sorted(turn.marks.keys()))
        print(f"\n[latency] turn={turn.turn_id} reason={reason} {' '.join(parts)} marks={marks}")
