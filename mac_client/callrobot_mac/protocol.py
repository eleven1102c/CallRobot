from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ClientEvent:
    type: str
    session_id: str
    audio_b64: str | None = None
    text: str | None = None
    meta: dict[str, Any] | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {"type": self.type, "session_id": self.session_id}
        if self.audio_b64 is not None:
            payload["audio_b64"] = self.audio_b64
        if self.text is not None:
            payload["text"] = self.text
        if self.meta:
            payload["meta"] = self.meta
        return json.dumps(payload, ensure_ascii=False)


def audio_event(session_id: str, pcm16: bytes) -> str:
    return ClientEvent(
        type="audio",
        session_id=session_id,
        audio_b64=base64.b64encode(pcm16).decode("ascii"),
    ).to_json()


def end_utterance_event(session_id: str) -> str:
    return ClientEvent(type="end_utterance", session_id=session_id).to_json()


def text_event(session_id: str, text: str) -> str:
    return ClientEvent(type="text", session_id=session_id, text=text).to_json()


def control_event(session_id: str, event_type: str) -> str:
    return ClientEvent(type=event_type, session_id=session_id).to_json()


def parse_server_event(raw: str) -> dict[str, Any]:
    return json.loads(raw)
