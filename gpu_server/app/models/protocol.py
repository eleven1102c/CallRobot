from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DialogueState(Enum):
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    BOT_SPEAKING = "BOT_SPEAKING"
    USER_INTERRUPTING = "USER_INTERRUPTING"


class ClientEvent(BaseModel):
    type: Literal["audio", "end_utterance", "cancel", "reset", "text"]
    session_id: str
    audio_b64: str | None = None
    text: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class ServerEvent(BaseModel):
    type: str
    session_id: str
    state: DialogueState | None = None
    text: str | None = None
    audio_b64: str | None = None
    is_final: bool = False
    request_id: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
