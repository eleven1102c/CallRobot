import asyncio
from dataclasses import dataclass, field
from time import monotonic
from uuid import uuid4

from gpu_server.app.models.protocol import DialogueState


@dataclass
class DialogueSession:
    session_id: str
    state: DialogueState = DialogueState.LISTENING
    history: list[dict[str, str]] = field(default_factory=list)
    active_llm_request_id: str | None = None
    bot_text_buffer: str = ""
    last_user_text: str = ""
    tts_cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    updated_at: float = field(default_factory=monotonic)

    def new_request_id(self) -> str:
        self.active_llm_request_id = f"{self.session_id}-{uuid4().hex}"
        self.tts_cancel_event.clear()
        self.updated_at = monotonic()
        return self.active_llm_request_id


class DialogueStateManager:
    def __init__(self) -> None:
        self._sessions: dict[str, DialogueSession] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> DialogueSession:
        async with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = DialogueSession(session_id=session_id)
            return self._sessions[session_id]

    async def transition(self, session_id: str, state: DialogueState) -> DialogueSession:
        session = await self.get(session_id)
        session.state = state
        session.updated_at = monotonic()
        return session

    async def reset(self, session_id: str) -> DialogueSession:
        async with self._lock:
            self._sessions[session_id] = DialogueSession(session_id=session_id)
            return self._sessions[session_id]

    async def cancel_active_output(self, session_id: str) -> str | None:
        session = await self.get(session_id)
        session.tts_cancel_event.set()
        request_id = session.active_llm_request_id
        session.active_llm_request_id = None
        session.bot_text_buffer = ""
        session.state = DialogueState.USER_INTERRUPTING
        session.updated_at = monotonic()
        return request_id
