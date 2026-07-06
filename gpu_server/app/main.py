from __future__ import annotations

import asyncio
import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from gpu_server.app.config import get_settings
from gpu_server.app.models.protocol import ClientEvent, DialogueState, ServerEvent
from gpu_server.app.services.asr_funasr import StreamingASR
from gpu_server.app.services.interrupt_vad import InterruptVAD
from gpu_server.app.services.llm_vllm import StreamingLLM
from gpu_server.app.services.state_manager import DialogueStateManager
from gpu_server.app.services.tts_cosyvoice import StreamingTTS

settings = get_settings()
state_manager = DialogueStateManager()
asr = StreamingASR(settings)
llm = StreamingLLM(settings)
tts = StreamingTTS(settings)
interrupt_vad = InterruptVAD(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    asr.load()
    await llm.load()
    tts.load()
    yield


app = FastAPI(title="CallRobot GPU Server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


async def send_event(ws: WebSocket, event: ServerEvent) -> None:
    await ws.send_text(event.model_dump_json())


async def cancel_generation(session_id: str) -> None:
    request_id = await state_manager.cancel_active_output(session_id)
    await llm.cancel(request_id)


def should_flush_tts(buffer: str) -> bool:
    return len(buffer) >= 18 or any(buffer.endswith(p) for p in ("。", "！", "？", "\n", ".", "!", "?"))


async def run_bot_turn(ws: WebSocket, session_id: str, user_text: str) -> None:
    session = await state_manager.get(session_id)
    request_id = session.new_request_id()
    await state_manager.transition(session_id, DialogueState.THINKING)
    await send_event(ws, ServerEvent(type="state", session_id=session_id, state=DialogueState.THINKING, request_id=request_id))

    full_text = ""
    tts_buffer = ""
    await state_manager.transition(session_id, DialogueState.BOT_SPEAKING)
    await send_event(ws, ServerEvent(type="state", session_id=session_id, state=DialogueState.BOT_SPEAKING, request_id=request_id))

    try:
        async for delta in llm.stream(request_id, session.history, user_text):
            current = await state_manager.get(session_id)
            if current.active_llm_request_id != request_id or current.tts_cancel_event.is_set():
                break

            full_text += delta
            current.bot_text_buffer = full_text
            tts_buffer += delta
            await send_event(ws, ServerEvent(type="llm_token", session_id=session_id, text=delta, request_id=request_id))

            if should_flush_tts(tts_buffer):
                async for audio in tts.stream(tts_buffer, current.tts_cancel_event):
                    await send_event(
                        ws,
                        ServerEvent(
                            type="tts_audio",
                            session_id=session_id,
                            audio_b64=base64.b64encode(audio).decode("ascii"),
                            request_id=request_id,
                        ),
                    )
                tts_buffer = ""

        current = await state_manager.get(session_id)
        if tts_buffer and not current.tts_cancel_event.is_set():
            async for audio in tts.stream(tts_buffer, current.tts_cancel_event):
                await send_event(
                    ws,
                    ServerEvent(
                        type="tts_audio",
                        session_id=session_id,
                        audio_b64=base64.b64encode(audio).decode("ascii"),
                        request_id=request_id,
                    ),
                )

        if full_text and not current.tts_cancel_event.is_set():
            session.history.append({"role": "user", "content": user_text})
            session.history.append({"role": "assistant", "content": full_text})
            await send_event(ws, ServerEvent(type="bot_final", session_id=session_id, text=full_text, is_final=True, request_id=request_id))
    finally:
        latest = await state_manager.get(session_id)
        if latest.active_llm_request_id == request_id:
            latest.active_llm_request_id = None
        if latest.state != DialogueState.USER_INTERRUPTING:
            await state_manager.transition(session_id, DialogueState.LISTENING)
            await send_event(ws, ServerEvent(type="state", session_id=session_id, state=DialogueState.LISTENING))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    bot_task: asyncio.Task[None] | None = None
    try:
        while True:
            raw = await ws.receive_text()
            event = ClientEvent.model_validate_json(raw)
            session = await state_manager.get(event.session_id)

            if event.type == "reset":
                if bot_task:
                    bot_task.cancel()
                await cancel_generation(event.session_id)
                await asr.reset(event.session_id)
                await state_manager.reset(event.session_id)
                await send_event(ws, ServerEvent(type="state", session_id=event.session_id, state=DialogueState.LISTENING))
                continue

            if event.type == "cancel":
                if bot_task:
                    bot_task.cancel()
                await cancel_generation(event.session_id)
                await send_event(ws, ServerEvent(type="cancelled", session_id=event.session_id, state=DialogueState.USER_INTERRUPTING))
                continue

            if event.type == "audio":
                pcm = base64.b64decode(event.audio_b64 or "")
                partial = await asr.transcribe_chunk(event.session_id, pcm, is_final=False)
                if partial:
                    session.last_user_text = partial
                    await send_event(ws, ServerEvent(type="asr_partial", session_id=event.session_id, text=partial))

                decision = interrupt_vad.decide(session, partial)
                if decision.is_interrupt:
                    if bot_task:
                        bot_task.cancel()
                    await cancel_generation(event.session_id)
                    await send_event(
                        ws,
                        ServerEvent(
                            type="interrupt",
                            session_id=event.session_id,
                            state=DialogueState.USER_INTERRUPTING,
                            text=partial,
                            meta={"confidence": decision.confidence, "reason": decision.reason},
                        ),
                    )
                continue

            if event.type in {"end_utterance", "text"}:
                text = event.text or session.last_user_text
                if event.type == "end_utterance" and event.audio_b64:
                    final_pcm = base64.b64decode(event.audio_b64)
                    text = await asr.transcribe_chunk(event.session_id, final_pcm, is_final=True) or text
                text = (text or "").strip()
                if not text:
                    continue
                await state_manager.transition(event.session_id, DialogueState.THINKING)
                await send_event(ws, ServerEvent(type="asr_final", session_id=event.session_id, text=text, is_final=True))
                bot_task = asyncio.create_task(run_bot_turn(ws, event.session_id, text))
    except WebSocketDisconnect:
        if bot_task:
            bot_task.cancel()
    except asyncio.CancelledError:
        if bot_task:
            bot_task.cancel()
        raise
