from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import suppress
from time import monotonic
from urllib.parse import urlparse, urlunparse
from uuid import uuid4

import websockets

from callrobot_mac.audio_io import AudioConfig, AudioPlayer, MicCapture, list_devices
from callrobot_mac.echo_control import EchoController
from callrobot_mac.protocol import ClientEvent, audio_event, control_event, parse_server_event, text_event
from callrobot_mac.vad import FastVadEndpoint


def device_arg(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def normalize_ws_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "https":
        return urlunparse(parsed._replace(scheme="wss"))
    if parsed.scheme == "http":
        return urlunparse(parsed._replace(scheme="ws"))
    return url


class FullDuplexMacClient:
    def __init__(
        self,
        server_url: str,
        session_id: str,
        audio_config: AudioConfig,
        vad_aggressiveness: int,
        speech_start_ms: int,
        speech_end_ms: int,
        send_chunk_ms: int,
        max_utterance_ms: int,
        aec_enabled: bool,
        aec_min_rms: float,
        aec_echo_rms: float,
        no_mic: bool = False,
        debug_audio: bool = False,
    ) -> None:
        self.server_url = normalize_ws_url(server_url)
        self.session_id = session_id
        self.audio_config = audio_config
        self.vad = FastVadEndpoint(
            sample_rate=audio_config.sample_rate,
            frame_ms=audio_config.frame_ms,
            aggressiveness=vad_aggressiveness,
            speech_start_ms=speech_start_ms,
            speech_end_ms=speech_end_ms,
        )
        self.send_chunk_frames = max(1, send_chunk_ms // audio_config.frame_ms)
        self.max_utterance_frames = max(1, max_utterance_ms // audio_config.frame_ms)
        self.no_mic = no_mic
        self.debug_audio = debug_audio
        self.mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self.player = AudioPlayer(output_device=audio_config.output_device)
        self.mic = MicCapture(audio_config, self.mic_queue)
        self.echo_controller = EchoController(
            enabled=aec_enabled,
            min_rms=aec_min_rms,
            echo_rms=aec_echo_rms,
        )
        self._last_state = "LISTENING"
        self._sent_audio_frames = 0
        self._utterance_frames = 0
        self._pending_audio_frames: list[bytes] = []
        self._utterance_audio_frames: list[bytes] = []
        self._last_audio_debug_at = 0.0

    async def run(self) -> None:
        async with websockets.connect(self.server_url, max_size=None, ping_interval=20, ping_timeout=20) as ws:
            await self.player.start()
            print(f"[client] connected: {self.server_url}")
            print(f"[client] session_id={self.session_id}")
            if not self.no_mic:
                self.mic.start()
                print("[mic] started. Use headphones for reliable full-duplex interrupt testing.")

            tasks = [
                asyncio.create_task(self._recv_loop(ws)),
                asyncio.create_task(self._keyboard_loop(ws)),
            ]
            if not self.no_mic:
                tasks.append(asyncio.create_task(self._mic_loop(ws)))

            try:
                await asyncio.gather(*tasks)
            finally:
                self.mic.stop()
                await self.player.stop()

    async def _mic_loop(self, ws) -> None:
        while True:
            frame = await self.mic_queue.get()
            echo_result = self.echo_controller.process(
                frame,
                playback_active=self.player.is_playing or self._last_state == "BOT_SPEAKING",
                user_speaking=self.vad.user_speaking,
            )
            self._debug_input_level(echo_result)
            for result in self.vad.process(echo_result.pcm16):
                if result.user_speaking_started:
                    print("[user] speaking")
                if result.should_send and result.pcm16:
                    self._sent_audio_frames += 1
                    self._utterance_frames += 1
                    self._pending_audio_frames.append(result.pcm16)
                    self._utterance_audio_frames.append(result.pcm16)
                    if len(self._pending_audio_frames) >= self.send_chunk_frames:
                        await self._flush_audio(ws)
                    if self.debug_audio and self._sent_audio_frames % 50 == 0:
                        print(
                            f"[mic] captured_frames={self._sent_audio_frames} "
                            f"pending_frames={len(self._pending_audio_frames)}"
                        )
                    if self._utterance_frames >= self.max_utterance_frames:
                        print(f"[vad] force_end_utterance captured_frames={self._sent_audio_frames}")
                        await self._end_utterance(ws)
                        continue
                if result.utterance_ended:
                    print("[user] stopped")
                    await self._end_utterance(ws)

    async def _flush_audio(self, ws) -> None:
        if not self._pending_audio_frames:
            return
        pcm = b"".join(self._pending_audio_frames)
        self._pending_audio_frames.clear()
        await ws.send(audio_event(self.session_id, pcm))

    async def _end_utterance(self, ws) -> None:
        await self._flush_audio(ws)
        print(f"[vad] end_utterance captured_frames={self._sent_audio_frames}")
        if self._sent_audio_frames:
            utterance_pcm = b"".join(self._utterance_audio_frames)
            if self.debug_audio:
                print(f"[mic] final_utterance_bytes={len(utterance_pcm)}")
            await ws.send(
                ClientEvent(
                    type="end_utterance",
                    session_id=self.session_id,
                    audio_b64=base64.b64encode(utterance_pcm).decode("ascii"),
                    meta={"sample_rate": self.audio_config.sample_rate},
                ).to_json()
            )
        self.vad.reset()
        self._sent_audio_frames = 0
        self._utterance_frames = 0
        self._pending_audio_frames.clear()
        self._utterance_audio_frames.clear()

    def _debug_input_level(self, echo_result) -> None:
        if not self.debug_audio:
            return
        now = monotonic()
        if now - self._last_audio_debug_at < 1.0:
            return
        self._last_audio_debug_at = now
        print(
            f"[mic] rms={echo_result.rms:.1f} peak={echo_result.peak} "
            f"noise={echo_result.noise_floor:.1f} playback={int(echo_result.playback_active)} "
            f"suppressed={int(echo_result.suppressed)} queue={self.mic_queue.qsize()}"
        )

    async def _recv_loop(self, ws) -> None:
        async for raw in ws:
            event = parse_server_event(raw)
            event_type = event.get("type")

            if event_type == "state":
                self._last_state = event.get("state") or self._last_state
                print(f"\n[state] {self._last_state}")
            elif event_type == "asr_partial":
                print(f"\r[asr] {event.get('text', '')}", end="", flush=True)
            elif event_type == "asr_final":
                print(f"\n[user] {event.get('text', '')}")
            elif event_type == "asr_empty":
                print(f"\n[asr_empty] {event.get('meta', {})}")
            elif event_type == "asr_error":
                print(f"\n[asr_error] {event.get('meta', {})}")
            elif event_type == "llm_token":
                print(event.get("text", ""), end="", flush=True)
            elif event_type == "tts_audio":
                audio_b64 = event.get("audio_b64")
                if audio_b64:
                    await self.player.enqueue_wav(base64.b64decode(audio_b64))
            elif event_type == "interrupt":
                print(f"\n[interrupt] {event.get('text', '')} {event.get('meta', {})}")
                self.player.clear()
            elif event_type == "cancelled":
                print("\n[cancelled]")
                self.player.clear()
            elif event_type == "bot_final":
                print(f"\n[bot_final] {event.get('text', '')}\n")
            else:
                print(f"\n[event] {event}")

    async def _keyboard_loop(self, ws) -> None:
        print("[keys] Enter text to send. Commands: /cancel /reset /quit")
        while True:
            line = await asyncio.to_thread(input, "> ")
            line = line.strip()
            if not line:
                continue
            if line == "/quit":
                await ws.close()
                return
            if line == "/cancel":
                self.player.clear()
                await ws.send(control_event(self.session_id, "cancel"))
                continue
            if line == "/reset":
                self.player.clear()
                await ws.send(control_event(self.session_id, "reset"))
                continue
            await ws.send(text_event(self.session_id, line))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CallRobot Mac full-duplex test client")
    parser.add_argument("--server", default="ws://127.0.0.1:9000/ws", help="GPU server WebSocket URL")
    parser.add_argument("--session-id", default=f"mac-{uuid4().hex[:8]}")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-ms", type=int, default=20, choices=[10, 20, 30])
    parser.add_argument("--vad", type=int, default=2, choices=[0, 1, 2, 3], help="WebRTC VAD aggressiveness")
    parser.add_argument("--speech-start-ms", type=int, default=120)
    parser.add_argument("--speech-end-ms", type=int, default=500)
    parser.add_argument("--send-chunk-ms", type=int, default=200, help="Aggregate mic audio before each WebSocket send")
    parser.add_argument("--max-utterance-ms", type=int, default=12000, help="Force endpointing after this duration")
    parser.add_argument("--no-aec", action="store_true", help="Disable local echo/noise suppression before VAD")
    parser.add_argument("--aec-min-rms", type=float, default=220.0, help="Minimum RMS required before VAD")
    parser.add_argument("--aec-echo-rms", type=float, default=900.0, help="RMS gate while local TTS playback is active")
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--no-mic", action="store_true", help="Text-only mode")
    parser.add_argument("--debug-audio", action="store_true", help="Print mic level and VAD diagnostics")
    parser.add_argument("--list-devices", action="store_true")
    return parser.parse_args()


async def amain() -> None:
    args = parse_args()
    if args.list_devices:
        list_devices()
        return

    config = AudioConfig(
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        input_device=device_arg(args.input_device),
        output_device=device_arg(args.output_device),
    )
    client = FullDuplexMacClient(
        server_url=args.server,
        session_id=args.session_id,
        audio_config=config,
        vad_aggressiveness=args.vad,
        speech_start_ms=args.speech_start_ms,
        speech_end_ms=args.speech_end_ms,
        send_chunk_ms=args.send_chunk_ms,
        max_utterance_ms=args.max_utterance_ms,
        aec_enabled=not args.no_aec,
        aec_min_rms=args.aec_min_rms,
        aec_echo_rms=args.aec_echo_rms,
        no_mic=args.no_mic,
        debug_audio=args.debug_audio,
    )
    await client.run()


def main() -> None:
    with suppress(KeyboardInterrupt):
        asyncio.run(amain())


if __name__ == "__main__":
    main()
