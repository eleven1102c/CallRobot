from __future__ import annotations

import argparse
import asyncio
import base64
from contextlib import suppress
from uuid import uuid4

import websockets

from callrobot_mac.audio_io import AudioConfig, AudioPlayer, MicCapture, list_devices
from callrobot_mac.protocol import audio_event, control_event, end_utterance_event, parse_server_event, text_event
from callrobot_mac.vad import FastVadEndpoint


def device_arg(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


class FullDuplexMacClient:
    def __init__(
        self,
        server_url: str,
        session_id: str,
        audio_config: AudioConfig,
        vad_aggressiveness: int,
        speech_end_ms: int,
        no_mic: bool = False,
    ) -> None:
        self.server_url = server_url
        self.session_id = session_id
        self.audio_config = audio_config
        self.vad = FastVadEndpoint(
            sample_rate=audio_config.sample_rate,
            frame_ms=audio_config.frame_ms,
            aggressiveness=vad_aggressiveness,
            speech_end_ms=speech_end_ms,
        )
        self.no_mic = no_mic
        self.mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self.player = AudioPlayer(output_device=audio_config.output_device)
        self.mic = MicCapture(audio_config, self.mic_queue)
        self._last_state = "LISTENING"

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
            for result in self.vad.process(frame):
                if result.should_send and result.pcm16:
                    await ws.send(audio_event(self.session_id, result.pcm16))
                if result.utterance_ended:
                    print("[vad] end_utterance")
                    await ws.send(end_utterance_event(self.session_id))

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
    parser.add_argument("--speech-end-ms", type=int, default=700)
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--output-device", default=None)
    parser.add_argument("--no-mic", action="store_true", help="Text-only mode")
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
        speech_end_ms=args.speech_end_ms,
        no_mic=args.no_mic,
    )
    await client.run()


def main() -> None:
    with suppress(KeyboardInterrupt):
        asyncio.run(amain())


if __name__ == "__main__":
    main()
