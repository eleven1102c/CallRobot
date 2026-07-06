from __future__ import annotations

import asyncio
import io
import wave
from contextlib import suppress
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    channels: int = 1
    dtype: str = "int16"
    input_device: int | str | None = None
    output_device: int | str | None = None

    @property
    def blocksize(self) -> int:
        return self.sample_rate * self.frame_ms // 1000


class MicCapture:
    def __init__(self, config: AudioConfig, queue: asyncio.Queue[bytes]) -> None:
        self.config = config
        self.queue = queue
        self._stream: sd.RawInputStream | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status) -> None:
            if status:
                print(f"[mic] {status}", flush=True)
            data = bytes(indata)
            assert self._loop is not None
            self._loop.call_soon_threadsafe(self._put_frame, data)

        self._stream = sd.RawInputStream(
            samplerate=self.config.sample_rate,
            blocksize=self.config.blocksize,
            channels=self.config.channels,
            dtype=self.config.dtype,
            device=self.config.input_device,
            callback=callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _put_frame(self, data: bytes) -> None:
        try:
            self.queue.put_nowait(data)
        except asyncio.QueueFull:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            with suppress(asyncio.QueueFull):
                self.queue.put_nowait(data)


class AudioPlayer:
    def __init__(self, output_device: int | str | None = None) -> None:
        self.output_device = output_device
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=32)
        self._cancel_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self.clear()
        await self.queue.put(None)
        if self._task is not None:
            await self._task

    async def enqueue_wav(self, wav_bytes: bytes) -> None:
        await self.queue.put(wav_bytes)

    def clear(self) -> None:
        self._cancel_event.set()
        sd.stop()
        while True:
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _run(self) -> None:
        while True:
            wav_bytes = await self.queue.get()
            if wav_bytes is None:
                return
            self._cancel_event.clear()
            try:
                audio, sample_rate = decode_wav(wav_bytes)
                await asyncio.to_thread(self._play_blocking, audio, sample_rate)
            except Exception as exc:
                print(f"[player] failed to play chunk: {exc}", flush=True)

    def _play_blocking(self, audio: np.ndarray, sample_rate: int) -> None:
        if self._cancel_event.is_set():
            return
        sd.play(audio, samplerate=sample_rate, device=self.output_device, blocking=True)


def decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(f"only 16-bit wav chunks are supported, got sample width {sample_width}")

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels)
    return audio, sample_rate


def list_devices() -> None:
    print(sd.query_devices())
