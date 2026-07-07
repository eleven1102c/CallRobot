from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd


def device_arg(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local microphone input level")
    parser.add_argument("--input-device", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    device = device_arg(args.input_device)
    blocksize = int(args.sample_rate * 0.1)

    print(sd.query_devices())
    print(f"[mic-check] device={device} sample_rate={args.sample_rate} seconds={args.seconds}")
    started_at = time.monotonic()

    def callback(indata, frames, time_info, status) -> None:
        if status:
            print(f"[mic-check] {status}")
        pcm = np.frombuffer(bytes(indata), dtype=np.int16).astype(np.float32)
        rms = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
        peak = int(np.max(np.abs(pcm))) if pcm.size else 0
        print(f"[mic-check] rms={rms:.1f} peak={peak}")

    with sd.RawInputStream(
        samplerate=args.sample_rate,
        blocksize=blocksize,
        channels=1,
        dtype="int16",
        device=device,
        callback=callback,
    ):
        while time.monotonic() - started_at < args.seconds:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
