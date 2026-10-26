#!/usr/bin/env python3
"""Exercise import, padding and overlap boundaries with synthetic audio only."""
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import wave

from smoke_audio import read_chunks

ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "Package.swift").is_file())
BINARY = ROOT / "Build/Build/Products/Release/ReuseProbe"
MODEL = ROOT / "models/model.safetensors"


def make_fixture(path, settings):
    values = []
    for index in range(settings["frames"]):
        seconds = index / settings["rate"]
        for channel in range(settings["channels"]):
            value = 0 if settings["silence"] else 0.08 * math.sin(2 * math.pi * (250 + 90 * channel) * seconds)
            values.append(round(value * 32767))
    with wave.open(str(path), "wb") as audio:
        audio.setparams((settings["channels"], 2, settings["rate"], settings["frames"], "NONE", "not compressed"))
        audio.writeframes(struct.pack(f"<{len(values)}h", *values))


def validate_audio(path, expected):
    chunks = read_chunks(path)
    encoding, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    if encoding == 65534:
        encoding = struct.unpack_from("<H", chunks[b"fmt "], 24)[0]
    assert (encoding, channels, rate, bits) == (3, 1, 16000, 32), "Incorrect output format"
    samples = struct.unpack(f"<{len(chunks[b'data']) // 4}f", chunks[b"data"])
    assert len(samples) == expected, f"Expected {expected} samples, received {len(samples)}"
    assert all(math.isfinite(value) for value in samples), "Nonfinite output"
    return len(samples)


def run_case(temporary, name, settings):
    source, output = temporary / f"{name}-input.wav", temporary / f"{name}-output.wav"
    make_fixture(source, settings)
    command = [str(BINARY), "--input", str(source), "--output", str(output), "--model", str(MODEL)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"{name}: {result.stderr}"
    expected = round(settings["frames"] * 16000 / settings["rate"])
    count = validate_audio(output, expected)
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["all_finite"] and metadata["samples"] == count
    return dict(name=name, samples=count, duration_seconds=count / 16000,
                seconds=metadata["timings_seconds"][0], device=metadata["device"], passed=True)


def main():
    cases = [("short", 80, 16000, 1, False), ("silence", 16000, 16000, 1, True),
             ("stereo48k", 192000, 48000, 2, False), ("cross_window", 65600, 16000, 1, False)]
    records = []
    with tempfile.TemporaryDirectory(prefix="reuse-boundary-") as directory:
        for name, frames, rate, channels, silence in cases:
            settings = dict(frames=frames, rate=rate, channels=channels, silence=silence)
            records.append(run_case(Path(directory), name, settings))
    report = dict(passed=True, fixture="synthetic tones and silence; no speech", cases=records)
    destination = ROOT / "Logs/boundary-smoke.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
