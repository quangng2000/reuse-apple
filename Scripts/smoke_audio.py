"""Standard-library audio fixtures and WAV validation; no speech recordings."""
import math
import struct
import wave


def make_fixture(path):
    samples = []
    for index in range(64000):
        seconds = index / 16000
        value = 0.09 * math.sin(2 * math.pi * (180 * seconds + 45 * seconds**2))
        value += 0.015 * math.sin(2 * math.pi * 2700 * seconds)
        samples.append(round(value * 32767))
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, len(samples), "NONE", "not compressed"))
        audio.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def read_chunks(path):
    contents = path.read_bytes()
    assert contents[:4] == b"RIFF" and contents[8:12] == b"WAVE", "Not a RIFF WAV"
    chunks, offset = {}, 12
    while offset + 8 <= len(contents):
        name, size = struct.unpack_from("<4sI", contents, offset)
        assert offset + 8 + size <= len(contents), "Truncated WAV chunk"
        chunks[name] = contents[offset + 8:offset + 8 + size]
        offset += 8 + size + size % 2
    return chunks


def validate_output(path):
    chunks = read_chunks(path)
    encoding, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", chunks[b"fmt "])
    if encoding == 65534:
        encoding = struct.unpack_from("<H", chunks[b"fmt "], 24)[0]
    assert (encoding, channels, rate, bits) == (3, 1, 16000, 32), "Expected mono float32 16 kHz WAV"
    data = chunks[b"data"]
    assert len(data) == 64000 * 4, "Inference changed the sample count"
    samples = struct.unpack(f"<{len(data) // 4}f", data)
    assert all(math.isfinite(value) for value in samples), "Nonfinite output sample"
    return {"sample_rate": rate, "channels": channels, "samples": len(samples), "all_finite": True}
