#!/usr/bin/env python3
"""One real GPU inference plus CLI safety checks; Python standard library only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile

from smoke_audio import make_fixture, validate_output

ROOT = Path(__file__).resolve().parents[1]


def invoke(binary, arguments, expected=0):
    result = subprocess.run([str(binary), *map(str, arguments)], text=True, capture_output=True)
    assert result.returncode == expected, (arguments, result.returncode, result.stderr)
    return result


def check_cli_errors(binary):
    assert "Swift/MLX/Metal" in invoke(binary, ["--help"]).stdout
    for arguments in ([], ["--input"], ["--unknown"], ["--input", "--output"]):
        invoke(binary, arguments, expected=1)


def run_fixture(binary, model, temporary):
    source, output = temporary / "fixture.wav", temporary / "enhanced.wav"
    report = output.with_suffix(".json")
    make_fixture(source)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    arguments = ["--input", source, "--output", output, "--model", model]
    invoke(binary, ["--input", source, "--output", source, "--model", model, "--force"], 1)
    output.write_bytes(b"existing WAV")
    report.write_text("existing report")
    invoke(binary, arguments, 1)
    assert output.read_bytes() == b"existing WAV" and report.read_text() == "existing report"
    invoke(binary, [*arguments, "--force"])
    metadata = json.loads(report.read_text())
    validated = validate_output(output)
    invoke(binary, arguments, 1)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash, "Input was overwritten"
    return validated, metadata


def validate_metadata(metadata):
    assert metadata["backend"] == "Swift/MLX/Metal"
    assert metadata["mlx_peak_bytes"] > 0 and metadata["all_finite"]
    assert len(metadata["timings_seconds"]) == 1
    assert all(math.isfinite(value) and value > 0 for value in metadata["timings_seconds"])
    assert (metadata["sample_rate"], metadata["channels"], metadata["samples"]) == (16000, 1, 64000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=ROOT / "Build/Build/Products/Release/ReuseProbe")
    parser.add_argument("--model", type=Path, default=ROOT / "Resources/model.safetensors")
    args = parser.parse_args()
    check_cli_errors(args.binary.resolve())
    with tempfile.TemporaryDirectory(prefix="reuse-smoke-") as temporary:
        validated, metadata = run_fixture(args.binary.resolve(), args.model.resolve(), Path(temporary))
    validate_metadata(metadata)
    print(json.dumps(dict(passed=True, fixture="synthetic chirp; no speech", audio=validated,
                         device=metadata["device"], timings_seconds=metadata["timings_seconds"],
                         checks=["help", "argument errors", "overwrite refusal", "force replacement",
                                 "input preservation", "GPU inference", "output format and finiteness"]), indent=2))


if __name__ == "__main__":
    main()
