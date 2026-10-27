#!/usr/bin/env python3
"""Compare two captured FP32 streaming outputs; no inference or audio realignment.

This is one synthetic noisy-speech diagnostic, not broad audio-quality validation.
Requires NumPy, SciPy and pystoi (for example /tmp/reuse-audio-onnx-env/bin/python).
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import re

import numpy as np
import pystoi
import scipy


SDK = "io.github.quangng2000:reuse-streaming-android:0.1.2"
BASELINE_MODEL = "18db8d32b638702909de484d2a9cffc97ed09e6454c4caed3f0008fb0aab53e8"
CANDIDATE_MODEL = "6b6cbd4117146f3137c1a0db71ecf04c23b8ba97bab7d973ece71c8739aa7756"
CANDIDATE_MODELS = {
    "frequency-v1": CANDIDATE_MODEL,
    "bulk-v2": "1fbd91be9c90568c58ab40e9ce70e07c54dafc599fffb694a84c1f4d3905b11d",
    "inline-v2": "55d910c30cd7144187e5f358dca6700ae6e7e1051eb8f7758d454f68e9144c11",
    "stable-v2": "db29c725c70608920e600787fcda625e98429cfbae2c2d09f1dc73726825319a",
}
CANDIDATE_MODEL_BYTES = {"frequency-v1": 11_590_965, "bulk-v2": 11_590_141,
                         "inline-v2": 11_584_702, "stable-v2": 11_586_326}
SAMPLE_RATE = 16_000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def pcm(raw, label):
    require(len(raw) > 0 and len(raw) % 4 == 0, f"{label}: expected nonempty FP32 little-endian PCM")
    values = np.frombuffer(raw, dtype="<f4")
    require(bool(np.isfinite(values).all()), f"{label}: nonfinite PCM")
    return values.astype(np.float64)


def read_capture(path, expected_model, expected_backend=None):
    require(path.is_file() and path.stat().st_size <= 10 * 1024 * 1024, "Invalid capture result file")
    raw_json = path.read_bytes()
    data = json.loads(raw_json)
    require(data.get("schemaVersion") == 1 and data.get("success") is True and data.get("status") == "complete",
            "Capture did not complete successfully")
    for field, expected in {
        "sdkCoordinate": SDK, "precision": "FP32", "executionProvider": "CPUExecutionProvider",
        "onnxRuntimeVersion": "1.23.2", "sampleRate": SAMPLE_RATE, "hopSamples": 320, "windowSamples": 640,
        "qualityCapture": True,
    }.items():
        require(data.get(field) == expected, f"Unexpected {field}: {data.get(field)!r}")
    native = expected_model != BASELINE_MODEL
    if native and expected_backend is None:
        expected_backend = "frequency-v1"  # Preserve the original explicit V1 comparison API.
    if native:
        require(expected_backend in CANDIDATE_MODELS and CANDIDATE_MODELS[expected_backend] == expected_model,
                "Unknown or mismatched candidate backend/model pin")
        allowed = {"native-v2-comparison"}
        if expected_backend == "frequency-v1":
            allowed.add("native-frequency-v1")
        require(data.get("sdkSource") == "local" and data.get("variant") in allowed and
                data.get("nativeFrequencyScan") is True, "Unexpected candidate SDK source or variant")
    else:
        require(expected_backend is None and data.get("sdkSource") == "published" and data.get("variant") == "published" and
                data.get("nativeFrequencyScan") is False, "Unexpected published SDK source or variant")
    require(len(data.get("cases", [])) == 1 and data.get("completedCases") == 1,
            "Quality comparison requires exactly one completed case per capture")
    case = data["cases"][0]
    if native:
        if data["variant"] == "native-v2-comparison":
            require(case.get("backend") == expected_backend, "Unexpected comparison candidate backend")
        else:
            require(case.get("backend", "frequency-v1") == expected_backend, "Unexpected legacy candidate backend")
    require(case.get("success") is True and case.get("layers") == 8 and case.get("threads") == 4 and
            case.get("qualityCapture") is True and case.get("allOutputFinite") is True,
            "Require successful eight-layer/four-thread finite quality output")
    require(case.get("modelSha256") == expected_model, "Unexpected captured model identity")
    require(case.get("modelBytes") == (CANDIDATE_MODEL_BYTES[expected_backend] if native else 11_615_514),
            "Unexpected captured model size")
    exported = case["export"]
    require(exported.get("dtype") == "float32-le" and exported.get("sampleRate") == SAMPLE_RATE,
            "Unexpected PCM export format")
    values, bytes_by_kind, hashes = {}, {}, {}
    folder = (path.parent / "pcm").resolve()
    for kind in ("input", "output"):
        record = exported[kind]
        count = record.get("samples")
        require(type(count) is int and count > 0 and type(record.get("bytes")) is int and record["bytes"] == count * 4,
                f"Invalid {kind} sample/byte count")
        expected_name = f"{data['runId']}-{case['caseId']}-{kind}.f32"
        require(re.fullmatch(r"[A-Za-z0-9_-]+\.f32", expected_name) and
                Path(record["file"]).name == expected_name, f"Unexpected {kind} export name")
        local = (folder / expected_name).resolve()
        require(local.parent == folder and local.is_file(), f"Missing local PCM export: {expected_name}")
        raw = local.read_bytes()
        sha = record.get("sha256")
        require(isinstance(sha, str) and re.fullmatch("[0-9a-f]{64}", sha) and
                len(raw) == record["bytes"] and digest(raw) == sha, f"{kind} PCM hash/size mismatch")
        values[kind], bytes_by_kind[kind], hashes[kind] = pcm(raw, kind), raw, sha
        require(case.get(kind + "Samples") == count, f"{kind} capture count disagrees with export")
    require(values["input"].size == values["output"].size and data.get("inputSha256") == hashes["input"],
            "Input/output length or input identity mismatch")
    receipt = {"resultsSha256": digest(raw_json), "runId": data["runId"], "sdkSource": data.get("sdkSource"),
               "variant": data.get("variant"), "modelSha256": expected_model, "pcmSha256": hashes}
    return values, bytes_by_kind, receipt


def db_ratio(signal_energy, error_energy):
    if error_energy == 0:
        return math.inf if signal_energy > 0 else math.nan
    return 10 * math.log10(signal_energy / error_energy) if signal_energy > 0 else -math.inf


def si_sdr(reference, estimate):
    target, prediction = reference - reference.mean(), estimate - estimate.mean()
    energy = float(np.dot(target, target))
    require(energy > 0, "Clean reference must have nonzero energy")
    projection = (float(np.dot(prediction, target)) / energy) * target
    residual = prediction - projection
    return db_ratio(float(np.dot(projection, projection)), float(np.dot(residual, residual)))


def reference_metrics(clean, output):
    error = output - clean
    return {"stoi": float(pystoi.stoi(clean, output, SAMPLE_RATE, extended=False)),
            "siSdrDb": si_sdr(clean, output),
            "snrDb": db_ratio(float(np.dot(clean, clean)), float(np.dot(error, error)))}


def safe_json(value):
    """Retain exact equality/degenerate metric information without nonstandard JSON NaN/Infinity."""
    if isinstance(value, float) and not math.isfinite(value):
        return "positiveInfinity" if value == math.inf else "negativeInfinity" if value == -math.inf else "undefined"
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    return value


def compare(baseline_path, candidate_path, clean_path, candidate_backend="frequency-v1"):
    require(candidate_backend in CANDIDATE_MODELS, "Choose a known, explicitly pinned candidate backend")
    baseline, baseline_raw, baseline_receipt = read_capture(baseline_path, BASELINE_MODEL)
    candidate, candidate_raw, candidate_receipt = read_capture(candidate_path, CANDIDATE_MODELS[candidate_backend], candidate_backend)
    require(baseline_raw["input"] == candidate_raw["input"], "Captures used different input bytes")
    require(baseline_receipt["pcmSha256"]["input"] == candidate_receipt["pcmSha256"]["input"],
            "Captures used different input hashes")
    raw_clean = clean_path.read_bytes()
    clean = pcm(raw_clean, "clean reference")
    require(clean.size == baseline["input"].size == candidate["input"].size, "Clean/input/output sample counts differ")
    require(clean.size >= 6400, "At least 0.4 seconds is required for the STOI diagnostic")
    error = candidate["output"] - baseline["output"]
    energy = float(np.dot(baseline["output"], baseline["output"]))
    error_energy = float(np.dot(error, error))
    require(energy > 0, "Baseline output must have nonzero energy")
    scores = {"input": reference_metrics(clean, baseline["input"]),
              "baseline": reference_metrics(clean, baseline["output"]),
              "candidate": reference_metrics(clean, candidate["output"])}
    return safe_json({
        "schemaVersion": 1,
        "scope": "One synthetic noisy-TTS diagnostic; not broad audio-quality or listening validation",
        "alignment": "Exact sample indices; no resampling, shift search, cropping or external gain normalization",
        "siSdrDefinition": "Zero-mean scale-invariant projection onto the clean reference",
        "sampleRate": SAMPLE_RATE, "samples": int(clean.size), "durationSeconds": clean.size / SAMPLE_RATE,
        "sdkCoordinate": SDK, "layers": 8, "threads": 4, "precision": "FP32", "executionProvider": "CPUExecutionProvider",
        "baseline": baseline_receipt, "candidate": candidate_receipt, "candidateBackend": candidate_backend,
        "cleanReferenceSha256": digest(raw_clean),
        "outputAgreement": {"bitwiseEqual": baseline_raw["output"] == candidate_raw["output"],
                            "maxAbs": float(np.max(np.abs(error))), "rmse": math.sqrt(error_energy / error.size),
                            "relativeRms": math.sqrt(error_energy / energy), "snrDb": db_ratio(energy, error_energy)},
        "againstClean": scores,
        "candidateMinusBaseline": {key: scores["candidate"][key] - scores["baseline"][key]
                                   for key in ("stoi", "siSdrDb", "snrDb")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "pystoi": pystoi.__version__},
        "scriptSha256": digest(Path(__file__).read_bytes()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path, help="Published SDK capture results.json with sibling pcm/")
    parser.add_argument("--candidate", required=True, type=Path, help="Experimental capture results.json with sibling pcm/")
    parser.add_argument("--candidate-backend", choices=sorted(CANDIDATE_MODELS), default="frequency-v1",
                        help="Explicit trusted candidate model/backend; no model pin is accepted from result JSON")
    parser.add_argument("--clean", required=True, type=Path, help="Aligned mono16kHz FP32 little-endian clean reference")
    parser.add_argument("--output", type=Path, help="Optional JSON report; also printed to stdout")
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, args.clean, args.candidate_backend)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
