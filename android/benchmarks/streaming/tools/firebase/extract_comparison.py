#!/usr/bin/env python3
"""Verify downloaded multi-case phone benchmark artifacts; no network or cloud calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


DEFAULT_CASES = "frequency-v1@4+stable-v2@4+stable-v2@8+stable-v2@6+frequency-v1@4"
DEFAULT_INPUT_SHA = "e3150c5d89ff157b323627de4e5ede75a35be2e44088b284daa9d47268ccb9ca"
MODEL_HASHES = {
    "frequency-v1": "6b6cbd4117146f3137c1a0db71ecf04c23b8ba97bab7d973ece71c8739aa7756",
    "bulk-v2": "1fbd91be9c90568c58ab40e9ce70e07c54dafc599fffb694a84c1f4d3905b11d",
    "inline-v2": "55d910c30cd7144187e5f358dca6700ae6e7e1051eb8f7758d454f68e9144c11",
    "stable-v2": "db29c725c70608920e600787fcda625e98429cfbae2c2d09f1dc73726825319a",
}
CLASS = "io.riftco.reusestreaming.benchmark.NativeComparisonBenchmarkTest"
SAFE_RUN = re.compile(r"[A-Za-z0-9_.-]{1,80}")
SAFE_CASE = re.compile(r"[A-Za-z0-9_.-]{1,100}")
CASE = re.compile(r"(frequency-v1|bulk-v2|inline-v2|stable-v2)@([1-8])")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def load_json(raw):
    def invalid(value):
        raise ValueError(f"Nonstandard JSON numeric constant: {value}")
    return json.loads(raw, parse_constant=invalid)


def number(value, name, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0),
            f"Invalid finite {name}")
    return value


def equal_number(claimed, actual, name):
    number(claimed, name)
    require(math.isclose(claimed, actual, rel_tol=1e-9, abs_tol=1e-6), f"Recomputed {name} disagrees: claimed {claimed}, actual {actual}")


def plan_entries(selection):
    require(isinstance(selection, str) and 1 <= len(selection) <= 300, "Invalid expected case selection")
    entries = selection.split("+")
    require(1 <= len(entries) <= 8, "Expected 1–8 ordered cases")
    result = []
    for entry in entries:
        match = CASE.fullmatch(entry)
        require(match is not None, "Invalid expected case entry")
        result.append((match[1], int(match[2])))
    return result


def verify_sidecar(path, raw):
    sidecar = path.with_suffix(".sha256")
    text = sidecar.read_text(encoding="utf-8")
    match = re.fullmatch(r"([a-f0-9]{64})[ \t]+([^\r\n]+)\n?", text)
    require(match is not None and match[2] == path.name and match[1] == sha256(raw), f"SHA256 sidecar integrity failure: {path.name}")
    return sidecar.read_bytes()


def timing_stats(values, count, name):
    require(isinstance(values, list) and len(values) == count, f"Incomplete {name} samples")
    for value in values:
        number(value, name)
    require(sum(values) > 0, f"No positive {name} work duration")
    ordered = sorted(values)
    def percentile(fraction):
        return ordered[math.ceil(fraction * len(ordered)) - 1]
    return {"mean": sum(values) / len(values), "min": ordered[0], "p50": percentile(.5),
            "p95": percentile(.95), "p99": percentile(.99), "max": ordered[-1]}


def verify_bundle(artifacts_root, run_id, selection=DEFAULT_CASES, frames=300,
                  expected_input_sha=DEFAULT_INPUT_SHA, allow_emulator=False):
    require(isinstance(run_id, str) and SAFE_RUN.fullmatch(run_id) is not None, "Unsafe expected run ID")
    require(type(frames) is int and 32 <= frames <= 1500, "Invalid expected frame count")
    require(re.fullmatch(r"[a-f0-9]{64}", expected_input_sha) is not None, "Invalid expected input SHA256")
    expected_plan = plan_entries(selection)
    receipts = list(Path(artifacts_root).rglob(f"{run_id}-receipt.json"))
    require(len(receipts) == 1, "Need exactly one matching comparison receipt")
    receipt_path = receipts[0]
    receipt_raw = receipt_path.read_bytes()
    receipt_sidecar = verify_sidecar(receipt_path, receipt_raw)
    receipt = load_json(receipt_raw)
    require(receipt.get("schemaVersion") == 1 and receipt.get("runId") == run_id and receipt.get("success") is True,
            "Comparison receipt failed or has unexpected identity")
    require(receipt.get("instrumentationClass") == CLASS, "Unexpected instrumentation class")
    requested = receipt["requested"]
    require(requested.get("comparisonCases") == selection and requested.get("timedFrames") == frames,
            "Receipt requested plan/frame count mismatch")
    require(requested.get("captureOutput") is False, "Performance comparison unexpectedly captures audio")
    require(requested.get("expectedInputSha256") == expected_input_sha, "Receipt expected input mismatch")
    if not allow_emulator:
        require(requested.get("requirePhysical") is True, "Receipt did not require a physical device")
    require(receipt.get("exports") == [], "Unexpected or unverified quality exports in performance comparison")
    record = receipt["result"]
    result_path = receipt_path.parent / f"{run_id}-result.json"
    require(Path(record["file"]).name == result_path.name, "Unexpected raw result artifact name")
    raw = result_path.read_bytes()
    require(len(raw) == record["bytes"] and sha256(raw) == record["sha256"], "Receipt/raw result integrity failure")
    result_sidecar = verify_sidecar(result_path, raw)
    result = load_json(raw)
    require(result.get("schemaVersion") == 1 and result.get("runId") == run_id and result.get("success") is True and
            result.get("status") == "complete", "Raw comparison did not complete successfully")
    expected_header = {"variant": "native-v2-comparison", "sdkSource": "local", "precision": "FP32",
                       "executionProvider": "CPUExecutionProvider", "onnxRuntimeVersion": "1.23.2",
                       "sdkCoordinate": "io.github.quangng2000:reuse-streaming-android:0.1.2",
                       "sampleRate": 16000, "hopSamples": 320, "windowSamples": 640,
                       "timedFramesPerCase": frames, "warmupHopsPerCase": 32, "completedCases": len(expected_plan)}
    for key, value in expected_header.items():
        require(result.get(key) == value, f"Unexpected result {key}")
    require(result.get("inputSha256", result.get("syntheticInputSha256")) == expected_input_sha, "Raw input SHA256 mismatch")
    require(result.get("qualityCapture") is False, "Raw performance result unexpectedly captures audio")
    device = result["device"]
    require(device.get("targetDebuggable") is False and "arm64-v8a" in device.get("abis", []), "Nondebuggable ARM64 target required")
    require(type(device.get("emulatorReported")) is bool, "Missing emulator identity")
    if not allow_emulator:
        require(device["emulatorReported"] is False, "Physical target required; emulator reported")
    require(isinstance(device.get("fingerprint"), str) and device["fingerprint"], "Missing device build fingerprint")
    cases, observed_plan = result["cases"], result["plan"]
    require(len(cases) == len(observed_plan) == len(expected_plan), "Incomplete comparison case count")
    ids, summaries = set(), []
    expected_samples = 640 + (32 + frames) * 320
    for index, ((backend, threads), declared, case) in enumerate(zip(expected_plan, observed_plan, cases)):
        for row in (declared, case):
            require(row.get("layers") == 8 and row.get("threads") == threads and row.get("backend") == backend,
                    f"Case {index + 1} order/backend/thread identity mismatch")
        case_id = case.get("caseId")
        require(isinstance(case_id, str) and SAFE_CASE.fullmatch(case_id) is not None and case_id not in ids,
                "Unsafe or duplicate comparison case ID")
        ids.add(case_id)
        require(case.get("success") is True and case.get("timedFrames") == frames and case.get("allOutputFinite") is True,
                f"Case {case_id} incomplete or nonfinite")
        require(case.get("modelSha256") == MODEL_HASHES[backend], f"Case {case_id} model pin mismatch")
        require(case.get("inputSamples") == case.get("outputSamples") == expected_samples, f"Case {case_id} output sample count mismatch")
        require(case.get("qualityCapture") is False, "Unexpected case output capture")
        fixture = case["fixture"]
        require(fixture.get("passed") is True and fixture.get("resetBitwiseRepeatPassed") is True and fixture.get("samples") == 16000,
                f"Case {case_id} reference/reset fixture failed")
        error = number(fixture["maximumAbsoluteError"], "fixture maximum absolute error")
        require(error <= .0003, "Fixture waveform error exceeds approved tolerance")
        if "absoluteTolerance" in fixture:
            equal_number(fixture["absoluteTolerance"], .0003, "fixture tolerance")
        number(fixture["rmsError"], "fixture RMS error")
        number(abs(case["timedOutputChecksum"]), "output checksum magnitude")
        push_stats = timing_stats(case["rawPushMs"], frames, "push")
        render_stats = timing_stats(case["rawRenderMs"], frames, "render")
        for claimed, actual, label in [(case["latencyMs"], push_stats, "push"), (case["renderLatencyMs"], render_stats, "render")]:
            for key, value in actual.items():
                equal_number(claimed[key], value, f"{case_id} {label} {key}")
        rtf = sum(case["rawPushMs"]) / (frames * 20.0)
        over20 = sum(value > 20.0 for value in case["rawPushMs"])
        equal_number(case["rtf"], rtf, f"{case_id} RTF")
        require(type(case.get("over20msCount")) is int and case["over20msCount"] == over20, f"Case {case_id} deadline miss count mismatch")
        equal_number(case["over20msPercent"], over20 * 100.0 / frames, f"{case_id} deadline miss percent")
        equal_number(case["timedAudioSeconds"], frames * .02, f"{case_id} timed audio duration")
        summaries.append({"order": index + 1, "caseId": case_id, "backend": backend, "threads": threads, "layers": 8,
                          "modelSha256": case["modelSha256"], "timedFrames": frames, "latencyMs": push_stats,
                          "renderLatencyMs": render_stats, "rtf": rtf, "over20msCount": over20,
                          "over20msPercent": over20 * 100.0 / frames,
                          "strictlyUnder20msCount": sum(value < 20.0 for value in case["rawPushMs"]),
                          "inputSamples": expected_samples, "outputSamples": case["outputSamples"],
                          "fixtureMaximumAbsoluteError": error, "resetBitwiseRepeatPassed": True})
    summary = {"schemaVersion": 1, "runId": run_id, "success": True, "receiptSha256": sha256(receipt_raw),
               "resultSha256": sha256(raw), "sourceReceipt": str(receipt_path.resolve()), "device": device,
               "comparisonCases": selection, "expectedInputSha256": expected_input_sha,
               "timedFramesPerCase": frames, "precision": "FP32", "executionProvider": "CPUExecutionProvider",
               "onnxRuntimeVersion": "1.23.2", "cases": summaries,
               "scope": "Ordered individual in-app SDK push cases on one execution; all controls/thread variants remain separate; no clock/thermal control or audio-device/network latency claim"}
    return summary, {"results.json": raw, "receipt.json": receipt_raw,
                     f"{run_id}-result.json": raw, f"{run_id}-receipt.json": receipt_raw,
                     f"{run_id}-result.sha256": result_sidecar, f"{run_id}-receipt.sha256": receipt_sidecar}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-cases", default=DEFAULT_CASES)
    parser.add_argument("--expected-frames", type=int, default=300)
    parser.add_argument("--expected-input-sha", default=DEFAULT_INPUT_SHA)
    parser.add_argument("--allow-emulator", action="store_true", help="Local runner smoke validation only; never a phone speed result")
    parser.add_argument("--output", type=Path, required=True, help="New local verification directory")
    args = parser.parse_args()
    summary, files = verify_bundle(args.artifacts_root, args.expected_run_id, args.expected_cases,
                                   args.expected_frames, args.expected_input_sha, args.allow_emulator)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        (args.output / name).write_bytes(raw)
    (args.output / "verified-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
