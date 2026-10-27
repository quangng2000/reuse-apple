#!/usr/bin/env python3
"""Verify already-downloaded Test Lab artifacts or logcat; never contacts Firebase."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

SAFE = re.compile(r"[A-Za-z0-9_.-]{1,80}")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def inspect_result(raw: bytes, run_id: str | None = None) -> tuple[dict, dict]:
    result = json.loads(raw)
    identity = result.get("runId")
    if not isinstance(identity, str) or not SAFE.fullmatch(identity) or (run_id is not None and identity != run_id):
        raise ValueError("Unexpected result runId")
    if result.get("success") is not True or result.get("status") != "complete":
        raise ValueError(f"Benchmark app did not complete successfully: {result.get('error')}")
    if result.get("precision") != "FP32" or result.get("executionProvider") != "CPUExecutionProvider":
        raise ValueError("Unexpected precision or execution provider")
    if result.get("device", {}).get("targetDebuggable") is not False:
        raise ValueError("Benchmark target is not proven nondebuggable")
    cases = result.get("cases", [])
    if len(cases) != 1:
        raise ValueError("Expected exactly one case per Firebase execution")
    case = cases[0]
    times = case.get("rawPushMs", [])
    if case.get("success") is not True or case.get("layers") != 8 or len(times) != case.get("timedFrames") or not times:
        raise ValueError("Incomplete eight-layer timing data")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 for value in times):
        raise ValueError("Invalid individual hop timing")
    if case.get("allOutputFinite") is not True or case.get("inputSamples") != case.get("outputSamples"):
        raise ValueError("Output invariants failed")
    if not case.get("fixture", {}).get("passed") or not case.get("fixture", {}).get("resetBitwiseRepeatPassed"):
        raise ValueError("Fixture validation did not pass")
    summary = {
        "runId": identity, "resultSha256": sha256(raw),
        "variant": result.get("variant"), "sdkSource": result.get("sdkSource"),
        "device": result.get("device"), "threads": case.get("threads"),
        "modelSha256": case.get("modelSha256"),
        "inputSha256": result.get("inputSha256", result.get("syntheticInputSha256")),
        "timedFrames": case.get("timedFrames"), "qualityCapture": case.get("qualityCapture", False),
        "latencyMs": case.get("latencyMs"), "rtf": case.get("rtf"),
        "over20msCount": case.get("over20msCount"),
        "scope": "Individual in-app FP32 CPU pushes; not remote-stream, call, microphone or end-to-end audio latency",
    }
    return result, summary


def extract_artifacts(root: Path, destination: Path) -> list[dict]:
    receipts = sorted(root.rglob("*-receipt.json"))
    if not receipts:
        raise ValueError("No Firebase benchmark receipts found; use --logcat if scoped storage prevented pulling")
    summaries = []
    for receipt_path in receipts:
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw)
        run_id = receipt.get("runId")
        if not isinstance(run_id, str) or not SAFE.fullmatch(run_id):
            raise ValueError("Unsafe receipt runId")
        if receipt.get("success") is not True:
            raise ValueError(f"Instrumentation receipt reports failure: {receipt_path}: {receipt.get('error')}")
        record = receipt["result"]
        result_path = receipt_path.parent / f"{run_id}-result.json"
        if Path(record["file"]).name != result_path.name:
            raise ValueError("Unexpected result artifact filename")
        raw = result_path.read_bytes()
        if len(raw) != record["bytes"] or sha256(raw) != record["sha256"]:
            raise ValueError(f"Result integrity failure: {result_path}")
        result, summary = inspect_result(raw, run_id)
        if receipt["requested"].get("requirePhysical", True) and result.get("device", {}).get("emulatorReported") is not False:
            raise ValueError("Physical target was requested but not proven")
        folder = destination / f"{run_id}-{sha256(receipt_raw)[:8]}"
        folder.mkdir()
        (folder / "results.json").write_bytes(raw)
        (folder / "receipt.json").write_bytes(receipt_raw)
        for case in result["cases"]:
            if not case.get("qualityCapture", False):
                continue
            for kind in ("input", "output"):
                pcm = case["export"][kind]
                name = f"{run_id}-{case['caseId']}-{kind}.f32"
                if Path(pcm["file"]).name != name or not SAFE.fullmatch(case["caseId"]):
                    raise ValueError("Unexpected PCM filename")
                source = receipt_path.parent / name
                pcm_raw = source.read_bytes()
                if len(pcm_raw) != pcm["bytes"] or len(pcm_raw) != pcm["samples"] * 4 or sha256(pcm_raw) != pcm["sha256"]:
                    raise ValueError(f"PCM integrity failure: {source}")
                shutil.copyfile(source, folder / name)
        summary["sourceReceipt"] = str(receipt_path.resolve())
        summary["verifiedDirectory"] = str(folder.resolve())
        summaries.append(summary)
    return summaries


def extract_logcat(path: Path, destination: Path) -> list[dict]:
    summaries = []
    active = None
    chunks: dict[int, str] = {}
    total = None
    seen = set()
    for line in path.read_text(errors="strict").splitlines():
        start = re.search(r"RUN_START ([A-Za-z0-9_.-]{1,80})\s*$", line)
        if start:
            active, chunks, total = start[1], {}, None
            continue
        if active is None:
            continue
        chunk = re.search(r"RESULT_CHUNK (\d+)/(\d+):([A-Za-z0-9+/=]+)\s*$", line)
        if chunk:
            index, count = int(chunk[1]), int(chunk[2])
            if total not in (None, count) or not 1 <= index <= count or index in chunks:
                raise ValueError("Invalid or duplicate result chunk")
            chunks[index], total = chunk[3], count
        checksum = re.search(r"RESULT_SHA256:([a-f0-9]{64})\s*$", line)
        if checksum:
            if total is None or set(chunks) != set(range(1, total + 1)):
                raise ValueError("Truncated result chunks")
            raw = base64.b64decode("".join(chunks[index] for index in range(1, total + 1)), validate=True)
            if sha256(raw) != checksum[1]:
                raise ValueError("Logcat result SHA256 mismatch")
            _, summary = inspect_result(raw, active)
            if active in seen:
                raise ValueError("Duplicate complete runId in logcat")
            seen.add(active)
            folder = destination / active
            folder.mkdir()
            (folder / "results.json").write_bytes(raw)
            summary.update(sourceLogcat=str(path.resolve()), verifiedDirectory=str(folder.resolve()),
                           pcmExportsVerified=False, artifactReceiptVerified=False)
            summaries.append(summary)
            active = None
    if active is not None or not summaries:
        raise ValueError("Missing complete, SHA256-verified benchmark result in logcat")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifacts-root", type=Path, help="Already-downloaded Test Lab results tree")
    source.add_argument("--logcat", type=Path, help="Already-downloaded test logcat, fallback for raw JSON only")
    parser.add_argument("--output", type=Path, required=True, help="New local verification directory")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = extract_artifacts(args.artifacts_root, args.output) if args.artifacts_root else extract_logcat(args.logcat, args.output)
    manifest = {"schemaVersion": 1, "verifiedRuns": summaries}
    (args.output / "verified-summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
