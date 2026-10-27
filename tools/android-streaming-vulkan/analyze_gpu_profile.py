#!/usr/bin/env python3
"""Validate and summarize a physical eight-layer Vulkan diagnostic capture.

This consumes existing artifacts only; it never launches a phone/GPU test. Timings
are instrumented GPU intervals, not isolated kernel or memory-barrier costs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import zipfile


class ValidationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValidationError(f"Nonfinite JSON value: {x}")))


def stats(values):
    values = [float(v) for v in values]
    require(bool(values) and all(math.isfinite(v) and v >= 0 for v in values), "Invalid timing samples")
    ordered = sorted(values)
    position = .95 * (len(values) - 1)
    lo, hi = math.floor(position), math.ceil(position)
    return {"count": len(values), "meanMs": statistics.mean(values), "medianMs": statistics.median(values),
            "p95Ms": ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo),
            "minMs": ordered[0], "maxMs": ordered[-1], "stdevMs": statistics.pstdev(values)}


def close(actual, expected, name):
    require(math.isfinite(actual) and math.isfinite(expected) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-7),
            f"Inconsistent {name}: {actual} versus independently summed {expected}")


def read_plan(raw):
    require(raw[:8] == b"RVKPLAN1" and len(raw) >= 32, "Invalid binary plan magic/header")
    version, arena, count, inputs, outputs, states = struct.unpack_from("<6I", raw, 8)
    require(version == 1 and (inputs, outputs, states) == (33, 33, 31), "Unexpected plan schema/state inventory")
    start = 32 + 8 * (inputs + outputs) + 12 * states
    require(len(raw) == start + count * 132, "Invalid binary plan record extent")
    return arena, [struct.unpack_from("<33I", raw, start + i * 132) for i in range(count)]


def dispatch(p):
    names = {1: "unary", 2: "binary", 3: "where", 4: "gather", 5: "pad", 6: "scatter", 7: "matmul_tiled", 8: "conv", 9: "layernorm", 10: "frequency", 11: "temporal"}
    name = names[p[0]]
    if p[0] == 7:
        return name, [(p[7] + 31) // 32, (p[6] + 15) // 16, p[9]]
    if p[0] == 8 and p[22] == 1 and p[24] == 0 and p[25] == 0 and p[26] == 0 and p[7] >= 16:
        return "conv_tiled", [(p[8] * p[9] + 31) // 32, (p[7] + 15) // 16, p[6]]
    if p[0] == 9:
        return name, [p[6], 1, 1]
    if p[0] == 10:
        return name, [(p[13] + 3) // 4, 1, 1]
    if p[0] == 11:
        return name, [(p[2] + 3) // 4, 1, 1]
    return name, [(p[2] + 127) // 128, 1, 1]


def verify_manifest(bundle, reported_sha):
    raw = (bundle / "manifest.json").read_bytes()
    require(digest(raw) == reported_sha, "Reported model manifest SHA does not match pinned bundle")
    manifest = json.loads(raw)
    for key, value in {"layers": 8, "precision": "float32", "sampleRate": 16000, "hopSize": 320, "stateCount": 31, "selectedWeightBytesUnchanged": True}.items():
        require(manifest.get(key) == value, f"Manifest {key} mismatch")
    for relative, expected in manifest["files"].items():
        path = (bundle / relative).resolve()
        require(path.is_relative_to(bundle.resolve()), "Manifest path escapes bundle")
        data = path.read_bytes()
        require(len(data) == expected["bytes"] and digest(data) == expected["sha256"], f"Pinned file digest/size mismatch: {relative}")
    return manifest


def verify_submission(path, bundle, native, result):
    submission = load_json(path)
    require(submission["runId"] == result["runId"], "Submission run ID mismatch")
    require(submission["device"] == {"model": "pa3q", "version": "35"}, "Submission was not Galaxy S25 Ultra/API35")
    for artifact in submission["apks"].values():
        raw = Path(artifact["path"]).read_bytes()
        require(len(raw) == artifact["bytes"] and digest(raw) == artifact["sha256"], "Submission APK digest/size mismatch")
    with zipfile.ZipFile(submission["apks"]["app"]["path"]) as apk:
        require(apk.read("assets/vulkan/manifest.json") == (bundle / "manifest.json").read_bytes(), "APK manifest differs from profile model")
        require(apk.read("lib/arm64-v8a/libreuse_streaming_vulkan.so") == native.read_bytes(), "APK native library differs from profiled binary")
    return {"path": str(path.resolve()), "sha256": digest(path.read_bytes()), "matrixIds": [r["matrixId"] for r in submission.get("requests", []) if "matrixId" in r]}


def analyze(result, plan, graph, flattened):
    require(result.get("schemaVersion") == 1 and result.get("success") is True, "Profile did not succeed")
    for key, value in {"layers": 8, "precision": "FP32", "sampleRate": 16000, "hopSamples": 320, "timedFrames": 12, "warmupHops": 4, "profiledOutputBitwiseEqual": True}.items():
        require(result.get(key) == value, f"Unexpected profile field {key}")
    device = result["device"]
    require(device.get("model") == "SM-S938U1" and device.get("sdk") == 35 and device.get("emulatorReported") is False,
            "Expected physical Galaxy S25 Ultra/API35")
    plain_gpu, measured_gpu = result["vulkan"], result["profiledVulkan"]
    require(plain_gpu.get("perDispatchProfiling") is False and measured_gpu.get("perDispatchProfiling") is True, "Invalid normal/profiled execution modes")
    require({k: v for k, v in plain_gpu.items() if k != "perDispatchProfiling"} == {k: v for k, v in measured_gpu.items() if k != "perDispatchProfiling"}, "GPU configurations differ")
    for key, value in {"deviceName": "Adreno (TM) 830", "hardwareGpu": True, "softwareDeviceAllowed": False, "subgroupSize": 64,
                       "timestampValidBits": 48, "operations": 530, "stateCount": 31, "precision": "float32", "tiledMatmulPipeline": True,
                       "tiledConvolutionPipeline": True, "gpuFiniteChecks": True, "statePingPong": True}.items():
        require(plain_gpu.get(key) == value, f"GPU identity/config mismatch: {key}")
    require(len(plan) == len(flattened) == 530, "Expected 530-op plans")
    changed = [i for i, (before, after) in enumerate(zip(plan, flattened)) if before != after]
    require(len(changed) == 40 and changed == graph["changedDescriptorIndices"], "Flattened operation list does not match actual plans")
    for i in changed:
        before, after = plan[i], flattened[i]
        require(before[0] == 7 and before[6] == 1 and before[9] == 161 and before[10] == before[6] * before[8] and before[11] == before[12] == 0, "Invalid flattenable GEMM")
        require(after[6] == before[6] * before[9] and after[9] == 1 and after[10] == 0, "Flattened GEMM shape mismatch")
        require(all(before[j] == after[j] for j in range(33) if j not in (6, 9, 10)), "Flattening changes arithmetic/addresses beyond safe batch merge")
    require(graph["baseline"]["dispatches"] == graph["flattened"]["dispatches"] == len(plan), "Graph dispatch count mismatch")
    expected_counts = Counter(str(p[0]) for p in plan)
    require(dict(expected_counts) == graph["baseline"]["opCounts"] == graph["flattened"]["opCounts"], "Static opcode counts mismatch")
    for key, descriptors in [("baseline", plan), ("flattened", flattened)]:
        groups = sum(math.prod(dispatch(p)[1]) for p in descriptors if p[0] == 7)
        require(groups == graph[key]["matmulWorkgroups"]["tiled"], "Static GEMM workgroup count mismatch")
    profiles = result["profiles"]
    require(len(profiles) == len(result["unprofiledPushMs"]) == len(result["profiledPushMs"]) == 12, "Expected 12 complete timing records")
    ordinals = [p["frameOrdinal"] for p in profiles]
    require(ordinals == list(range(ordinals[0], ordinals[0] + 12)), "Frame ordinals are not consecutive")
    frame_categories = defaultdict(list)
    per_op = [[] for _ in plan]
    fields = ["wholeGpuMs", "inputAndStatusInitMs", "inputDependencyMs", "computeGraphMs", "summedDispatchMs", "unattributedGraphMs", "finiteValidationMs", "outputReadbackMs", "completionMs"]
    for frame in profiles:
        require(frame.get("schemaVersion") == 1 and frame.get("profiled") is True and frame.get("queryCount") == 1067 and frame.get("operationCount") == 530,
                "Profile frame schema/query/operation count mismatch")
        require(frame["timestampValidBits"] == 48 and frame["timestampPeriodNs"] == profiles[0]["timestampPeriodNs"] and frame["timestampPeriodNs"] > 0, "Timestamp clock changed")
        require(all(isinstance(frame[k], (float, int)) and math.isfinite(frame[k]) and frame[k] >= 0 for k in fields), "Invalid interval duration")
        require(len(frame["operations"]) == 530, "Truncated operation capture")
        sums, counts = defaultdict(float), Counter()
        for i, (op, p) in enumerate(zip(frame["operations"], plan)):
            kernel, groups = dispatch(p)
            require(op["index"] == i and op["opcode"] == p[0] and op["outputElements"] == p[2] and op["kernel"] == kernel and op["workgroups"] == groups,
                    f"Operation {i} identity does not match independently parsed plan")
            duration = op["gpuMs"]
            require(isinstance(duration, (float, int)) and math.isfinite(duration) and duration >= 0, "Invalid operation timing")
            per_op[i].append(duration); sums[kernel] += duration; counts[kernel] += 1
        require(len(frame["categories"]) == len(sums), "Category inventory mismatch")
        seen = set()
        for category in frame["categories"]:
            kernel = category["kernel"]
            require(kernel in sums and kernel not in seen and category["dispatches"] == counts[kernel], "Category identity/count mismatch")
            seen.add(kernel); close(category["gpuMs"], sums[kernel], f"{kernel} category")
            frame_categories[kernel].append(sums[kernel])
        close(frame["summedDispatchMs"], sum(sums.values()), "total dispatch interval")
        close(frame["computeGraphMs"], frame["summedDispatchMs"] + frame["unattributedGraphMs"], "graph interval")
        close(frame["wholeGpuMs"], sum(frame[k] for k in fields if k not in ("wholeGpuMs", "summedDispatchMs", "unattributedGraphMs")), "whole GPU interval")
    whole_mean = statistics.mean(p["wholeGpuMs"] for p in profiles)
    categories = [{"kernel": kernel, "dispatchesPerFrame": sum(dispatch(p)[0] == kernel for p in plan), **stats(times),
                   "percentWholeGpu": statistics.mean(times) / whole_mean * 100} for kernel, times in frame_categories.items()]
    categories.sort(key=lambda c: c["meanMs"], reverse=True)
    operations = []
    for i, times in enumerate(per_op):
        p = plan[i]
        op = {"index": i, "opcode": p[0], "kernel": dispatch(p)[0], "outputElements": p[2], "workgroups": dispatch(p)[1], **stats(times)}
        if p[0] == 7:
            op["matrix"] = {"m": p[6], "n": p[7], "k": p[8], "batch": p[9]}
        operations.append(op)
    flatten_samples = [sum(per_op[i][f] for i in changed) for f in range(12)]
    flatten_stats = stats(flatten_samples)
    flatten_stats.update({"descriptorIndices": changed, "operationCount": len(changed), "percentWholeGpu": flatten_stats["meanMs"] / whole_mean * 100,
                         "percentAllMatmul": flatten_stats["meanMs"] / statistics.mean(frame_categories["matmul_tiled"]) * 100,
                         "operations": [operations[i] for i in changed], "staticTiledWorkgroupsBefore": graph["baseline"]["matmulWorkgroups"]["tiled"],
                         "staticTiledWorkgroupsAfter": graph["flattened"]["matmulWorkgroups"]["tiled"],
                         "note": "Measured baseline cost of affected operations; not a measured or promised speedup for the candidate."})
    plain, instrumented = stats(result["unprofiledPushMs"]), stats(result["profiledPushMs"])
    return {"validation": {"passed": True, "frames": 12, "operationsPerFrame": 530, "queriesPerFrame": 1067,
                            "operationIdentityCheckedAgainstBinaryPlan": True, "categoryAndIntervalSumsChecked": True,
                            "profiledOutputBitwiseEqual": True, "flattenedDescriptorSafetyChecked": True},
            "device": device, "vulkan": measured_gpu, "publicPush": {"unprofiled": plain, "profiled": instrumented,
            "profiledMinusUnprofiledMeanMs": instrumented["meanMs"] - plain["meanMs"],
            "note": "Sequential short samples; negative delta reflects run/clock variability and does not establish zero instrumentation overhead."},
            "profiledGpuIntervals": {field: stats(p[field] for p in profiles) for field in fields},
            "unprofiledWholeGpu": None,
            "unprofiledWholeGpuNote": "The Android capture contains unprofiled public push duration but no unprofiled whole-GPU query series. These are different timing scopes.",
            "categories": categories, "topOperations": sorted(operations, key=lambda op: op["meanMs"], reverse=True)[:25],
            "flattenableGemmContribution": flatten_stats, "firstScatter": operations[0],
            "limitations": ["Per-dispatch GPU timestamps perturb scheduling and include implementation-dependent waits. These are not isolated kernel costs.",
                "Unattributed graph time includes dependencies and timestamp overhead; it must not be labeled isolated barrier cost.",
                "The 12 profiled hops are a bottleneck diagnostic, not the final steady-state latency benchmark.",
                "Raw hardware ticks were not exported; timestamp arithmetic is validated via reported interval sums, not independently reconstructed ticks.",
                "The separate Android emulator smoke capture (profile-emulator-artifacts/vulkan-v2-profile-smoke-result.json, first profile) attributed 14.221875 ms to a 321-element first scatter within 22.08325 ms whole GPU time. This implausible emulator per-op attribution is not used for physical-phone conclusions; the native host CLI capture is a different observation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("result", "bundle", "native", "graph-analysis", "flattened-plan", "submission-plan", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--host-profile", type=Path)
    parser.add_argument("--emulator-profile", type=Path)
    args = parser.parse_args()
    result_bytes = args.result.read_bytes(); result = load_json(args.result)
    sidecar = args.result.with_suffix(".sha256")
    require(sidecar.read_text().strip() == digest(result_bytes), "Profile result SHA sidecar mismatch")
    require(result["runId"] == args.run_id, "Unexpected result run ID")
    manifest = verify_manifest(args.bundle, result["bundleManifestSha256"])
    require(digest(args.native.read_bytes()) == result["nativeLibrarySha256"], "Profile native SHA mismatch")
    submission = verify_submission(args.submission_plan, args.bundle, args.native, result)
    arena, plan = read_plan((args.bundle / "plan.bin").read_bytes())
    flat_arena, flattened = read_plan(args.flattened_plan.read_bytes())
    require(arena == flat_arena and arena * 4 == result["vulkan"]["arenaBytes"], "Arena size mismatch")
    report = analyze(result, plan, load_json(args.graph_analysis), flattened)
    report.update({"schemaVersion": 1, "createdUtc": datetime.now(timezone.utc).isoformat(), "runId": result["runId"],
                   "provenance": {"analyzerSha256": digest(Path(__file__).read_bytes()), "result": {"path": str(args.result.resolve()), "sha256": digest(result_bytes)},
                     "bundleManifestSha256": result["bundleManifestSha256"], "nativeLibrarySha256": result["nativeLibrarySha256"],
                     "planSha256": manifest["files"]["plan.bin"]["sha256"], "weightsSha256": manifest["files"]["weights.bin"]["sha256"],
                     "flattenedPlanSha256": digest(args.flattened_plan.read_bytes()), "graphAnalysisSha256": digest(args.graph_analysis.read_bytes()),
                     "submission": submission}})
    if args.host_profile:
        host = load_json(args.host_profile); first = host["profile"]["operations"][0]; graph_time = host["profile"]["computeGraphMs"]
        report["hostTimestampObservation"] = {"path": str(args.host_profile.resolve()), "sha256": digest(args.host_profile.read_bytes()),
            "device": host["device"]["deviceName"], "firstScatter": first, "computeGraphMs": graph_time,
            "firstScatterPercentGraph": first["gpuMs"] / graph_time * 100,
            "firstScatterDominatesGraph": first["gpuMs"] > 1 and first["gpuMs"] / graph_time > .5,
            "note": "This native host CLI capture is separate from the Android emulator smoke capture; no host timing is used as phone kernel cost.",
            "notUsedAsPhoneKernelCost": True}
    if args.emulator_profile:
        emulator_raw = args.emulator_profile.read_bytes()
        require(args.emulator_profile.with_suffix(".sha256").read_text().strip() == digest(emulator_raw), "Emulator result SHA sidecar mismatch")
        emulator = load_json(args.emulator_profile)
        require(emulator["device"]["emulatorReported"] is True, "Expected separate Android emulator capture")
        observation = emulator["profiles"][0]
        report["emulatorTimestampLimitationEvidence"] = {
            "path": str(args.emulator_profile.resolve()), "sha256": digest(emulator_raw), "shaSidecarVerified": True,
            "device": emulator["device"], "profileIndex": 0, "firstScatter": observation["operations"][0],
            "wholeGpuMs": observation["wholeGpuMs"], "notUsedAsPhoneKernelCost": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    args.output.with_suffix(".sha256").write_text(digest(args.output.read_bytes()) + "\n")
    print(json.dumps({"report": str(args.output.resolve()), "validated": True, "flattenableGemmMs": report["flattenableGemmContribution"]["meanMs"],
                      "wholeGpuMs": report["profiledGpuIntervals"]["wholeGpuMs"]["meanMs"]}))


if __name__ == "__main__":
    main()
