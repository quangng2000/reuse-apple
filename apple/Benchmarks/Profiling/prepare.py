#!/usr/bin/env python3
"""Prepare an ignored, instrumented snapshot without changing shipping sources."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REVISION = "2446c83"
# Git-object locator at REVISION, intentionally retaining that revision's old layout.
SOURCE = "Sources/ReuseStreaming/"
DESTINATION = HERE / "Generated"


def git(*arguments):
    return subprocess.check_output(["git", *arguments], cwd=ROOT)


def patch(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError(f"Pinned profiling injection no longer matches: {before[:80]!r}")
    return text.replace(before, after, 1)


def instrument(relative, text):
    if relative == "Model/StreamingReuseModel.swift":
        return patch(text, "final class StreamingReuseModel {", "final class StreamingReuseModel {\n    var benchmarkProfiler: BenchmarkStageProfiler?")
    if relative == "Model/StreamingReuseInference.swift":
        text = patch(text, "        var x = encode(magnitude, phase)", """        let encodeStarted = benchmarkProfiler.map { _ in benchmarkNow() }
        var x = encode(magnitude, phase)
        if let profiler = benchmarkProfiler, let started = encodeStarted {
            profiler.complete("encode", started: started, arrays: [x] + frameStateArrays)
        }""")
        text = patch(text, '            x = timeFrequency(x, "TSMamba.\\(index)")', """            let blockStarted = benchmarkProfiler.map { _ in benchmarkNow() }
            x = timeFrequency(x, "TSMamba.\\(index)")
            if let profiler = benchmarkProfiler, let started = blockStarted {
                profiler.complete("tf\\(index)", started: started, arrays: [x] + frameStateArrays)
            }""")
        return patch(text, "        let (mag, pha) = decode(x, frequencies: magnitude.dim(0))", """        let decodeStarted = benchmarkProfiler.map { _ in benchmarkNow() }
        let (mag, pha) = decode(x, frequencies: magnitude.dim(0))
        if let profiler = benchmarkProfiler, let started = decodeStarted {
            profiler.complete("decode", started: started, arrays: [mag, pha] + frameStateArrays)
        }""")
    if relative == "Pipeline/StreamingTransform.swift":
        text = patch(text, "final class StreamingTransform {", "final class StreamingTransform {\n    var benchmarkProfiler: BenchmarkStageProfiler?")
        text = patch(text, "    func analyze(_ samples: [Float]) -> (MLXArray, MLXArray) {", """    func analyze(_ samples: [Float]) -> (MLXArray, MLXArray) {
        let started = benchmarkProfiler.map { _ in benchmarkNow() }""")
        text = patch(text, """        return (log1p(abs(spectrum)).expandedDimensions(axis: 1),
                atan2(imaginary, spectrum.realPart()).expandedDimensions(axis: 1))""", """        let result = (log1p(abs(spectrum)).expandedDimensions(axis: 1),
                      atan2(imaginary, spectrum.realPart()).expandedDimensions(axis: 1))
        if let profiler = benchmarkProfiler, let started {
            profiler.complete("analyze", started: started, arrays: [result.0, result.1])
        }
        return result""")
        text = patch(text, "                    evaluating stateArrays: [MLXArray] = []) -> [Float] {", """                    evaluating stateArrays: [MLXArray] = []) -> [Float] {
        let started = benchmarkProfiler.map { _ in benchmarkNow() }""")
        text = patch(text, """        if !stateArrays.isEmpty { eval([samples] + stateArrays) }
        return overlapAdd(samples.asArray(Float.self))""", """        if let profiler = benchmarkProfiler, let started {
            let index = profiler.complete("synthesis", started: started, arrays: [samples] + stateArrays)
            let cpuStarted = benchmarkNow()
            let output = overlapAdd(samples.asArray(Float.self))
            profiler.finishCPU(index: index, started: cpuStarted)
            return output
        }
        if !stateArrays.isEmpty { eval([samples] + stateArrays) }
        return overlapAdd(samples.asArray(Float.self))""")
        return patch(text, "    private func overlapAdd(_ samples: [Float]) -> [Float] {", """    func benchmarkStateDigests() -> [BenchmarkStateDigest] {
        [benchmarkDigest("transform/overlap", shape: [overlap.count], values: overlap),
         benchmarkDigest("transform/envelope", shape: [envelope.count], values: envelope),
         benchmarkDigest("transform/first", shape: [1], values: [first ? Float(1) : Float(0)])]
    }

    private func overlapAdd(_ samples: [Float]) -> [Float] {""")
    if relative == "StreamingEnhancer.swift":
        text = patch(text, "public final class StreamingEnhancer {", """public final class StreamingEnhancer {
    public var benchmarkProfilingEnabled = false
    public private(set) var benchmarkFrameProfiles = [[BenchmarkStageTiming]]()

    public func benchmarkStateDigests() -> [BenchmarkStateDigest] {
        Device.withDefaultDevice(.gpu) {
            benchmarkModelStateDigests(model) + transform.benchmarkStateDigests()
        }
    }""")
        text = text.replace("        lastFrameMilliseconds.removeAll(keepingCapacity: true)", "        lastFrameMilliseconds.removeAll(keepingCapacity: true)\n        benchmarkFrameProfiles.removeAll(keepingCapacity: true)")
        return patch(text, "    private func renderFrame(_ input: [Float]) -> [Float] {", """    private func renderFrame(_ input: [Float]) -> [Float] {
        let profiler = benchmarkProfilingEnabled ? BenchmarkStageProfiler() : nil
        model.benchmarkProfiler = profiler
        transform.benchmarkProfiler = profiler
        defer {
            if let profiler { benchmarkFrameProfiles.append(profiler.timings) }
            model.benchmarkProfiler = nil
            transform.benchmarkProfiler = nil
        }""")
    return text


def main():
    revision = git("rev-parse", REVISION).decode().strip()
    paths = git("ls-tree", "-r", "--name-only", revision, SOURCE).decode().splitlines()
    if not paths or any(not path.startswith(SOURCE) for path in paths):
        raise RuntimeError("Missing pinned streaming sources")
    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    source_files, generated_files = [], []
    for path in paths:
        contents = git("show", f"{revision}:{path}")
        relative = path.removeprefix(SOURCE)
        target = DESTINATION / "Sources/ReuseStreaming" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        modified = instrument(relative, contents.decode()).encode() if path.endswith(".swift") else contents
        target.write_bytes(modified)
        source_files.append({"path": path, "sha256": hashlib.sha256(contents).hexdigest()})
        generated_files.append({"path": relative, "sha256": hashlib.sha256(modified).hexdigest()})
    canonical = json.dumps(source_files, sort_keys=True, separators=(",", ":")).encode()
    manifest = {
        "schema_version": 1, "source_revision": revision,
        "source_file_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_file_manifest_hash_format": "SHA256 of UTF8 JSON file list, sorted_keys=True, separators=(',',':')",
        "source_files": source_files, "generated_files": generated_files,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "support_sha256": hashlib.sha256((HERE / "Support/StageProfiler.swift").read_bytes()).hexdigest(),
        "warning": "Diagnostic forced completion boundaries distort normal fusion, dispatch and overlap. Evaluation wait is not isolated GPU execution time.",
    }
    (DESTINATION / "profile-source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(paths)} pinned source files from {revision}; shipping sources untouched.")


if __name__ == "__main__":
    main()
