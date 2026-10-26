import XCTest
import Foundation
import CryptoKit
@testable import ReuseStreaming
import ReuseStreamingReference

private protocol CompiledBenchmarkEngine: AnyObject {
    var lastFrameMilliseconds: [Double] { get }
    func push(_ samples: [Float]) throws -> [Float]
    func finish() throws -> [Float]
    func reset()
    func warmUp() throws
}
extension ReuseStreaming.StreamingEnhancer: CompiledBenchmarkEngine {}
extension ReuseStreamingReference.StreamingEnhancer: CompiledBenchmarkEngine {}

/// Opt-in measurement, separate from the ordinary regression suite. Timings have
/// no pass threshold; every completed waveform must match the frozen reference.
final class StreamingCompiledBenchmarkTests: XCTestCase {
    func testMacWholeStreamABBA() throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["REUSE_COMPILED_BENCHMARK"] == "1" else {
            throw XCTSkip("Set REUSE_COMPILED_BENCHMARK=1 for the full-stream Mac ABBA measurement.")
        }
        guard let checkpoint = environment["REUSE_STREAMING_CHECKPOINT"] else {
            throw XCTSkip("Set REUSE_STREAMING_CHECKPOINT to the pinned checkpoint.")
        }
        let root = try repositoryRoot()
        let inputURL = environment["REUSE_STREAMING_BENCHMARK_INPUT"].map { URL(fileURLWithPath: $0) }
            ?? root.appendingPathComponent("apple/Benchmarks/StreamingEight/BenchAssets/input.f32")
        let inputBytes = try Data(contentsOf: inputURL)
        guard inputBytes.count >= 640 * 4, inputBytes.count.isMultiple(of: 4) else {
            XCTFail("Expected finite, little-endian float32 input."); return
        }
        var source = [Float](repeating: 0, count: inputBytes.count / 4)
        _ = source.withUnsafeMutableBytes { inputBytes.copyBytes(to: $0) }
        guard source.allSatisfy(\.isFinite) else { XCTFail("Input contains nonfinite samples."); return }
        let repetitions = Int(environment["REUSE_COMPILED_BENCHMARK_REPETITIONS"] ?? "8") ?? 8
        guard repetitions > 0 else { XCTFail("Repetitions must be positive."); return }
        let audio = Array(repeating: source, count: repetitions).flatMap { $0 }
        guard let mode = StreamingFrameCompilation(rawValue: environment["REUSE_COMPILED_BENCHMARK_MODE"] ?? "body"),
              mode != .disabled else { XCTFail("Choose body or decoder_features compilation."); return }
        let pooledFrequency = environment["REUSE_COMPILED_BENCHMARK_POOLED_FREQUENCY"] == "1"
        let modelURL = URL(fileURLWithPath: checkpoint)
        let engines: [String: any CompiledBenchmarkEngine] = [
            "baseline": try ReuseStreamingReference.StreamingEnhancer(modelURL: modelURL, layers: 8),
            "candidate": try ReuseStreaming.StreamingEnhancer(modelURL: modelURL, layers: 8,
                                                                optimized: true, compilation: mode,
                                                                pooledFrequency: pooledFrequency)
        ]
        var reference: [Float]?
        var trials = [[String: Any]]()
        let order = ["baseline", "candidate", "candidate", "baseline"]
        for (index, variant) in order.enumerated() {
            let engine = engines[variant]!
            engine.reset()
            try engine.warmUp()
            let thermalBefore = thermal()
            let powerBefore = ProcessInfo.processInfo.isLowPowerModeEnabled
            var output = [Float](), frames = [Double]()
            var thermalCounts = [String: Int]()
            output.reserveCapacity(audio.count)
            frames.reserveCapacity(audio.count / 320 + 2)
            let started = ProcessInfo.processInfo.systemUptime
            for position in stride(from: 0, to: audio.count, by: 320) {
                output += try engine.push(Array(audio[position..<min(position + 320, audio.count)]))
                frames += engine.lastFrameMilliseconds
                thermalCounts[thermal(), default: 0] += 1
            }
            output += try engine.finish()
            frames += engine.lastFrameMilliseconds
            let wall = ProcessInfo.processInfo.systemUptime - started
            guard output.count == audio.count, output.allSatisfy(\.isFinite),
                  !frames.isEmpty, frames.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                XCTFail("Invalid waveform or timing in trial \(index)."); return
            }
            if reference == nil { reference = output }
            let expected = reference!
            let differences = zip(expected, output).filter { $0.bitPattern != $1.bitPattern }
            let bytes = output.withUnsafeBytes { Data($0) }
            trials.append([
                "variant": variant, "order_index": index, "raw_frame_ms": frames,
                "frame_timing": statistics(frames), "wall_seconds": wall,
                "real_time_factor": wall / (Double(audio.count) / 16000),
                "thermal_before": thermalBefore, "thermal_after": thermal(),
                "thermal_push_counts": thermalCounts, "low_power_mode_before": powerBefore,
                "low_power_mode_after": ProcessInfo.processInfo.isLowPowerModeEnabled,
                "output_samples": output.count, "output_sha256": digest(bytes), "all_finite": true,
                "bitwise_equal": differences.isEmpty,
                "different_float_bit_patterns": differences.count,
                "max_abs_difference": differences.map { abs($0 - $1) }.max() ?? 0
            ])
            guard differences.isEmpty else {
                try writeReport(trials: trials, root: root, input: inputBytes, audioSamples: audio.count,
                                repetitions: repetitions, mode: mode, status: "parity_failed", environment: environment)
                XCTFail("Trial \(index) changed \(differences.count) float bit patterns."); return
            }
        }
        try writeReport(trials: trials, root: root, input: inputBytes, audioSamples: audio.count,
                        repetitions: repetitions, mode: mode, status: "passed", environment: environment)
    }

    private func writeReport(trials: [[String: Any]], root: URL, input: Data, audioSamples: Int,
                             repetitions: Int, mode: StreamingFrameCompilation, status: String,
                             environment: [String: String]) throws {
        var pooled = [String: Any]()
        for variant in ["baseline", "candidate"] {
            let selected = trials.filter { $0["variant"] as? String == variant }
            let frames = selected.flatMap { $0["raw_frame_ms"] as! [Double] }
            guard !frames.isEmpty else { continue }
            let wall = selected.reduce(0.0) { $0 + ($1["wall_seconds"] as! Double) }
            pooled[variant] = ["frame_timing": statistics(frames), "wall_seconds": wall,
                               "real_time_factor": wall / (Double(audioSamples * selected.count) / 16000)]
        }
        let manifestURL = root.appendingPathComponent("apple/Benchmarks/Reference/baseline-manifest.json")
        let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
        let sourceHashes = try engineHashes(root: root)
        let report: [String: Any] = [
            "status": status, "created_at": ISO8601DateFormatter().string(from: Date()),
            "platform": "macOS", "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
            "baseline_revision": manifest["revision"]!,
            "candidate_source_sha256": digest(try JSONSerialization.data(withJSONObject: sourceHashes, options: .sortedKeys)),
            "candidate_source_files_sha256": sourceHashes,
            "checkpoint_sha256": "6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade",
            "baseline_optimized": true, "candidate_optimized": true, "candidate_compiled": true,
            "candidate_pooled_frequency": environment["REUSE_COMPILED_BENCHMARK_POOLED_FREQUENCY"] == "1",
            "layers": 8, "precision": "float32", "lookahead_frames": 0,
            "global_compile_mode_unchanged": true, "compiled_boundary": mode.rawValue,
            "sample_rate": 16000, "audio_samples_per_trial": audioSamples,
            "audio_seconds_per_trial": Double(audioSamples) / 16000,
            "input_repetitions_without_state_reset": repetitions, "input_sha256": digest(input),
            "paced": false, "trial_order": ["baseline", "candidate", "candidate", "baseline"],
            "warmup": "Three dummy frames then reset per trial; load, warmup, validation and IO excluded.",
            "scope": "GPU-completed full streaming push plus finish, file replay; not live call latency.",
            "limitations": "Two trials per variant; frames are correlated; no frame or outlier exclusions.",
            "trials": trials, "pooled": pooled
        ]
        let path = environment["REUSE_COMPILED_BENCHMARK_REPORT"] ?? "/tmp/reuse-eight-compiled-frame-mac-benchmark.json"
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path), options: .atomic)
        print("REUSE_COMPILED_BENCHMARK_REPORT=\(path)")
        print("REUSE_COMPILED_BENCHMARK_POOLED=\(pooled)")
    }

    private func repositoryRoot() throws -> URL {
        var directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let files = FileManager.default
        while !files.fileExists(atPath: directory.appendingPathComponent("Package.swift").path) {
            let parent = directory.deletingLastPathComponent()
            guard parent.path != directory.path else {
                throw NSError(domain: "StreamingBenchmark", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Cannot find the package root for benchmark fixtures."])
            }
            directory = parent
        }
        return directory
    }

    private func engineHashes(root: URL) throws -> [String: String] {
        let source = root.appendingPathComponent("apple/Sources/ReuseStreaming")
        let files = FileManager.default.enumerator(at: source, includingPropertiesForKeys: nil)!
        var hashes = [String: String]()
        for case let file as URL in files where file.pathExtension == "swift" {
            hashes[String(file.path.dropFirst(root.path.count + 1))] = try digest(Data(contentsOf: file))
        }
        return hashes
    }

    private func statistics(_ values: [Double]) -> [String: Any] {
        let sorted = values.sorted()
        func quantile(_ p: Double) -> Double { sorted[max(0, Int(ceil(p * Double(sorted.count))) - 1)] }
        return ["count": values.count, "mean_ms": values.reduce(0, +) / Double(values.count),
                "p95_ms": quantile(0.95), "p99_ms": quantile(0.99),
                "minimum_ms": sorted[0], "maximum_ms": sorted[sorted.count - 1],
                "frames_over_20ms": values.filter { $0 > 20 }.count,
                "quantile_definition": "nearest rank ceil(P*N)"]
    }

    private func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
