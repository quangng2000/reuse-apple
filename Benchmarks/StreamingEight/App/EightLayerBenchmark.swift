import Foundation
import CryptoKit
import Darwin
import UIKit
import ReuseStreaming
import ReuseStreamingReference

private protocol EightLayerEngine: AnyObject {
    var lastFrameMilliseconds: [Double] { get }
    func push(_ samples: [Float]) throws -> [Float]
    func finish() throws -> [Float]
    func reset()
    func warmUp() throws
}
extension ReuseStreaming.StreamingEnhancer: EightLayerEngine {}
extension ReuseStreamingReference.StreamingEnhancer: EightLayerEngine {}

private struct Configuration: Decodable {
    struct Phase: Decodable { let id: String; let repetitions: Int; let trialOrder: [String] }
    struct EngineOptions: Decodable {
        let compilation: String
        let pooledFrequency: Bool
        var report: [String: Any] { ["compilation": compilation, "pooled_frequency": pooledFrequency] }
    }
    let schemaVersion: Int
    let baselineRevision: String
    let candidateRevision: String
    let checkpointFile: String
    let checkpointSHA256: String
    let inputFile: String
    let layers: Int
    let paced: Bool
    let cooldownSeconds: Double
    let maximumAbsoluteOutputDifference: Double
    let requireBitwiseOutputEquality: Bool
    let baselineOptions: EngineOptions
    let candidateOptions: EngineOptions
    let phases: [Phase]
}

private enum BenchmarkError: LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let text) = self { return text }; return nil }
}

private func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
private func elapsed(_ start: UInt64) -> Double { Double(now() - start) / 1_000_000_000 }
private func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
private func floatData(_ samples: [Float]) -> Data { samples.withUnsafeBytes { Data($0) } }
private func thermal() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}
private func battery() -> Float {
    if Thread.isMainThread { return UIDevice.current.batteryLevel }
    return DispatchQueue.main.sync { UIDevice.current.batteryLevel }
}
private func stats(_ values: [Double]) -> [String: Any] {
    let s = values.sorted()
    guard !s.isEmpty else { return ["count": 0] }
    let middle = s.count / 2
    let median = s.count.isMultiple(of: 2) ? (s[middle - 1] + s[middle]) / 2 : s[middle]
    func percentile(_ p: Double) -> Double { s[max(0, Int(ceil(p * Double(s.count))) - 1)] }
    return ["count": s.count, "mean_ms": s.reduce(0, +) / Double(s.count),
            "median_ms": median, "p95_ms": percentile(0.95), "p99_ms": percentile(0.99),
            "minimum_ms": s[0], "maximum_ms": s[s.count - 1],
            "over_20ms": values.filter { $0 > 20 }.count,
            "percent_over_20ms": Double(values.filter { $0 > 20 }.count) / Double(s.count) * 100,
            "percentile_definition": "nearest rank ceil(P*N)"]
}
private func emit(_ value: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: .sortedKeys),
          let string = String(data: data, encoding: .utf8) else { return }
    print("REUSE_EIGHT_BENCHMARK_EVENT=" + string)
    fflush(stdout)
}

final class EightLayerBenchmark {
    let runID: String
    private let resources: URL
    private let documents: URL
    private let configuration: Configuration
    private let input: [Float]
    private let progress: (String) -> Void
    private var trials = [[String: Any]]()
    private var baseReport = [String: Any]()

    init(progress: @escaping (String) -> Void, runID: String = UUID().uuidString) throws {
        self.progress = progress
        self.runID = runID
        guard let url = Bundle.main.resourceURL?.appendingPathComponent("BenchAssets") else {
            throw BenchmarkError.invalid("Missing benchmark assets")
        }
        resources = url
        documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        configuration = try JSONDecoder().decode(Configuration.self,
            from: Data(contentsOf: url.appendingPathComponent("config.json")))
        guard configuration.schemaVersion == 1, configuration.layers == 8,
              !configuration.phases.isEmpty, configuration.maximumAbsoluteOutputDifference > 0,
              configuration.cooldownSeconds >= 0, UUID(uuidString: runID) != nil,
              configuration.requireBitwiseOutputEquality,
              configuration.baselineRevision == "2446c8383ac65da5376d9159fa1d079535c4e36b",
              configuration.baselineOptions.compilation == "disabled", !configuration.baselineOptions.pooledFrequency,
              configuration.candidateOptions.compilation == "decoderFeatures", configuration.candidateOptions.pooledFrequency,
              configuration.candidateRevision.range(of: "^source-sha256:[0-9a-f]{64}$", options: .regularExpression) != nil
        else { throw BenchmarkError.invalid("Invalid eight-layer benchmark configuration or unfrozen candidate source") }
        let bytes = try Data(contentsOf: url.appendingPathComponent(configuration.inputFile))
        guard bytes.count >= 640 * 4, bytes.count.isMultiple(of: 4) else {
            throw BenchmarkError.invalid("Fixture must be nonempty little-endian float32 audio")
        }
        var samples = [Float](repeating: 0, count: bytes.count / 4)
        _ = samples.withUnsafeMutableBytes { bytes.copyBytes(to: $0) }
        guard samples.allSatisfy({ $0.isFinite }) else { throw BenchmarkError.invalid("Input fixture contains nonfinite samples") }
        input = samples
    }

    private func persist(_ name: String, status: String) throws {
        var report = baseReport
        report["status"] = status
        report["updated_at"] = ISO8601DateFormatter().string(from: Date())
        report["trials"] = trials
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: documents.appendingPathComponent(name), options: .atomic)
    }

    static func recordFailure(_ error: Error, runID: String) {
        emit(["event": "failed", "run_id": runID, "error": error.localizedDescription])
        let document = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let record: [String: Any] = ["status": "failed", "run_id": runID, "error": error.localizedDescription,
            "time": ISO8601DateFormatter().string(from: Date()),
            "partial_results_file": "Documents/latest-progress.json"]
        if let bytes = try? JSONSerialization.data(withJSONObject: record, options: [.prettyPrinted, .sortedKeys]) {
            try? bytes.write(to: document.appendingPathComponent("latest-failure.json"), options: .atomic)
        }
    }

    func run() throws -> Int {
        var system = utsname()
        uname(&system)
        let hardware = withUnsafePointer(to: &system.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        let modelURL = resources.appendingPathComponent(configuration.checkpointFile)
        let modelData = try Data(contentsOf: modelURL, options: .mappedIfSafe)
        let checkpointHash = sha256(modelData)
        guard checkpointHash == configuration.checkpointSHA256 else { throw BenchmarkError.invalid("Checkpoint SHA256 mismatch") }
        baseReport = ["schema_version": 1, "run_id": runID, "started_at": ISO8601DateFormatter().string(from: Date()),
            "device": ["hardware_model": hardware, "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
                       "physical_device": !hardware.hasPrefix("arm64") && !hardware.hasPrefix("x86"),
                       "battery_level_before": battery(), "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled],
            "baseline_revision": configuration.baselineRevision, "candidate_revision": configuration.candidateRevision,
            "baseline_options": configuration.baselineOptions.report, "candidate_options": configuration.candidateOptions.report,
            "swift_compilation_mode": "wholemodule", "require_bitwise_output_equality": true,
            "output_parity_requirement": "bitwise_equal", "absolute_tolerance_is_acceptance_gate": false,
            "layers": configuration.layers, "precision": "float32", "lookahead_frames": 0,
            "optimized_argument_for_both_variants": true,
            "checkpoint_sha256": checkpointHash, "input_sha256": sha256(floatData(input)),
            "source_fixture_samples": input.count, "sample_rate": 16000,
            "hop_samples": 320, "startup_window_samples": 640,
            "paced_input": configuration.paced,
            "measurement_scope": "Memory file replay, GPU-completed per-frame analysis/network/synthesis. Not microphone, echo cancellation, codec, transport or speaker latency.",
            "timing_clock": "DispatchTime uptimeNanoseconds for wall time; package lastFrameMilliseconds for GPU-completed frames",
            "warmup_scope": "Public warmUp: three dummy render frames, then reset, before every trial; excluded from trial time.",
            "output_max_absolute_tolerance": configuration.maximumAbsoluteOutputDifference]
        emit(["event": "run_started", "run_id": runID, "hardware": hardware, "layers": configuration.layers,
              "baseline_revision": configuration.baselineRevision, "candidate_revision": configuration.candidateRevision,
              "baseline_options": configuration.baselineOptions.report, "candidate_options": configuration.candidateOptions.report])
        var engines = [String: EightLayerEngine]()
        var loadTimes = [String: Double]()
        var started = now()
        engines["baseline"] = try ReuseStreamingReference.StreamingEnhancer(modelURL: modelURL, layers: 8, optimized: true)
        loadTimes["baseline_ms"] = elapsed(started) * 1000
        started = now()
        engines["candidate"] = try ReuseStreaming.StreamingEnhancer(modelURL: modelURL, layers: 8, optimized: true,
                                                                  compilation: .decoderFeatures, pooledFrequency: true)
        loadTimes["candidate_ms"] = elapsed(started) * 1000
        baseReport["load_times_ms_excluded"] = loadTimes
        try persist("latest-progress.json", status: "running")
        for phase in configuration.phases {
            guard phase.repetitions > 0, !phase.trialOrder.isEmpty,
                  phase.trialOrder.first == "baseline",
                  phase.trialOrder.allSatisfy({ engines[$0] != nil }) else {
                throw BenchmarkError.invalid("Each phase must begin with the baseline and use valid variants")
            }
            let audio = Array(repeating: input, count: phase.repetitions).flatMap { $0 }
            var reference: [Float]?
            for (index, variant) in phase.trialOrder.enumerated() {
                let identifier = phase.id + "-" + String(index) + "-" + variant
                guard let engine = engines[variant] else { throw BenchmarkError.invalid("Missing engine") }
                progress("\(phase.id): \(variant), trial \(index + 1)/\(phase.trialOrder.count)")
                engine.reset()
                let warmStart = now()
                try engine.warmUp()
                let warmMS = elapsed(warmStart) * 1000
                let before = thermal(), batteryBefore = battery()
                emit(["event": "trial_started", "run_id": runID, "trial": identifier, "samples": audio.count, "thermal": before])
                let trialStart = now()
                var output = [Float](), frames = [Double](), arrivalLateness = [Double]()
                var thermalCounts = [String: Int]()
                output.reserveCapacity(audio.count)
                frames.reserveCapacity(audio.count / 320 + 2)
                for offset in stride(from: 0, to: audio.count, by: 320) {
                    let end = min(offset + 320, audio.count)
                    if configuration.paced {
                        let expectedArrival = Double(end) / 16000
                        let wait = expectedArrival - elapsed(trialStart)
                        if wait > 0 { Thread.sleep(forTimeInterval: wait) }
                        arrivalLateness.append(max(0, elapsed(trialStart) - expectedArrival) * 1000)
                    }
                    output += try engine.push(Array(audio[offset..<end]))
                    frames += engine.lastFrameMilliseconds
                    thermalCounts[thermal(), default: 0] += 1
                }
                output += try engine.finish()
                frames += engine.lastFrameMilliseconds
                let wall = elapsed(trialStart)
                guard output.count == audio.count, output.allSatisfy({ $0.isFinite }),
                      !frames.isEmpty, frames.allSatisfy({ $0.isFinite && $0 >= 0 }) else {
                    throw BenchmarkError.invalid("Invalid output or timing in " + identifier)
                }
                if reference == nil { reference = output }
                guard let expected = reference, expected.count == output.count else { throw BenchmarkError.invalid("Missing baseline reference") }
                var maximum = 0.0, squared = 0.0, expectedSquared = 0.0, different = 0
                for (a, b) in zip(output, expected) {
                    let delta = Double(a) - Double(b)
                    maximum = max(maximum, abs(delta)); squared += delta * delta
                    expectedSquared += Double(b) * Double(b)
                    if a.bitPattern != b.bitPattern { different += 1 }
                }
                let outputBytes = floatData(output)
                let outputFile = "output-" + runID + "-" + identifier + ".f32"
                try outputBytes.write(to: documents.appendingPathComponent(outputFile), options: .atomic)
                let passed = different == 0
                let options = variant == "baseline" ? configuration.baselineOptions : configuration.candidateOptions
                var record: [String: Any] = ["id": identifier, "run_id": runID, "phase": phase.id, "variant": variant,
                    "engine_options": options.report,
                    "order_index": index, "input_repetitions": phase.repetitions,
                    "audio_seconds": Double(audio.count) / 16000, "output_samples": output.count,
                    "output_sha256": sha256(outputBytes), "output_file": "Documents/" + outputFile,
                    "all_finite": true, "frame_timing": stats(frames), "raw_frame_ms": frames,
                    "wall_seconds_excluding_load_warmup_io_validation": wall,
                    "real_time_factor": wall / (Double(audio.count) / 16000), "warmup_ms_excluded": warmMS,
                    "thermal_before": before, "thermal_after": thermal(), "thermal_push_counts": thermalCounts,
                    "battery_before": batteryBefore, "battery_after": battery(),
                    "parity": ["reference_trial": phase.id + "-0-baseline", "bitwise_equal": different == 0,
                               "different_float_bit_patterns": different, "max_abs_difference": maximum,
                               "rmse": sqrt(squared / Double(output.count)),
                               "relative_rmse": sqrt(squared / max(expectedSquared, 1e-30)),
                               "maximum_absolute_tolerance": configuration.maximumAbsoluteOutputDifference,
                               "absolute_tolerance_passed": maximum <= configuration.maximumAbsoluteOutputDifference,
                               "gate": "bitwise_equal",
                               "passed": passed]]
                if !arrivalLateness.isEmpty { record["synthetic_input_arrival_lateness_ms"] = stats(arrivalLateness) }
                trials.append(record)
                try persist("latest-progress.json", status: passed ? "running" : "parity_failed")
                emit(["event": "trial_completed", "run_id": runID, "trial": identifier, "timing": stats(frames),
                      "parity_passed": passed, "different_float_bit_patterns": different,
                      "max_abs_difference": maximum, "wall_seconds": wall])
                guard passed else { throw BenchmarkError.invalid("Bitwise output parity failed in " + identifier) }
                if configuration.cooldownSeconds > 0 { Thread.sleep(forTimeInterval: configuration.cooldownSeconds) }
            }
        }
        baseReport["completed_at"] = ISO8601DateFormatter().string(from: Date())
        baseReport["battery_level_after"] = battery()
        try persist("latest-results.json", status: "completed")
        try persist("latest-progress.json", status: "completed")
        emit(["event": "run_completed", "run_id": runID, "trials": trials.count, "result_file": "Documents/latest-results.json"])
        return trials.count
    }
}
