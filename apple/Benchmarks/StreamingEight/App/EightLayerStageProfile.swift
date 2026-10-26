import Foundation
import CryptoKit
import Darwin
import ReuseStreamingDiagnostic

private enum StageProfileError: LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}

private func profileClock() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
private func profileSeconds(_ start: UInt64) -> Double { Double(profileClock() - start) / 1_000_000_000 }
private func profileBytes(_ values: [Float]) -> Data { values.withUnsafeBytes { Data($0) } }
private func profileHash(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
private func profileThermal() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}
private func profileStats(_ values: [Double]) -> [String: Any] {
    let sorted = values.sorted(), count = values.count
    guard count > 0 else { return ["count": 0] }
    let middle = count / 2
    return ["count": count, "mean_ms": sorted.reduce(0, +) / Double(count),
            "p50_ms": count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle],
            "p95_ms": sorted[Int(ceil(0.95 * Double(count))) - 1],
            "minimum_ms": sorted[0], "maximum_ms": sorted[count - 1]]
}
private func profileEvent(_ event: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: event, options: .sortedKeys),
          let text = String(data: data, encoding: .utf8) else { return }
    print("REUSE_EIGHT_PROFILE_EVENT=" + text)
    fflush(stdout)
}

final class EightLayerStageProfile {
    private let progress: (String) -> Void
    private let resources: URL
    private let documents: URL
    private let manifest: [String: Any]
    private let checkpointHash: String
    private let modelURL: URL
    private let input: [Float]
    private let warmFrames = 8
    private let measuredFrames = 40
    private let expectedStages = ["analyze", "encode"] + (0..<8).map { "tf\($0)" } + ["decode", "synthesis"]

    init(progress: @escaping (String) -> Void, assetsURL: URL? = nil,
         sourceManifestURL: URL? = nil, outputURL: URL? = nil) throws {
        self.progress = progress
        guard let resources = assetsURL ?? Bundle.main.resourceURL?.appendingPathComponent("BenchAssets"),
              let manifestURL = sourceManifestURL ?? Bundle.main.url(forResource: "profile-source", withExtension: "json"),
              let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as? [String: Any],
              let revision = manifest["source_revision"] as? String, revision.hasPrefix("2446c83") else {
            throw StageProfileError.invalid("Missing pinned diagnostic source manifest")
        }
        self.resources = resources
        self.manifest = manifest
        documents = outputURL ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try FileManager.default.createDirectory(at: documents, withIntermediateDirectories: true)
        modelURL = resources.appendingPathComponent("realtime-reuse.safetensors")
        checkpointHash = profileHash(try Data(contentsOf: modelURL, options: .mappedIfSafe))
        guard checkpointHash == "6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade" else {
            throw StageProfileError.invalid("Diagnostic checkpoint SHA256 mismatch")
        }
        let bytes = try Data(contentsOf: resources.appendingPathComponent("input.f32"))
        guard bytes.count.isMultiple(of: 4), bytes.count >= (8 + 40 - 1) * 320 * 4 else {
            throw StageProfileError.invalid("Diagnostic requires at least 47 input hops")
        }
        var samples = [Float](repeating: 0, count: bytes.count / 4)
        _ = samples.withUnsafeMutableBytes { bytes.copyBytes(to: $0) }
        guard samples.allSatisfy({ $0.isFinite }) else { throw StageProfileError.invalid("Nonfinite diagnostic input") }
        input = Array(samples.prefix((8 + 40 - 1) * 320))
    }

    private struct Replay {
        let output: [Float]
        let states: [BenchmarkStateDigest]
        let measuredFrameMS: [Double]
        let stages: [[BenchmarkStageTiming]]
        let wallSeconds: Double
        let thermalBefore: String
        let thermalAfter: String
        let thermalFrames: [String: Int]
    }

    private func replay(profile: Bool) throws -> Replay {
        let engine = try ReuseStreamingDiagnostic.StreamingEnhancer(modelURL: modelURL, layers: 8, optimized: true)
        // Compile the diagnostic boundary kernels before timed replay, then reset.
        engine.benchmarkProfilingEnabled = profile
        try engine.warmUp()
        let before = profileThermal()
        let start = profileClock()
        var output = [Float](), allFrameMS = [Double](), stages = [[BenchmarkStageTiming]]()
        var thermalFrames = [String: Int]()
        for offset in stride(from: 0, to: input.count, by: 320) {
            engine.benchmarkProfilingEnabled = profile && allFrameMS.count >= warmFrames
            output += try engine.push(Array(input[offset..<offset + 320]))
            allFrameMS += engine.lastFrameMilliseconds
            stages += engine.benchmarkFrameProfiles
            thermalFrames[profileThermal(), default: 0] += engine.lastFrameMilliseconds.count
        }
        engine.benchmarkProfilingEnabled = profile
        output += try engine.finish()
        allFrameMS += engine.lastFrameMilliseconds
        stages += engine.benchmarkFrameProfiles
        thermalFrames[profileThermal(), default: 0] += engine.lastFrameMilliseconds.count
        let wall = profileSeconds(start)
        guard output.count == input.count, output.allSatisfy({ $0.isFinite }),
              allFrameMS.count == warmFrames + measuredFrames,
              allFrameMS.allSatisfy({ $0.isFinite && $0 >= 0 }),
              !profile || stages.count == measuredFrames else {
            throw StageProfileError.invalid("Incomplete diagnostic frames or output")
        }
        if profile {
            guard stages.allSatisfy({ $0.map(\.name) == expectedStages }) else {
                throw StageProfileError.invalid("Diagnostic stage sequence mismatch")
            }
        }
        // Hash all FP32 caches only after replay; readback and hashing are untimed.
        let states = engine.benchmarkStateDigests()
        return Replay(output: output, states: states, measuredFrameMS: Array(allFrameMS.suffix(measuredFrames)),
                      stages: stages, wallSeconds: wall, thermalBefore: before,
                      thermalAfter: profileThermal(), thermalFrames: thermalFrames)
    }

    func run() throws -> Int {
        profileEvent(["event": "profile_started", "source_revision": manifest["source_revision"] ?? "unknown"])
        progress("Profiling: uninstrumented pinned reference and cache gate…")
        let normal = try replay(profile: false)
        progress("Profiling: 40 frames with explicit stage completions…")
        let profiled = try replay(profile: true)
        var different = 0, maximum = 0.0
        for (left, right) in zip(normal.output, profiled.output) {
            if left.bitPattern != right.bitPattern { different += 1 }
            maximum = max(maximum, abs(Double(left) - Double(right)))
        }
        let normalStates = Dictionary(uniqueKeysWithValues: normal.states.map { ($0.name, $0) })
        let profileStates = Dictionary(uniqueKeysWithValues: profiled.states.map { ($0.name, $0) })
        let allNames = Set(normalStates.keys).union(profileStates.keys)
        let stateDifferences = allNames.filter { normalStates[$0] != profileStates[$0] }.sorted()
        let exact = different == 0 && stateDifferences.isEmpty
        let normalBytes = profileBytes(normal.output), measuredBytes = profileBytes(profiled.output)
        try normalBytes.write(to: documents.appendingPathComponent("profile-normal-output.f32"), options: .atomic)
        try measuredBytes.write(to: documents.appendingPathComponent("profile-stage-output.f32"), options: .atomic)
        var aggregates = [[String: Any]]()
        for (index, name) in expectedStages.enumerated() {
            let stages = profiled.stages.map { $0[index] }
            aggregates.append(["name": name,
                "graph_construction": profileStats(stages.map(\.graphConstructionMilliseconds)),
                "evaluation_wait": profileStats(stages.map(\.evaluationWaitMilliseconds)),
                "cpu_postprocessing": profileStats(stages.map(\.cpuPostprocessingMilliseconds)),
                "total_completed_wall": profileStats(stages.map(\.totalMilliseconds))])
        }
        let rawFrames: [[String: Any]] = profiled.stages.enumerated().map { index, timings in
            ["frame_index_after_warmup": index, "frame_wall_ms": profiled.measuredFrameMS[index],
             "stages": timings.map { stage -> [String: Any] in
                ["name": stage.name, "graph_construction_ms": stage.graphConstructionMilliseconds,
                 "evaluation_wait_ms": stage.evaluationWaitMilliseconds,
                 "cpu_postprocessing_ms": stage.cpuPostprocessingMilliseconds,
                 "total_completed_wall_ms": stage.totalMilliseconds]
             }]
        }
        func encodedStates(_ values: [BenchmarkStateDigest]) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(values))
        }
        var hardware = utsname(); uname(&hardware)
        let machine = withUnsafePointer(to: &hardware.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        let report: [String: Any] = [
            "schema_version": 1, "status": exact ? "completed" : "parity_failed",
            "created_at": ISO8601DateFormatter().string(from: Date()),
            "source_manifest": manifest, "checkpoint_sha256": checkpointHash,
            "input_sha256": profileHash(profileBytes(input)), "layers": 8, "precision": "float32",
            "hardware_model": machine, "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
            "low_power_mode": ProcessInfo.processInfo.isLowPowerModeEnabled,
            "sample_rate": 16000, "input_samples": input.count,
            "measured_frames": measuredFrames, "real_stream_warmup_frames": warmFrames,
            "dummy_warmup_frames_excluded": 3,
            "normal_frame_timing": profileStats(normal.measuredFrameMS),
            "normal_raw_frame_ms": normal.measuredFrameMS,
            "profiled_frame_timing": profileStats(profiled.measuredFrameMS), "raw_frames": rawFrames,
            "normal_full_replay_wall_seconds": normal.wallSeconds,
            "profiled_full_replay_wall_seconds": profiled.wallSeconds,
            "stage_aggregates": aggregates,
            "thermal": ["normal_before": normal.thermalBefore, "normal_after": normal.thermalAfter,
                        "profiled_before": profiled.thermalBefore, "profiled_after": profiled.thermalAfter,
                        "normal_rendered_frame_counts": normal.thermalFrames,
                        "profiled_rendered_frame_counts": profiled.thermalFrames],
            "parity": ["bitwise_equal": different == 0, "different_float_bit_patterns": different,
                       "max_abs_difference": maximum, "output_samples_each": input.count,
                       "normal_output_sha256": profileHash(normalBytes),
                       "profiled_output_sha256": profileHash(measuredBytes),
                       "state_sha256_equal": stateDifferences.isEmpty,
                       "state_arrays_each": normal.states.count, "different_state_names": stateDifferences],
            "normal_state_digests": try encodedStates(normal.states),
            "profiled_state_digests": try encodedStates(profiled.states),
            "measurement_notes": [
                "Diagnostic only: stage eval+synchronize boundaries inhibit normal fusion/overlap and add dispatch overhead.",
                "graph_construction_ms is host lazy graph creation; evaluation_wait_ms includes MLX traversal/encoding, GPU execution and completion wait. It is not isolated GPU time.",
                "Each model boundary evaluates its output and every updated convolution/recurrent cache. Synthesis includes CPU readback and overlap-add as cpu_postprocessing_ms.",
                "Three dummy frames warm kernels and reset; eight real stream frames warm recurrent state before the forty measured frames, including EOF.",
                "Normal replay uses the same pinned generated module with hooks disabled. Uninstrumented balanced ABBA remains authoritative for optimization speedup.",
                "Output and all model caches plus transform overlap/envelope are compared byte-for-byte via sample bits and full-state SHA256 after replay; state hashing is untimed."
            ]]
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            .write(to: documents.appendingPathComponent("latest-profile.json"), options: .atomic)
        profileEvent(["event": "profile_completed", "measured_frames": measuredFrames,
                      "output_bitwise_equal": different == 0, "state_sha256_equal": stateDifferences.isEmpty,
                      "result_file": "Documents/latest-profile.json"])
        guard exact else { throw StageProfileError.invalid("Profile changed output bits or state; diagnostic evidence saved") }
        return measuredFrames
    }

    static func recordFailure(_ error: Error) {
        let value: [String: Any] = ["event": "profile_failed", "error": error.localizedDescription,
                                   "time": ISO8601DateFormatter().string(from: Date())]
        profileEvent(value)
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        if let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: documents.appendingPathComponent("latest-profile-failure.json"), options: .atomic)
        }
    }
}
