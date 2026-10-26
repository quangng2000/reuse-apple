import ReuseAudio
import Foundation
import Metal
import MLX

enum ProbeRunner {
    static func run(_ options: ProbeOptions) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw ReuseError.invalid("A Metal GPU is required.") }
        GPU.set(cacheLimit: 64 * 1024 * 1024)
        let samples = try AudioFileIO.read(options.input)
        let start = ProcessInfo.processInfo.systemUptime
        let engine = try options.optimization.load(modelURL: options.model)
        let loadSeconds = ProcessInfo.processInfo.systemUptime - start
        let result = try infer(engine, samples: samples, repeats: options.benchmark ? 3 : 1)
        guard result.output.count == samples.count, result.output.allSatisfy({ $0.isFinite }) else {
            throw ReuseError.invalid("Inference returned invalid audio or changed its duration.")
        }
        let report = try makeReport(options, result: result, loadSeconds: loadSeconds, device: device.name)
        try save(result.output, report: report, options: options)
        print(String(decoding: report, as: UTF8.self))
    }

    private static func infer(_ engine: ReuseEngine, samples: [Float], repeats: Int) throws -> ProbeInferenceResult {
        var result = ProbeInferenceResult()
        for index in 0..<repeats {
            let thermal = ProcessInfo.processInfo.thermalState.rawValue
            let start = ProcessInfo.processInfo.systemUptime
            result.output = try engine.enhance(samples) { _ in }
            let seconds = ProcessInfo.processInfo.systemUptime - start
            result.append(seconds: seconds, startingThermalState: thermal)
            fputs(String(format: "Run %d: %.3f seconds\n", index + 1, seconds), stderr)
        }
        return result
    }

    private static func makeReport(_ options: ProbeOptions, result: ProbeInferenceResult,
                                   loadSeconds: Double, device: String) throws -> Data {
        let count = result.output.count
        let report: [String: Any] = ["input": options.input.path, "output": options.output.path,
            "model": options.model.path, "backend": "Swift/MLX/Metal", "device": device,
            "timings_seconds": result.seconds, "load_seconds": loadSeconds, "samples": count,
            "optimization": options.optimization.rawValue, "thermal_states_before_after": result.thermalStates,
            "thermal_state_key": ["0": "nominal", "1": "fair", "2": "serious", "3": "critical"],
            "sample_rate": 16000, "channels": 1, "duration_seconds": Double(count) / 16000,
            "mlx_peak_bytes": GPU.peakMemory, "dtype": "float32", "all_finite": true]
        return try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    }

    private static func save(_ samples: [Float], report: Data, options: ProbeOptions) throws {
        try options.validatePaths()
        let temporary = options.output.deletingLastPathComponent().appendingPathComponent(".reuse-\(UUID()).wav")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try AudioFileIO.write(samples, to: temporary)
        if options.force {
            guard rename(temporary.path, options.output.path) == 0 else {
                throw ReuseError.invalid("Cannot replace output: \(String(cString: strerror(errno)))")
            }
        } else {
            try FileManager.default.moveItem(at: temporary, to: options.output)
        }
        try report.write(to: options.report, options: options.force ? .atomic : .withoutOverwriting)
    }
}
