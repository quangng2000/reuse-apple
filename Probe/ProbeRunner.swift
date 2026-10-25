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
        let engine = try ReuseEngine(modelURL: options.model)
        let loadSeconds = ProcessInfo.processInfo.systemUptime - start
        let (output, timings) = try infer(engine, samples: samples, repeats: options.benchmark ? 3 : 1)
        guard output.count == samples.count, output.allSatisfy({ $0.isFinite }) else {
            throw ReuseError.invalid("Inference returned invalid audio or changed its duration.")
        }
        let report = try makeReport(options, timings: timings, loadSeconds: loadSeconds, device: device.name, count: output.count)
        try save(output, report: report, options: options)
        print(String(decoding: report, as: UTF8.self))
    }

    private static func infer(_ engine: ReuseEngine, samples: [Float], repeats: Int) throws -> ([Float], [Double]) {
        var timings = [Double](), output = [Float]()
        for index in 0..<repeats {
            let start = ProcessInfo.processInfo.systemUptime
            output = try engine.enhance(samples) { _ in }
            let seconds = ProcessInfo.processInfo.systemUptime - start
            timings.append(seconds)
            fputs(String(format: "Run %d: %.3f seconds\n", index + 1, seconds), stderr)
        }
        return (output, timings)
    }

    private static func makeReport(_ options: ProbeOptions, timings: [Double], loadSeconds: Double,
                                   device: String, count: Int) throws -> Data {
        let report: [String: Any] = ["input": options.input.path, "output": options.output.path,
            "model": options.model.path, "backend": "Swift/MLX/Metal", "device": device,
            "timings_seconds": timings, "load_seconds": loadSeconds, "samples": count,
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
