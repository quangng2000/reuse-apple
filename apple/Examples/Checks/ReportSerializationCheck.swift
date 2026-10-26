import Foundation

enum ReuseError: Error { case invalid(String) }
struct RenderedAudio { var samples = [Float](); var frameMilliseconds = [Double]() }
struct RunTiming { let load: Double; let warmup: Double; let processing: Double }

@main
enum ReportSerializationCheck {
    static func main() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let options = try makeOptions(directory)
        for frames in [[], [1.25], [0.1, 1.4, 20.5]] {
            try checkCLI(options, frames: frames)
            try checkRecording(directory, frames: frames)
        }
        for invalid in [Double.nan, .infinity, -.infinity] {
            try requireFailure { try cliData(options, frames: [invalid]) }
            try requireFailure { try RecordingReport.save(directory, frames: [invalid], samples: 640,
                                                          diagnostics: (0, 0.02)) }
        }
        print("PASS: CLI/live empty, singleton, and multi-frame JSON; NaN/infinities rejected safely.")
    }

    private static func makeOptions(_ directory: URL) throws -> StreamingOptions {
        for name in ["input.wav", "model.safetensors", "output.wav"] {
            try Data([1, 2, 3]).write(to: directory.appendingPathComponent(name))
        }
        return try StreamingOptions(["--input", directory.appendingPathComponent("input.wav").path,
            "--model", directory.appendingPathComponent("model.safetensors").path,
            "--output", directory.appendingPathComponent("output.wav").path, "--benchmark"])
    }

    private static func cliData(_ options: StreamingOptions, frames: [Double]) throws -> Data {
        try StreamingReport.data(options, rendered: RenderedAudio(samples: [0.1], frameMilliseconds: frames),
            timing: RunTiming(load: 1, warmup: 0.1, processing: 0.2), optimized: true, outputURL: options.output)
    }

    private static func checkCLI(_ options: StreamingOptions, frames: [Double]) throws {
        let report = try JSONSerialization.jsonObject(with: cliData(options, frames: frames)) as! [String: Any]
        let timing = report["frame_timing"] as! [String: Any]
        guard timing["maximum_ms"] as? Double == (frames.max() ?? 0) else {
            throw ReuseError.invalid("CLI maximum is missing or incorrectly typed.")
        }
    }

    private static func checkRecording(_ directory: URL, frames: [Double]) throws {
        try RecordingReport.save(directory, frames: frames, samples: 640, diagnostics: (0, 0.02))
        let data = try Data(contentsOf: directory.appendingPathComponent("report.json"))
        let report = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        guard report["maximum_frame_ms"] as? Double == (frames.max() ?? 0) else {
            throw ReuseError.invalid("Live maximum is missing or incorrectly typed.")
        }
    }

    private static func requireFailure<T>(_ operation: () throws -> T) throws {
        do { _ = try operation() } catch { return }
        throw ReuseError.invalid("Invalid metadata was unexpectedly accepted.")
    }
}
