import CryptoKit
import Foundation
import ReuseAudio

enum StreamingReport {
    static func data(_ options: StreamingOptions, rendered: RenderedAudio, timing: RunTiming,
                     optimized: Bool, outputURL: URL) throws -> Data {
        let duration = Double(rendered.samples.count) / 16000
        var report: [String: Any] = [
            "sample_rate": 16000, "samples": rendered.samples.count, "duration_seconds": duration,
            "layers": options.layers, "lookahead_frames": 0, "precision": "float32",
            "optimized": optimized,
            "backend": "MLX / Metal", "model_sha256": try hash(options.model),
            "input_sha256": try hash(options.input), "output_sha256": try hash(outputURL),
            "load_seconds": timing.load, "warmup_seconds": timing.warmup,
            "processing_seconds": timing.processing, "real_time_factor": timing.processing / duration,
            "all_finite": true, "noncommercial_research": true,
            "timing_scope": "File replay, analysis + model + synthesis + GPU completion. No microphone, network, echo cancellation, or playback latency."
        ]
        if options.benchmark { report["frame_timing"] = frameTiming(rendered.frameMilliseconds) }
        guard JSONSerialization.isValidJSONObject(report) else {
            throw ReuseError.invalid("Cannot serialize nonfinite or unsupported benchmark metadata.")
        }
        return try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
    }

    private static func frameTiming(_ values: [Double]) -> [String: Any] {
        let sorted = values.sorted()
        let maximumFrameMilliseconds: Double = sorted.isEmpty ? 0 : sorted[sorted.count - 1]
        return ["count": values.count, "milliseconds": values,
                "mean_ms": values.reduce(0, +) / Double(max(1, values.count)),
                "p95_ms": percentile(sorted, fraction: 0.95), "p99_ms": percentile(sorted, fraction: 0.99),
                "maximum_ms": maximumFrameMilliseconds, "frames_over_20ms": values.filter { $0 > 20 }.count,
                "fraction_over_20ms": Double(values.filter { $0 > 20 }.count) / Double(max(1, values.count))]
    }

    private static func percentile(_ sorted: [Double], fraction: Double) -> Double {
        guard !sorted.isEmpty else { return 0 }
        return sorted[max(0, min(sorted.count - 1, Int(ceil(fraction * Double(sorted.count))) - 1))]
    }

    private static func hash(_ url: URL) throws -> String {
        SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined()
    }
}
