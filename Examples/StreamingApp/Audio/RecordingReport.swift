import Foundation
import ReuseAudio

enum RecordingReport {
    static func save(_ directory: URL, frames: [Double], samples: Int,
                     diagnostics: (backlog: Double, callback: Double)) throws {
        let sorted = frames.sorted()
        let maximumFrameMilliseconds: Double = sorted.isEmpty ? 0 : sorted[sorted.count - 1]
        let report: [String: Any] = [
            "kind": "live_microphone", "sample_rate": 16000, "layers": 4,
            "samples": samples, "duration_seconds": Double(samples) / 16000,
            "mean_frame_ms": frames.reduce(0, +) / Double(max(1, frames.count)),
            "maximum_frame_ms": maximumFrameMilliseconds, "frame_milliseconds": frames,
            "frames_over_20ms": frames.filter { $0 > 20 }.count,
            "maximum_backlog_seconds": diagnostics.backlog,
            "maximum_microphone_callback_seconds": diagnostics.callback,
            "note": "Package public API; 16kHz, four-layer optimized FP32, zero lookahead. No echo cancellation or connected call. No clean reference."
        ]
        guard JSONSerialization.isValidJSONObject(report) else {
            throw ReuseError.invalid("Cannot serialize nonfinite or unsupported recording metadata.")
        }
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: directory.appendingPathComponent("report.json"), options: .atomic)
    }
}
