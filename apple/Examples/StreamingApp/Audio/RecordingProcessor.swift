import Foundation
import ReuseAudio
import ReuseStreaming

final class RecordingProcessor {
    private let enhancer: StreamingEnhancer
    private let progress: RecordingProgress
    private var original = [Float]()
    private var restored = [Float]()
    private var frameMilliseconds = [Double]()

    init(modelURL: URL, progress: RecordingProgress) throws {
        self.progress = progress
        enhancer = try StreamingEnhancer(modelURL: modelURL)
        try enhancer.warmUp()
    }

    func append(_ samples: [Float]) throws {
        guard !progress.isCancelled else { return }
        original += samples
        restored += try enhancer.push(samples)
        frameMilliseconds += enhancer.lastFrameMilliseconds
        progress.completed(restored.count)
    }

    func finish(directory: URL) throws -> URL {
        guard !progress.isCancelled else { throw ReuseError.invalid("Recording was interrupted.") }
        restored += try enhancer.finish()
        frameMilliseconds += enhancer.lastFrameMilliseconds
        guard restored.count == original.count else { throw ReuseError.invalid("Output duration changed.") }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try saveAudio(directory)
        try RecordingReport.save(directory, frames: frameMilliseconds, samples: restored.count,
                                 diagnostics: progress.diagnostics())
        progress.completed(restored.count)
        return directory
    }

    private func saveAudio(_ directory: URL) throws {
        try AudioFileIO.write(original, to: directory.appendingPathComponent("original.wav"))
        try AudioFileIO.write(restored, to: directory.appendingPathComponent("restored.wav"))
        let peak = max(original.map { abs($0) }.max() ?? 0, restored.map { abs($0) }.max() ?? 0)
        let gain: Float = min(1, 0.89 / max(peak, 1e-6))
        try AudioFileIO.write(original.map { $0 * gain }, to: directory.appendingPathComponent("play-original.wav"))
        try AudioFileIO.write(restored.map { $0 * gain }, to: directory.appendingPathComponent("play-restored.wav"))
    }
}
