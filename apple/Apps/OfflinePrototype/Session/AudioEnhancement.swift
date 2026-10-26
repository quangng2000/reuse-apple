import ReuseAudio
import Foundation

struct EnhancementResult {
    let elapsed: Double
    let duration: Double
    let reducedLevel: Bool
}

enum AudioEnhancement {
    static func run(_ input: URL, model: URL, target: URL,
                    progress: @escaping (Double) -> Void) throws -> EnhancementResult {
        let start = Date()
        let samples = try AudioFileIO.read(input)
        try Task.checkCancellation()
        // FP32 remains the default until FP16 parity is validated.
        let engine = try ReuseEngine(modelURL: model)
        let output = try restore(samples, engine: engine, progress: progress)
        let reducedLevel = try save(output, target: target)
        return EnhancementResult(elapsed: Date().timeIntervalSince(start),
                                 duration: Double(samples.count) / 16000, reducedLevel: reducedLevel)
    }

    private static func restore(_ samples: [Float], engine: ReuseEngine,
                                progress: @escaping (Double) -> Void) throws -> [Float] {
        var lastReported = -1
        return try engine.enhance(samples) { fraction in
            try Task.checkCancellation()
            let percent = Int(fraction * 100)
            if percent != lastReported {
                lastReported = percent
                progress(fraction)
            }
        }
    }

    private static func save(_ output: [Float], target: URL) throws -> Bool {
        try Task.checkCancellation()
        let peak = output.reduce(Float(0)) { max($0, abs($1)) }
        let gain: Float = peak > 0.98 ? 0.98 / peak : 1
        try AudioFileIO.write(output.map { $0 * gain }, to: target)
        try Task.checkCancellation()
        return gain < 1
    }
}
