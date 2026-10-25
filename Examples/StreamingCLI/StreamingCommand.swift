import Foundation
import ReuseAudio
import ReuseStreaming

enum StreamingCommand {
    static func run(_ options: StreamingOptions) throws {
        try options.validatePaths()
        let input = try AudioFileIO.read(options.input)
        guard input.count >= StreamingEnhancer.windowSamples else {
            throw ReuseError.invalid("Streaming input must contain at least 40 ms of audio.")
        }
        let start = ProcessInfo.processInfo.systemUptime
        let engine = try StreamingEnhancer(modelURL: options.model, layers: options.layers, optimized: !options.baseline)
        let loaded = ProcessInfo.processInfo.systemUptime
        try engine.warmUp()
        let warmed = ProcessInfo.processInfo.systemUptime
        let rendered = try render(input, engine: engine)
        let finished = ProcessInfo.processInfo.systemUptime
        try StreamingOutput.save(options, rendered: rendered,
            timing: RunTiming(load: loaded - start, warmup: warmed - loaded, processing: finished - warmed),
            optimized: engine.isOptimized)
        print("Saved \(options.output.path)\nReport: \(options.report.path)")
    }

    private static func render(_ input: [Float], engine: StreamingEnhancer) throws -> RenderedAudio {
        var result = RenderedAudio()
        for start in stride(from: 0, to: input.count, by: StreamingEnhancer.hopSamples) {
            let end = min(start + StreamingEnhancer.hopSamples, input.count)
            result.samples += try engine.push(Array(input[start..<end]))
            result.frameMilliseconds += engine.lastFrameMilliseconds
        }
        result.samples += try engine.finish()
        result.frameMilliseconds += engine.lastFrameMilliseconds
        guard result.samples.count == input.count, result.samples.allSatisfy({ $0.isFinite }) else {
            throw ReuseError.invalid("Enhancement produced nonfinite audio or changed the input duration.")
        }
        return result
    }
}

struct RenderedAudio {
    var samples = [Float]()
    var frameMilliseconds = [Double]()
}
struct RunTiming {
    let load: Double
    let warmup: Double
    let processing: Double
}
