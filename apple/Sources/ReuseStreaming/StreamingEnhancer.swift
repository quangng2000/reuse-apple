// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// See LICENSE in this module. Noncommercial research and education only.
import Foundation
import MLX

/// Stateful FP32 speech enhancement for finite, 16 kHz mono samples.
/// Call from one serial worker, never an audio render callback. Instances are
/// not thread-safe. Each push returns only newly available audio; finish flushes
/// the tail. Audio/model state is bounded; output and timing storage scale with
/// the latest chunk size. Use hop-sized chunks for low-latency audio processing.
public final class StreamingEnhancer {
    public static let sampleRate = 16_000
    public static let hopSamples = 320
    public static let windowSamples = 640
    public let layers: Int
    public let isOptimized: Bool
    public let optimization: StreamingOptimization
    /// Actual frame times from the latest push/finish, including GPU completion.
    public private(set) var lastFrameMilliseconds = [Double]()
    private let model: StreamingReuseModel
    private let compilation: StreamingFrameCompilation
    private var compiledFrame: StreamingCompiledFrame?
    private var transform: StreamingTransform
    private var session = StreamingSession()

    /// Explicit package configuration for benchmark and regression comparisons.
    package init(modelURL: URL, layers: Int, optimized: Bool, compilation: StreamingFrameCompilation,
                 pooledFrequency: Bool = false, optimization: StreamingOptimization = .standard) throws {
        guard (3...12).contains(layers) else { throw StreamingError.invalidLayers(layers) }
        try optimization.validate(layers: layers, optimized: optimized)
        self.layers = layers
        isOptimized = optimized
        self.optimization = optimization
        self.compilation = optimized ? compilation : .disabled
        model = try Device.withDefaultDevice(.gpu) {
            let model = try StreamingReuseModel(weightsURL: modelURL, layers: layers, lookahead: 0,
                                                optimized: optimized, optimization: optimization)
            model.useBatchedFrequency = pooledFrequency && optimized && optimization == .standard
            return model
        }
        transform = Device.withDefaultDevice(.gpu) { StreamingTransform() }
    }

    /// Accepts any chunk size. The first output waits for 640 input samples.
    /// Nonfinite input is rejected without changing stream state.
    public func push(_ samples: [Float]) throws -> [Float] {
        lastFrameMilliseconds.removeAll(keepingCapacity: true)
        return try Device.withDefaultDevice(.gpu) { try session.push(samples, render: renderFrame) }
    }

    /// Flushes the tail once, preserving exactly the total input sample count.
    /// Requires at least 640 samples; reset before starting the next recording.
    public func finish() throws -> [Float] {
        lastFrameMilliseconds.removeAll(keepingCapacity: true)
        return try Device.withDefaultDevice(.gpu) { try session.finish(render: renderFrame) }
    }

    /// Discards pending audio and recurrent state. Retains the loaded weights.
    public func reset() {
        Device.withDefaultDevice(.gpu) {
            model.reset()
            transform = StreamingTransform()
        }
        // The compiled graph only retains weights and placeholders, with caches
        // supplied on each call. Keep it warm across independent recordings.
        session = StreamingSession()
        lastFrameMilliseconds.removeAll(keepingCapacity: true)
    }

    /// Prepares GPU kernels and the eight-layer graph before recording.
    /// Valid only on a fresh/reset stream.
    /// Three dummy frames are excluded from later timing and audio output.
    public func warmUp() throws {
        guard session.isFresh else { throw StreamingError.warmupRequiresReset }
        defer { reset() }
        Device.withDefaultDevice(.gpu) {
            for _ in 0..<3 { _ = renderFrame([Float](repeating: 0, count: 640)) }
        }
    }

    private func renderFrame(_ input: [Float]) -> [Float] {
        let started = ProcessInfo.processInfo.systemUptime
        let (magnitude, phase) = transform.analyze(input)
        let restored: (MLXArray, MLXArray)
        if compilation != .disabled && !model.isFirstFrame {
            if compiledFrame == nil {
                compiledFrame = StreamingCompiledFrame(model: model, mode: compilation)
            }
            restored = compiledFrame!(magnitude: magnitude, phase: phase)
        } else {
            restored = model(magnitude: magnitude, phase: phase, deferEvaluation: isOptimized)
        }
        let output = transform.synthesize(magnitude: restored.0, phase: restored.1,
                                         evaluating: isOptimized ? model.frameStateArrays : [])
        lastFrameMilliseconds.append((ProcessInfo.processInfo.systemUptime - started) * 1000)
        return output
    }
}
