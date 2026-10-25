// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// See LICENSE in this module. Noncommercial research and education only.
import MLX

package enum StreamingFrameCompilation: String, Equatable {
    case disabled
    case body
    case decoderFeatures = "decoder_features"
}

/// Experimental fixed-shape graph. Every cache is an explicit input and output,
/// so replay never captures a previous recording's recurrent state.
final class StreamingCompiledFrame {
    private unowned let model: StreamingReuseModel
    private let convolutionKeys: [String]
    private let recurrentKeys: [String]
    private let mode: StreamingFrameCompilation
    private let compiled: ([MLXArray]) -> [MLXArray]

    init(model: StreamingReuseModel, mode: StreamingFrameCompilation = .body) {
        precondition(mode != .disabled)
        precondition(!model.isFirstFrame, "Initialize caches with an eager first frame.")
        precondition(model.lookahead == 0 && model.useSingleEvaluation)
        let convolutionKeys = model.convolutionStates.keys.sorted()
        let recurrentKeys = model.recurrentStates.keys.sorted()
        precondition(!convolutionKeys.isEmpty && !recurrentKeys.isEmpty)
        self.model = model
        self.convolutionKeys = convolutionKeys
        self.recurrentKeys = recurrentKeys
        self.mode = mode
        compiled = compile(shapeless: false) { [unowned model] (arrays: [MLXArray]) -> [MLXArray] in
            Self.trace(arrays, model: model, convolutionKeys: convolutionKeys,
                       recurrentKeys: recurrentKeys, mode: mode)
        }
    }

    func callAsFunction(magnitude: MLXArray, phase: MLXArray) -> (MLXArray, MLXArray) {
        if mode == .body {
            let x = features(magnitude: magnitude, phase: phase)
            return model.decodeFrame(x, frequencies: magnitude.dim(0))
        }
        let (mag, phaseFeatures) = decoderFeatures(magnitude: magnitude, phase: phase)
        return (mag, model.decodePhase(phaseFeatures, frequencies: magnitude.dim(0)))
    }

    /// Keep the unchanged decoder outside compilation to preserve output bits.
    func features(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        precondition(mode == .body)
        return replay(magnitude: magnitude, phase: phase, outputs: 1)[0]
    }

    func decoderFeatures(magnitude: MLXArray, phase: MLXArray) -> (MLXArray, MLXArray) {
        precondition(mode == .decoderFeatures)
        let result = replay(magnitude: magnitude, phase: phase, outputs: 2)
        return (result[0], result[1])
    }

    private func replay(magnitude: MLXArray, phase: MLXArray, outputs: Int) -> [MLXArray] {
        precondition(!model.isFirstFrame)
        precondition(model.convolutionStates.count == convolutionKeys.count
                     && model.recurrentStates.count == recurrentKeys.count)
        let state = convolutionKeys.map { model.convolutionStates[$0]! }
            + recurrentKeys.map { model.recurrentStates[$0]! }
        let result = compiled([magnitude, phase] + state)
        Self.install(result, stateStart: outputs, model: model, convolutionKeys: convolutionKeys,
                     recurrentKeys: recurrentKeys)
        return result
    }

    /// Tracing temporarily borrows the normal frame implementation, then restores
    /// real model state. Cached calls only replace the graph's explicit inputs.
    private static func trace(_ arrays: [MLXArray], model: StreamingReuseModel,
                              convolutionKeys: [String], recurrentKeys: [String],
                              mode: StreamingFrameCompilation) -> [MLXArray] {
        let convolution = model.convolutionStates
        let recurrent = model.recurrentStates
        let firstFrame = model.isFirstFrame
        defer {
            model.convolutionStates = convolution
            model.recurrentStates = recurrent
            model.isFirstFrame = firstFrame
        }
        install(arrays, stateStart: 2, model: model, convolutionKeys: convolutionKeys,
                recurrentKeys: recurrentKeys)
        let x = model.frameFeatures(magnitude: arrays[0], phase: arrays[1])
        let output: [MLXArray]
        if mode == .decoderFeatures {
            let features = model.decodeFrameFeatures(x, frequencies: arrays[0].dim(0))
            output = [features.0, features.1]
        } else {
            output = [x]
        }
        precondition(model.convolutionStates.keys.sorted() == convolutionKeys
                     && model.recurrentStates.keys.sorted() == recurrentKeys)
        return output + convolutionKeys.map { model.convolutionStates[$0]! }
            + recurrentKeys.map { model.recurrentStates[$0]! }
    }

    private static func install(_ arrays: [MLXArray], stateStart: Int, model: StreamingReuseModel,
                                convolutionKeys: [String], recurrentKeys: [String]) {
        precondition(arrays.count == stateStart + convolutionKeys.count + recurrentKeys.count)
        let recurrentStart = stateStart + convolutionKeys.count
        model.convolutionStates = Dictionary(uniqueKeysWithValues:
            zip(convolutionKeys, arrays[stateStart..<recurrentStart]))
        model.recurrentStates = Dictionary(uniqueKeysWithValues:
            zip(recurrentKeys, arrays[recurrentStart...]))
    }
}
