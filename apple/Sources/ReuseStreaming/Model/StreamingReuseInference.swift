// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import Foundation
import MLX

extension StreamingReuseModel {
    /// First call consumes lookahead + 1 frames, then exactly one frame per call.
    /// Magnitudes are log1p compressed; inputs and outputs are [frequency, time].
    /// Direct callers retain eager evaluation. The streaming frame wrapper may
    /// defer it only when synthesis also evaluates every updated state array.
    func callAsFunction(magnitude: MLXArray, phase: MLXArray,
                        deferEvaluation: Bool = false) -> (MLXArray, MLXArray) {
        let x = frameFeatures(magnitude: magnitude, phase: phase)
        let (mag, pha) = decodeFrame(x, frequencies: magnitude.dim(0))
        if !deferEvaluation { eval([mag, pha] + frameStateArrays) }
        isFirstFrame = false
        return (mag, pha)
    }

    /// A separate boundary permits exact checks of the compiled hidden features.
    func frameFeatures(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        precondition(magnitude.ndim == 2 && phase.shape == magnitude.shape)
        precondition(magnitude.dim(1) == (isFirstFrame ? lookahead + 1 : 1))
        var x = encode(magnitude, phase)
        for index in 0..<layers {
            x = timeFrequency(x, "TSMamba.\(index)")
            if !useSingleEvaluation { eval(x) }
        }
        return x
    }

    /// These caches include graph branches not necessarily needed by this
    /// frame's audio output. Materialize all of them before the next frame.
    var frameStateArrays: [MLXArray] {
        Array(convolutionStates.values) + Array(recurrentStates.values)
    }

    private func encode(_ magnitude: MLXArray, _ phase: MLXArray) -> MLXArray {
        var x = stacked([magnitude.T, phase.T], axis: -1).expandedDimensions(axis: 0)
        x = padded(x, widths: [0, 0, .init((0, 2)), 0])
        let initial = "dense_encoder.dense_conv_1_\(lookahead + 1)"
        x = causalConvolution(x, initial + ".0", initialPadding: 2 - lookahead)
        x = normalizedPrelu(x, norm: initial + ".1.norm", activation: initial + ".2")
        x = dense(x, "dense_encoder.dense_block")
        x = convolution(x, "dense_encoder.dense_conv_2.0", stride: .init((1, 2)))
        return normalizedPrelu(x, norm: "dense_encoder.dense_conv_2.1.norm", activation: "dense_encoder.dense_conv_2.2")
    }

    func decodeFrame(_ x: MLXArray, frequencies: Int) -> (MLXArray, MLXArray) {
        let (mag, phaseFeatures) = decodeFrameFeatures(x, frequencies: frequencies)
        return (mag, decodePhase(phaseFeatures, frequencies: frequencies))
    }

    func decodeFrameFeatures(_ x: MLXArray, frequencies: Int) -> (MLXArray, MLXArray) {
        let magPrefix = "mask_decoder_list.\(layers - 1)"
        let phasePrefix = "phase_decoder_list.\(layers - 1)"
        let magFeatures = decoder(x, magPrefix)
        let phaseFeatures = decoder(x, phasePrefix)
        let mag = convolution(magFeatures, magPrefix + ".final_conv")[0, 0, 0..<frequencies, 0]
            .expandedDimensions(axis: 1)
        return (mag, phaseFeatures)
    }

    func decodePhase(_ phaseFeatures: MLXArray, frequencies: Int) -> MLXArray {
        let phasePrefix = "phase_decoder_list.\(layers - 1)"
        let real = convolution(phaseFeatures, phasePrefix + ".phase_conv_r")
        let imaginary = convolution(phaseFeatures, phasePrefix + ".phase_conv_i")
        let pha = atan2(imaginary, real)[0, 0, 0..<frequencies, 0].expandedDimensions(axis: 1)
        return pha
    }
}
