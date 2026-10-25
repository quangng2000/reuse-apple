// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import Foundation
import MLX

extension StreamingReuseModel {
    /// First call consumes lookahead + 1 frames, then exactly one frame per call.
    /// Magnitudes are log1p compressed; inputs and outputs are [frequency, time].
    func callAsFunction(magnitude: MLXArray, phase: MLXArray) -> (MLXArray, MLXArray) {
        precondition(magnitude.ndim == 2 && phase.shape == magnitude.shape)
        precondition(magnitude.dim(1) == (isFirstFrame ? lookahead + 1 : 1))
        var x = encode(magnitude, phase)
        for index in 0..<layers {
            x = timeFrequency(x, "TSMamba.\(index)")
            if !useSingleEvaluation { eval(x) }
        }
        let (mag, pha) = decode(x, frequencies: magnitude.dim(0))
        eval([mag, pha] + Array(convolutionStates.values) + Array(recurrentStates.values))
        isFirstFrame = false
        return (mag, pha)
    }

    private func encode(_ magnitude: MLXArray, _ phase: MLXArray) -> MLXArray {
        var x = stacked([magnitude.T, phase.T], axis: -1).expandedDimensions(axis: 0)
        x = padded(x, widths: [0, 0, .init((0, 2)), 0])
        let initial = "dense_encoder.dense_conv_1_\(lookahead + 1)"
        x = causalConvolution(x, initial + ".0", initialPadding: 2 - lookahead)
        x = prelu(normalize(x, initial + ".1.norm"), initial + ".2")
        x = dense(x, "dense_encoder.dense_block")
        x = convolution(x, "dense_encoder.dense_conv_2.0", stride: .init((1, 2)))
        return prelu(normalize(x, "dense_encoder.dense_conv_2.1.norm"), "dense_encoder.dense_conv_2.2")
    }

    private func decode(_ x: MLXArray, frequencies: Int) -> (MLXArray, MLXArray) {
        let magPrefix = "mask_decoder_list.\(layers - 1)"
        let phasePrefix = "phase_decoder_list.\(layers - 1)"
        let magFeatures = decoder(x, magPrefix)
        let phaseFeatures = decoder(x, phasePrefix)
        let mag = convolution(magFeatures, magPrefix + ".final_conv")[0, 0, 0..<frequencies, 0]
            .expandedDimensions(axis: 1)
        let real = convolution(phaseFeatures, phasePrefix + ".phase_conv_r")
        let imaginary = convolution(phaseFeatures, phasePrefix + ".phase_conv_i")
        let pha = atan2(imaginary, real)[0, 0, 0..<frequencies, 0].expandedDimensions(axis: 1)
        return (mag, pha)
    }
}
