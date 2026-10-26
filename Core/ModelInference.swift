// Swift/MLX adaptation of NVIDIA-NemotronLabs-RE-USE, revision
// 31f82f2efad1095034492f236e8dac9239ba713b. Original model materials:
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
import MLX

extension ReuseModel {
    /// Inputs/outputs [frequency, frame]. Each block evaluates to bound the lazy graph.
    func callAsFunction(magnitude: MLXArray, phase: MLXArray,
                        progress: ((Int) throws -> Void)? = nil,
                        inspect: ((String, MLXArray) -> Void)? = nil) throws -> (MLXArray, MLXArray) {
        var x = encode(magnitude: magnitude, phase: phase)
        inspect?("encoder", x.transposed(0, 3, 1, 2))
        for i in 0..<30 {
            try progress?(i)
            x = timeFrequencyBlock(x, index: i)
            eval(x)
            if i == 0 || i == 29 { inspect?("block\(i+1)", x.transposed(0, 3, 1, 2)) }
        }
        let result = decode(x, frames: magnitude.dim(1), frequencies: magnitude.dim(0))
        try progress?(30)
        return result
    }
    private func encode(magnitude: MLXArray, phase: MLXArray) -> MLXArray {
        var x = stacked([magnitude.T, phase.T], axis: -1).expandedDimensions(axis: 0).asType(dtype)
        x = padded(x, widths: [0, .init((0, 2)), .init((0, 2)), 0])
        x = prelu(instance(conv(x, "dense_encoder.dense_conv_1.0"), "dense_encoder.dense_conv_1.1"), "dense_encoder.dense_conv_1.2")
        x = dense(x, "dense_encoder.dense_block")
        x = prelu(instance(conv(x, "dense_encoder.dense_conv_2.0", stride: .init((4, 2))), "dense_encoder.dense_conv_2.1"), "dense_encoder.dense_conv_2.2")
        eval(x)
        return x
    }
    private func timeFrequencyBlock(_ x: MLXArray, index: Int) -> MLXArray {
        let nt = x.dim(1), nf = x.dim(2)
        let tx = x.transposed(0, 2, 1, 3).reshaped(nf, nt, 64)
        let ty = bidirectionalResidual(tx, "TSMamba.\(index).time_mamba")
        let fx = ty.reshaped(1, nf, nt, 64).transposed(0, 2, 1, 3).reshaped(nt, nf, 64)
        return bidirectionalResidual(fx, "TSMamba.\(index).freq_mamba").reshaped(1, nt, nf, 64)
    }
    private func decode(_ x: MLXArray, frames: Int, frequencies: Int) -> (MLXArray, MLXArray) {
        let magnitude = conv(decoder(x, "mask_decoder"), "mask_decoder.final_conv")[0, 0..<frames, 0..<frequencies, 0].T.asType(.float32)
        let phaseFeatures = decoder(x, "phase_decoder")
        let real = conv(phaseFeatures, "phase_decoder.phase_conv_r")[0, 0..<frames, 0..<frequencies, 0].T.asType(.float32)
        let imag = conv(phaseFeatures, "phase_decoder.phase_conv_i")[0, 0..<frames, 0..<frequencies, 0].T.asType(.float32)
        let phase = atan2(imag, real)
        eval(magnitude, phase)
        return (magnitude, phase)
    }
}
