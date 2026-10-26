// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

struct StreamingNormalizationWeights {
    let weight: MLXArray
    let bias: MLXArray
}

/// Experimental FP32 normalization for rows of exactly 64 channels.
/// The SIMD reduction changes floating-point summation order; validate parity
/// before enabling. Inputs are made row contiguous by the custom-kernel API.
final class StreamingNormalizationFusion {
    private let kernel = MLXFast.metalKernel(
        name: "reuse_normalization64_fused", inputNames: ["x", "weight", "bias", "other"],
        outputNames: ["out"], source: StreamingNormalizationSource.body,
        header: "#include <metal_simdgroup>\n", ensureRowContiguous: true)

    /// Applies layer normalization, learned affine parameters, and PReLU.
    func prelu(_ input: MLXArray, normalization: StreamingNormalizationWeights,
               slope: MLXArray) -> MLXArray {
        validate(input, normalization: normalization)
        precondition(slope.shape == [64] && slope.dtype == .float32)
        return run([input, normalization.weight, normalization.bias, slope], prelu: true)
    }

    /// Adds the residual after normalization and its learned affine transform.
    func residual(_ input: MLXArray, normalization: StreamingNormalizationWeights,
                  residual: MLXArray) -> MLXArray {
        validate(input, normalization: normalization)
        precondition(residual.shape == input.shape && residual.dtype == .float32)
        return run([input, normalization.weight, normalization.bias, residual], prelu: false)
    }

    private func run(_ inputs: [MLXArray], prelu: Bool) -> MLXArray {
        let input = inputs[0]
        return kernel(inputs, template: [("PRELU", prelu)],
            grid: (32, input.size / 64, 1), threadGroup: (32, 1, 1),
            outputShapes: [input.shape], outputDTypes: [.float32])[0]
    }

    private func validate(_ input: MLXArray, normalization: StreamingNormalizationWeights) {
        precondition(input.ndim >= 2 && input.shape.last == 64 && input.size > 0)
        precondition(input.dtype == .float32)
        precondition(normalization.weight.shape == [64] && normalization.bias.shape == [64])
        precondition(normalization.weight.dtype == .float32 && normalization.bias.dtype == .float32)
    }
}
