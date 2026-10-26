// Independent fusion of the offline Mamba LayerNorm and its outer residual.
// SPDX-License-Identifier: OpenMDW-1.1
import MLX

/// Normalizes the 64 output channels of each Mamba position, then adds its input.
final class OfflineNormalizationResidual {
    private let kernel = MLXFast.metalKernel(
        name: "reuse_offline_normalization_residual64",
        inputNames: ["values", "scale", "offset", "residual"], outputNames: ["result"],
        source: """
            uint row = thread_position_in_grid.y;
            uint lane = thread_position_in_grid.x;
            uint first = row * 64 + lane, second = first + 32;
            float left = values[first], right = values[second];
            float mean = simd_sum(left + right) / 64.0f;
            float centeredLeft = left - mean, centeredRight = right - mean;
            float variance = simd_sum(centeredLeft * centeredLeft
                                   + centeredRight * centeredRight) / 64.0f;
            float inverseDeviation = precise::rsqrt(variance + 1e-5f);
            float normalizedLeft = centeredLeft * inverseDeviation;
            float normalizedRight = centeredRight * inverseDeviation;
            result[first] = (normalizedLeft * scale[lane] + offset[lane]) + residual[first];
            result[second] = (normalizedRight * scale[lane + 32] + offset[lane + 32]) + residual[second];
            """)

    func callAsFunction(_ values: MLXArray, affine: [MLXArray], residual: MLXArray) -> MLXArray {
        precondition(values.ndim > 0 && values.shape.last == 64 && values.size > 0)
        precondition(affine.count == 2 && affine.allSatisfy { $0.shape == [64] })
        precondition(values.shape == residual.shape)
        precondition(([values, residual] + affine).allSatisfy { $0.dtype == .float32 })
        return kernel([values, affine[0], affine[1], residual],
            grid: (32, values.size / 64, 1), threadGroup: (32, 1, 1),
            outputShapes: [values.shape], outputDTypes: [.float32])[0]
    }
}
