// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

extension StreamingReuseModel {
    func normalizedPrelu(_ input: MLXArray, norm: String, activation: String) -> MLXArray {
        guard useSpecializedMetal else { return prelu(normalize(input, norm), activation) }
        return normalizationFusion.prelu(input,
            normalization: .init(weight: w(norm + ".weight"), bias: w(norm + ".bias")),
            slope: w(activation + ".weight"))
    }

    func normalizedResidual(_ input: MLXArray, norm: String, residual: MLXArray) -> MLXArray {
        guard useSpecializedMetal else { return normalize(input, norm) + residual }
        return normalizationFusion.residual(input,
            normalization: .init(weight: w(norm + ".weight"), bias: w(norm + ".bias")),
            residual: residual)
    }
}
