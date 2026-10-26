// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

enum StreamingNormalizationSource {
    static let body = """
        // One complete SIMD group owns a row; no shared-memory barriers.
        uint channel = thread_position_in_grid.x * 2;
        uint row = thread_position_in_grid.y;
        uint index = row * 64 + channel;
        float first = x[index];
        float second = x[index + 1];
        float mean = simd_sum(first + second) / 64.0f;
        // Match LayerNorm's centered, two-pass variance rather than E[x*x]-E[x]^2.
        float firstCentered = first - mean;
        float secondCentered = second - mean;
        float variance = simd_sum(firstCentered * firstCentered
            + secondCentered * secondCentered) / 64.0f;
        float inverseDeviation = metal::precise::rsqrt(variance + 1e-5f);
        float firstNormalized = firstCentered * inverseDeviation;
        float secondNormalized = secondCentered * inverseDeviation;
        float firstAffine = weight[channel] * firstNormalized + bias[channel];
        float secondAffine = weight[channel + 1] * secondNormalized + bias[channel + 1];
        if constexpr (PRELU) {
            out[index] = firstAffine >= 0.0f ? firstAffine : firstAffine * other[channel];
            out[index + 1] = secondAffine >= 0.0f ? secondAffine : secondAffine * other[channel + 1];
        } else {
            out[index] = firstAffine + other[index];
            out[index + 1] = secondAffine + other[index + 1];
        }
        """
}
