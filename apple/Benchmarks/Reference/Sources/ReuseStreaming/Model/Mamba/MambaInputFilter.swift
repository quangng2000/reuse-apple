// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

final class MambaInputFilter {
    private let temporal = MLXFast.metalKernel(
        name: "reuse_mamba_temporal_filter", inputNames: ["xz", "old", "w", "bias"],
        outputNames: ["out", "gate", "next"], source: """
        uint c = thread_position_in_grid.x;
        uint batch = thread_position_in_grid.y;
        uint channels = xz_shape[2] / 2;
        if (c >= channels || batch >= xz_shape[0]) return;
        uint ix = batch * channels + c;
        uint base = batch * 3 * channels + c;
        float latest = xz[batch * channels * 2 + c];
        float v = old[base] * w[c * 4];
        v += old[base + channels] * w[c * 4 + 1];
        v += old[base + 2 * channels] * w[c * 4 + 2];
        v += latest * w[c * 4 + 3];
        v += bias[c];
        out[ix] = v * (1.0f / (1.0f + exp(-v)));
        gate[ix] = xz[batch * channels * 2 + channels + c];
        next[base] = old[base + channels];
        next[base + channels] = old[base + 2 * channels];
        next[base + 2 * channels] = latest;
        """)
    private let frequency = MLXFast.metalKernel(
        name: "reuse_mamba_frequency_filter", inputNames: ["xz", "w", "bias"],
        outputNames: ["out", "gate"], source: """
        uint c = thread_position_in_grid.x;
        uint t = thread_position_in_grid.y;
        uint batch = thread_position_in_grid.z;
        uint channels = xz_shape[2] / 2;
        uint length = xz_shape[1];
        if (c >= channels || t >= length || batch >= xz_shape[0]) return;
        float v = 0.0f;
        for (int k = 0; k < 4; ++k) {
            int position = int(t) - 3 + k;
            if (position >= 0) v += xz[(batch * length + position) * channels * 2 + c] * w[c * 4 + k];
        }
        v += bias[c];
        uint ix = (batch * length + t) * channels + c;
        out[ix] = v * (1.0f / (1.0f + exp(-v)));
        gate[ix] = xz[(batch * length + t) * channels * 2 + channels + c];
        """)

    func step(_ xz: MLXArray, state: MLXArray, weights: [MLXArray]) -> [MLXArray] {
        let shape = [xz.dim(0), 1, xz.dim(2) / 2]
        return temporal([xz, state] + weights, grid: (shape[2], shape[0], 1),
            threadGroup: (128, 1, 1), outputShapes: [shape, shape, state.shape],
            outputDTypes: [.float32, .float32, .float32])
    }

    func sequence(_ xz: MLXArray, weights: [MLXArray]) -> [MLXArray] {
        let shape = [xz.dim(0), xz.dim(1), xz.dim(2) / 2]
        return frequency([xz] + weights, grid: (shape[2], shape[1], shape[0]),
            threadGroup: (128, 1, 1), outputShapes: [shape, shape], outputDTypes: [.float32, .float32])
    }
}
