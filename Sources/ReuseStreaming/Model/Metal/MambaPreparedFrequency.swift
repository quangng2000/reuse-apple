// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

/// Compute channel-wise softplus and gate once rather than in all 16 state lanes.
/// The frequency recurrence and its reduction order are unchanged.
final class MambaPreparedFrequency {
    private let prepare = MLXFast.metalKernel(
        name: "reuse_frequency_delta_gate", inputNames: ["u", "p", "z", "w", "bias"],
        outputNames: ["delta", "gate"], source: """
        uint i = thread_position_in_grid.x;
        uint count = u_shape[0] * u_shape[1] * u_shape[2];
        if (i >= count) return;
        uint channel = i % u_shape[2];
        uint pi = (i / u_shape[2]) * 36;
        float v = p[pi] * w[channel * 4];
        for (uint rank = 1; rank < 4; ++rank) v += p[pi + rank] * w[channel * 4 + rank];
        v += bias[channel];
        delta[i] = max(v, 0.0f) + log1p(exp(-abs(v)));
        float zz = z[i];
        gate[i] = zz / (1.0f + exp(-zz));
        """)
    private let scan = MLXFast.metalKernel(
        name: "reuse_prepared_frequency_scan", inputNames: ["u", "p", "a", "d", "delta", "gate"],
        outputNames: ["out"], source: """
        uint lane = thread_position_in_grid.x;
        uint channel = lane / 16;
        uint component = lane % 16;
        uint batch = thread_position_in_grid.y;
        uint channels = u_shape[2];
        uint length = u_shape[1];
        if (channel >= channels || batch >= u_shape[0]) return;
        float state = 0.0f;
        float aa = a[channel * 16 + component];
        for (uint t = 0; t < length; ++t) {
            uint ix = (batch * length + t) * channels + channel;
            uint pi = (batch * length + t) * 36;
            float dt = delta[ix];
            float uu = u[ix];
            state = exp(dt * aa) * state + dt * p[pi + 4 + component] * uu;
            float yy = state * p[pi + 20 + component];
            for (uint offset = 8; offset > 0; offset >>= 1) {
                float other = simd_shuffle_down(yy, offset);
                if (component < offset) yy += other;
            }
            if (component == 0) out[ix] = (yy + d[channel] * uu) * gate[ix];
        }
        """)

    func callAsFunction(_ inputs: [MLXArray]) -> MLXArray {
        precondition(inputs.count == 7 && inputs.allSatisfy { $0.dtype == .float32 })
        let input = inputs[0]
        precondition(input.ndim == 3 && input.dim(2) == 256 && inputs[1].dim(2) == 36)
        let prepared = prepare([input, inputs[1], inputs[4], inputs[5], inputs[6]],
            grid: (input.size, 1, 1), threadGroup: (128, 1, 1),
            outputShapes: [input.shape, input.shape], outputDTypes: [.float32, .float32])
        return scan(Array(inputs.prefix(4)) + prepared, grid: (input.dim(2) * 16, input.dim(0), 1),
            threadGroup: (128, 1, 1), outputShapes: [input.shape], outputDTypes: [.float32])[0]
    }
}
