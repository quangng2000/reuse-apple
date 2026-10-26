// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

final class MambaProjectionTemporal {
    private let kernel = MLXFast.metalKernel(
        name: "reuse_mamba_projected_temporal", inputNames: ["u", "p", "a", "d", "z", "w", "bias", "state"],
        outputNames: ["out", "next_state"], source: """
        uint channel = thread_position_in_grid.x;
        uint batch = thread_position_in_grid.y;
        uint channels = u_shape[2];
        if (channel >= channels || batch >= u_shape[0]) return;
        uint ix = batch * channels + channel;
        uint pi = batch * 36;
        float delta = p[pi] * w[channel * 4];
        for (uint rank = 1; rank < 4; ++rank) delta += p[pi + rank] * w[channel * 4 + rank];
        float v = delta + bias[channel];
        float dt = max(v, 0.0f) + log1p(exp(-abs(v)));
        float uu = u[ix];
        float yy = 0.0f;
        for (uint n = 0; n < 16; ++n) {
            uint si = ix * 16 + n;
            float updated = exp(dt * a[channel * 16 + n]) * state[si] + dt * p[pi + 4 + n] * uu;
            next_state[si] = updated;
            yy += updated * p[pi + 20 + n];
        }
        float zz = z[ix];
        out[ix] = (yy + d[channel] * uu) * (zz / (1.0f + exp(-zz)));
        """)
    func callAsFunction(_ inputs: [MLXArray], state: MLXArray) -> (MLXArray, MLXArray) {
        let input = inputs[0]
        let output = kernel(inputs + [state], grid: (input.dim(2), input.dim(0), 1),
            threadGroup: (128, 1, 1), outputShapes: [input.shape, state.shape], outputDTypes: [.float32, .float32])
        return (output[0], output[1])
    }
}
