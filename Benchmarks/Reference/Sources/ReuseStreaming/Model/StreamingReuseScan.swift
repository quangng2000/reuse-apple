// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

final class StreamingReuseScan {
    private let kernel = MLXFast.metalKernel(
        name: "realtime_reuse_mamba_step_16",
        inputNames: ["u", "delta", "a", "b", "c", "d", "z", "bias", "state"],
        outputNames: ["out", "next_state"], source: """
        uint channel = thread_position_in_grid.x;
        uint batch = thread_position_in_grid.y;
        uint channels = u_shape[2];
        if (channel >= channels || batch >= u_shape[0]) return;
        uint ix = batch * channels + channel;
        uint bc = batch * 16;
        float v = delta[ix] + bias[channel];
        float dt = max(v, 0.0f) + log1p(exp(-abs(v)));
        float uu = u[ix];
        float yy = 0.0f;
        for (uint n = 0; n < 16; ++n) {
            uint si = ix * 16 + n;
            float updated = exp(dt * a[channel * 16 + n]) * state[si] + dt * b[bc+n] * uu;
            next_state[si] = updated;
            yy += updated * c[bc+n];
        }
        float zz = z[ix];
        out[ix] = (yy + d[channel] * uu) * (zz / (1.0f + exp(-zz)));
        """)

    /// Inputs follow the named Metal kernel arguments through `bias`.
    func callAsFunction(_ inputs: [MLXArray], state: MLXArray) -> (MLXArray, MLXArray) {
        precondition(inputs.count == 8)
        let input = inputs[0]
        precondition(input.dim(1) == 1 && state.shape == [input.dim(0), input.dim(2), 16])
        let result = kernel(inputs + [state],
                            grid: (input.dim(2), input.dim(0), 1), threadGroup: (128, 1, 1),
                            outputShapes: [input.shape, state.shape], outputDTypes: [.float32, .float32])
        return (result[0], result[1])
    }
}
