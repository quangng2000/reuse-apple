// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

final class MambaProjectionFrequency {
    private let kernel = MLXFast.metalKernel(
        name: "reuse_mamba_projected_frequency_shared_delta_gather", inputNames: ["u", "p", "a", "d", "z", "w", "bias"],
        outputNames: ["out"], source: """
        uint lane = thread_position_in_grid.x;
        uint channel = lane / 16;
        uint component = lane % 16;
        uint batch = thread_position_in_grid.y;
        uint channels = u_shape[2];
        uint length = u_shape[1];
        if (channel >= channels || batch >= u_shape[0]) return;
        // One 16-lane subgroup owns a channel's state. A SIMD group can
        // contain multiple channels, so share relative to component zero,
        // never relative to the first lane of the whole SIMD group.
        bool share_delta = threads_per_simdgroup >= 16;
        ushort leader = ushort(thread_index_in_simdgroup & ~15u);
        float state = 0.0f;
        float aa = a[channel * 16 + component];
        for (uint t = 0; t < length; ++t) {
            uint ix = (batch * length + t) * channels + channel;
            uint pi = (batch * length + t) * 36;
            float dt = 0.0f;
            if (!share_delta || component == 0) {
                float delta = p[pi] * w[channel * 4];
                for (uint rank = 1; rank < 4; ++rank) delta += p[pi + rank] * w[channel * 4 + rank];
                float v = delta + bias[channel];
                dt = max(v, 0.0f) + log1p(exp(-abs(v)));
            }
            if (share_delta) dt = simd_shuffle(dt, leader);
            float uu = u[ix];
            state = exp(dt * aa) * state + dt * p[pi + 4 + component] * uu;
            float yy = state * p[pi + 20 + component];
            for (uint offset = 8; offset > 0; offset >>= 1) {
                float other = simd_shuffle_down(yy, offset);
                if (component < offset) yy += other;
            }
            if (component == 0) {
                float zz = z[ix];
                out[ix] = (yy + d[channel] * uu) * (zz / (1.0f + exp(-zz)));
            }
        }
        """)
    func callAsFunction(_ inputs: [MLXArray]) -> MLXArray {
        let input = inputs[0]
        return kernel(inputs, grid: (input.dim(2) * 16, input.dim(0), 1),
            threadGroup: (128, 1, 1), outputShapes: [input.shape], outputDTypes: [.float32])[0]
    }
}
