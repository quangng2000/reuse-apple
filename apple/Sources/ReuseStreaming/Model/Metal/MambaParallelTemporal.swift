// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

/// Experimental SIMD16 temporal step with contiguous component loads/stores.
/// State equations are unchanged; the output reduction uses a different order.
final class MambaParallelTemporal {
    private let kernel = MLXFast.metalKernel(
        name: "reuse_mamba_projected_temporal_simd16",
        inputNames: ["u", "p", "a", "d", "z", "w", "bias", "state"],
        outputNames: ["out", "next_state"], source: """
        uint lane = thread_position_in_grid.x;
        uint channel = lane / 16;
        uint component = lane % 16;
        uint batch = thread_position_in_grid.y;
        uint channels = u_shape[2];
        uint ix = batch * channels + channel;
        uint pi = batch * 36;
        // Two independent channels share a SIMD32 group.
        ushort leader = ushort((lane % 32) - component);
        float dt = 0.0f;
        if (component == 0) {
            float delta = p[pi] * w[channel * 4];
            for (uint rank = 1; rank < 4; ++rank) delta += p[pi + rank] * w[channel * 4 + rank];
            float v = delta + bias[channel];
            dt = max(v, 0.0f) + log1p(exp(-abs(v)));
        }
        dt = simd_shuffle(dt, leader);
        float uu = u[ix];
        uint si = ix * 16 + component;
        float updated = exp(dt * a[channel * 16 + component]) * state[si]
            + dt * p[pi + 4 + component] * uu;
        next_state[si] = updated;
        float yy = updated * p[pi + 20 + component];
        for (uint offset = 8; offset > 0; offset >>= 1) {
            float other = simd_shuffle_down(yy, offset);
            if (component < offset) yy += other;
        }
        if (component == 0) {
            float zz = z[ix];
            out[ix] = (yy + d[channel] * uu) * (zz / (1.0f + exp(-zz)));
        }
        """, header: "#include <metal_simdgroup>\n", ensureRowContiguous: true)

    /// Matches MambaProjectionTemporal's seven inputs and explicit FP32 state.
    func callAsFunction(_ inputs: [MLXArray], state: MLXArray) -> (MLXArray, MLXArray) {
        validate(inputs, state: state)
        let input = inputs[0]
        let output = kernel(inputs + [state], grid: (256 * 16, input.dim(0), 1),
            threadGroup: (128, 1, 1), outputShapes: [input.shape, state.shape],
            outputDTypes: [.float32, .float32])
        return (output[0], output[1])
    }

    private func validate(_ inputs: [MLXArray], state: MLXArray) {
        precondition(inputs.count == 7 && inputs.allSatisfy { $0.dtype == .float32 })
        let input = inputs[0]
        precondition(input.ndim == 3 && input.dim(0) > 0)
        precondition(input.dim(1) == 1 && input.dim(2) == 256)
        precondition(inputs[1].shape == [input.dim(0), 1, 36])
        precondition(inputs[2].shape == [256, 16] && inputs[3].shape == [256])
        precondition(inputs[4].shape == input.shape && inputs[5].shape == [256, 4])
        precondition(inputs[6].shape == [256])
        precondition(state.shape == [input.dim(0), 256, 16] && state.dtype == .float32)
    }
}
