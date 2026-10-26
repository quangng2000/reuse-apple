// Native Metal implementation of Mamba-1 recurrence, checked against the
// Apache-2.0 Mamba reference by Tri Dao and Albert Gu (Copyright 2023).
// Swift/Metal adaptation: SPDX-License-Identifier: OpenMDW-1.1
// See models/THIRD-PARTY-NOTICES.txt for upstream attribution.
import MLX

/// Mamba-1 recurrence: one Metal thread per batch/channel, 16 FP32 states.
/// It fuses delta softplus, recurrent state update, skip connection and SiLU gate.
/// No CUDA, Python, or approximation of the recurrence is used.
package final class SelectiveScan {
    package init() {}
    private let kernel = MLXFast.metalKernel(
        name: "reuse_selective_scan_16",
        inputNames: ["u", "delta", "a", "b", "c", "d", "z", "bias"],
        outputNames: ["out"],
        source: """
        uint channel = thread_position_in_grid.x;
        uint batch = thread_position_in_grid.y;
        uint channels = u_shape[2];
        uint length = u_shape[1];
        if (channel >= channels || batch >= u_shape[0]) return;
        float state[16];
        float aa[16];
        for (uint n = 0; n < 16; ++n) {
            state[n] = 0.0f;
            aa[n] = a[channel * 16 + n];
        }
        for (uint t = 0; t < length; ++t) {
            uint ix = (batch * length + t) * channels + channel;
            uint bc = (batch * length + t) * 16;
            float v = delta[ix] + bias[channel];
            float dt = max(v, 0.0f) + log1p(exp(-abs(v)));
            float uu = u[ix];
            float yy = 0.0f;
            for (uint n = 0; n < 16; ++n) {
                state[n] = exp(dt * aa[n]) * state[n] + dt * b[bc+n] * uu;
                yy += state[n] * c[bc+n];
            }
            float zz = z[ix];
            out[ix] = (yy + d[channel] * uu) * (zz / (1.0f + exp(-zz)));
        }
        """
    )

    package func callAsFunction(_ u: MLXArray, _ delta: MLXArray, _ a: MLXArray,
                        _ b: MLXArray, _ c: MLXArray, _ d: MLXArray,
                        _ z: MLXArray, _ bias: MLXArray) -> MLXArray {
        precondition(a.shape == [u.dim(2), 16] && b.dim(2) == 16 && c.dim(2) == 16)
        return kernel([u, delta, a, b, c, d, z, bias].map { $0.asType(.float32) },
                      grid: (u.dim(2), u.dim(0), 1), threadGroup: (128, 1, 1),
                      outputShapes: [u.shape], outputDTypes: [.float32])[0].asType(u.dtype)
    }
}
