// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

/// Sixteen SIMD lanes cooperate on each channel instead of one mostly idle lane.
/// Time remains sequential; state arithmetic and model parameters stay FP32.
final class MambaFrequencyScan {
    private var constants: [String: MLXArray] = [:]
    private var reversal: [Int: MLXArray] = [:]
    private let kernel = MLXFast.metalKernel(
        name: "realtime_reuse_frequency_scan_simd16",
        inputNames: ["u", "delta", "a", "b", "c", "d", "z", "bias"],
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
            uint bc = (batch * length + t) * 16 + component;
            float v = delta[ix] + bias[channel];
            float dt = max(v, 0.0f) + log1p(exp(-abs(v)));
            float uu = u[ix];
            state = exp(dt * aa) * state + dt * b[bc] * uu;
            float yy = state * c[bc];
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
        precondition(inputs.count == 8 && inputs[2].shape == [inputs[0].dim(2), 16])
        let input = inputs[0]
        return kernel(inputs, grid: (input.dim(2) * 16, input.dim(0), 1),
            threadGroup: (128, 1, 1), outputShapes: [input.shape], outputDTypes: [.float32])[0]
    }

    func matrix(_ logarithm: MLXArray, key: String) -> MLXArray {
        if let result = constants[key] { return result }
        let result = -exp(logarithm)
        eval(result)
        constants[key] = result
        return result
    }

    func reversedIndices(_ count: Int) -> MLXArray {
        if let result = reversal[count] { return result }
        let result = MLXArray(Array((0..<count).reversed()).map(Int32.init))
        eval(result)
        reversal[count] = result
        return result
    }
}
