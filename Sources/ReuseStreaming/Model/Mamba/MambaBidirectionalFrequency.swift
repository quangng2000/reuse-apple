// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

/// Pools the independent forward/backward frequency branches along the batch
/// axis for filter and scan only. Original per-branch GEMMs retain their exact
/// operands and shapes, avoiding a different batched-GEMM reduction strategy.
/// It stores evaluated constants only and never retains the owning model.
final class MambaBidirectionalFrequency {
    private let inputWeight: [MLXArray]
    private let inputBias: [MLXArray?]
    private let filterWeight: MLXArray
    private let filterBias: MLXArray
    private let projectionWeight: [MLXArray]
    private let transition: MLXArray
    private let skip: MLXArray
    private let deltaWeight: MLXArray
    private let deltaBias: MLXArray
    private let outputWeight: [MLXArray]
    private let outputBias: [MLXArray?]

    init(model: StreamingReuseModel, prefix: String) {
        let branches = [prefix + ".forward_blocks", prefix + ".backward_blocks"]
        func pack(_ suffix: String) -> MLXArray {
            stacked(branches.map { model.w($0 + suffix) }, axis: 0)
        }
        inputWeight = branches.map { model.w($0 + ".in_proj.weight").T }
        inputBias = branches.map { model.weights[$0 + ".in_proj.bias"] }
        filterWeight = pack(".conv1d.weight").reshaped(2, 256, 4)
        filterBias = pack(".conv1d.bias")
        projectionWeight = branches.map { model.w($0 + ".x_proj.weight").T }
        transition = stacked(branches.map {
            model.frequencyScan.matrix(model.w($0 + ".A_log"), key: $0)
        }, axis: 0)
        skip = pack(".D")
        deltaWeight = pack(".dt_proj.weight")
        deltaBias = pack(".dt_proj.bias")
        outputWeight = branches.map { model.w($0 + ".out_proj.weight").T }
        outputBias = branches.map { model.weights[$0 + ".out_proj.bias"] }
        var constants: [MLXArray] = [filterWeight, filterBias, transition, skip, deltaWeight, deltaBias]
        constants.append(contentsOf: inputWeight)
        constants.append(contentsOf: projectionWeight)
        constants.append(contentsOf: outputWeight)
        constants.append(contentsOf: inputBias.compactMap { $0 })
        constants.append(contentsOf: outputBias.compactMap { $0 })
        eval(constants)
    }

    private let filter = MLXFast.metalKernel(
        name: "reuse_mamba_frequency_filter_bidirectional_separate_gemm", inputNames: ["xf", "xb", "w", "bias"],
        outputNames: ["out", "gate"], source: """
        uint c = thread_position_in_grid.x;
        uint t = thread_position_in_grid.y;
        uint batch = thread_position_in_grid.z;
        uint channels = xf_shape[2] / 2;
        uint length = xf_shape[1];
        if (c >= channels || t >= length || batch >= 2) return;
        device const float* xz = batch == 0 ? xf : xb;
        uint wc = batch * channels + c;
        float v = 0.0f;
        for (int k = 0; k < 4; ++k) {
            int position = int(t) - 3 + k;
            if (position >= 0) v += xz[position * channels * 2 + c] * w[wc * 4 + k];
        }
        v += bias[wc];
        uint ix = (batch * length + t) * channels + c;
        out[ix] = v * (1.0f / (1.0f + exp(-v)));
        gate[ix] = xz[t * channels * 2 + channels + c];
        """)

    private let scan = MLXFast.metalKernel(
        name: "reuse_mamba_projected_frequency_bidirectional_separate_gemm", inputNames: ["u", "pf", "pb", "a", "d", "z", "w", "bias"],
        outputNames: ["out"], source: """
        uint lane = thread_position_in_grid.x;
        uint channel = lane / 16;
        uint component = lane % 16;
        uint batch = thread_position_in_grid.y;
        uint channels = u_shape[2];
        uint length = u_shape[1];
        if (channel >= channels || batch >= u_shape[0]) return;
        uint wc = batch * channels + channel;
        device const float* p = batch == 0 ? pf : pb;
        bool share_delta = threads_per_simdgroup >= 16;
        ushort leader = ushort(thread_index_in_simdgroup & ~15u);
        float state = 0.0f;
        float aa = a[wc * 16 + component];
        for (uint t = 0; t < length; ++t) {
            uint ix = (batch * length + t) * channels + channel;
            uint pi = t * 36;
            float dt = 0.0f;
            if (!share_delta || component == 0) {
                float projected = p[pi] * w[wc * 4];
                for (uint rank = 1; rank < 4; ++rank) projected += p[pi + rank] * w[wc * 4 + rank];
                float v = projected + bias[wc];
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
                out[ix] = (yy + d[wc] * uu) * (zz / (1.0f + exp(-zz)));
            }
        }
        """)

    func callAsFunction(_ input: MLXArray, reversedIndices: MLXArray) -> (MLXArray, MLXArray) {
        precondition(input.dim(0) == 1 && input.dim(2) == 64)
        let inputs = [input, take(input, reversedIndices, axis: 1)]
        let xz = (0..<2).map { branch -> MLXArray in
            let projected = matmul(inputs[branch], inputWeight[branch])
            if let bias = inputBias[branch] { return projected + bias }
            return projected
        }
        let shape = [2, input.dim(1), 256]
        let filtered = filter(xz + [filterWeight, filterBias],
            grid: (shape[2], shape[1], shape[0]), threadGroup: (128, 1, 1),
            outputShapes: [shape, shape], outputDTypes: [.float32, .float32])
        let projection = (0..<2).map { branch in
            matmul(filtered[0][branch..<(branch + 1), 0..., 0...], projectionWeight[branch])
        }
        let scanned = scan([filtered[0]] + projection + [transition, skip, filtered[1], deltaWeight, deltaBias],
            grid: (shape[2] * 16, shape[0], 1), threadGroup: (128, 1, 1),
            outputShapes: [shape], outputDTypes: [.float32])[0]
        let residual = (0..<2).map { branch -> MLXArray in
            var output = matmul(scanned[branch..<(branch + 1), 0..., 0...], outputWeight[branch])
            if let bias = outputBias[branch] { output = output + bias }
            return output + inputs[branch]
        }
        return (residual[0], take(residual[1], reversedIndices, axis: 1))
    }
}
