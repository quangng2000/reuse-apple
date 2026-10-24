// Swift/MLX adaptation of NVIDIA-NemotronLabs-RE-USE, revision
// 31f82f2efad1095034492f236e8dac9239ba713b. Original model materials:
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Mamba reference: Copyright (c) 2023 Tri Dao, Albert Gu (Apache-2.0).
// SPDX-License-Identifier: OpenMDW-1.1
import MLX

extension ReuseModel {
    private func reversed(_ x: MLXArray) -> MLXArray {
        take(x, MLXArray(Array((0..<x.dim(1)).reversed()).map(Int32.init)), axis: 1)
    }
    private func mamba(_ input: MLXArray, _ prefix: String) -> MLXArray {
        let xz = linear(input, prefix + ".in_proj")
        let length = xz.dim(1)
        let xx = xz[0..., 0..., 0..<256]
        let z = xz[0..., 0..., 256..<512]
        var x = conv1d(xx, w(prefix + ".conv1d.weight"), padding: 3, groups: 256)
        x = x[0..., 0..<length, 0...] + w(prefix + ".conv1d.bias")
        x = x * sigmoid(x)
        eval(x, z)
        let y = scanMamba(x, z, prefix)
        eval(y)
        let result = linear(y, prefix + ".out_proj")
        eval(result)
        return result
    }
    private func scanMamba(_ x: MLXArray, _ z: MLXArray, _ prefix: String) -> MLXArray {
        let projection = linear(x, prefix + ".x_proj", bias: false)
        let dt = linear(projection[0..., 0..., 0..<4], prefix + ".dt_proj", bias: false)
        let b = projection[0..., 0..., 4..<20]
        let c = projection[0..., 0..., 20..<36]
        let a = -exp(w(prefix + ".A_log"))
        return scan(x, dt, a, b, c, w(prefix + ".D"), z, w(prefix + ".dt_proj.bias"))
    }
    func bidirectional(_ x: MLXArray, _ prefix: String) -> MLXArray {
        let fw = mamba(x, prefix + ".forward_blocks") + x
        eval(fw)
        let rx = reversed(x)
        let bw = reversed(mamba(rx, prefix + ".backward_blocks") + rx)
        let output = linear(concatenated([fw, bw], axis: 2), prefix + ".output_proj")
        return MLXFast.layerNorm(output, weight: w(prefix + ".norm.weight"),
                                 bias: w(prefix + ".norm.bias"), eps: 1e-5)
    }
}
