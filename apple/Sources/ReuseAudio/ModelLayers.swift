// Swift/MLX adaptation of NVIDIA-NemotronLabs-RE-USE, revision
// 31f82f2efad1095034492f236e8dac9239ba713b. Original model materials:
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
import MLX

extension ReuseModel {
    func instance(_ x: MLXArray, _ prefix: String) -> MLXArray {
        let channelMajor = x.transposed(0, 3, 1, 2).reshaped(x.dim(0), x.dim(3), -1).asType(.float32)
        let normalized = MLXFast.layerNorm(channelMajor, eps: 1e-5)
            .reshaped(x.dim(0), x.dim(3), x.dim(1), x.dim(2)).transposed(0, 2, 3, 1)
        let result = (normalized * w(prefix + ".weight").asType(.float32)
                + w(prefix + ".bias").asType(.float32)).asType(dtype)
        eval(result)
        return result
    }
    func prelu(_ x: MLXArray, _ prefix: String) -> MLXArray {
        preluKernel([x, w(prefix + ".weight")], grid: (x.size, 1, 1), threadGroup: (256, 1, 1),
                    outputShapes: [x.shape], outputDTypes: [x.dtype])[0]
    }
    func dense(_ input: MLXArray, _ prefix: String) -> MLXArray {
        var skip = input, x = input
        for i in 0..<4 {
            let block = "\(prefix).dense_block.\(i)"
            let dilation = 1 << i
            x = conv(skip, block + ".0", padding: .init((dilation, 1)), dilation: .init((dilation, 1)))
            x = prelu(instance(x, block + ".1"), block + ".2")
            eval(x)
            if i < 3 { skip = concatenated([x, skip], axis: 3); eval(skip) }
        }
        return x
    }
    func up(_ input: MLXArray, _ prefix: String, ratio: Int) -> MLXArray {
        let x = padded(input, widths: [0, 0, 1, 0])
        let y = conv(x, prefix + ".0.conv")
        // Channel ordering in the checkpoint is [subpixel, output channel].
        let shuffled = y.reshaped(y.dim(0), y.dim(1), y.dim(2) * ratio, 64)
        return prelu(instance(shuffled, prefix + ".1"), prefix + ".2")
    }
    func decoder(_ input: MLXArray, _ prefix: String) -> MLXArray {
        var x = dense(input, prefix + ".dense_block")
        x = up(x, prefix + ".up_conv1", ratio: 2)
        x = up(x.transposed(0, 2, 1, 3), prefix + ".up_conv2", ratio: 4).transposed(0, 2, 1, 3)
        return x
    }
}
