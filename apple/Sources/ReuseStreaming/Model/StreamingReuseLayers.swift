// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

extension StreamingReuseModel {
    func convolution(_ x: MLXArray, _ prefix: String, stride: IntOrPair = 1,
                     padding: IntOrPair = 0, dilation: IntOrPair = 1) -> MLXArray {
        conv2d(x, w(prefix + ".weight"), stride: stride, padding: padding, dilation: dilation)
            + w(prefix + ".bias")
    }

    /// NHWC causal convolution. A separate cache belongs to every layer.
    /// `transposeKernel` expresses the decoder's F/T transposition in weights,
    /// preserving temporal cache layout and avoiding an activation transpose.
    func causalConvolution(_ x: MLXArray, _ prefix: String, dilation: Int = 1,
                           initialPadding: Int? = nil, transposeKernel: Bool = false) -> MLXArray {
        var kernel = w(prefix + ".weight")
        if transposeKernel { kernel = kernel.transposed(0, 2, 1, 3) }
        let cacheLength = (kernel.dim(1) - 1) * dilation
        let frequencyPadding = (kernel.dim(2) - 1) / 2
        let previous = convolutionStates[prefix] ?? MLXArray.zeros(
            [x.dim(0), initialPadding ?? cacheLength, x.dim(2), x.dim(3)])
        let joined = concatenated([previous, x], axis: 1)
        let y = conv2d(joined, kernel, padding: .init((0, frequencyPadding)),
                       dilation: .init((dilation, 1))) + w(prefix + ".bias")
        if cacheLength > 0 {
            convolutionStates[prefix] = joined[0..., (joined.dim(1) - cacheLength)..., 0..., 0...]
        }
        return y
    }

    func dense(_ input: MLXArray, _ prefix: String) -> MLXArray {
        var skip = input
        var x = input
        for index in 0..<4 {
            let block = "\(prefix).dense_block.\(index)"
            x = causalConvolution(skip, block + ".1", dilation: 1 << index)
            x = normalizedPrelu(x, norm: block + ".2.norm", activation: block + ".3")
            if index < 3 { skip = concatenated([x, skip], axis: 3) }
        }
        return x
    }

    func decoder(_ input: MLXArray, _ prefix: String) -> MLXArray {
        var x = dense(input, prefix + ".dense_block")
        // Subpixel convolution expands frequency only. PyTorch ordering is
        // [subpixel, output channel], so NHWC can directly reshape F*2.
        x = padded(x, widths: [0, 0, 1, 0])
        x = convolution(x, prefix + ".up_conv1.0.conv")
        x = x.reshaped(x.dim(0), x.dim(1), x.dim(2) * 2, 64)
        x = normalizedPrelu(x, norm: prefix + ".up_conv1.1.norm", activation: prefix + ".up_conv1.2")
        x = causalConvolution(x, prefix + ".up_conv2.0.conv", transposeKernel: true)
        x = normalizedPrelu(x, norm: prefix + ".up_conv2.1.norm", activation: prefix + ".up_conv2.2")
        return x
    }
}
