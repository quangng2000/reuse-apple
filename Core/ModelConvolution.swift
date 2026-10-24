// Swift/MLX adaptation of NVIDIA-NemotronLabs-RE-USE, revision
// 31f82f2efad1095034492f236e8dac9239ba713b. Original model materials:
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
import MLX

struct ConvolutionSettings {
    let stride: IntOrPair
    let padding: IntOrPair
    let dilation: IntOrPair
}

extension ReuseModel {
    func conv(_ x: MLXArray, _ prefix: String,
              stride: IntOrPair = 1, padding: IntOrPair = 0,
              dilation: IntOrPair = 1) -> MLXArray {
        let kernel = w(prefix + ".weight")
        if x.dim(1) > 96 && kernel.dim(1) > 1 {
            let settings = ConvolutionSettings(stride: stride, padding: padding, dilation: dilation)
            return tiledConvolution(x, kernel, prefix, settings)
        }
        return conv2d(x, kernel, stride: stride, padding: padding,
                      dilation: dilation) + w(prefix + ".bias")
    }

    private func tiledConvolution(_ x: MLXArray, _ kernel: MLXArray,
                                  _ prefix: String, _ settings: ConvolutionSettings) -> MLXArray {
        // Exact halos bound workspace without resetting context or normalization.
        let paddedX = padded(x, widths: [0, .init(settings.padding.first), .init(settings.padding.second), 0])
        eval(paddedX)
        let receptive = (kernel.dim(1)-1)*settings.dilation.first + 1
        let tiles = convolutionTiles(paddedX, kernel, prefix, settings, receptive)
        let result = concatenated(tiles, axis: 1)
        eval(result)
        return result
    }

    private func convolutionTiles(_ x: MLXArray, _ kernel: MLXArray, _ prefix: String,
                                  _ settings: ConvolutionSettings, _ receptive: Int) -> [MLXArray] {
        let count = (x.dim(1)-receptive)/settings.stride.first + 1
        var tiles = [MLXArray]()
        for start in Swift.stride(from: 0, to: count, by: 32) {
            let rows = min(32, count-start)
            let lo = start*settings.stride.first
            let hi = lo + (rows-1)*settings.stride.first + receptive
            let y = conv2d(x[0..., lo..<hi, 0..., 0...], kernel,
                           stride: settings.stride, dilation: settings.dilation) + w(prefix + ".bias")
            eval(y)
            tiles.append(y)
        }
        return tiles
    }
}
