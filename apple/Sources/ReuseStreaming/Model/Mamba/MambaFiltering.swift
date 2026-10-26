// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

extension StreamingReuseModel {
    func filterMamba(_ xz: MLXArray, prefix: String, streaming: Bool) -> (MLXArray, MLXArray) {
        if !useFusedMambaFilter { return originalFilter(xz, prefix: prefix, streaming: streaming) }
        let weights = [w(prefix + ".conv1d.weight"), w(prefix + ".conv1d.bias")]
        if streaming {
            let key = prefix + ".conv1d"
            let old = convolutionStates[key] ?? MLXArray.zeros([xz.dim(0), 3, 256])
            let result = inputFilter.step(xz, state: old, weights: weights)
            convolutionStates[key] = result[2]
            return (result[0], result[1])
        }
        let result = inputFilter.sequence(xz, weights: weights)
        return (result[0], result[1])
    }

    private func originalFilter(_ xz: MLXArray, prefix: String, streaming: Bool) -> (MLXArray, MLXArray) {
        let xx = xz[0..., 0..., 0..<256], z = xz[0..., 0..., 256..<512]
        var x: MLXArray
        if streaming {
            let key = prefix + ".conv1d"
            let old = convolutionStates[key] ?? MLXArray.zeros([xz.dim(0), 3, 256])
            let joined = concatenated([old, xx], axis: 1)
            x = conv1d(joined, w(prefix + ".conv1d.weight"), groups: 256)
            convolutionStates[key] = joined[0..., 1..., 0...]
        } else {
            x = conv1d(xx, w(prefix + ".conv1d.weight"), padding: 3, groups: 256)
            x = x[0..., 0..<xz.dim(1), 0...]
        }
        x = x + w(prefix + ".conv1d.bias")
        return (x * sigmoid(x), z)
    }
}
