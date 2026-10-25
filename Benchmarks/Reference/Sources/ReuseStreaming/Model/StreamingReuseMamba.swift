// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

extension StreamingReuseModel {
    private func reverseSequence(_ x: MLXArray) -> MLXArray {
        take(x, frequencyScan.reversedIndices(x.dim(1)), axis: 1)
    }

    private func stateSpace(_ x: MLXArray, _ z: MLXArray, _ prefix: String,
                            streaming: Bool) -> MLXArray {
        let projection = linear(x, prefix + ".x_proj", bias: false)
        if useFusedMambaProjection {
            return projectedScan(x, z: z, projection: projection, prefix: prefix, streaming: streaming)
        }
        let dt = linear(projection[0..., 0..., 0..<4], prefix + ".dt_proj", bias: false)
        let b = projection[0..., 0..., 4..<20]
        let c = projection[0..., 0..., 20..<36]
        let a = frequencyScan.matrix(w(prefix + ".A_log"), key: prefix)
        return scanStateSpace([x, dt, a, b, c, w(prefix + ".D"), z, w(prefix + ".dt_proj.bias")],
                              prefix: prefix, streaming: streaming)
    }

    private func scanStateSpace(_ inputs: [MLXArray], prefix: String, streaming: Bool) -> MLXArray {
        if streaming {
            let state = recurrentStates[prefix] ?? MLXArray.zeros([inputs[0].dim(0), 256, 16])
            let (output, nextState) = stepScan(inputs, state: state)
            recurrentStates[prefix] = nextState
            return output
        }
        if useParallelFrequencyScan { return frequencyScan(inputs) }
        return fullScan(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], inputs[7])
    }

    private func projectedScan(_ x: MLXArray, z: MLXArray, projection: MLXArray,
                               prefix: String, streaming: Bool) -> MLXArray {
        let a = frequencyScan.matrix(w(prefix + ".A_log"), key: prefix)
        let inputs = [x, projection, a, w(prefix + ".D"), z, w(prefix + ".dt_proj.weight"), w(prefix + ".dt_proj.bias")]
        if !streaming { return projectionScan.sequence(inputs) }
        let state = recurrentStates[prefix] ?? MLXArray.zeros([x.dim(0), 256, 16])
        let result = projectionScan.step(inputs, state: state)
        recurrentStates[prefix] = result.1
        return result.0
    }

    private func mamba(_ input: MLXArray, _ prefix: String, streaming: Bool) -> MLXArray {
        let xz = linear(input, prefix + ".in_proj")
        let (x, z) = filterMamba(xz, prefix: prefix, streaming: streaming)
        let y = stateSpace(x, z, prefix, streaming: streaming)
        return linear(y, prefix + ".out_proj")
    }

    func timeFrequency(_ input: MLXArray, _ prefix: String) -> MLXArray {
        precondition(input.dim(1) == 1)
        let frequencies = input.dim(2)
        let temporalInput = input.transposed(0, 2, 1, 3).reshaped(frequencies, 1, 64)
        let temporalOutput = temporalBlock(temporalInput, prefix + ".time_mamba")
        let frequencyInput = temporalOutput.reshaped(1, frequencies, 64)
        return frequencyBlock(frequencyInput, prefix + ".freq_mamba").reshaped(1, 1, frequencies, 64)
    }

    private func temporalBlock(_ input: MLXArray, _ prefix: String) -> MLXArray {
        let temporal = mamba(input, prefix + ".forward_blocks", streaming: true) + input
        let projected = linear(temporal, prefix + ".output_proj")
        return normalize(projected, prefix + ".norm") + input
    }

    private func frequencyBlock(_ input: MLXArray, _ prefix: String) -> MLXArray {
        let forward = mamba(input, prefix + ".forward_blocks", streaming: false) + input
        let reversed = reverseSequence(input)
        let backward = reverseSequence(mamba(reversed, prefix + ".backward_blocks", streaming: false) + reversed)
        let output = linear(concatenated([forward, backward], axis: 2), prefix + ".output_proj")
        return normalize(output, prefix + ".norm") + input
    }
}
