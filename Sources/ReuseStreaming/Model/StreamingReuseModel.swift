// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import Foundation
import MLX
import ReuseAudio

final class StreamingReuseModel {
    enum ModelError: LocalizedError {
        case invalid(String)
        var errorDescription: String? {
            switch self { case .invalid(let message): return message }
        }
    }
    let layers: Int
    let lookahead: Int
    var weights: [String: MLXArray]
    var convolutionStates: [String: MLXArray] = [:]
    var recurrentStates: [String: MLXArray] = [:]
    let frequencyScan = MambaFrequencyScan()
    let inputFilter = MambaInputFilter()
    let projectionScan = MambaProjectionScan()
    // Experimental batching stays internal and off until exactness and target
    // device latency pass. Constants are retained across stream resets.
    var useBatchedFrequency = false
    var bidirectionalFrequency: [String: MambaBidirectionalFrequency] = [:]
    let useParallelFrequencyScan: Bool
    let useFusedMambaFilter: Bool
    let useFusedMambaProjection: Bool
    let useSingleEvaluation: Bool
    let fullScan = SelectiveScan()
    let stepScan = StreamingReuseScan()
    var isFirstFrame = true
    let preluKernel = MLXFast.metalKernel(
        name: "realtime_reuse_prelu", inputNames: ["x", "weight"], outputNames: ["out"], source: """
        uint i = thread_position_in_grid.x;
        if (i < x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]) {
            float v = x[i];
            out[i] = v >= 0.0f ? v : v * weight[i % weight_shape[0]];
        }
        """)

    init(weightsURL: URL, layers: Int, lookahead: Int, optimized: Bool = true) throws {
        guard (3...12).contains(layers), (0...2).contains(lookahead) else {
            throw ModelError.invalid("Real-time RE-USE supports exits 3–12 and 0–2 lookahead frames.")
        }
        self.layers = layers
        self.lookahead = lookahead
        useParallelFrequencyScan = optimized
        useFusedMambaFilter = optimized
        useFusedMambaProjection = optimized
        useSingleEvaluation = optimized
        weights = try Self.loadCheckpoint(weightsURL, layers: layers, lookahead: lookahead)
    }

    func reset() {
        convolutionStates.removeAll()
        recurrentStates.removeAll()
        isFirstFrame = true
    }

    func w(_ key: String) -> MLXArray {
        guard let value = weights[key] else { preconditionFailure("Missing weight: \(key)") }
        return value
    }
    func linear(_ x: MLXArray, _ prefix: String, bias: Bool = true) -> MLXArray {
        let output = matmul(x, w(prefix + ".weight").T)
        return bias && weights[prefix + ".bias"] != nil ? output + w(prefix + ".bias") : output
    }
    func normalize(_ x: MLXArray, _ prefix: String) -> MLXArray {
        MLXFast.layerNorm(x, weight: w(prefix + ".weight"), bias: w(prefix + ".bias"), eps: 1e-5)
    }
    func prelu(_ x: MLXArray, _ prefix: String) -> MLXArray {
        preluKernel([x, w(prefix + ".weight")], grid: (x.size, 1, 1),
                    threadGroup: (256, 1, 1), outputShapes: [x.shape], outputDTypes: [.float32])[0]
    }

}
