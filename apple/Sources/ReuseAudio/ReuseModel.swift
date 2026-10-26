// Swift/MLX adaptation of NVIDIA-NemotronLabs-RE-USE, revision
// 31f82f2efad1095034492f236e8dac9239ba713b. Original model materials:
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
import Foundation
import CryptoKit
import MLX

/// Invalid input, unsupported checkpoint, or restoration failure.
public enum ReuseError: LocalizedError {
    case invalid(String)
    public var errorDescription: String? { if case .invalid(let s) = self { return s }; return nil }
}

/// Inference-only, batch-one RE-USE. Activations use NHWC / NLC layouts.
final class ReuseModel {
    var weights: [String: MLXArray]
    let scan = SelectiveScan()
    let specializedScan = OfflineSelectiveScan()
    let preparedSerialScan = OfflineSelectiveScan(execution: .serial)
    let normalizationResidual = OfflineNormalizationResidual()
    let optimization: ReuseOptimization
    let configuration: ReuseMetalConfiguration
    let dtype: DType
    let preluKernel = MLXFast.metalKernel(name: "reuse_prelu", inputNames: ["x", "weight"], outputNames: ["out"], source: """
        uint i = thread_position_in_grid.x;
        if (i < x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3]) {
            float v = float(x[i]);
            out[i] = v >= 0.0f ? v : v * float(weight[i % weight_shape[0]]);
        }
        """)

    init(url: URL, half: Bool = false, optimization: ReuseOptimization = .standard,
         configuration: ReuseMetalConfiguration = .combined) throws {
        guard !half || optimization == .standard else {
            throw ReuseError.invalid("Specialized Metal execution requires FP32 weights and activations.")
        }
        self.optimization = optimization
        self.configuration = configuration
        try Self.verifyCheckpoint(url)
        weights = try loadArrays(url: url)
        guard weights.count > 1000,
              weights["TSMamba.29.freq_mamba.backward_blocks.A_log"]?.shape == [256, 16],
              weights.values.reduce(0, { $0 + $1.size }) == 9_607_747 else {
            throw ReuseError.invalid("This is not the supported 30-layer commercial RE-USE checkpoint.")
        }
        dtype = half ? .float16 : .float32
        convertWeights()
        eval(Array(weights.values))
    }

    private static func verifyCheckpoint(_ url: URL) throws {
        let hash = SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe))
            .map { String(format: "%02x", $0) }.joined()
        guard hash == "06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1" else {
            throw ReuseError.invalid("Model integrity check failed. Restore the pinned commercial checkpoint.")
        }
    }

    private func convertWeights() {
        for (key, value) in weights {
            // PyTorch OIHW -> MLX OHWI; depthwise O1K -> OK1.
            var weight = value
            if value.ndim == 4 { weight = value.transposed(0, 2, 3, 1) }
            else if value.ndim == 3 { weight = value.transposed(0, 2, 1) }
            // Preserve state coefficients and tiny delta biases in full precision.
            let full = key.hasSuffix("A_log") || key.hasSuffix(".D") || key.hasSuffix("dt_proj.bias")
            weights[key] = weight.asType(full ? .float32 : dtype)
        }
    }

    func w(_ key: String) -> MLXArray { weights[key]! }
    func linear(_ x: MLXArray, _ prefix: String, bias: Bool = true) -> MLXArray {
        let y = matmul(x, w(prefix + ".weight").T)
        return bias && weights[prefix + ".bias"] != nil ? y + w(prefix + ".bias") : y
    }
}
