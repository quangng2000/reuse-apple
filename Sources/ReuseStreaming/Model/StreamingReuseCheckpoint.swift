// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import Foundation
import CryptoKit
import MLX

extension StreamingReuseModel {
    static func loadCheckpoint(_ url: URL, layers: Int, lookahead: Int) throws -> [String: MLXArray] {
        try verifyCheckpoint(url)
        let all = try loadArrays(url: url)
        guard all["TSMamba.11.time_mamba.forward_blocks.A_log"]?.shape == [256, 16],
              all["dense_encoder.dense_conv_1_1.0.weight"]?.shape == [64, 2, 3, 3] else {
            throw ModelError.invalid("Unsupported Real-time RE-USE checkpoint layout.")
        }
        let selected = all.filter { needed($0.key, layers: layers, lookahead: lookahead) }
        let converted = selected.mapValues { convertWeight($0) }
        eval(Array(converted.values))
        return converted
    }

    private static func verifyCheckpoint(_ url: URL) throws {
        let digest = SHA256.hash(data: try Data(contentsOf: url, options: .mappedIfSafe))
            .map { String(format: "%02x", $0) }.joined()
        guard digest == "6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade" else {
            throw ModelError.invalid("Real-time RE-USE checkpoint integrity check failed.")
        }
    }

    private static func convertWeight(_ value: MLXArray) -> MLXArray {
        var weight = value
        if value.ndim == 4 { weight = value.transposed(0, 2, 3, 1) }
        else if value.ndim == 3 { weight = value.transposed(0, 2, 1) }
        return weight.asType(.float32)
    }

    private static func needed(_ key: String, layers: Int, lookahead: Int) -> Bool {
        if key.hasPrefix("dense_encoder.dense_conv_1_") {
            return key.hasPrefix("dense_encoder.dense_conv_1_\(lookahead + 1).")
        }
        if key.hasPrefix("dense_encoder.") { return true }
        if key.hasPrefix("mask_decoder_list.\(layers - 1).") ||
           key.hasPrefix("phase_decoder_list.\(layers - 1).") { return true }
        if key.hasPrefix("TSMamba.") {
            let pieces = key.split(separator: ".")
            return pieces.count > 1 && (Int(pieces[1]) ?? 99) < layers
        }
        return false
    }
}
