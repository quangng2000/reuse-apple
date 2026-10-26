// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

final class MambaProjectionScan {
    let temporal = MambaProjectionTemporal()
    let frequency = MambaProjectionFrequency()
    func step(_ inputs: [MLXArray], state: MLXArray) -> (MLXArray, MLXArray) {
        temporal(inputs, state: state)
    }
    func sequence(_ inputs: [MLXArray]) -> MLXArray { frequency(inputs) }
}
