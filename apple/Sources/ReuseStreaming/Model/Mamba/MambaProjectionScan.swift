// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import MLX

final class MambaProjectionScan {
    let temporal = MambaProjectionTemporal()
    let frequency = MambaProjectionFrequency()
    private let parallelTemporal = MambaParallelTemporal()
    private let preparedFrequency = MambaPreparedFrequency()
    private let specialized: Bool

    init(specialized: Bool = false) { self.specialized = specialized }

    func step(_ inputs: [MLXArray], state: MLXArray) -> (MLXArray, MLXArray) {
        specialized ? parallelTemporal(inputs, state: state) : temporal(inputs, state: state)
    }
    func sequence(_ inputs: [MLXArray]) -> MLXArray {
        specialized ? preparedFrequency(inputs) : frequency(inputs)
    }
}
