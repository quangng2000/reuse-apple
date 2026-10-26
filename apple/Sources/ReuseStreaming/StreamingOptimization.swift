// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// See LICENSE in this module. Noncommercial research and education only.

/// Selects streaming execution without changing the checkpoint or layer count.
public enum StreamingOptimization: String, Sendable {
    /// Preserves the existing kernels and automatic eight-layer configuration.
    case standard
    /// Opt-in FP32 kernels for optimized eight- and twelve-layer streams.
    /// Changes reduction order, so output can differ by floating-point rounding.
    /// Requires Apple silicon with 32-lane SIMD groups; timing depends on device.
    case specializedMetal = "specialized_metal"

    func validate(layers: Int, optimized: Bool, lookahead: Int = 0) throws {
        guard self == .specializedMetal else { return }
        guard optimized, [8, 12].contains(layers), lookahead == 0 else {
            throw StreamingError.invalidOptimization
        }
    }
}
