// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// See LICENSE in this module. Noncommercial research and education only.
import Foundation

extension StreamingEnhancer {
    /// Loads the pinned checkpoint. The standard configuration preserves existing
    /// defaults; specialized Metal is opt-in for optimized eight/twelve-layer streams.
    /// Calls such as `StreamingEnhancer(modelURL: url, layers: 8,
    /// optimization: .specializedMetal)` retain FP32 and zero lookahead.
    public convenience init(modelURL: URL, layers: Int = 4, optimized: Bool = true,
                            optimization: StreamingOptimization = .standard) throws {
        let specialized = optimization == .specializedMetal
        let compiled = optimized && (layers == 8 || specialized)
        try self.init(modelURL: modelURL, layers: layers, optimized: optimized,
                      compilation: compiled ? .decoderFeatures : .disabled,
                      pooledFrequency: optimized && layers == 8 && !specialized,
                      optimization: optimization)
    }
}
