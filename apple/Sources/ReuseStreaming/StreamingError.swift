// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// See LICENSE in this module. Noncommercial research and education only.
import Foundation

/// Validation and stream-lifecycle failures from the streaming API.
public enum StreamingError: LocalizedError, Equatable {
    case invalidLayers(Int)
    case invalidOptimization
    case nonfiniteInput
    case tooShort
    case finished
    case resetRequired
    case invalidOutput
    case warmupRequiresReset

    public var errorDescription: String? {
        switch self {
        case .invalidLayers(let count): return "Choose a streaming exit between 3 and 12 layers, not \(count)."
        case .invalidOptimization: return "Specialized Metal requires optimized eight- or twelve-layer streaming with zero lookahead."
        case .nonfiniteInput: return "Input must contain only finite 16 kHz mono samples."
        case .tooShort: return "Supply at least 640 samples (40 ms) before finishing."
        case .finished: return "The stream is finished. Reset it before sending more audio."
        case .resetRequired: return "Inference failed. Reset the stream before continuing."
        case .invalidOutput: return "Inference returned invalid or incomplete audio. Reset the stream."
        case .warmupRequiresReset: return "Warm up before sending audio, or reset the stream first."
        }
    }
}
