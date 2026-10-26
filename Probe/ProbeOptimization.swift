import Foundation
import ReuseAudio

enum ProbeOptimization: String {
    case standard
    case specializedMetal = "specialized-metal"
    case scanOnly = "scan-only"
    case normalizationOnly = "normalization-only"
    case preparedSerial = "prepared-serial"
    case schedulingOnly = "scheduling-only"

    static func parse(_ value: String) throws -> Self {
        guard let mode = Self(rawValue: value) else {
            throw ReuseError.invalid("Unknown optimization: \(value). See --help for supported modes.")
        }
        return mode
    }

    func load(modelURL: URL) throws -> ReuseEngine {
        switch self {
        case .standard: return try ReuseEngine(modelURL: modelURL)
        case .specializedMetal: return try ReuseEngine(modelURL: modelURL, optimization: .specializedMetal)
        case .scanOnly: return try ReuseEngine(modelURL: modelURL, configuration: .scanOnly)
        case .normalizationOnly: return try ReuseEngine(modelURL: modelURL, configuration: .normalizationOnly)
        case .preparedSerial: return try ReuseEngine(modelURL: modelURL, configuration: .preparedSerial)
        case .schedulingOnly: return try ReuseEngine(modelURL: modelURL, configuration: .schedulingOnly)
        }
    }
}
