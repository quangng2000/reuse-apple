// SPDX-License-Identifier: OpenMDW-1.1

/// Execution strategy for the unchanged FP32, bidirectional 30-layer model.
public enum ReuseOptimization: String, Sendable {
    /// Original implementation and the default for existing callers.
    case standard
    /// Experimental SIMD16 scan and normalization fusion; not a speed guarantee.
    /// Slower than standard in the recorded M4 Pro offline benchmark.
    case specializedMetal
}

/// Package-only controls for measuring the independent kernel changes.
package enum ReuseMetalConfiguration: Sendable, Equatable {
    case scanOnly
    case normalizationOnly
    case combined
    case preparedSerial
    case schedulingOnly

    var usesScan: Bool { self == .scanOnly || self == .combined || self == .preparedSerial }
    var usesNormalization: Bool { self == .normalizationOnly || self == .combined }
}
