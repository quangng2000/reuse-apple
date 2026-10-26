import XCTest
@testable import ReuseStreaming

final class StreamingMetalConfigurationTests: XCTestCase {
    func testUnsupportedSpecializedConfigurationsFailBeforeCheckpointAccess() {
        let missing = URL(fileURLWithPath: "/nonexistent/reuse-checkpoint.safetensors")
        for (layers, optimized) in [(4, true), (8, false), (12, false)] {
            XCTAssertThrowsError(try StreamingEnhancer(modelURL: missing, layers: layers,
                                                      optimized: optimized, optimization: .specializedMetal)) {
                XCTAssertEqual($0 as? StreamingError, .invalidOptimization)
            }
        }
    }

    func testSpecializedConfigurationSupportsOnlyValidatedDepthsAndZeroLookahead() throws {
        for layers in [8, 12] {
            try StreamingOptimization.specializedMetal.validate(layers: layers, optimized: true)
            XCTAssertThrowsError(try StreamingOptimization.specializedMetal.validate(
                layers: layers, optimized: true, lookahead: 1))
        }
        for layers in 3...12 { try StreamingOptimization.standard.validate(layers: layers, optimized: false) }
    }
}
