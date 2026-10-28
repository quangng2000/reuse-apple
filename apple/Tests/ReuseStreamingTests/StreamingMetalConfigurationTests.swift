import XCTest
@testable import ReuseStreaming

final class StreamingMetalConfigurationTests: XCTestCase {
    func testUnsupportedSpecializedConfigurationsFailBeforeCheckpointAccess() {
        let missing = URL(fileURLWithPath: "/nonexistent/reuse-checkpoint.safetensors")
        for (layers, optimized) in [(2, true), (13, true)] + (3...12).map { ($0, false) } {
            XCTAssertThrowsError(try StreamingEnhancer(modelURL: missing, layers: layers,
                                                      optimized: optimized, optimization: .specializedMetal)) {
                XCTAssertEqual($0 as? StreamingError, (3...12).contains(layers) ? .invalidOptimization : .invalidLayers(layers))
            }
        }
    }

    func testSpecializedConfigurationSupportsEveryExitAndOnlyZeroLookahead() throws {
        for layers in 3...12 {
            try StreamingOptimization.specializedMetal.validate(layers: layers, optimized: true)
            XCTAssertThrowsError(try StreamingOptimization.specializedMetal.validate(
                layers: layers, optimized: true, lookahead: 1))
            XCTAssertThrowsError(try StreamingOptimization.specializedMetal.validate(
                layers: layers, optimized: true, lookahead: 2))
        }
        for layers in [Int.min, -1, 0, 1, 2, 13, Int.max] {
            XCTAssertThrowsError(try StreamingOptimization.specializedMetal.validate(layers: layers, optimized: true))
        }
        for layers in 3...12 { try StreamingOptimization.standard.validate(layers: layers, optimized: false) }
    }
}
