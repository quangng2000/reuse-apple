import XCTest
@testable import ReuseStreaming

/// Opt-in numerical checks: weights stay outside Git and ordinary unit tests.
final class StreamingEnhancerModelTests: XCTestCase {
    func testOptimizedOutputMatchesBaselineAcrossChunkBoundaries() throws {
        let input = (0..<16037).map { Float(sin(Double($0) * 0.13) * 0.08) }
        let baseline = try StreamingEnhancer(modelURL: checkpoint(), optimized: false)
        let optimized = try StreamingEnhancer(modelURL: checkpoint(), optimized: true)
        let expected = try render(input, engine: baseline, chunkSize: 320)
        let actual = try render(input, engine: optimized, chunkSize: 197)
        XCTAssertEqual(actual.count, input.count)
        XCTAssertTrue(actual.allSatisfy({ $0.isFinite }))
        XCTAssertLessThan(zip(expected, actual).map { abs($0 - $1) }.max() ?? 1, 0.00001)
    }

    func testResetReproducesFreshOutputAndRejectsMidstreamWarmup() throws {
        let engine = try StreamingEnhancer(modelURL: checkpoint())
        let input = (0..<961).map { Float(cos(Double($0) * 0.07) * 0.05) }
        try engine.warmUp()
        XCTAssertTrue(engine.lastFrameMilliseconds.isEmpty)
        let first = try render(input, engine: engine, chunkSize: 320)
        engine.reset()
        let second = try render(input, engine: engine, chunkSize: 640)
        XCTAssertEqual(first, second)
        engine.reset()
        _ = try engine.push(Array(input.prefix(640)))
        XCTAssertEqual(engine.lastFrameMilliseconds.count, 2)
        XCTAssertThrowsError(try engine.warmUp()) {
            XCTAssertEqual($0 as? StreamingError, .warmupRequiresReset)
        }
    }

    private func render(_ input: [Float], engine: StreamingEnhancer, chunkSize: Int) throws -> [Float] {
        var output = [Float]()
        for start in stride(from: 0, to: input.count, by: chunkSize) {
            output += try engine.push(Array(input[start..<min(start + chunkSize, input.count)]))
        }
        output += try engine.finish()
        return output
    }

    private func checkpoint() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["REUSE_STREAMING_CHECKPOINT"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Set REUSE_STREAMING_CHECKPOINT to the verified Real-time RE-USE safetensors file.")
        }
        return URL(fileURLWithPath: path)
    }
}
