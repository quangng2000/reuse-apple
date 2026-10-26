import Foundation
import XCTest
import ReuseStreaming
import ReuseStreamingReference

/// Compare public optimized and explicit legacy paths with matching frozen baselines.
final class EightLayerReferenceTests: XCTestCase {
    func testEightLayerUnoptimizedPublicFallbackMatchesFrozenExactly() throws {
        let checkpoint = try checkpointURL()
        let reference = try ReuseStreamingReference.StreamingEnhancer(modelURL: checkpoint,
                                                                       layers: 8, optimized: false)
        let candidate = try ReuseStreaming.StreamingEnhancer(modelURL: checkpoint,
                                                              layers: 8, optimized: false)
        let audio = (0..<961).map { Float(sin(Double($0) * 0.031) * 0.08 + cos(Double($0) * 0.107) * 0.02) }
        var expected = [Float](), actual = [Float]()
        for start in stride(from: 0, to: audio.count, by: 320) {
            expected += try reference.push(Array(audio[start..<min(start + 320, audio.count)]))
        }
        expected += try reference.finish()
        for start in stride(from: 0, to: audio.count, by: 197) {
            actual += try candidate.push(Array(audio[start..<min(start + 197, audio.count)]))
        }
        actual += try candidate.finish()
        XCTAssertEqual(expected.count, audio.count)
        XCTAssertEqual(actual.count, audio.count)
        XCTAssertTrue(actual.allSatisfy(\.isFinite))
        XCTAssertEqual(expected.map(\.bitPattern), actual.map(\.bitPattern))
    }

    func testEightLayersPreserveOutputAcrossChunkBoundariesAndReset() throws {
        let checkpoint = try checkpointURL()
        let reference = try ReuseStreamingReference.StreamingEnhancer(modelURL: checkpoint, layers: 8)
        let candidate = try ReuseStreaming.StreamingEnhancer(modelURL: checkpoint, layers: 8)
        try reference.warmUp()
        try candidate.warmUp()
        let audio = (0..<16_037).map { Float(sin(Double($0) * 0.031) * 0.08 + cos(Double($0) * 0.107) * 0.02) }
        var expected = [Float]()
        for start in stride(from: 0, to: audio.count, by: 320) {
            expected += try reference.push(Array(audio[start..<min(start + 320, audio.count)]))
        }
        expected += try reference.finish()
        var actual = [Float]()
        for start in stride(from: 0, to: audio.count, by: 197) {
            actual += try candidate.push(Array(audio[start..<min(start + 197, audio.count)]))
        }
        actual += try candidate.finish()
        XCTAssertEqual(expected.count, audio.count)
        XCTAssertEqual(actual.count, audio.count)
        XCTAssertTrue(actual.allSatisfy { $0.isFinite })
        XCTAssertEqual(expected.map(\.bitPattern), actual.map(\.bitPattern))

        reference.reset()
        candidate.reset()
        let silence = [Float](repeating: 0, count: 961)
        let resetExpected = try reference.push(silence) + reference.finish()
        let resetActual = try candidate.push(silence) + candidate.finish()
        XCTAssertEqual(resetExpected.map(\.bitPattern), resetActual.map(\.bitPattern))
        XCTAssertEqual(resetActual.count, silence.count)
    }

    private func checkpointURL() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["REUSE_STREAMING_CHECKPOINT"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Set REUSE_STREAMING_CHECKPOINT for the eight-layer GPU comparison.")
        }
        return URL(fileURLWithPath: path)
    }
}
