import XCTest
@testable import ReuseStreaming
import ReuseStreamingReference

final class StreamingMetalEnhancerTests: XCTestCase {
    func testExit3PublicChunkFinishWarmupAndResetMatchFrozen() throws { try compare(layers: 3) }
    func testExit4PublicChunkFinishWarmupAndResetMatchFrozen() throws { try compare(layers: 4) }
    func testExit5PublicChunkFinishWarmupAndResetMatchFrozen() throws { try compare(layers: 5) }
    func testExit6PublicChunkFinishWarmupAndResetMatchFrozen() throws { try compare(layers: 6) }
    func testExit7PublicChunkFinishWarmupAndResetMatchFrozen() throws { try compare(layers: 7) }
    func testExit8PublicChunkFinishWarmupAndResetMatchFrozen() throws { try compare(layers: 8) }
    func testExit9PublicChunkFinishWarmupAndResetMatchFrozen() throws { try compare(layers: 9) }
    func testExit10PublicChunkFinishWarmupAndResetMatchFrozen() throws { try compare(layers: 10) }
    func testExit11PublicChunkFinishWarmupAndResetMatchFrozen() throws { try compare(layers: 11) }
    func testExit12PublicChunkFinishWarmupAndResetMatchFrozen() throws { try compare(layers: 12) }

    private func compare(layers: Int) throws {
        let checkpoint = try StreamingMetalTestSupport.checkpoint()
        let reference = try ReuseStreamingReference.StreamingEnhancer(modelURL: checkpoint, layers: layers,
                                                                      optimized: true)
        let specialized = try ReuseStreaming.StreamingEnhancer(modelURL: checkpoint, layers: layers,
                                                               optimization: .specializedMetal)
        XCTAssertEqual(specialized.optimization, .specializedMetal)
        try reference.warmUp()
        try specialized.warmUp()
        let input = StreamingMetalTestSupport.signal(1643)
        var expected = [Float]()
        for start in stride(from: 0, to: input.count, by: 320) {
            expected += try reference.push(Array(input[start..<min(start + 320, input.count)]))
        }
        expected += try reference.finish()
        let actual = try render(input, engine: specialized, chunkSize: 197)
        XCTAssertEqual(expected.count, input.count)
        XCTAssertEqual(actual.count, input.count)
        StreamingMetalTestSupport.assertNear(actual, expected)
        specialized.reset()
        let replay = try render(input, engine: specialized, chunkSize: 640)
        StreamingMetalTestSupport.assertExact(replay, actual)
        reference.reset()
        var referenceReplay = try reference.push(input)
        referenceReplay += try reference.finish()
        StreamingMetalTestSupport.assertExact(referenceReplay, expected)
        StreamingMetalTestSupport.assertNear(replay, referenceReplay)
    }

    private func render(_ input: [Float], engine: ReuseStreaming.StreamingEnhancer, chunkSize: Int) throws -> [Float] {
        var output = [Float]()
        for start in stride(from: 0, to: input.count, by: chunkSize) {
            output += try engine.push(Array(input[start..<min(start + chunkSize, input.count)]))
        }
        output += try engine.finish()
        return output
    }
}
