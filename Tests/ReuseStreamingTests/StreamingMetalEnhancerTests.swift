import XCTest
@testable import ReuseStreaming

final class StreamingMetalEnhancerTests: XCTestCase {
    func testPublicOptInPreservesSamplesChunkingWarmupAndResetAtBothDepths() throws {
        let checkpoint = try StreamingMetalTestSupport.checkpoint()
        for layers in [8, 12] { try compare(layers: layers, checkpoint: checkpoint) }
    }

    private func compare(layers: Int, checkpoint: URL) throws {
        let standard = try StreamingEnhancer(modelURL: checkpoint, layers: layers)
        let specialized = try StreamingEnhancer(modelURL: checkpoint, layers: layers,
                                                optimization: .specializedMetal)
        XCTAssertEqual(standard.optimization, .standard)
        XCTAssertEqual(specialized.optimization, .specializedMetal)
        try specialized.warmUp()
        let input = StreamingMetalTestSupport.signal(1643)
        let expected = try render(input, engine: standard, chunkSize: 320)
        let actual = try render(input, engine: specialized, chunkSize: 197)
        XCTAssertEqual(actual.count, input.count)
        StreamingMetalTestSupport.assertNear(actual, expected)
        specialized.reset()
        let replay = try render(input, engine: specialized, chunkSize: 640)
        StreamingMetalTestSupport.assertExact(replay, actual)
    }

    private func render(_ input: [Float], engine: StreamingEnhancer, chunkSize: Int) throws -> [Float] {
        var output = [Float]()
        for start in stride(from: 0, to: input.count, by: chunkSize) {
            output += try engine.push(Array(input[start..<min(start + chunkSize, input.count)]))
        }
        output += try engine.finish()
        return output
    }
}
