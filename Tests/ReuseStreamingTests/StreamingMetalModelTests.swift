import XCTest
import MLX
@testable import ReuseStreaming

final class StreamingMetalModelTests: XCTestCase {
    func testEightLayerCompiledAudioAndEveryStateMatchAcrossReset() throws { try compare(layers: 8) }
    func testTwelveLayerCompiledAudioAndEveryStateMatchAcrossReset() throws { try compare(layers: 12) }

    private func compare(layers: Int) throws {
        let checkpoint = try StreamingMetalTestSupport.checkpoint()
        try Device.withDefaultDevice(.gpu) {
            let models = try StreamingMetalModelPair(layers: layers, checkpoint: checkpoint)
            var previous = [Float]()
            for recording in 0..<2 {
                models.reset()
                var output = [Float]()
                for frame in 0..<4 {
                    let input = frame == 2 ? [Float](repeating: 0, count: 640)
                        : StreamingMetalTestSupport.signal(640, offset: frame * 320)
                    output += models.render(input)
                }
                if recording > 0 { StreamingMetalTestSupport.assertExact(output, previous) }
                previous = output
            }
        }
    }
}
