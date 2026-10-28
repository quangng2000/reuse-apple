import XCTest
import MLX
@testable import ReuseStreaming

final class StreamingMetalModelTests: XCTestCase {
    func testExit3CompiledAudioAndEveryStateMatchFrozenAcrossReset() throws { try compare(layers: 3) }
    func testExit4CompiledAudioAndEveryStateMatchFrozenAcrossReset() throws { try compare(layers: 4) }
    func testExit5CompiledAudioAndEveryStateMatchFrozenAcrossReset() throws { try compare(layers: 5) }
    func testExit6CompiledAudioAndEveryStateMatchFrozenAcrossReset() throws { try compare(layers: 6) }
    func testExit7CompiledAudioAndEveryStateMatchFrozenAcrossReset() throws { try compare(layers: 7) }
    func testExit8CompiledAudioAndEveryStateMatchFrozenAcrossReset() throws { try compare(layers: 8) }
    func testExit9CompiledAudioAndEveryStateMatchFrozenAcrossReset() throws { try compare(layers: 9) }
    func testExit10CompiledAudioAndEveryStateMatchFrozenAcrossReset() throws { try compare(layers: 10) }
    func testExit11CompiledAudioAndEveryStateMatchFrozenAcrossReset() throws { try compare(layers: 11) }
    func testExit12CompiledAudioAndEveryStateMatchFrozenAcrossReset() throws { try compare(layers: 12) }

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
