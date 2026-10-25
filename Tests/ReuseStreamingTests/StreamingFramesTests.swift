import XCTest
@testable import ReuseStreaming

final class StreamingFramesTests: XCTestCase {
    func testPartitionsMatchIndependentReflectedFrames() {
        for count in [640, 641, 959, 960, 64_000, 64_037] {
            let input = (0..<count).map { Float($0 % 997) / 997 }
            for chunkSize in [80, 160, 320, 641, 1000, 3200] {
                XCTAssertEqual(frame(input, chunkSize: chunkSize), reference(input),
                               "count=\(count), chunk=\(chunkSize)")
            }
        }
    }

    func testFramesWaitForRealStartupSamples() {
        var source = StreamingFrames()
        XCTAssertTrue(source.push([Float](repeating: 1, count: 639)).isEmpty)
        XCTAssertEqual(source.push([1]).count, 2)
        XCTAssertTrue(source.push([Float](repeating: 2, count: 319)).isEmpty)
        XCTAssertEqual(source.push([2]).count, 1)
    }

    func testFutureInputCannotChangeAvailableFrames() {
        let prefix = (0..<2400).map(Float.init)
        var source = StreamingFrames()
        let available = source.push(prefix)
        let positive = frame(prefix + [Float](repeating: 7, count: 2000), chunkSize: 320)
        let negative = frame(prefix + [Float](repeating: -7, count: 2000), chunkSize: 641)
        XCTAssertEqual(available, Array(positive.prefix(available.count)))
        XCTAssertEqual(available, Array(negative.prefix(available.count)))
    }

    private func frame(_ input: [Float], chunkSize: Int) -> [[Float]] {
        var source = StreamingFrames()
        var frames = [[Float]]()
        for start in stride(from: 0, to: input.count, by: chunkSize) {
            frames += source.push(Array(input[start..<min(start + chunkSize, input.count)]))
        }
        return frames + source.finish()
    }

    private func reference(_ input: [Float]) -> [[Float]] {
        (0..<(1 + (input.count + 319) / 320)).map { frame in
            (0..<640).map { offset in
                var index = frame * 320 - 320 + offset
                while index < 0 || index >= input.count {
                    if index < 0 { index = -index }
                    if index >= input.count { index = 2 * input.count - 2 - index }
                }
                return input[index]
            }
        }
    }
}
