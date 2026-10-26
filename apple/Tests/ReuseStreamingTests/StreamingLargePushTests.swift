import XCTest
@testable import ReuseStreaming

final class StreamingLargePushTests: XCTestCase {
    private final class IdentityRenderer {
        var count = 0
        func render(_ frame: [Float]) -> [Float] {
            defer { count += 1 }
            return count == 0 ? [] : Array(frame.prefix(320))
        }
    }

    func testWholeMinutePushMatchesSmallChunksAndPreservesTail() throws {
        let input = (0..<960_037).map { Float($0 % 997) / 997 }
        let whole = try render(input, chunkSize: input.count)
        let chunked = try render(input, chunkSize: 197)
        XCTAssertEqual(whole, chunked)
        XCTAssertEqual(whole, input)
    }

    func testLateNonfiniteSampleRejectsWholePushBeforeRendering() throws {
        var session = StreamingSession()
        let renderer = IdentityRenderer()
        var invalid = [Float](repeating: 1, count: 64_000)
        invalid[invalid.count - 1] = .infinity
        XCTAssertThrowsError(try session.push(invalid, render: renderer.render)) {
            XCTAssertEqual($0 as? StreamingError, .nonfiniteInput)
        }
        XCTAssertEqual(renderer.count, 0)
        XCTAssertTrue(session.isFresh)
        let valid = [Float](repeating: 2, count: 640)
        let output = try session.push(valid, render: renderer.render) + session.finish(render: renderer.render)
        XCTAssertEqual(output, valid)
    }

    private func render(_ input: [Float], chunkSize: Int) throws -> [Float] {
        var session = StreamingSession()
        let renderer = IdentityRenderer()
        var output = [Float]()
        for start in stride(from: 0, to: input.count, by: chunkSize) {
            output += try session.push(Array(input[start..<min(start + chunkSize, input.count)]), render: renderer.render)
        }
        output += try session.finish(render: renderer.render)
        return output
    }
}
