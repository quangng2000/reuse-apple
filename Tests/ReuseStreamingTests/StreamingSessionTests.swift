import XCTest
@testable import ReuseStreaming

final class StreamingSessionTests: XCTestCase {
    private final class IdentityRenderer {
        var count = 0
        func render(_ frame: [Float]) -> [Float] {
            defer { count += 1 }
            return count == 0 ? [] : Array(frame.prefix(320))
        }
    }

    func testChunkedPushAndFinishPreserveEverySample() throws {
        for count in [640, 641, 959, 960, 64_037] {
            let input = (0..<count).map(Float.init)
            for chunkSize in [1, 80, 320, 641, 3200] {
                var session = StreamingSession()
                let renderer = IdentityRenderer()
                var output = [Float]()
                for start in stride(from: 0, to: input.count, by: chunkSize) {
                    output += try session.push(Array(input[start..<min(start + chunkSize, count)]), render: renderer.render)
                }
                output += try session.finish(render: renderer.render)
                XCTAssertEqual(output, input, "count=\(count), chunk=\(chunkSize)")
            }
        }
    }

    func testNonfiniteInputDoesNotAdvanceState() throws {
        var session = StreamingSession()
        let renderer = IdentityRenderer()
        XCTAssertThrowsError(try session.push([.nan], render: renderer.render)) {
            XCTAssertEqual($0 as? StreamingError, .nonfiniteInput)
        }
        let input = [Float](repeating: 1, count: 640)
        var output = try session.push(input, render: renderer.render)
        output += try session.finish(render: renderer.render)
        XCTAssertEqual(output, input)
    }

    func testShortFinishCanBeRetriedAfterMoreInput() throws {
        var session = StreamingSession()
        let renderer = IdentityRenderer()
        XCTAssertTrue(try session.push([Float](repeating: 1, count: 639), render: renderer.render).isEmpty)
        XCTAssertThrowsError(try session.finish(render: renderer.render)) {
            XCTAssertEqual($0 as? StreamingError, .tooShort)
        }
        let output = try session.push([1], render: renderer.render) + session.finish(render: renderer.render)
        XCTAssertEqual(output.count, 640)
    }

    func testFinishedStreamRejectsPushAndSecondFinish() throws {
        var session = StreamingSession()
        let renderer = IdentityRenderer()
        _ = try session.push([Float](repeating: 0, count: 640), render: renderer.render)
        _ = try session.finish(render: renderer.render)
        XCTAssertThrowsError(try session.push([], render: renderer.render)) {
            XCTAssertEqual($0 as? StreamingError, .finished)
        }
        XCTAssertThrowsError(try session.finish(render: renderer.render)) {
            XCTAssertEqual($0 as? StreamingError, .finished)
        }
    }

    func testInvalidOutputRequiresReset() throws {
        var session = StreamingSession()
        XCTAssertThrowsError(try session.push([Float](repeating: 0, count: 640), render: { _ in [.infinity] })) {
            XCTAssertEqual($0 as? StreamingError, .invalidOutput)
        }
        XCTAssertThrowsError(try session.push([], render: { _ in [] })) {
            XCTAssertEqual($0 as? StreamingError, .resetRequired)
        }
    }

    func testInvalidExitRejectedBeforeCheckpointAccess() {
        XCTAssertThrowsError(try StreamingEnhancer(modelURL: URL(fileURLWithPath: "/missing"), layers: 2)) {
            XCTAssertEqual($0 as? StreamingError, .invalidLayers(2))
        }
    }
}
