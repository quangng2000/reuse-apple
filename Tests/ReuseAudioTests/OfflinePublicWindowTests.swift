// Public bounded-file integration contract. SPDX-License-Identifier: OpenMDW-1.1
import Foundation
import ReuseAudio
import XCTest

final class OfflinePublicWindowTests: XCTestCase {
    func testPaddingPrefixesAndAllThirtyBlocksMatchTheExistingCompleteRecordingAPI() throws {
        let engine = try ReuseEngine(modelURL: OfflineMetalTestSupport.checkpoint(), optimization: .standard)
        for count in [1, 639, 640, 641, 1_643] {
            let input = OfflineMetalTestSupport.signal(count)
            var blocks: [Int] = []
            let window = try engine.enhanceWindow(input) { blocks.append($0) }
            XCTAssertEqual(window.count, max(640, ((count + 79) / 80) * 80))
            XCTAssertEqual(blocks, Array(0...30))
            XCTAssertTrue(window.allSatisfy(\.isFinite))
            OfflineMetalTestSupport.assertExact(Array(window.prefix(count)), try engine.enhance(input))
        }
    }

    func testInvalidOrOversizedWindowsNeverBeginInference() throws {
        let engine = try ReuseEngine(modelURL: OfflineMetalTestSupport.checkpoint(), optimization: .standard)
        let invalid: [[Float]] = [[], [.nan], [.infinity], [-.infinity], [Float](repeating: 0, count: 64_001)]
        for input in invalid {
            XCTAssertThrowsError(try engine.enhanceWindow(input) { _ in XCTFail("Invalid window entered the model") }) {
                XCTAssertTrue($0 is ReuseError)
            }
        }
    }

    func testCallbackCancellationStopsRemainingBlocksAndLeavesIndependentCallsUsable() throws {
        struct Stop: Error {}
        let engine = try ReuseEngine(modelURL: OfflineMetalTestSupport.checkpoint(), optimization: .standard)
        let input = OfflineMetalTestSupport.signal(641)
        let expected = try engine.enhanceWindow(input)
        var blocks: [Int] = []
        XCTAssertThrowsError(try engine.enhanceWindow(input) {
            blocks.append($0)
            if $0 == 3 { throw Stop() }
        }) { XCTAssertTrue($0 is Stop) }
        XCTAssertEqual(blocks, Array(0...3))
        OfflineMetalTestSupport.assertExact(try engine.enhanceWindow(input), expected)
    }

    func testAlreadyCancelledTaskDoesNotEnterTheModel() async throws {
        let engine = try ReuseEngine(modelURL: OfflineMetalTestSupport.checkpoint(), optimization: .standard)
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try engine.enhanceWindow([0]) { _ in XCTFail("Cancelled work entered the model") }
        }
        do { _ = try await task.value; XCTFail("Expected cancellation") }
        catch is CancellationError { }
    }
}
