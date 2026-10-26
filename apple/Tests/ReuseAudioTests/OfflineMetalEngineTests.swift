// Independent offline regression tests. SPDX-License-Identifier: OpenMDW-1.1
import Foundation
import XCTest
@testable import ReuseAudio

final class OfflineMetalEngineTests: XCTestCase {
    func testPublicOptInPreservesOddSampleCountAndIndependentRepeatedCalls() throws {
        let checkpoint = try OfflineMetalTestSupport.checkpoint()
        let standard = try ReuseEngine(modelURL: checkpoint)
        let specialized = try ReuseEngine(modelURL: checkpoint, optimization: .specializedMetal)
        let input = OfflineMetalTestSupport.signal(1643)
        let expected = try standard.enhance(input)
        let actual = try specialized.enhance(input)
        XCTAssertEqual(actual.count, input.count)
        OfflineMetalTestSupport.assertClose(actual, expected)
        let other = try specialized.enhance(OfflineMetalTestSupport.signal(641).map { -$0 })
        XCTAssertEqual(other.count, 641)
        XCTAssertTrue(other.allSatisfy(\.isFinite))
        OfflineMetalTestSupport.assertExact(try specialized.enhance(input), actual)
    }

    func testSpecializedHalfPrecisionIsRejectedBeforeCheckpointAccess() {
        let missing = URL(fileURLWithPath: "/nonexistent/offline-checkpoint.safetensors")
        XCTAssertThrowsError(try ReuseModel(url: missing, half: true, optimization: .specializedMetal)) {
            XCTAssertTrue($0 is ReuseError, "Expected configuration failure before reading weights")
        }
    }
}
