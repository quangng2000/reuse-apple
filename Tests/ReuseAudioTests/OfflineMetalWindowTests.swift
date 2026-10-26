// Independent offline regression tests. SPDX-License-Identifier: OpenMDW-1.1
import XCTest
@testable import ReuseAudio

final class OfflineMetalWindowTests: XCTestCase {
    /// The shortest input crossing the public 64,000-sample window boundary.
    /// Run separately from device benchmarks: this performs two windows per path.
    func testSpecializedAudioMatchesAcrossTheHalfSecondCrossfadeAndFinalPartialWindow() throws {
        let checkpoint = try OfflineMetalTestSupport.checkpoint()
        let input = OfflineMetalTestSupport.signal(64_001)
        let reference = try render(ReuseEngine(modelURL: checkpoint), input: input)
        let candidate = try render(ReuseEngine(modelURL: checkpoint, optimization: .specializedMetal), input: input)
        XCTAssertEqual(candidate.count, input.count)
        OfflineMetalTestSupport.assertClose(candidate, reference)
        // Evaluate the blend independently, so a long prefix cannot hide boundary error.
        OfflineMetalTestSupport.assertClose(Array(candidate[55_999..<64_001]),
                                            Array(reference[55_999..<64_001]))
    }

    private func render(_ engine: ReuseEngine, input: [Float]) throws -> [Float] {
        var progress = [Double]()
        let output = try engine.enhance(input, progress: { progress.append($0) })
        XCTAssertEqual(progress.first, 0)
        XCTAssertEqual(progress.last, 1)
        XCTAssertTrue(progress.contains(0.5), "Expected the second overlapping window")
        XCTAssertTrue(zip(progress, progress.dropFirst()).allSatisfy { $0 <= $1 })
        return output
    }
}
