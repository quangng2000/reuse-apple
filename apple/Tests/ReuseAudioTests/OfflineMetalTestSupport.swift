// Independent offline regression tests. SPDX-License-Identifier: OpenMDW-1.1
import Foundation
import MLX
import XCTest
@testable import ReuseAudio

enum OfflineMetalTestSupport {
    static func checkpoint() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["REUSE_OFFLINE_CHECKPOINT"] else {
            throw XCTSkip("Set REUSE_OFFLINE_CHECKPOINT to the pinned offline checkpoint.")
        }
        return URL(fileURLWithPath: path)
    }

    static func values(_ count: Int, seed: Int, scale: Float = 1) -> [Float] {
        (0..<count).map { index in
            let angle = Double((index * 29 + seed * 47) % 1009) * 0.043
            return Float(sin(angle) + 0.3 * cos(angle * 0.17)) * scale
        }
    }

    static func signal(_ count: Int) -> [Float] {
        (0..<count).map { index in
            let time = Double(index) / 16_000
            let envelope = index % 877 < 91 ? 0.0 : 0.08
            let chirp = sin(2 * Double.pi * (193 * time + 83 * time * time))
            let impulse = index % 1237 == 7 ? 0.13 : 0.0
            return Float(envelope * (chirp + 0.25 * cos(2 * Double.pi * 691 * time)) + impulse)
        }
    }

    static func assertClose(_ actual: MLXArray, _ expected: MLXArray,
                            file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, .float32, file: file, line: line)
        assertClose(actual.asArray(Float.self), expected.asArray(Float.self), file: file, line: line)
    }

    static func assertClose(_ actual: [Float], _ expected: [Float],
                            file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        guard actual.count == expected.count, !actual.isEmpty else { return }
        XCTAssertTrue(actual.allSatisfy(\.isFinite), file: file, line: line)
        XCTAssertTrue(expected.allSatisfy(\.isFinite), file: file, line: line)
        let difference = zip(actual, expected).map { Double($0) - Double($1) }
        let error = difference.reduce(0) { $0 + $1 * $1 }
        let power = expected.reduce(0.0) { $0 + Double($1) * Double($1) }
        let relative = sqrt(error / max(power, 1e-30))
        let maximum = difference.map { abs($0) }.max() ?? 0
        XCTAssertLessThanOrEqual(relative, 1e-4, "Relative RMS error", file: file, line: line)
        XCTAssertLessThan(maximum, 1e-4, "Maximum sample error", file: file, line: line)
    }

    static func assertExact(_ actual: [Float], _ expected: [Float],
                            file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.map(\.bitPattern), expected.map(\.bitPattern), file: file, line: line)
    }
}
