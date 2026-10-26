import Foundation
import MLX
import XCTest
@testable import ReuseStreaming

enum StreamingMetalTestSupport {
    static func tensor(_ shape: [Int], salt: Int = 11, scale: Float = 0.3) -> MLXArray {
        let values = (0..<shape.reduce(1, *)).map {
            Float(sin(Double(($0 * 17 + salt) % 997) * 0.071)) * scale
        }
        return MLXArray(values).reshaped(shape)
    }

    static func inputs(batch: Int, length: Int, biasScale: Float = 3) -> [MLXArray] {
        let shape = [batch, length, 256]
        return [tensor(shape), tensor([batch, length, 36], salt: 23, scale: 0.15),
                -abs(tensor([256, 16], salt: 31)) - 0.01, tensor([256], salt: 37),
                tensor(shape, salt: 43, scale: 2), tensor([256, 4], salt: 59, scale: 0.8),
                MLXArray((0..<256).map { Float(($0 % 7) - 3) * biasScale })]
    }

    static func assertNear(_ actual: MLXArray, _ expected: MLXArray,
                           file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, .float32, file: file, line: line)
        assertNear(actual.asArray(Float.self), expected.asArray(Float.self), file: file, line: line)
    }

    static func assertNear(_ actual: [Float], _ expected: [Float],
                           file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        XCTAssertTrue(actual.allSatisfy(\.isFinite), file: file, line: line)
        let pairs = zip(actual, expected).map { (Double($0), Double($1)) }
        let error = pairs.reduce(0) { $0 + pow($1.0 - $1.1, 2) }
        let power = pairs.reduce(0) { $0 + $1.1 * $1.1 }
        let maximum = pairs.map { abs($0.0 - $0.1) }.max() ?? 0
        let relative = sqrt(error / max(power, 1e-20))
        XCTAssertTrue(relative < 1e-4 || maximum < 1e-5,
                      "Relative RMS \(relative), maximum error \(maximum)", file: file, line: line)
    }

    static func assertExact(_ actual: [Float], _ expected: [Float],
                            file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.map(\.bitPattern), expected.map(\.bitPattern), file: file, line: line)
    }

    static func checkpoint() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["REUSE_STREAMING_CHECKPOINT"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Set REUSE_STREAMING_CHECKPOINT to the verified streaming checkpoint.")
        }
        return URL(fileURLWithPath: path)
    }

    static func signal(_ count: Int, offset: Int = 0) -> [Float] {
        (0..<count).map { index in
            let position = Double(index + offset)
            return Float(sin(position * 0.13) * 0.08 + cos(position * 0.017) * 0.015)
        }
    }
}
