import XCTest
import MLX
@testable import ReuseStreaming
@testable import ReuseStreamingReference

final class MambaBidirectionalFrequencyTests: XCTestCase {
    func testEveryTimeFrequencyBlockMatchesFrozen2446Exactly() throws {
        let checkpoint = try checkpoint()
        try Device.withDefaultDevice(.gpu) {
            let frozen = try ReuseStreamingReference.StreamingReuseModel(weightsURL: checkpoint, layers: 8, lookahead: 0)
            let candidate = try ReuseStreaming.StreamingReuseModel(weightsURL: checkpoint, layers: 8, lookahead: 0)
            candidate.useBatchedFrequency = true
            for frame in 0..<4 {
                var expected = features(frame: frame)
                var actual = expected
                for block in 0..<8 {
                    let prefix = "TSMamba.\(block)"
                    expected = frozen.timeFrequency(expected, prefix)
                    actual = candidate.timeFrequency(actual, prefix)
                    eval([expected, actual] + Array(frozen.convolutionStates.values)
                         + Array(frozen.recurrentStates.values) + candidate.frameStateArrays)
                    assertBitsEqual(expected, actual, "TF\(block), frame\(frame)")
                }
                assertStatesEqual(frozen.convolutionStates, candidate.convolutionStates)
                assertStatesEqual(frozen.recurrentStates, candidate.recurrentStates)
            }
        }
    }

    func testEightLayerModelMatchesFrozen2446AcrossResetExactly() throws {
        let checkpoint = try checkpoint()
        try Device.withDefaultDevice(.gpu) {
            let frozen = try ReuseStreamingReference.StreamingReuseModel(weightsURL: checkpoint, layers: 8, lookahead: 0)
            let candidate = try ReuseStreaming.StreamingReuseModel(weightsURL: checkpoint, layers: 8, lookahead: 0)
            candidate.useBatchedFrequency = true
            for recording in 0..<2 {
                frozen.reset()
                candidate.reset()
                for frame in 0..<4 {
                    let magnitude = MLXArray((0..<321).map {
                        frame == 2 ? Float(0) : Float(abs(sin(Double($0 + frame * 13) * 0.031))) * 0.3
                    }).reshaped(321, 1)
                    let phase = MLXArray((0..<321).map {
                        Float(sin(Double($0 + frame * 17) * 0.017)) * 3
                    }).reshaped(321, 1)
                    let expected = frozen(magnitude: magnitude, phase: phase)
                    let actual = candidate(magnitude: magnitude, phase: phase)
                    assertBitsEqual(expected.0, actual.0, "Magnitude recording\(recording), frame\(frame)")
                    assertBitsEqual(expected.1, actual.1, "Phase recording\(recording), frame\(frame)")
                    assertStatesEqual(frozen.convolutionStates, candidate.convolutionStates)
                    assertStatesEqual(frozen.recurrentStates, candidate.recurrentStates)
                }
            }
        }
    }

    func testOptionalBatchedFrequencyTimeFrequencyMicrobenchmark() throws {
        guard ProcessInfo.processInfo.environment["REUSE_KERNEL_BENCHMARK"] == "1" else {
            throw XCTSkip("Set TEST_RUNNER_REUSE_KERNEL_BENCHMARK=1 for the opt-in GPU microbenchmark")
        }
        let checkpoint = try checkpoint()
        try Device.withDefaultDevice(.gpu) {
            let current = try ReuseStreaming.StreamingReuseModel(weightsURL: checkpoint, layers: 8, lookahead: 0)
            let batched = try ReuseStreaming.StreamingReuseModel(weightsURL: checkpoint, layers: 8, lookahead: 0)
            batched.useBatchedFrequency = true
            let models = [current, batched]
            let input = features(frame: 0)
            eval(input)
            func sample(_ model: ReuseStreaming.StreamingReuseModel) -> Double {
                Stream.gpu.synchronize()
                let start = DispatchTime.now().uptimeNanoseconds
                var output = input
                for block in 0..<8 { output = model.timeFrequency(output, "TSMamba.\(block)") }
                eval([output] + model.frameStateArrays)
                Stream.gpu.synchronize()
                return Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            }
            for model in models { for _ in 0..<3 { _ = sample(model) } }
            var samples: [[Double]] = [[], []]
            for _ in 0..<8 {
                for index in [0, 1, 1, 0] { samples[index].append(sample(models[index])) }
            }
            func statistics(_ values: [Double]) -> [String: Double] {
                let sorted = values.sorted()
                return ["mean_ms": values.reduce(0, +) / Double(values.count),
                        "p50_ms": sorted[sorted.count / 2], "minimum_ms": sorted[0],
                        "maximum_ms": sorted[sorted.count - 1]]
            }
            let report: [String: Any] = [
                "kind": "bidirectional_frequency_b2_eight_tf_microbenchmark",
                "shape": [1, 1, 162, 64], "blocks": 8,
                "current": statistics(samples[0]), "batched": statistics(samples[1]),
                "current_samples_ms": samples[0], "batched_samples_ms": samples[1],
                "timing": "Eight sequential TF blocks including graph construction, dispatch and GPU/state completion"
            ]
            let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
            print("REUSE_KERNEL_BENCHMARK " + String(decoding: data, as: UTF8.self))
        }
    }

    private func features(frame: Int) -> MLXArray {
        let values: [Float] = (0..<(162 * 64)).map { (index: Int) -> Float in
            if frame == 2 { return 0 }
            let position = Double(index * 17 + frame * 31) * 0.019
            return Float(sin(position)) * 0.2
        }
        return MLXArray(values).reshaped(1, 1, 162, 64)
    }

    private func assertStatesEqual(_ expected: [String: MLXArray], _ actual: [String: MLXArray],
                                   file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(Set(expected.keys), Set(actual.keys), file: file, line: line)
        for (key, value) in expected {
            guard let observed = actual[key] else { continue }
            assertBitsEqual(value, observed, key, file: file, line: line)
        }
    }

    private func assertBitsEqual(_ expected: MLXArray, _ actual: MLXArray, _ message: String,
                                 file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(expected.shape, actual.shape, message, file: file, line: line)
        XCTAssertEqual(actual.dtype, .float32, message, file: file, line: line)
        let a = expected.asArray(Float.self), b = actual.asArray(Float.self)
        XCTAssertTrue(b.allSatisfy(\.isFinite), message, file: file, line: line)
        var unequal = 0
        var maximum: Float = 0
        for (left, right) in zip(a, b) {
            if left.bitPattern != right.bitPattern { unequal += 1 }
            maximum = max(maximum, abs(left - right))
        }
        XCTAssertEqual(unequal, 0, "\(message): max error=\(maximum)", file: file, line: line)
    }

    private func checkpoint() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["REUSE_STREAMING_CHECKPOINT"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Set REUSE_STREAMING_CHECKPOINT to the verified Real-time RE-USE checkpoint.")
        }
        return URL(fileURLWithPath: path)
    }
}
