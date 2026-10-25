import XCTest
import MLX
@testable import ReuseStreaming
@testable import ReuseStreamingReference

final class StreamingCompiledPooledParityTests: XCTestCase {
    func testCombinedDecoderFeaturesAndPooledFrequencyMatchFrozenWaveformStatesAndReset() throws {
        guard let path = ProcessInfo.processInfo.environment["REUSE_STREAMING_CHECKPOINT"] else {
            throw XCTSkip("Set REUSE_STREAMING_CHECKPOINT to the pinned checkpoint.")
        }
        try Device.withDefaultDevice(.gpu) {
            let checkpoint = URL(fileURLWithPath: path)
            let reference = try ReuseStreamingReference.StreamingReuseModel(weightsURL: checkpoint,
                                                                            layers: 8, lookahead: 0)
            let candidate = try ReuseStreaming.StreamingReuseModel(weightsURL: checkpoint,
                                                                   layers: 8, lookahead: 0)
            candidate.useBatchedFrequency = true
            var compiled: ReuseStreaming.StreamingCompiledFrame?
            var expectedTransform = ReuseStreamingReference.StreamingTransform()
            var actualTransform = ReuseStreaming.StreamingTransform()
            func render(_ input: [Float]) -> [Float] {
                let expectedSpectrum = expectedTransform.analyze(input)
                let expected = reference(magnitude: expectedSpectrum.0, phase: expectedSpectrum.1,
                                         deferEvaluation: true)
                let actualSpectrum = actualTransform.analyze(input)
                let actual: (MLXArray, MLXArray)
                if candidate.isFirstFrame {
                    actual = candidate(magnitude: actualSpectrum.0, phase: actualSpectrum.1,
                                       deferEvaluation: true)
                } else {
                    if compiled == nil {
                        compiled = ReuseStreaming.StreamingCompiledFrame(model: candidate, mode: .decoderFeatures)
                    }
                    actual = compiled!(magnitude: actualSpectrum.0, phase: actualSpectrum.1)
                }
                let expectedAudio = expectedTransform.synthesize(magnitude: expected.0, phase: expected.1,
                                                                evaluating: reference.frameStateArrays)
                let actualAudio = actualTransform.synthesize(magnitude: actual.0, phase: actual.1,
                                                            evaluating: candidate.frameStateArrays)
                assertBitsEqual(expected.0.asArray(Float.self), actual.0.asArray(Float.self), "Magnitude")
                assertBitsEqual(expected.1.asArray(Float.self), actual.1.asArray(Float.self), "Phase")
                assertBitsEqual(expectedAudio, actualAudio, "Waveform")
                assertStatesEqual(reference.convolutionStates, candidate.convolutionStates)
                assertStatesEqual(reference.recurrentStates, candidate.recurrentStates)
                return actualAudio
            }
            for _ in 0..<3 { _ = render([Float](repeating: 0, count: 640)) }
            let input = (0..<4117).map { index -> Float in
                if (1600..<2240).contains(index) { return 0 }
                return Float(sin(Double(index) * 0.13) * 0.08 + cos(Double(index) * 0.017) * 0.015)
            }
            var previous = [Float]()
            for recording in 0..<2 {
                reference.reset()
                candidate.reset()
                expectedTransform = ReuseStreamingReference.StreamingTransform()
                actualTransform = ReuseStreaming.StreamingTransform()
                var session = ReuseStreaming.StreamingSession()
                var position = 0
                var output = [Float]()
                for count in [17, 623, 1, 319, 509, 640, 161, 1847] {
                    output += try session.push(Array(input[position..<(position + count)]), render: render)
                    position += count
                }
                output += try session.finish(render: render)
                XCTAssertEqual(output.count, input.count)
                if recording > 0 { assertBitsEqual(previous, output, "Reset replay") }
                previous = output
            }
        }
    }

    private func assertStatesEqual(_ expected: [String: MLXArray], _ actual: [String: MLXArray],
                                   file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(Set(expected.keys), Set(actual.keys), file: file, line: line)
        for (key, value) in expected {
            guard let observed = actual[key] else { continue }
            XCTAssertEqual(observed.shape, value.shape, file: file, line: line)
            XCTAssertEqual(observed.dtype, .float32, file: file, line: line)
            assertBitsEqual(value.asArray(Float.self), observed.asArray(Float.self), key,
                            file: file, line: line)
        }
    }

    private func assertBitsEqual(_ expected: [Float], _ actual: [Float], _ message: String,
                                 file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(expected.count, actual.count, message, file: file, line: line)
        XCTAssertTrue(actual.allSatisfy(\.isFinite), message, file: file, line: line)
        XCTAssertTrue(zip(expected, actual).allSatisfy { $0.bitPattern == $1.bitPattern },
                      message, file: file, line: line)
    }
}
