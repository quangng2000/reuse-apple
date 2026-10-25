import XCTest
import MLX
@testable import ReuseStreaming

final class StreamingCompiledFrameTests: XCTestCase {
    func testEightLayerCompiledFramesMatchAudioAndEveryStateAcrossReset() throws {
        try assertFrames(mode: .body)
    }

    func testEightLayerCompiledDecoderFeaturesMatchAudioAndEveryStateAcrossReset() throws {
        try assertFrames(mode: .decoderFeatures)
    }

    private func assertFrames(mode: StreamingFrameCompilation) throws {
        let checkpoint = try checkpoint()
        try Device.withDefaultDevice(.gpu) {
            let eager = try StreamingReuseModel(weightsURL: checkpoint, layers: 8, lookahead: 0)
            let candidate = try StreamingReuseModel(weightsURL: checkpoint, layers: 8, lookahead: 0)
            var compiled: StreamingCompiledFrame?
            for recording in 0..<2 {
                eager.reset()
                candidate.reset()
                let expectedTransform = StreamingTransform()
                let actualTransform = StreamingTransform()
                for frame in 0..<5 {
                    let input = signal(count: 640, offset: frame * 320, silent: frame == 2)
                    let expectedSpectrum = expectedTransform.analyze(input)
                    let expected: (MLXArray, MLXArray)
                    var expectedHidden: MLXArray?
                    if eager.isFirstFrame {
                        expected = eager(magnitude: expectedSpectrum.0, phase: expectedSpectrum.1,
                                         deferEvaluation: true)
                    } else {
                        let hidden = eager.frameFeatures(magnitude: expectedSpectrum.0, phase: expectedSpectrum.1)
                        if mode == .body {
                            expectedHidden = hidden
                            expected = eager.decodeFrame(hidden, frequencies: expectedSpectrum.0.dim(0))
                        } else {
                            let features = eager.decodeFrameFeatures(hidden, frequencies: expectedSpectrum.0.dim(0))
                            expectedHidden = features.1
                            expected = (features.0, eager.decodePhase(features.1, frequencies: expectedSpectrum.0.dim(0)))
                        }
                    }
                    let actualSpectrum = actualTransform.analyze(input)
                    let actual: (MLXArray, MLXArray)
                    var actualHidden: MLXArray?
                    if candidate.isFirstFrame {
                        actual = candidate(magnitude: actualSpectrum.0, phase: actualSpectrum.1,
                                           deferEvaluation: true)
                    } else {
                        if compiled == nil {
                            compiled = StreamingCompiledFrame(model: candidate, mode: mode)
                        }
                        if mode == .body {
                            let hidden = compiled!.features(magnitude: actualSpectrum.0, phase: actualSpectrum.1)
                            actualHidden = hidden
                            actual = candidate.decodeFrame(hidden, frequencies: actualSpectrum.0.dim(0))
                        } else {
                            let features = compiled!.decoderFeatures(magnitude: actualSpectrum.0, phase: actualSpectrum.1)
                            actualHidden = features.1
                            actual = (features.0, candidate.decodePhase(features.1, frequencies: actualSpectrum.0.dim(0)))
                        }
                    }
                    let expectedAudio = expectedTransform.synthesize(magnitude: expected.0, phase: expected.1,
                                                                    evaluating: eager.frameStateArrays)
                    let actualAudio = actualTransform.synthesize(magnitude: actual.0, phase: actual.1,
                                                                evaluating: candidate.frameStateArrays)
                    if let expectedHidden, let actualHidden {
                        assertBitsEqual(expectedHidden.asArray(Float.self), actualHidden.asArray(Float.self), "Hidden")
                    }
                    assertBitsEqual(expected.0.asArray(Float.self), actual.0.asArray(Float.self), "Magnitude")
                    assertBitsEqual(expected.1.asArray(Float.self), actual.1.asArray(Float.self), "Phase")
                    assertBitsEqual(expectedAudio, actualAudio, "Audio recording \(recording), frame \(frame)")
                    assertStatesEqual(eager.convolutionStates, candidate.convolutionStates)
                    assertStatesEqual(eager.recurrentStates, candidate.recurrentStates)
                }
            }
        }
    }

    func testCompiledEnhancerMatchesDefaultThroughWarmupFinishAndReset() throws {
        try assertEnhancer(mode: .body)
    }

    func testCompiledDecoderFeaturesEnhancerMatchesDefaultThroughWarmupFinishAndReset() throws {
        try assertEnhancer(mode: .decoderFeatures)
    }

    private func assertEnhancer(mode: StreamingFrameCompilation) throws {
        let checkpoint = try checkpoint()
        let eager = try StreamingEnhancer(modelURL: checkpoint, layers: 8, optimized: true,
                                          compilation: .disabled, pooledFrequency: false)
        let compiled = try StreamingEnhancer(modelURL: checkpoint, layers: 8,
                                             optimized: true, compilation: mode)
        try eager.warmUp()
        try compiled.warmUp()
        let input = signal(count: 2282)
        var previous = [Float]()
        for recording in 0..<2 {
            if recording > 0 { eager.reset(); compiled.reset() }
            var expected = [Float](), actual = [Float]()
            var position = 0
            for size in [127, 513, 319, 640, 43, 640] {
                let chunk = Array(input[position..<(position + size)])
                expected += try eager.push(chunk)
                actual += try compiled.push(chunk)
                position += size
            }
            expected += try eager.finish()
            actual += try compiled.finish()
            XCTAssertEqual(actual.count, input.count)
            assertBitsEqual(expected, actual, "Enhancer recording \(recording)")
            if recording > 0 { assertBitsEqual(previous, actual, "Reset replay") }
            previous = actual
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
        let differences = zip(expected, actual).filter { $0.bitPattern != $1.bitPattern }
        let maximum = differences.map { abs($0 - $1) }.max() ?? 0
        XCTAssertTrue(differences.isEmpty, "\(message): \(differences.count) unequal bits, max abs \(maximum)",
                      file: file, line: line)
    }

    private func signal(count: Int, offset: Int = 0, silent: Bool = false) -> [Float] {
        (0..<count).map { index in
            if silent { return 0 }
            let position = Double(offset + index)
            return Float(sin(position * 0.13) * 0.08 + cos(position * 0.017) * 0.015)
        }
    }

    private func checkpoint() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["REUSE_STREAMING_CHECKPOINT"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("Set REUSE_STREAMING_CHECKPOINT to the verified Real-time RE-USE checkpoint.")
        }
        return URL(fileURLWithPath: path)
    }
}
