import MLX
import XCTest
@testable import ReuseStreaming
@testable import ReuseStreamingReference

final class StreamingMetalModelPair {
    private let reference: ReuseStreamingReference.StreamingReuseModel
    private let candidate: ReuseStreaming.StreamingReuseModel
    private var compiled: ReuseStreaming.StreamingCompiledFrame?
    private var expectedTransform = ReuseStreamingReference.StreamingTransform()
    private var actualTransform = ReuseStreaming.StreamingTransform()

    init(layers: Int, checkpoint: URL) throws {
        reference = try ReuseStreamingReference.StreamingReuseModel(weightsURL: checkpoint, layers: layers, lookahead: 0)
        candidate = try ReuseStreaming.StreamingReuseModel(weightsURL: checkpoint, layers: layers, lookahead: 0,
                                           optimization: .specializedMetal)
        XCTAssertTrue(candidate.useSpecializedMetal)
    }

    func reset() {
        reference.reset()
        candidate.reset()
        expectedTransform = ReuseStreamingReference.StreamingTransform()
        actualTransform = ReuseStreaming.StreamingTransform()
    }

    func render(_ input: [Float]) -> [Float] {
        let expectedSpectrum = expectedTransform.analyze(input), actualSpectrum = actualTransform.analyze(input)
        let expected = reference(magnitude: expectedSpectrum.0, phase: expectedSpectrum.1, deferEvaluation: true)
        let actual = runCandidate(actualSpectrum)
        StreamingMetalTestSupport.assertNear(actual.0, expected.0)
        StreamingMetalTestSupport.assertNear(actual.1, expected.1)
        let expectedAudio = expectedTransform.synthesize(magnitude: expected.0, phase: expected.1,
                                                        evaluating: reference.frameStateArrays)
        let actualAudio = actualTransform.synthesize(magnitude: actual.0, phase: actual.1,
                                                    evaluating: candidate.frameStateArrays)
        StreamingMetalTestSupport.assertNear(actualAudio, expectedAudio)
        assertStates(candidate.convolutionStates, reference.convolutionStates)
        assertStates(candidate.recurrentStates, reference.recurrentStates)
        XCTAssertEqual(candidate.recurrentStates.count, candidate.layers)
        return actualAudio
    }

    private func runCandidate(_ spectrum: (MLXArray, MLXArray)) -> (MLXArray, MLXArray) {
        if candidate.isFirstFrame {
            return candidate(magnitude: spectrum.0, phase: spectrum.1, deferEvaluation: true)
        }
        if compiled == nil { compiled = ReuseStreaming.StreamingCompiledFrame(model: candidate, mode: .decoderFeatures) }
        return compiled!(magnitude: spectrum.0, phase: spectrum.1)
    }

    private func assertStates(_ actual: [String: MLXArray], _ expected: [String: MLXArray]) {
        XCTAssertEqual(Set(actual.keys), Set(expected.keys))
        for key in expected.keys.sorted() {
            guard let observed = actual[key], let reference = expected[key] else { continue }
            StreamingMetalTestSupport.assertNear(observed, reference)
        }
    }
}
