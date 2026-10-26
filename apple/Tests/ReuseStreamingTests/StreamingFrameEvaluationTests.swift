import XCTest
import MLX
@testable import ReuseStreaming

/// Isolate frame evaluation from kernel changes: both models use the same
/// optimized kernels, while only one defers completion until synthesis.
final class StreamingFrameEvaluationTests: XCTestCase {
    func testEightLayerDeferredFrameMatchesEagerAudioAndEveryState() throws {
        let checkpoint = try checkpoint()
        try Device.withDefaultDevice(.gpu) {
            let eager = try StreamingReuseModel(weightsURL: checkpoint, layers: 8, lookahead: 0)
            let deferred = try StreamingReuseModel(weightsURL: checkpoint, layers: 8, lookahead: 0)
            let eagerTransform = StreamingTransform()
            let deferredTransform = StreamingTransform()

            for frame in 0..<7 {
                let input = (0..<640).map { sample -> Float in
                    if frame == 3 { return 0 }
                    let position = Double(frame * 320 + sample)
                    return Float(sin(position * 0.13) * 0.08 + cos(position * 0.017) * 0.015)
                }
                let expected = render(input, model: eager, transform: eagerTransform, deferred: false)
                let actual = render(input, model: deferred, transform: deferredTransform, deferred: true)
                XCTAssertEqual(actual, expected, "Audio changed at frame \(frame)")
                XCTAssertTrue(actual.allSatisfy(\.isFinite))
                XCTAssertEqual(actual.count, frame == 0 ? 0 : 320)
                assertStatesEqual(eager.convolutionStates, deferred.convolutionStates, frame: frame)
                assertStatesEqual(eager.recurrentStates, deferred.recurrentStates, frame: frame)
            }
        }
    }

    private func render(_ input: [Float], model: StreamingReuseModel,
                        transform: StreamingTransform, deferred: Bool) -> [Float] {
        let spectrum = transform.analyze(input)
        let restored = model(magnitude: spectrum.0, phase: spectrum.1, deferEvaluation: deferred)
        return transform.synthesize(magnitude: restored.0, phase: restored.1,
                                    evaluating: deferred ? model.frameStateArrays : [])
    }

    private func assertStatesEqual(_ expected: [String: MLXArray], _ actual: [String: MLXArray],
                                   frame: Int, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(Set(actual.keys), Set(expected.keys), file: file, line: line)
        for (key, value) in expected {
            guard let observed = actual[key] else { continue }
            XCTAssertEqual(observed.shape, value.shape, file: file, line: line)
            XCTAssertEqual(observed.dtype, .float32, file: file, line: line)
            XCTAssertEqual(observed.asArray(Float.self), value.asArray(Float.self),
                           "State \(key) changed at frame \(frame)", file: file, line: line)
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
