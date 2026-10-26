// Independent offline regression tests. SPDX-License-Identifier: OpenMDW-1.1
import MLX
import XCTest
@testable import ReuseAudio

private struct OfflineModelObservation {
    let magnitude: MLXArray
    let phase: MLXArray
    let waveform: [Float]
    let features: [String: MLXArray]
}

final class OfflineMetalModelTests: XCTestCase {
    func testAllThirtyLayersPreserveAudioForEachSpecializedComponentAndCombination() throws {
        let checkpoint = try OfflineMetalTestSupport.checkpoint()
        try Device.withDefaultDevice(.gpu) {
            let input = OfflineMetalTestSupport.signal(1600)
            let reference = try observe(ReuseModel(url: checkpoint), input: input)
            let configurations: [ReuseMetalConfiguration] = [.scanOnly, .normalizationOnly,
                                                             .combined, .preparedSerial, .schedulingOnly]
            for configuration in configurations {
                let model = try ReuseModel(url: checkpoint, optimization: .specializedMetal,
                                          configuration: configuration)
                try compare(observe(model, input: input), reference)
            }
        }
    }

    private func observe(_ model: ReuseModel, input: [Float]) throws -> OfflineModelObservation {
        let spectrum = AudioTransform.analyze(input)
        var features: [String: MLXArray] = [:], visited = [Int]()
        let output = try model(magnitude: spectrum.0, phase: spectrum.1,
                               progress: { visited.append($0) }, inspect: { features[$0] = $1 })
        XCTAssertEqual(visited, Array(0...30), "Every offline model block must execute")
        XCTAssertEqual(Set(features.keys), Set(["encoder", "block1", "block30"]))
        let waveform = AudioTransform.synthesize(magnitude: output.0, phase: output.1, count: input.count)
        return OfflineModelObservation(magnitude: output.0, phase: output.1,
                                       waveform: waveform, features: features)
    }

    private func compare(_ actual: OfflineModelObservation, _ expected: OfflineModelObservation) {
        compareSpectrum(actual, expected)
        OfflineMetalTestSupport.assertClose(actual.waveform, expected.waveform)
        for key in expected.features.keys.sorted() {
            guard let observed = actual.features[key], let reference = expected.features[key] else {
                XCTFail("Missing captured feature: \(key)"); continue
            }
            OfflineMetalTestSupport.assertClose(observed, reference)
        }
    }

    private func compareSpectrum(_ actual: OfflineModelObservation, _ expected: OfflineModelObservation) {
        OfflineMetalTestSupport.assertClose(actual.magnitude, expected.magnitude)
        // Phasors handle pi wrapping and avoid overstating phase error at zero magnitude.
        let actualAmplitude = expm1(maximum(actual.magnitude, 0))
        let expectedAmplitude = expm1(maximum(expected.magnitude, 0))
        OfflineMetalTestSupport.assertClose(actualAmplitude * sin(actual.phase), expectedAmplitude * sin(expected.phase))
        OfflineMetalTestSupport.assertClose(actualAmplitude * cos(actual.phase), expectedAmplitude * cos(expected.phase))
    }
}
