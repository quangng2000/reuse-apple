import XCTest
import MLX
@testable import ReuseStreaming

final class StreamingNormalizationFusionTests: XCTestCase {
    func testFusedEpiloguesPreserveConstantAndStridedRows() {
        Device.withDefaultDevice(.gpu) {
            let inputs = [MLX.full([2, 64], values: Float(-2)),
                          StreamingMetalTestSupport.tensor([64, 3]).T,
                          StreamingMetalTestSupport.tensor([1, 1, 161, 64])]
            for input in inputs { compare(input) }
        }
    }

    private func compare(_ input: MLXArray) {
        let weights = StreamingNormalizationWeights(weight: StreamingMetalTestSupport.tensor([64], salt: 19),
                                                    bias: StreamingMetalTestSupport.tensor([64], salt: 23))
        let slope = StreamingMetalTestSupport.tensor([64], salt: 29)
        let residual = StreamingMetalTestSupport.tensor(input.shape, salt: 31)
        let normalized = MLXFast.layerNorm(input, weight: weights.weight, bias: weights.bias, eps: 1e-5)
        let expectedPrelu = which(normalized .>= 0, normalized, normalized * slope)
        let fusion = StreamingNormalizationFusion()
        let actualPrelu = fusion.prelu(input, normalization: weights, slope: slope)
        let actualResidual = fusion.residual(input, normalization: weights, residual: residual)
        eval(expectedPrelu, actualPrelu, actualResidual)
        StreamingMetalTestSupport.assertNear(actualPrelu, expectedPrelu)
        StreamingMetalTestSupport.assertNear(actualResidual, normalized + residual)
    }
}
