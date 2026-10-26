// Independent offline regression tests. SPDX-License-Identifier: OpenMDW-1.1
import MLX
import XCTest
@testable import ReuseAudio

final class OfflineNormalizationResidualTests: XCTestCase {
    func testNormalizationResidualPreservesConstantStridedAndBatchedRows() {
        Device.withDefaultDevice(.gpu) {
            let strided = MLXArray(OfflineMetalTestSupport.values(64 * 7, seed: 3)).reshaped(64, 7).T
            let batched = MLXArray(OfflineMetalTestSupport.values(2 * 5 * 64, seed: 5)).reshaped(2, 5, 64)
            for input in [MLX.full([3, 64], values: Float(7)), strided, batched] { compare(input) }
        }
    }

    private func compare(_ input: MLXArray) {
        let weight = MLXArray(OfflineMetalTestSupport.values(64, seed: 11, scale: 0.7))
        let bias = MLXArray(OfflineMetalTestSupport.values(64, seed: 17, scale: 0.3))
        let residual = MLXArray(OfflineMetalTestSupport.values(input.size, seed: 29)).reshaped(input.shape)
        let normalized = MLXFast.layerNorm(input, weight: weight, bias: bias, eps: 1e-5)
        let actual = OfflineNormalizationResidual()(input, affine: [weight, bias], residual: residual)
        OfflineMetalTestSupport.assertClose(actual, normalized + residual)
    }
}
