import XCTest
import MLX
@testable import ReuseStreaming

final class MambaPreparedFrequencyTests: XCTestCase {
    func testPreparedFrequencyPreservesExistingScanBitsAcrossLengthsAndBatches() {
        Device.withDefaultDevice(.gpu) {
            for (batch, length, bias) in [(1, 1, Float(3)), (2, 13, Float(1000)), (1, 161, Float(0))] {
                let inputs = StreamingMetalTestSupport.inputs(batch: batch, length: length, biasScale: bias)
                let expected = MambaProjectionFrequency()(inputs)
                let actual = MambaPreparedFrequency()(inputs)
                eval(expected, actual)
                StreamingMetalTestSupport.assertNear(actual, expected)
                StreamingMetalTestSupport.assertExact(actual.asArray(Float.self), expected.asArray(Float.self))
            }
        }
    }
}
