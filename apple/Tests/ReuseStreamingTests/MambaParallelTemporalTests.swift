import XCTest
import MLX
@testable import ReuseStreaming

final class MambaParallelTemporalTests: XCTestCase {
    func testEveryStateComponentMatchesSerialUpdatesAcrossStepsAndBatches() {
        Device.withDefaultDevice(.gpu) {
            for (batch, bias) in [(1, Float(3)), (2, Float(1000)), (161, Float(0))] {
                compare(batch: batch, bias: bias)
            }
        }
    }

    private func compare(batch: Int, bias: Float) {
        let inputs = StreamingMetalTestSupport.inputs(batch: batch, length: 1, biasScale: bias)
        var expectedState = StreamingMetalTestSupport.tensor([batch, 256, 16], salt: 67)
        var actualState = expectedState
        let serial = MambaProjectionTemporal(), parallel = MambaParallelTemporal()
        for _ in 0..<3 {
            let expected = serial(inputs, state: expectedState)
            let actual = parallel(inputs, state: actualState)
            eval(expected.0, expected.1, actual.0, actual.1)
            StreamingMetalTestSupport.assertNear(actual.0, expected.0)
            StreamingMetalTestSupport.assertNear(actual.1, expected.1)
            expectedState = expected.1
            actualState = actual.1
        }
    }
}
