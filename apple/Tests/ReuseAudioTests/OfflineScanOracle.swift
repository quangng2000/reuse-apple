// Scalar Double oracle for the offline Mamba equations. SPDX-License-Identifier: OpenMDW-1.1
import Foundation

enum OfflineScanOracle {
    static func evaluate(_ fixture: OfflineScanFixture) -> [Float] {
        var output = [Float](repeating: 0, count: fixture.input.count)
        for batch in 0..<fixture.batch {
            for channel in 0..<256 {
                var state = [Double](repeating: 0, count: 16)
                for time in 0..<fixture.length {
                    let point = batch * fixture.length + time
                    output[point * 256 + channel] = step(fixture, at: (point, channel), state: &state)
                }
            }
        }
        return output
    }

    private static func step(_ fixture: OfflineScanFixture, at position: (Int, Int),
                             state: inout [Double]) -> Float {
        let (point, channel) = position, index = point * 256 + channel
        let value = Double(fixture.delta[index]) + Double(fixture.bias[channel])
        let delta = max(value, 0) + log1p(exp(-abs(value)))
        let input = Double(fixture.input[index])
        var output = 0.0
        for component in 0..<16 {
            let coefficient = point * 16 + component
            let decay = exp(delta * Double(fixture.transition[channel * 16 + component]))
            state[component] = decay * state[component] + delta * Double(fixture.drive[coefficient]) * input
            output += state[component] * Double(fixture.readout[coefficient])
        }
        let gate = Double(fixture.gate[index])
        return Float((output + Double(fixture.skip[channel]) * input) * gate / (1 + exp(-gate)))
    }
}
