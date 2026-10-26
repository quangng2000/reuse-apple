// Independent offline regression tests. SPDX-License-Identifier: OpenMDW-1.1
import MLX

struct OfflineScanFixture {
    let batch: Int
    let length: Int
    let input: [Float]
    let delta: [Float]
    let transition: [Float]
    let drive: [Float]
    let readout: [Float]
    let skip: [Float]
    let gate: [Float]
    let bias: [Float]

    init(batch: Int, length: Int) {
        self.batch = batch
        self.length = length
        let count = batch * length * 256
        input = OfflineMetalTestSupport.values(count, seed: 2, scale: 0.4)
        delta = OfflineMetalTestSupport.values(count, seed: 7, scale: 2)
        transition = (0..<(256 * 16)).map { -Float(1 + $0 % 37) * 0.031 }
        drive = OfflineMetalTestSupport.values(batch * length * 16, seed: 13, scale: 0.1)
        readout = OfflineMetalTestSupport.values(batch * length * 16, seed: 19, scale: 0.1)
        skip = OfflineMetalTestSupport.values(256, seed: 23, scale: 0.2)
        gate = OfflineMetalTestSupport.values(count, seed: 31, scale: 5)
        bias = (0..<256).map { Float([-80, -8, 0, 8, 80][($0 * 3) % 5]) }
    }

    var arrays: [MLXArray] {
        let shape = [batch, length, 256], coefficients = [batch, length, 16]
        return [MLXArray(input).reshaped(shape), MLXArray(delta).reshaped(shape),
                MLXArray(transition).reshaped(256, 16), MLXArray(drive).reshaped(coefficients),
                MLXArray(readout).reshaped(coefficients), MLXArray(skip),
                MLXArray(gate).reshaped(shape), MLXArray(bias)]
    }
}
