import XCTest
import MLX
@testable import ReuseStreaming

/// The reference is the current optimized kernel at repository revision
/// bdf56d19f4dbc16fb9ee949d42b70b4b6b5fe43e, not the older unfused baseline.
final class MambaProjectionFrequencyTests: XCTestCase {
    private let reference = MLXFast.metalKernel(
        name: "reuse_mamba_projected_frequency_reference_bdf56d1",
        inputNames: ["u", "p", "a", "d", "z", "w", "bias"],
        outputNames: ["out"], source: """
        uint lane = thread_position_in_grid.x;
        uint channel = lane / 16;
        uint component = lane % 16;
        uint batch = thread_position_in_grid.y;
        uint channels = u_shape[2];
        uint length = u_shape[1];
        if (channel >= channels || batch >= u_shape[0]) return;
        float state = 0.0f;
        float aa = a[channel * 16 + component];
        for (uint t = 0; t < length; ++t) {
            uint ix = (batch * length + t) * channels + channel;
            uint pi = (batch * length + t) * 36;
            float delta = p[pi] * w[channel * 4];
            for (uint rank = 1; rank < 4; ++rank) delta += p[pi + rank] * w[channel * 4 + rank];
            float v = delta + bias[channel];
            float dt = max(v, 0.0f) + log1p(exp(-abs(v)));
            float uu = u[ix];
            state = exp(dt * aa) * state + dt * p[pi + 4 + component] * uu;
            float yy = state * p[pi + 20 + component];
            for (uint offset = 8; offset > 0; offset >>= 1) {
                float other = simd_shuffle_down(yy, offset);
                if (component < offset) yy += other;
            }
            if (component == 0) {
                float zz = z[ix];
                out[ix] = (yy + d[channel] * uu) * (zz / (1.0f + exp(-zz)));
            }
        }
        """)

    func testChannelSubgroupsMatchCurrentKernelExactly() {
        // Adjacent channels have deliberately different projections and biases.
        // Odd channel counts cover a final half-full SIMD group; batches must
        // never exchange projections either.
        for (batch, length, channels) in [(1, 1, 1), (1, 7, 2), (2, 13, 3),
                                           (1, 161, 17), (2, 162, 256)] {
            compare(batch: batch, length: length, channels: channels, biasScale: 3)
        }
    }

    func testStableSoftplusExtremesAndZeroInputMatchExactly() {
        compare(batch: 2, length: 19, channels: 9, biasScale: 1000)
        compare(batch: 1, length: 37, channels: 32, biasScale: 0, silent: true)
    }

    private func compare(batch: Int, length: Int, channels: Int,
                         biasScale: Float, silent: Bool = false,
                         file: StaticString = #filePath, line: UInt = #line) {
        Device.withDefaultDevice(.gpu) {
            let count = batch * length * channels
            let inputs = [
                tensor(count, shape: [batch, length, channels], salt: 11, scale: silent ? 0 : 0.35),
                tensor(batch * length * 36, shape: [batch, length, 36], salt: 23, scale: 0.15),
                MLXArray((0..<(channels * 16)).map { -Float(1 + $0 % 31) / 17 }).reshaped(channels, 16),
                tensor(channels, shape: [channels], salt: 37, scale: 0.2),
                tensor(count, shape: [batch, length, channels], salt: 43, scale: 2),
                tensor(channels * 4, shape: [channels, 4], salt: 59, scale: 0.8),
                MLXArray((0..<channels).map { Float(($0 % 7) - 3) * biasScale })
            ]
            let expected = reference(inputs, grid: (channels * 16, batch, 1),
                threadGroup: (128, 1, 1), outputShapes: [[batch, length, channels]],
                outputDTypes: [.float32])[0]
            let actual = MambaProjectionFrequency()(inputs)
            eval(expected, actual)
            XCTAssertEqual(actual.dtype, .float32, file: file, line: line)
            XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
            let expectedValues = expected.asArray(Float.self)
            let actualValues = actual.asArray(Float.self)
            XCTAssertTrue(actualValues.allSatisfy(\.isFinite), file: file, line: line)
            var mismatches = 0
            var maximumError: Float = 0
            for (a, b) in zip(actualValues, expectedValues) {
                if a.bitPattern != b.bitPattern { mismatches += 1 }
                maximumError = max(maximumError, abs(a - b))
            }
            XCTAssertEqual(mismatches, 0,
                "batch=\(batch), length=\(length), channels=\(channels), max error=\(maximumError)",
                file: file, line: line)
        }
    }

    private func tensor(_ count: Int, shape: [Int], salt: Int, scale: Float) -> MLXArray {
        let values = (0..<count).map { index in
            Float(sin(Double((index * 17 + salt) % 997) * 0.071)) * scale
        }
        return MLXArray(values).reshaped(shape)
    }
}
