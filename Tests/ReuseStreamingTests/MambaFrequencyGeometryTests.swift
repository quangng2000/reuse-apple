import XCTest
import MLX
@testable import ReuseStreaming

/// Test-only duplicate of the pooled frequency scan. Only launch geometry
/// varies; production defaults and ordered FP32 expressions stay unchanged.
final class MambaFrequencyGeometryTests: XCTestCase {
    private let kernel = MLXFast.metalKernel(
        name: "reuse_pooled_frequency_geometry_reference", inputNames: ["u", "pf", "pb", "a", "d", "z", "w", "bias"],
        outputNames: ["out"], source: """
        uint lane = thread_position_in_grid.x;
        uint channel = lane / 16;
        uint component = lane % 16;
        uint batch = thread_position_in_grid.y;
        uint channels = u_shape[2];
        uint length = u_shape[1];
        if (channel >= channels || batch >= u_shape[0]) return;
        uint wc = batch * channels + channel;
        device const float* p = batch == 0 ? pf : pb;
        bool share_delta = threads_per_simdgroup >= 16;
        ushort leader = ushort(thread_index_in_simdgroup & ~15u);
        float state = 0.0f;
        float aa = a[wc * 16 + component];
        for (uint t = 0; t < length; ++t) {
            uint ix = (batch * length + t) * channels + channel;
            uint pi = t * 36;
            float dt = 0.0f;
            if (!share_delta || component == 0) {
                float projected = p[pi] * w[wc * 4];
                for (uint rank = 1; rank < 4; ++rank) projected += p[pi + rank] * w[wc * 4 + rank];
                float v = projected + bias[wc];
                dt = max(v, 0.0f) + log1p(exp(-abs(v)));
            }
            if (share_delta) dt = simd_shuffle(dt, leader);
            float uu = u[ix];
            state = exp(dt * aa) * state + dt * p[pi + 4 + component] * uu;
            float yy = state * p[pi + 20 + component];
            for (uint offset = 8; offset > 0; offset >>= 1) {
                float other = simd_shuffle_down(yy, offset);
                if (component < offset) yy += other;
            }
            if (component == 0) {
                float zz = z[ix];
                out[ix] = (yy + d[wc] * uu) * (zz / (1.0f + exp(-zz)));
            }
        }
        """)

    func testAllGeometriesMatchCurrentKernelBitsExactly() {
        Device.withDefaultDevice(.gpu) {
            for (length, channels) in [(162, 256), (13, 17)] {
                let inputs = makeInputs(length: length, channels: channels)
                let branches = (0..<2).map { branch in
                    MambaProjectionFrequency()([
                        inputs[0][branch..<(branch + 1), 0..., 0...], inputs[1 + branch], inputs[3][branch],
                        inputs[4][branch], inputs[5][branch..<(branch + 1), 0..., 0...],
                        inputs[6][branch], inputs[7][branch]
                    ])
                }
                let expected = concatenated(branches, axis: 0)
                let variants = [32, 64, 128].map { launch(inputs, group: $0) }
                eval([expected] + variants)
                let expectedBits = expected.asArray(Float.self).map(\.bitPattern)
                for value in variants {
                    XCTAssertEqual(value.dtype, .float32)
                    XCTAssertEqual(value.asArray(Float.self).map(\.bitPattern), expectedBits)
                }
            }
        }
    }

    func testOptionalFrequencyGeometryMicrobenchmark() throws {
        guard ProcessInfo.processInfo.environment["REUSE_KERNEL_BENCHMARK"] == "1" else {
            throw XCTSkip("Set TEST_RUNNER_REUSE_KERNEL_BENCHMARK=1 for the opt-in GPU microbenchmark")
        }
        Device.withDefaultDevice(.gpu) {
            let inputs = makeInputs(length: 162, channels: 256)
            eval(inputs)
            let groups = [32, 64, 128]
            let chainLength = 64
            func sample(_ group: Int) -> Double {
                var output = inputs[0]
                for _ in 0..<chainLength {
                    var arguments = inputs
                    arguments[0] = output
                    output = launch(arguments, group: group)
                }
                Stream.gpu.synchronize()
                let start = DispatchTime.now().uptimeNanoseconds
                eval(output)
                Stream.gpu.synchronize()
                return Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000 / Double(chainLength)
            }
            for group in groups { for _ in 0..<3 { _ = sample(group) } }
            var samples = groups.map { _ in [Double]() }
            for _ in 0..<8 {
                for index in [2, 1, 0, 0, 1, 2] { samples[index].append(sample(groups[index])) }
            }
            var measurements: [String: Any] = [:]
            for (index, group) in groups.enumerated() {
                let values = samples[index]
                let sorted = values.sorted()
                measurements[String(group)] = [
                    "mean_ms": values.reduce(0, +) / Double(values.count),
                    "p50_ms": sorted[sorted.count / 2], "minimum_ms": sorted[0],
                    "maximum_ms": sorted[sorted.count - 1], "samples_ms": values
                ]
            }
            let report: [String: Any] = [
                "kind": "pooled_frequency_threadgroup_geometry_microbenchmark",
                "shape": [2, 162, 256], "chain_length": chainLength,
                "measurements": measurements,
                "timing": "Prebuilt dependent chain evaluation plus GPU completion, milliseconds per scan"
            ]
            let data = try! JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
            print("REUSE_KERNEL_BENCHMARK " + String(decoding: data, as: UTF8.self))
        }
    }

    private func launch(_ inputs: [MLXArray], group: Int) -> MLXArray {
        let input = inputs[0]
        return kernel(inputs, grid: (input.dim(2) * 16, input.dim(0), 1),
                      threadGroup: (group, 1, 1), outputShapes: [input.shape], outputDTypes: [.float32])[0]
    }

    private func makeInputs(length: Int, channels: Int) -> [MLXArray] {
        func tensor(_ count: Int, _ shape: [Int], salt: Int, scale: Float) -> MLXArray {
            let values: [Float] = (0..<count).map { index in
                let position = Double((index * 17 + salt) % 997) * 0.071
                return Float(sin(position)) * scale
            }
            return MLXArray(values).reshaped(shape)
        }
        let count = 2 * length * channels
        return [
            tensor(count, [2, length, channels], salt: 11, scale: 0.35),
            tensor(length * 36, [1, length, 36], salt: 23, scale: 0.15),
            tensor(length * 36, [1, length, 36], salt: 31, scale: 0.15),
            MLXArray((0..<(2 * channels * 16)).map { -Float(1 + $0 % 31) / 17 }).reshaped(2, channels, 16),
            tensor(2 * channels, [2, channels], salt: 37, scale: 0.2),
            tensor(count, [2, length, channels], salt: 43, scale: 2),
            tensor(2 * channels * 4, [2, channels, 4], salt: 59, scale: 0.8),
            MLXArray((0..<(2 * channels)).map { Float(($0 % 7) - 3) * 3 }).reshaped(2, channels)
        ]
    }
}
