// Benchmark-only support compiled into the generated diagnostic framework.
import Foundation
import CryptoKit
import MLX

public struct BenchmarkStageTiming: Codable {
    public let name: String
    public let graphConstructionMilliseconds: Double
    public let evaluationWaitMilliseconds: Double
    public var cpuPostprocessingMilliseconds: Double
    public var totalMilliseconds: Double {
        graphConstructionMilliseconds + evaluationWaitMilliseconds + cpuPostprocessingMilliseconds
    }
}

public struct BenchmarkStateDigest: Codable, Equatable {
    public let name: String
    public let shape: [Int]
    public let sha256: String
}

func benchmarkNow() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
private func milliseconds(_ start: UInt64, _ end: UInt64) -> Double {
    Double(end - start) / 1_000_000
}

final class BenchmarkStageProfiler {
    private(set) var timings = [BenchmarkStageTiming]()

    @discardableResult
    func complete(_ name: String, started: UInt64, arrays: [MLXArray]) -> Int {
        let constructed = benchmarkNow()
        eval(arrays)
        Stream.gpu.synchronize()
        let completed = benchmarkNow()
        timings.append(BenchmarkStageTiming(name: name,
            graphConstructionMilliseconds: milliseconds(started, constructed),
            evaluationWaitMilliseconds: milliseconds(constructed, completed),
            cpuPostprocessingMilliseconds: 0))
        return timings.count - 1
    }

    func finishCPU(index: Int, started: UInt64) {
        timings[index].cpuPostprocessingMilliseconds = milliseconds(started, benchmarkNow())
    }
}

func benchmarkDigest(_ name: String, shape: [Int], values: [Float]) -> BenchmarkStateDigest {
    let bytes = values.withUnsafeBytes { Data($0) }
    let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    return BenchmarkStateDigest(name: name, shape: shape, sha256: hash)
}

func benchmarkModelStateDigests(_ model: StreamingReuseModel) -> [BenchmarkStateDigest] {
    eval(model.frameStateArrays)
    Stream.gpu.synchronize()
    var result = [BenchmarkStateDigest]()
    for (prefix, dictionary) in [("convolution", model.convolutionStates), ("recurrent", model.recurrentStates)] {
        for key in dictionary.keys.sorted() {
            guard let array = dictionary[key] else { continue }
            precondition(array.dtype == .float32, "Diagnostic state gate requires FP32 caches")
            result.append(benchmarkDigest("model/" + prefix + "/" + key,
                                          shape: array.shape, values: array.asArray(Float.self)))
        }
    }
    result.append(benchmarkDigest("model/first", shape: [1],
                                  values: [model.isFirstFrame ? Float(1) : Float(0)]))
    return result
}
