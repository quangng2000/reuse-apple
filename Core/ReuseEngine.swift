// Offline windowing wrapper for the NVIDIA-NemotronLabs-RE-USE Apple adaptation.
// SPDX-License-Identifier: OpenMDW-1.1
import Foundation
import MLX

private struct WindowPosition {
    let start: Int
    let valid: Int
    let index: Int
    let count: Int
}

/// Four-second windows with 0.5-second crossfades. This offline adaptation
/// restarts bidirectional context and instance normalization for each window.
final class ReuseEngine {
    let model: ReuseModel
    private let window = 64_000
    private let overlap = 8_000
    init(modelURL: URL, half: Bool = false) throws {
        GPU.set(cacheLimit: 64 * 1024 * 1024)
        model = try ReuseModel(url: modelURL, half: half)
    }
    func enhance(_ input: [Float], progress: @escaping (Double) throws -> Void) throws -> [Float] {
        guard !input.isEmpty, input.allSatisfy({ $0.isFinite }) else {
            throw ReuseError.invalid("The recording is empty or contains invalid samples.")
        }
        let result = try renderWindows(input, progress: progress)
        try progress(1)
        return result
    }
    private func renderWindows(_ input: [Float], progress: @escaping (Double) throws -> Void) throws -> [Float] {
        let starts = Array(Swift.stride(from: 0, to: max(1, input.count - overlap), by: window - overlap))
        var result = [Float](repeating: 0, count: input.count)
        var weight = result
        for (index, start) in starts.enumerated() {
            try progress(Double(index) / Double(starts.count))
            let end = min(start + window, input.count), valid = end - start
            let output = try processWindow(Array(input[start..<end]), progress: { block in
                try progress((Double(index) + Double(block)/30) / Double(starts.count))
            })
            blend(output, into: &result, weight: &weight,
                  position: WindowPosition(start: start, valid: valid, index: index, count: starts.count))
            GPU.clearCache()
        }
        for i in result.indices { result[i] /= max(weight[i], 1e-8) }
        return result
    }
    private func processWindow(_ input: [Float], progress: @escaping (Int) throws -> Void) throws -> [Float] {
        var chunk = input
        let paddedCount = max(640, ((chunk.count + 79) / 80) * 80)
        chunk += [Float](repeating: 0, count: paddedCount - chunk.count)
        let (mag, pha) = AudioTransform.analyze(chunk)
        let (pm, pp) = try model(magnitude: mag, phase: pha, progress: progress)
        let output = AudioTransform.synthesize(magnitude: pm, phase: pp, count: chunk.count)
        guard output.allSatisfy({ $0.isFinite }) else {
            throw ReuseError.invalid("The model produced invalid audio. The input is unchanged.")
        }
        return output
    }
    private func blend(_ output: [Float], into result: inout [Float], weight: inout [Float],
                       position: WindowPosition) {
        for i in 0..<position.valid {
            var gain: Float = 1
            if position.index > 0 && i < overlap { gain *= Float(i+1) / Float(overlap+1) }
            if position.index < position.count-1 && i >= position.valid-overlap {
                gain *= Float(position.valid-i) / Float(overlap+1)
            }
            result[position.start+i] += output[i] * gain
            weight[position.start+i] += gain
        }
    }
}
