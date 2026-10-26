// NVIDIA Real-time RE-USE adaptation; NVIDIA One-Way Noncommercial License (NSCLv1).
// Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
// See LICENSE in this module. Noncommercial research and education only.
import Foundation
import MLX

/// Online Hann STFT/weighted overlap-add, 40 ms window and 20 ms hop at 16 kHz.
final class StreamingTransform {
    private let window: MLXArray
    private let coefficients: [Float]
    private var overlap = [Float](repeating: 0, count: 640)
    private var envelope = [Float](repeating: 0, count: 640)
    private var first = true

    init() {
        coefficients = (0..<640).map { 0.5 - 0.5 * cos(2 * Float.pi * Float($0) / 640) }
        window = MLXArray(coefficients)
        eval(window)
    }

    func analyze(_ samples: [Float]) -> (MLXArray, MLXArray) {
        let spectrum = MLX.rfft(MLXArray(samples) * window, n: 640)
        // Real FFT endpoints are real-valued. Canonical positive zero matches
        // PyTorch's +pi phase for negative DC/Nyquist coefficients; Metal may
        // otherwise emit -0 at Nyquist and give the model a -pi input instead.
        let imaginary = concatenated([MLXArray.zeros([1]), spectrum.imaginaryPart()[1..<320],
                                      MLXArray.zeros([1])])
        return (log1p(abs(spectrum)).expandedDimensions(axis: 1),
                atan2(imaginary, spectrum.realPart()).expandedDimensions(axis: 1))
    }

    func synthesize(magnitude: MLXArray, phase: MLXArray,
                    evaluating stateArrays: [MLXArray] = []) -> [Float] {
        var mag = expm1(maximum(magnitude, 0)).reshaped([321])
        let empty = mean((mag .== 0).asType(.float32)) .> 0.5
        mag = which(empty, 0, mag)
        let angle = phase.reshaped([321])
        let spectrum = mag * cos(angle) + MLXArray(real: 0, imaginary: 1) * mag * sin(angle)
        let samples = MLX.irfft(spectrum, n: 640)
        // One completion point covers synthesis and every model cache. The
        // following CPU read consumes evaluated samples without new GPU work.
        if !stateArrays.isEmpty { eval([samples] + stateArrays) }
        return overlapAdd(samples.asArray(Float.self))
    }

    private func overlapAdd(_ samples: [Float]) -> [Float] {
        for index in 0..<640 {
            overlap[index] += samples[index] * coefficients[index]
            envelope[index] += coefficients[index] * coefficients[index]
        }
        let output = (0..<320).map { overlap[$0] / max(envelope[$0], 1e-12) }
        overlap = Array(overlap.suffix(320)) + [Float](repeating: 0, count: 320)
        envelope = Array(envelope.suffix(320)) + [Float](repeating: 0, count: 320)
        defer { first = false }
        return first ? [] : output
    }
}
