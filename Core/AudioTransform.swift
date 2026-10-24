// Swift/MLX adaptation of NVIDIA-NemotronLabs-RE-USE STFT and sweep suppression,
// revision 31f82f2efad1095034492f236e8dac9239ba713b. Original model materials:
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: OpenMDW-1.1
import Foundation
import MLX

enum AudioTransform {
    static let sampleRate = 16000
    static let fftSize = 640
    static let hop = 80
    static func window() -> MLXArray {
        let n = MLXArray(0..<fftSize).asType(.float32)
        return 0.5 - 0.5 * cos(n * (2 * Float.pi / Float(fftSize)))
    }
    /// Periodic Hann, centered reflect padding, unnormalized FFT, matching PyTorch.
    static func analyze(_ samples: [Float]) -> (MLXArray, MLXArray) {
        precondition(samples.count > fftSize / 2)
        let pad = fftSize / 2
        let paddedSamples = Array(samples[1...pad].reversed()) + samples
            + Array(samples[(samples.count-pad-1)..<(samples.count-1)].reversed())
        let frames = 1 + samples.count / hop
        let framed = asStrided(MLXArray(paddedSamples), [frames, fftSize], strides: [hop, 1])
        let spectrum = MLX.rfft(framed * window(), n: fftSize)
        return (log1p(abs(spectrum)).T, atan2(spectrum.imaginaryPart(), spectrum.realPart()).T)
    }
    static func synthesize(magnitude: MLXArray, phase: MLXArray, count: Int,
                           suppressSweep: Bool = true) -> [Float] {
        var mag = expm1(maximum(magnitude, 0))
        if suppressSweep {
            let empty = mean((mag .== 0).asType(.float32), axis: 0) .> 0.5
            mag = which(empty.expandedDimensions(axis: 0), 0, mag)
        }
        let complex = mag * cos(phase) + MLXArray(real: 0, imaginary: 1) * mag * sin(phase)
        let frames = MLX.irfft(complex.T, n: fftSize).asArray(Float.self)
        let win = window().asArray(Float.self)
        let nFrames = magnitude.dim(1)
        let (signal, envelope) = overlapFrames(frames, window: win, count: nFrames)
        return reconstruct(signal, envelope: envelope, frames: nFrames, count: count)
    }
    private static func reconstruct(_ signal: [Float], envelope: [Float], frames: Int, count: Int) -> [Float] {
        let available = min(count, (frames - 1) * hop)
        var result = [Float](repeating: 0, count: count)
        for i in 0..<available {
            let j = i + fftSize / 2
            result[i] = signal[j] / max(envelope[j], 1e-12)
        }
        return result
    }
    private static func overlapFrames(_ frames: [Float], window: [Float], count: Int) -> ([Float], [Float]) {
        let total = (count - 1) * hop + fftSize
        var signal = [Float](repeating: 0, count: total)
        var envelope = signal
        for t in 0..<count {
            for k in 0..<fftSize {
                signal[t*hop+k] += frames[t*fftSize+k] * window[k]
                envelope[t*hop+k] += window[k] * window[k]
            }
        }
        return (signal, envelope)
    }
}
