// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

import kotlin.math.*

internal data class StreamingSpectrum(val magnitude: FloatArray, val phase: FloatArray) {
    init {
        require(magnitude.size == 321 && phase.size == 321)
        require(magnitude.all(Float::isFinite) && phase.all(Float::isFinite)) { "Invalid spectral frame." }
    }
}

/** Online Hann STFT and weighted overlap-add. Retains only one frame of overlap. */
internal class StreamingTransform {
    private val window = PeriodicHann640.values
    private val overlap = FloatArray(640)
    private val envelope = FloatArray(640)
    private var first = true

    fun analyze(samples: FloatArray): StreamingSpectrum {
        require(samples.size == 640)
        val real = FloatArray(640) { samples[it] * window[it] }
        val imaginary = FloatArray(640)
        StreamingFourier640.forward(real, imaginary)
        return StreamingSpectrum(
            FloatArray(321) { ln1p(sqrt(real[it] * real[it] + imaginary[it] * imaginary[it])) },
            FloatArray(321) { atan2(if (it == 0 || it == 320) 0f else imaginary[it], real[it]) },
        )
    }

    fun synthesize(spectrum: StreamingSpectrum): FloatArray {
        val magnitude = FloatArray(321) { expm1(max(spectrum.magnitude[it], 0f)) }
        val empty = magnitude.count { it == 0f } > 160
        val real = FloatArray(640)
        val imaginary = FloatArray(640)
        for (i in 0..320) {
            val mag = if (empty) 0f else magnitude[i]
            real[i] = mag * cos(spectrum.phase[i])
            imaginary[i] = if (i == 0 || i == 320) 0f else mag * sin(spectrum.phase[i])
        }
        StreamingFourier640.inverse(real, imaginary)
        for (i in 0 until 640) {
            overlap[i] += real[i] * window[i]
            envelope[i] += window[i] * window[i]
        }
        val output = if (first) FloatArray(0) else FloatArray(320) { overlap[it] / max(envelope[it], 1e-12f) }
        overlap.copyInto(overlap, 0, 320, 640)
        envelope.copyInto(envelope, 0, 320, 640)
        overlap.fill(0f, 320)
        envelope.fill(0f, 320)
        first = false
        return output
    }
}
