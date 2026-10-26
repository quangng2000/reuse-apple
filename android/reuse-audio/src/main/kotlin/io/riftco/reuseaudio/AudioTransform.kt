// RE-USE STFT adaptation, source revision 31f82f2efad1095034492f236e8dac9239ba713b.
// SPDX-License-Identifier: OpenMDW-1.1
package io.riftco.reuseaudio

import kotlin.math.*

internal data class Spectrum(val frames: Int, val magnitude: FloatArray, val phase: FloatArray) {
    init {
        require(frames in 9..801 && magnitude.size == 321 * frames && phase.size == magnitude.size)
        require(magnitude.all(Float::isFinite) && phase.all(Float::isFinite)) { "Invalid spectrum." }
    }
}

/** Float32 transforms, periodic Hann, centered reflect padding, unnormalized forward FFT. */
internal object AudioTransform {
    const val FFT = 640
    const val HOP = 80
    private val window = PeriodicHann640.values.copyOf()

    fun analyze(samples: FloatArray, cancellation: CancellationToken): Spectrum {
        require(samples.size in 640..64_000 && samples.size % HOP == 0)
        require(samples.all(Float::isFinite)) { "Invalid audio samples." }
        val frames = 1 + samples.size / HOP
        val magnitude = FloatArray(321 * frames)
        val phase = FloatArray(magnitude.size)
        val real = FloatArray(FFT)
        val imag = FloatArray(FFT)
        val fft = Fourier640()
        for (frame in 0 until frames) {
            cancellation.throwIfCancelled()
            for (k in 0 until FFT) {
                val raw = frame * HOP + k - FFT / 2
                val index = when {
                    raw < 0 -> -raw
                    raw >= samples.size -> 2 * samples.size - 2 - raw
                    else -> raw
                }
                real[k] = samples[index] * window[k]
                imag[k] = 0f
            }
            fft.transform(real, imag)
            for (frequency in 0..320) {
                val at = frequency * frames + frame
                magnitude[at] = ln1p(sqrt(real[frequency] * real[frequency] + imag[frequency] * imag[frequency]))
                // A real FFT's DC/Nyquist bins are real with +0 imaginary part. A generic
                // complex FFT leaves roundoff here; atan2(-epsilon, negative) would flip
                // +pi to -pi even though the complex spectrum is almost unchanged.
                phase[at] = atan2(if (frequency == 0 || frequency == 320) 0f else imag[frequency], real[frequency])
            }
        }
        return Spectrum(frames, magnitude, phase)
    }

    fun synthesize(spectrum: Spectrum, count: Int, cancellation: CancellationToken): FloatArray {
        require(count in 640..64_000 && count == (spectrum.frames - 1) * HOP)
        val real = FloatArray(FFT)
        val imag = FloatArray(FFT)
        val fft = Fourier640()
        val signal = FloatArray((spectrum.frames - 1) * HOP + FFT)
        val envelope = FloatArray(signal.size)
        val magnitudes = FloatArray(321)
        for (frame in 0 until spectrum.frames) {
            cancellation.throwIfCancelled()
            var zeros = 0
            for (frequency in 0..320) {
                magnitudes[frequency] = expm1(max(spectrum.magnitude[frequency * spectrum.frames + frame], 0f))
                if (magnitudes[frequency] == 0f) zeros++
            }
            // Same frame-wide sweep suppression: more than half of bins are exactly zero.
            for (frequency in 0..320) {
                val mag = if (zeros > 160) 0f else magnitudes[frequency]
                val phase = spectrum.phase[frequency * spectrum.frames + frame]
                real[frequency] = mag * cos(phase)
                imag[frequency] = if (frequency == 0 || frequency == 320) 0f else mag * sin(phase)
            }
            for (frequency in 321 until FFT) {
                real[frequency] = real[FFT - frequency]
                imag[frequency] = -imag[FFT - frequency]
            }
            fft.transform(real, imag, inverse = true)
            for (k in 0 until FFT) {
                val at = frame * HOP + k
                signal[at] += real[k] * window[k]
                envelope[at] += window[k] * window[k]
            }
        }
        return FloatArray(count) { i -> signal[i + FFT / 2] / max(envelope[i + FFT / 2], 1e-12f) }
            .also { require(it.all(Float::isFinite)) { "Model produced invalid audio." } }
    }
}

/** Same Float32 PocketFFT backend as the reference; no FP16 or fast-math path. */
internal class Fourier640 {
    fun transform(real: FloatArray, imag: FloatArray, inverse: Boolean = false) {
        require(real.size == 640 && imag.size == 640)
        if (inverse) NativeFourier640.inverse(real, imag) else NativeFourier640.forward(real, imag)
    }
}
