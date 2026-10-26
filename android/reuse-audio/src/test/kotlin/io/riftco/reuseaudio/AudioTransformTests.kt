package io.riftco.reuseaudio

import org.junit.Assert.*
import org.junit.Test
import kotlin.math.*

class AudioTransformTests {
    @Test fun mixedRadixMatchesIndependentDftAndInverts() {
        val source = FloatArray(640) { sin(it * .071).toFloat() + .2f * cos(it * .293).toFloat() }
        val real = source.copyOf(); val imaginary = FloatArray(640)
        val fft = Fourier640(); fft.transform(real, imaginary)
        for (bin in listOf(0, 1, 5, 13, 64, 127, 128, 220, 319, 320, 639)) {
            var re = 0.0; var im = 0.0
            for (i in source.indices) {
                re += source[i] * cos(-2 * PI * bin * i / 640)
                im += source[i] * sin(-2 * PI * bin * i / 640)
            }
            assertEquals(re, real[bin].toDouble(), 5e-5)
            assertEquals(im, imaginary[bin].toDouble(), 5e-5)
        }
        fft.transform(real, imaginary, inverse = true)
        assertTrue(source.indices.maxOf { abs(source[it] - real[it]) } < 8e-7f)
    }

    @Test fun centeredReflectHannMatchesDirectTransform() {
        val input = FloatArray(720) { sin(.03 * it).toFloat() + .1f * cos(.19 * it).toFloat() }
        val spectrum = AudioTransform.analyze(input, CancellationToken())
        assertEquals(10, spectrum.frames)
        for (frame in listOf(0, 1, 8, 9)) for (bin in listOf(0, 1, 3, 19, 61, 320)) {
            var real = 0.0; var imaginary = 0.0
            for (i in 0 until 640) {
                val raw = frame * 80 + i - 320
                val reflected = when { raw < 0 -> -raw; raw >= input.size -> 2 * input.size - 2 - raw; else -> raw }
                val angle = i * (2f * PI.toFloat() / 640)
                val window = .5f - .5f * cos(angle)
                real += input[reflected] * window * cos(-2 * PI * bin * i / 640)
                imaginary += input[reflected] * window * sin(-2 * PI * bin * i / 640)
            }
            val at = bin * spectrum.frames + frame
            assertEquals(ln1p(hypot(real, imaginary)), spectrum.magnitude[at].toDouble(), 2e-5)
            if (hypot(real, imaginary) > .01 && bin != 320)
                assertTrue(abs(atan2(sin(spectrum.phase[at] - atan2(imaginary, real)), cos(spectrum.phase[at] - atan2(imaginary, real)))) < .001)
        }
    }

    @Test fun overlapReconstructionPreservesFinitePcm() {
        for (count in listOf(640, 720, 64_000)) {
            val input = FloatArray(count) { .42f * sin(it * .17).toFloat() }
            val spectrum = AudioTransform.analyze(input, CancellationToken())
            val actual = AudioTransform.synthesize(spectrum, count, CancellationToken())
            assertEquals(count, actual.size)
            assertTrue(input.indices.maxOf { abs(input[it] - actual[it]) } < 7e-7f)
        }
    }

    @Test fun sweepSuppressionZerosWholeFrameWhenMajorityBinsAreEmpty() {
        val frames = 9
        val magnitude = FloatArray(321 * frames) { ln1p(1f) }
        for (bin in 0..160) for (t in 0 until frames) magnitude[bin * frames + t] = -1f
        val actual = AudioTransform.synthesize(Spectrum(frames, magnitude, FloatArray(magnitude.size)), 640, CancellationToken())
        assertTrue(actual.all { it == 0f })
    }

    @Test fun rejectsNonfiniteSpectra() {
        try { Spectrum(9, FloatArray(321 * 9) { Float.NaN }, FloatArray(321 * 9)); fail() }
        catch (_: IllegalArgumentException) { }
    }

    @Test fun realFftEndpointsUsePositiveZeroImaginaryPhase() {
        val input = FloatArray(640) { if (it % 2 == 0) -1f else .5f }
        val spectrum = AudioTransform.analyze(input, CancellationToken())
        for (frame in 0 until spectrum.frames) {
            assertEquals(PI.toFloat(), spectrum.phase[frame], 0f)
            assertEquals(PI.toFloat(), spectrum.phase[320 * spectrum.frames + frame], 0f)
        }
    }
}
