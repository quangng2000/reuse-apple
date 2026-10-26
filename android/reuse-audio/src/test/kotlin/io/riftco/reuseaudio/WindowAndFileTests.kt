package io.riftco.reuseaudio

import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.util.concurrent.CancellationException
import kotlin.math.*

class WindowAndFileTests {
    private val identity = SpectralProcessor { input, _ -> input }

    @Test fun boundaryWindowsAndLongClipRetainLengthAndBlend() {
        for (count in listOf(1, 639, 640, 641, 8_000, 56_000, 56_001, 64_000, 64_001, 112_000, 112_001, 960_001)) {
            val input = FloatArray(count) { .33f * sin(it * .073).toFloat() }
            val actual = ArrayList<Float>()
            var largestRead = 0; var lastProgress = -1.0
            WindowedEnhancer.render(count.toLong(), { start, length ->
                largestRead = max(largestRead, length)
                input.copyOfRange(start.toInt(), start.toInt() + length)
            }, { block, length -> repeat(length) { actual.add(block[it]) } }, identity, CancellationToken()) {
                assertTrue(it.isFinite() && it in 0.0..1.0 && it >= lastProgress); lastProgress = it
            }
            assertEquals(count, actual.size); assertTrue(largestRead <= 64_000)
            assertTrue(input.indices.maxOf { abs(input[it] - actual[it]) } < 1e-6f)
            assertEquals(1.0, lastProgress, 0.0)
        }
    }

    @Test fun cancelledWindowDoesNotEmitLateOutput() {
        val token = CancellationToken()
        var emitted = false
        try {
            WindowedEnhancer.render(64_001, { _, count -> FloatArray(count) { .1f } }, { _, _ -> emitted = true },
                SpectralProcessor { input, _ -> token.cancel(); input }, token, {})
            fail()
        } catch (_: CancellationException) { }
        assertFalse(emitted)
    }

    @Test fun stereo48kResamplingIsBoundedAndPreservesTone() = withDirectory { directory ->
        val source = File(directory, "stereo.wav")
        val input = FloatArray(48_000) { .4f * sin(2 * PI * 440 * it / 48_000).toFloat() }
        writePcm16Stereo(source, input, 48_000)
        val decoded = WavFiles.decode(source, File(directory, "decoded.f32"), CancellationToken())
        val normalized = PcmResampler.normalize(decoded, File(directory, "normalized.f32"), CancellationToken())
        assertEquals(16_000, normalized.sampleCount)
        FloatReader(normalized.file).use { reader ->
            val result = reader.read(0, 16_000)
            assertTrue((100 until 15_900).maxOf { abs(result[it] - .4f * sin(2 * PI * 440 * it / 16_000).toFloat()) } < .0002)
        }
    }

    @Test fun fileUsesOneGlobalGainAndKeepsInput() = withDirectory { directory ->
        val raw = File(directory, "source.f32")
        val input = FloatArray(72_123) { if (it < 56_000) .8f * sin(.11 * it).toFloat() else .2f * sin(.11 * it).toFloat() }
        FloatWriter(raw).use { it.write(input) }
        val source = File(directory, "input.wav"); WavFiles.write(raw, source, 1f, CancellationToken())
        val originalBytes = source.readBytes()
        val output = File(directory, "output.wav")
        val amplifier = SpectralProcessor { spectrum, _ ->
            Spectrum(spectrum.frames, FloatArray(spectrum.magnitude.size) { ln1p(expm1(spectrum.magnitude[it]) * 4f) }, spectrum.phase)
        }
        val result = FileEnhancement.render(source, output, directory, amplifier, ModelManifest.MODEL_ID, CancellationToken(), {})
        assertEquals(input.size.toLong(), result.sampleCount)
        assertEquals(16_000, result.sampleRate)
        assertTrue(result.appliedGain < 1 && result.peakBeforeGain > 3)
        val decoded = WavFiles.decode(output, File(directory, "result.f32"), CancellationToken())
        FloatReader(decoded.file).use { reader ->
            val actual = reader.read(0, input.size)
            assertTrue(input.indices.maxOf { abs(actual[it] - input[it] * 4f * result.appliedGain) } < 4e-6)
            assertTrue(actual.maxOf { abs(it) } <= .980001f)
        }
        assertArrayEquals(originalBytes, source.readBytes())
        assertFalse(directory.listFiles()!!.any { it.name.startsWith("reuse-") || it.name.startsWith(".reuse-") })
    }

    @Test fun cancellationAndProcessorFailurePreserveExistingDestination() = withDirectory { directory ->
        val raw = File(directory, "source.f32"); FloatWriter(raw).use { it.write(FloatArray(640) { .2f }) }
        val source = File(directory, "input.wav"); WavFiles.write(raw, source, 1f, CancellationToken())
        val output = File(directory, "saved.wav"); output.writeText("existing")
        for (cancel in listOf(false, true)) {
            val token = CancellationToken()
            try {
                FileEnhancement.render(source, output, directory, SpectralProcessor { _, _ ->
                    if (cancel) { token.cancel(); token.throwIfCancelled() }
                    error("Expected failure")
                }, ModelManifest.MODEL_ID, token, {})
                fail()
            } catch (_: IllegalStateException) { } catch (_: CancellationException) { }
            assertEquals("existing", output.readText())
            assertFalse(directory.listFiles()!!.any { it.name.startsWith("reuse-") || it.name.startsWith(".reuse-") })
        }
    }

    @Test fun failedDestinationWriteCleansScratchAndKeepsSource() = withDirectory { directory ->
        val raw = File(directory, "source.f32"); FloatWriter(raw).use { it.write(FloatArray(640) { .2f }) }
        val source = File(directory, "input.wav"); WavFiles.write(raw, source, 1f, CancellationToken())
        val output = File(directory, "occupied").also { it.mkdir(); File(it, "keep").writeText("keep") }
        try { FileEnhancement.render(source, output, directory, identity, ModelManifest.MODEL_ID, CancellationToken(), {}); fail() }
        catch (_: java.io.IOException) { }
        assertEquals("keep", File(output, "keep").readText())
        assertTrue(source.isFile)
        assertFalse(directory.listFiles()!!.any { it.name.startsWith("reuse-") || it.name.startsWith(".reuse-") })
    }

    private fun withDirectory(block: (File) -> Unit) {
        val directory = Files.createTempDirectory("reuse-test").toFile()
        try { block(directory) } finally { directory.deleteRecursively() }
    }
    private fun writePcm16Stereo(file: File, input: FloatArray, rate: Int) {
        val bytes = ByteBuffer.allocate(44 + input.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        bytes.put("RIFF".toByteArray()).putInt(36 + input.size * 4).put("WAVEfmt ".toByteArray())
            .putInt(16).putShort(1).putShort(2).putInt(rate).putInt(rate * 4).putShort(4).putShort(16)
            .put("data".toByteArray()).putInt(input.size * 4)
        for (sample in input) repeat(2) { bytes.putShort((sample * 32767).toInt().toShort()) }
        file.writeBytes(bytes.array())
    }
}
