package io.riftco.reuseaudio

import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.security.MessageDigest
import kotlin.math.*

class ModelAndParityTests {
    @Test fun integrityRejectsTruncationAndWrongSha() {
        val file = Files.createTempFile("reuse-model", ".onnx").toFile()
        try {
            file.writeBytes(byteArrayOf(1, 2, 3))
            try { ModelManifest("0".repeat(64), 3).verify(file); fail() } catch (_: IllegalArgumentException) { }
            val hash = MessageDigest.getInstance("SHA-256").digest(file.readBytes()).joinToString("") { "%02x".format(it.toInt() and 255) }
            ModelManifest(hash, 3).verify(file)
            try { ModelManifest(hash, 4).verify(file); fail() } catch (_: IllegalArgumentException) { }
        } finally { file.delete() }
    }

    @Test fun tokenInterruptIsBoundAndCannotBeReusedConcurrently() {
        val token = CancellationToken(); var cancelled = false
        token.bind { cancelled = true }; token.cancel(); assertTrue(cancelled)
        try { token.bind {}; fail() } catch (_: IllegalStateException) { }
        token.unbind()
        try { token.throwIfCancelled(); fail() } catch (_: java.util.concurrent.CancellationException) { }
    }

    @Test fun independentlyExportedPytorchFixturesMatchKotlinDsp() {
        val path = System.getProperty("REUSE_PARITY_DIR")
        assumeTrue("Set REUSE_PARITY_DIR to generated independent oracle fixtures.", path != null)
        val directory = File(path!!)
        for (count in listOf(640, 8000, 64000)) {
            val source = File(directory, "$count-input.f32")
            if (!source.exists()) continue
            val input = readFloats(source)
            val analyzed = AudioTransform.analyze(input, CancellationToken())
            val expectedMagnitude = readFloats(File(directory, "$count-magnitude.f32"))
            val expectedPhase = readFloats(File(directory, "$count-phase.f32"))
            assertEquals(expectedMagnitude.size, analyzed.magnitude.size)
            assertTrue(analyzed.magnitude.indices.maxOf { abs(analyzed.magnitude[it] - expectedMagnitude[it]) } < 5e-5f)
            // Phase of near-zero bins is ill-conditioned; compare complex values instead.
            assertTrue(expectedMagnitude.indices.maxOf { i ->
                val mag = expm1(expectedMagnitude[i]); val actualMag = expm1(analyzed.magnitude[i])
                hypot(mag * cos(expectedPhase[i]) - actualMag * cos(analyzed.phase[i]),
                    mag * sin(expectedPhase[i]) - actualMag * sin(analyzed.phase[i]))
            } < .0003f)
            val expected = Spectrum(analyzed.frames,
                readFloats(File(directory, "$count-enhanced_magnitude.f32")),
                readFloats(File(directory, "$count-enhanced_phase.f32")))
            val actual = AudioTransform.synthesize(expected, count, CancellationToken())
            val waveform = readFloats(File(directory, "$count-waveform.f32"))
            assertTrue(actual.indices.maxOf { abs(actual[it] - waveform[it]) } < 2e-5f)
        }
    }

    @Test fun fullThirtyLayerOnnxCpuMatchesIndependentOracle() {
        val model = System.getProperty("REUSE_MODEL_FILE")
        val manifest = System.getProperty("REUSE_MODEL_MANIFEST")
        val fixtures = System.getProperty("REUSE_PARITY_DIR")
        assumeTrue("Set REUSE_MODEL_FILE, REUSE_MODEL_MANIFEST and REUSE_PARITY_DIR for real inference parity.",
            model != null && manifest != null && fixtures != null)
        OnnxSpectralProcessor(File(model!!), ModelManifest.read(File(manifest!!)), InferenceOptions(2)).use { processor ->
            for (count in listOf(640, 8000, 64000)) {
                val directory = File(fixtures!!)
                if (!File(directory, "$count-input.f32").exists()) continue
                val magnitude = readFloats(File(directory, "$count-magnitude.f32"))
                val phase = readFloats(File(directory, "$count-phase.f32"))
                val spectrum = Spectrum(count / 80 + 1, magnitude, phase)
                val output = processor.process(spectrum, CancellationToken())
                val expectedMag = readFloats(File(directory, "$count-enhanced_magnitude.f32"))
                val expectedPhase = readFloats(File(directory, "$count-enhanced_phase.f32"))
                assertTrue(output.magnitude.indices.maxOf { abs(output.magnitude[it] - expectedMag[it]) } < 1e-4f)
                val actualWave = AudioTransform.synthesize(output, count, CancellationToken())
                val expectedWave = readFloats(File(directory, "$count-waveform.f32"))
                assertTrue(actualWave.indices.maxOf { abs(actualWave[it] - expectedWave[it]) } < 5e-5f)
                assertTrue(expectedPhase.all(Float::isFinite))
                val input = readFloats(File(directory, "$count-input.f32"))
                val actualSpectrum = AudioTransform.analyze(input, CancellationToken())
                val fullOutput = processor.process(actualSpectrum, CancellationToken())
                val fullWave = AudioTransform.synthesize(fullOutput, count, CancellationToken())
                val error = fullWave.indices.maxOf { abs(fullWave[it] - expectedWave[it]) }
                val diagnostics = File("build/parity-diagnostics").also { it.mkdirs() }
                fun save(name: String, values: FloatArray) {
                    val buffer = ByteBuffer.allocate(values.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                    values.forEach(buffer::putFloat)
                    File(diagnostics, "$count-$name.f32").writeBytes(buffer.array())
                }
                save("magnitude", actualSpectrum.magnitude); save("phase", actualSpectrum.phase); save("waveform", fullWave)
                println("Full PCM $count samples max=$error")
                if (error >= 1e-4f) {
                    println("Full PCM phase branch differences: " + phase.indices.filter { abs(phase[it] - actualSpectrum.phase[it]) > 3f }
                        .take(20).map { "f=${it / spectrum.frames},t=${it % spectrum.frames},expected=${phase[it]},actual=${actualSpectrum.phase[it]}" })
                }
                assertTrue("Full PCM $count samples max=$error", error < 1e-4f)
            }
        }
    }

    @Test fun activeNativeInferenceCancellationDrainsAndSessionRemainsUsable() {
        val model = System.getProperty("REUSE_MODEL_FILE")
        val manifest = System.getProperty("REUSE_MODEL_MANIFEST")
        assumeTrue(model != null && manifest != null)
        OnnxSpectralProcessor(File(model!!), ModelManifest.read(File(manifest!!)), InferenceOptions(2)).use { processor ->
            val token = CancellationToken()
            val failure = java.util.concurrent.atomic.AtomicReference<Throwable>()
            val started = java.util.concurrent.CountDownLatch(1)
            val worker = Thread {
                started.countDown()
                try { processor.process(Spectrum(801, FloatArray(321 * 801) { .1f }, FloatArray(321 * 801)), token) }
                catch (error: Throwable) { failure.set(error) }
            }
            worker.start(); started.await(); Thread.sleep(50); token.cancel(); worker.join(5000)
            assertFalse("Native inference did not drain after cancellation.", worker.isAlive)
            assertTrue(failure.get() is java.util.concurrent.CancellationException)
            val next = processor.process(Spectrum(9, FloatArray(321 * 9) { .1f }, FloatArray(321 * 9)), CancellationToken())
            assertEquals(9, next.frames)
        }
    }

    private fun readFloats(file: File): FloatArray {
        val bytes = file.readBytes(); require(bytes.size % 4 == 0)
        return FloatArray(bytes.size / 4).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
    }
}
