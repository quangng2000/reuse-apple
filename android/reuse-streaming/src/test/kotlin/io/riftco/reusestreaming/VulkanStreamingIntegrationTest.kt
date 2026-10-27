package io.riftco.reusestreaming

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

/** Uses a real local Vulkan GPU. The exact same native sources and shaders build for Android. */
class VulkanStreamingIntegrationTest {
    private fun value(name: String) = System.getProperty(name) ?: System.getenv(name)
    private data class Fixture(val directory: File, val library: File, val model: File,
                               val manifest: StreamingModelManifest, val parity: File)
    private fun fixture(): Fixture {
        val path = value("REUSE_VULKAN_DIRECTORY")
        assumeTrue("Provide the separately built Vulkan artifacts for GPU integration tests.", path != null)
        return Fixture(File(path!!), File(requireNotNull(value("REUSE_VULKAN_LIBRARY"))),
            File(requireNotNull(value("REUSE_STREAMING_MODEL_FILE"))),
            StreamingModelManifest.read(File(requireNotNull(value("REUSE_STREAMING_MODEL_MANIFEST")))),
            File(requireNotNull(value("REUSE_STREAMING_PARITY_DIR"))))
    }
    private fun options() = StreamingEnhancerConfiguration(8, 4)
    private fun gpu(f: Fixture) = ExperimentalVulkanStreaming.create(f.directory, f.library, options())
    private fun gpuTiled(f: Fixture) = ExperimentalVulkanStreaming.createTiled(f.directory, f.library, options())
    private fun floats(path: File): FloatArray {
        val raw = path.readBytes(); require(raw.size % 4 == 0)
        val buffer = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }
    private fun compare(a: FloatArray, b: FloatArray, maximum: Double = 5e-6, relative: Double = 1e-5) {
        assertEquals(a.size, b.size); var error = 0.0; var signal = 0.0; var worst = 0.0
        a.indices.forEach { i ->
            assertTrue(b[i].isFinite()); val delta = b[i].toDouble() - a[i]
            error += delta * delta; signal += a[i].toDouble() * a[i]; worst = maxOf(worst, abs(delta))
        }
        val rms = sqrt(error / maxOf(signal, 1e-30))
        println("Vulkan waveform samples=${a.size} maxAbs=$worst relativeRms=$rms")
        assertTrue("maxAbs=$worst", worst <= maximum); assertTrue("relativeRms=$rms", rms <= relative)
    }
    private fun bits(a: FloatArray, b: FloatArray) {
        assertEquals(a.size, b.size)
        a.indices.forEach { i -> if (a[i].toRawBits() != b[i].toRawBits()) fail("Different replay bits at $i") }
    }

    @Test fun publishedWaveformsPartialChunksAndResetMatchAcrossSevenLengths() {
        val f = fixture()
        StreamingEnhancer(f.model, f.manifest, options()).use { cpu -> gpu(f).use { vk ->
            val receipt = JSONObject(File(f.parity, "parity.json").readText())
            val cases = receipt.getJSONArray("fixtures")
            assertEquals(7, cases.length())
            repeat(cases.length()) { i ->
                val spec = cases.getJSONObject(i); val input = floats(File(f.parity, spec.getString("inputFile")))
                cpu.reset(); vk.reset()
                val expected = cpu.push(input) + cpu.finish()
                val actual = vk.push(input) + vk.finish()
                compare(expected, actual)
                vk.reset(); val replay = ArrayList<Float>(); var offset = 0
                val chunks = spec.getJSONArray("chunks")
                repeat(chunks.length()) { index ->
                    val end = offset + chunks.getInt(index)
                    replay.addAll(vk.push(input.copyOfRange(offset, end)).toList()); offset = end
                }
                replay.addAll(vk.finish().toList()); bits(actual, replay.toFloatArray())
            }
            println("Vulkan device=${vk.backendDiagnosticsJson}")
        } }
    }

    @Test fun allCarriedStatesMatchAndResetReplaysOnGPU() {
        val f = fixture()
        ExperimentalVulkanStreaming.verify(f.directory, f.library, options())
        // Factory verifies/loads the native library before the internal diagnostic processor is used.
        gpu(f).close()
        val receipt = JSONObject(File(f.parity, "parity.json").readText()).getJSONObject("spectral")
        val magnitude = floats(File(f.parity, receipt.getString("magnitudeFile")))
        val phase = floats(File(f.parity, receipt.getString("phaseFile")))
        OnnxStreamingProcessor(f.model, f.manifest, options()).use { cpu ->
            VulkanStreamingProcessor(f.directory).use { vk ->
                var first: StreamingSpectrum? = null
                var firstStates: List<FloatArray>? = null
                repeat(receipt.getInt("frames")) { frame ->
                    val offset = frame * 321
                    val input = StreamingSpectrum(magnitude.copyOfRange(offset, offset + 321), phase.copyOfRange(offset, offset + 321))
                    val reference = cpu.process(input, StreamingCancellationToken())
                    val actual = vk.process(input, StreamingCancellationToken())
                    for (i in 0 until 321) {
                        assertTrue(abs(reference.magnitude[i] - actual.magnitude[i]) <= .003)
                        val delta = reference.phase[i].toDouble() - actual.phase[i]
                        assertTrue(abs(atan2(sin(delta), cos(delta))) <= .003)
                    }
                    val expected = cpu.copyStateForValidation(); val observed = vk.copyStateForValidation()
                    assertEquals(31, observed.size)
                    expected.indices.forEach { state ->
                        assertEquals(expected[state].size, observed[state].size)
                        expected[state].indices.forEach { i ->
                            val delta = abs(expected[state][i].toDouble() - observed[state][i])
                            if (!observed[state][i].isFinite() || delta > .0003 + .0002 * abs(expected[state][i]))
                                fail("Vulkan state $state frame $frame element $i error $delta")
                        }
                    }
                    if (frame == 0) { first = actual; firstStates = observed }
                }
                vk.reset()
                val replay = vk.process(StreamingSpectrum(magnitude.copyOfRange(0, 321), phase.copyOfRange(0, 321)), StreamingCancellationToken())
                bits(first!!.magnitude, replay.magnitude); bits(first!!.phase, replay.phase)
                val replayStates = vk.copyStateForValidation()
                firstStates!!.indices.forEach { bits(firstStates!![it], replayStates[it]) }
            }
        }
    }

    @Test fun cancelledAndTooShortInputsPreserveStreamingLifecycle() {
        val f = fixture(); gpu(f).use { vk ->
            val cancelled = StreamingCancellationToken().apply { cancel() }
            assertThrows(java.util.concurrent.CancellationException::class.java) { vk.push(FloatArray(640), cancelled) }
            vk.reset()
            for (n in listOf(0, 1, 319, 320, 639)) {
                vk.reset(); assertEquals(0, vk.push(FloatArray(n)).size)
                assertEquals(StreamingException.Reason.TOO_SHORT,
                    assertThrows(StreamingException::class.java) { vk.finish() }.reason)
                assertEquals(640, (vk.push(FloatArray(640 - n)) + vk.finish()).size)
            }
            vk.close(); vk.close()
            assertThrows(IllegalStateException::class.java) { vk.reset() }
        }
    }

    @Test fun corruptArtifactAndWrongLayerFailBeforeNativeLoad() {
        val f = fixture()
        assertThrows(IllegalArgumentException::class.java) {
            ExperimentalVulkanStreaming.verify(f.directory, f.library, StreamingEnhancerConfiguration(4, 1))
        }
        val temp = kotlin.io.path.createTempDirectory("vulkan-corrupt-").toFile()
        try {
            File(temp, "manifest.json").writeText("{}")
            assertThrows(IllegalArgumentException::class.java) { ExperimentalVulkanStreaming.verify(temp, f.library, options()) }
        } finally { temp.deleteRecursively() }
    }

    @Test fun tiledKernelsMatchDirectWithExactResetAndChunkReplay() {
        val f = fixture()
        gpu(f).use { direct -> gpuTiled(f).use { tiled ->
            val input = floats(File(f.parity, "16000-input.f32"))
            val reference = direct.push(input) + direct.finish()
            val actual = tiled.push(input) + tiled.finish()
            compare(reference, actual)
            tiled.reset(); val replay = ArrayList<Float>(); var offset = 0
            while (offset < input.size) {
                val end = minOf(offset + 317, input.size)
                replay.addAll(tiled.push(input.copyOfRange(offset, end)).toList()); offset = end
            }
            replay.addAll(tiled.finish().toList()); bits(actual, replay.toFloatArray())
        } }
    }

    @Test fun noisySpeechMatchesPublishedWithUnchangedFP32QualityThresholds() {
        val f = fixture()
        val inputFile = value("REUSE_VULKAN_QUALITY_INPUT")
        assumeTrue("Provide the fixed four-second synthetic noisy speech sample.", inputFile != null)
        val input = floats(File(inputFile!!)); assertEquals(64000, input.size)
        val outputs = File(requireNotNull(value("REUSE_VULKAN_QUALITY_OUTPUT"))).apply { mkdirs() }
        fun run(enhancer: StreamingEnhancer): FloatArray {
            val output = ArrayList<Float>(); var offset = 0
            while (offset < input.size) {
                val end = minOf(offset + 320, input.size)
                output.addAll(enhancer.push(input.copyOfRange(offset, end)).toList()); offset = end
            }
            output.addAll(enhancer.finish().toList()); return output.toFloatArray()
        }
        StreamingEnhancer(f.model, f.manifest, options()).use { cpu -> gpu(f).use { vk -> gpuTiled(f).use { tiled ->
            val reference = run(cpu); val candidate = run(vk); val tiledOutput = run(tiled)
            for ((name, audio) in listOf("reference" to reference, "vulkan" to candidate, "vulkan-tiled" to tiledOutput)) {
                val buffer = ByteBuffer.allocate(audio.size * 4).order(ByteOrder.LITTLE_ENDIAN)
                buffer.asFloatBuffer().put(audio); File(outputs, "$name.f32").writeBytes(buffer.array())
            }
            compare(reference, candidate)
            compare(reference, tiledOutput)
        } } }
    }
}
