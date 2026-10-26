package io.riftco.reusestreaming

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import java.util.concurrent.CancellationException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.ExecutionException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.math.*

/** Actual FP32 ORT graph against independent pinned-source fixtures; no generated model in the AAR. */
class StreamingModelIntegrationTest {
    private data class Fixture(val model: File, val manifest: StreamingModelManifest, val directory: File, val json: JSONObject)

    private fun fixture(): Fixture {
        fun setting(name: String) = System.getProperty(name) ?: System.getenv(name)
        val names = listOf("REUSE_STREAMING_MODEL_FILE", "REUSE_STREAMING_MODEL_MANIFEST", "REUSE_STREAMING_PARITY_DIR")
        val values = names.map(::setting)
        assumeTrue("Provide real streaming model and parity fixtures to run the integration suite.", values.any { it != null })
        require(values.all { !it.isNullOrBlank() }) { "Provide all three REUSE_STREAMING model/manifest/parity paths." }
        val model = File(values[0]!!)
        val receipt = File(values[1]!!)
        val directory = File(values[2]!!)
        require(model.isFile && receipt.isFile && directory.isDirectory) { "Missing real-model fixture files." }
        val manifest = StreamingModelManifest.read(receipt)
        val json = JSONObject(File(directory, "parity.json").readText())
        require(json.getInt("schemaVersion") == 1 && json.getInt("layers") == manifest.layers &&
            json.getString("modelSha256") == manifest.modelSha256 &&
            json.getString("checkpointSha256") == StreamingModelManifest.CHECKPOINT_SHA256 &&
            json.getString("dtype") == "float32" && json.getBoolean("passed")) { "Fixture and model contract differ." }
        val files = json.getJSONObject("files")
        for (name in files.keys()) {
            require(name.matches(Regex("[A-Za-z0-9_.-]+")))
            val record = files.getJSONObject(name)
            val file = File(directory, name)
            require(record.getString("file") == name && file.isFile && file.length() == record.getLong("bytes"))
            val sha = MessageDigest.getInstance("SHA-256").digest(file.readBytes()).joinToString("") { "%02x".format(it.toInt() and 255) }
            require(sha == record.getString("sha256")) { "Fixture hash mismatch: $name" }
        }
        return Fixture(model, manifest, directory, json)
    }

    private fun floats(directory: File, name: String): FloatArray {
        require(name.matches(Regex("[A-Za-z0-9_.-]+")))
        val bytes = File(directory, name).readBytes()
        require(bytes.isNotEmpty() && bytes.size % 4 == 0)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it); require(it.all(Float::isFinite)) }
    }

    private fun assertClose(expected: FloatArray, actual: FloatArray, absolute: Float, relative: Float,
                            label: String, phase: Boolean = false) {
        assertEquals("$label sample count", expected.size, actual.size)
        var worst = 0f
        for (i in expected.indices) {
            assertTrue("$label nonfinite at $i", actual[i].isFinite())
            var error = abs(expected[i] - actual[i])
            if (phase) error = minOf(error, abs(error - (2 * PI).toFloat()))
            worst = maxOf(worst, error)
            assertTrue("$label at $i: expected=${expected[i]}, actual=${actual[i]}, error=$error",
                error <= absolute + relative * abs(expected[i]))
        }
        println("$label max absolute error=$worst")
    }

    @Test fun actualStatefulOnnxMatchesIndependentSpectralCarryAndReset() {
        val fixture = fixture()
        val spec = fixture.json.getJSONObject("spectral")
        val frames = spec.getInt("frames")
        assertTrue(frames >= 3)
        val magnitude = floats(fixture.directory, spec.getString("magnitudeFile"))
        val phase = floats(fixture.directory, spec.getString("phaseFile"))
        val expectedMagnitude = floats(fixture.directory, spec.getString("enhancedMagnitudeFile"))
        val expectedPhase = floats(fixture.directory, spec.getString("enhancedPhaseFile"))
        assertEquals(frames * 321, magnitude.size)
        assertEquals(magnitude.size, phase.size)
        val options = StreamingEnhancerConfiguration(fixture.manifest.layers, 2)
        OnnxStreamingProcessor(fixture.model, fixture.manifest, options).use { processor ->
            var carriedSecond: StreamingSpectrum? = null
            repeat(frames) { frame ->
                val start = frame * 321
                val output = processor.process(StreamingSpectrum(magnitude.copyOfRange(start, start + 321),
                    phase.copyOfRange(start, start + 321)), StreamingCancellationToken())
                assertClose(expectedMagnitude.copyOfRange(start, start + 321), output.magnitude, 5e-4f, 2e-4f, "spectral magnitude hop $frame")
                assertClose(expectedPhase.copyOfRange(start, start + 321), output.phase, 2e-3f, 1e-4f, "spectral phase hop $frame", true)
                if (frame == 1) carriedSecond = output
            }
            processor.reset()
            val repeated = processor.process(StreamingSpectrum(magnitude.copyOf(321), phase.copyOf(321)), StreamingCancellationToken())
            assertClose(expectedMagnitude.copyOf(321), repeated.magnitude, 5e-4f, 2e-4f, "reset first magnitude")
            processor.reset()
            val resetSecond = processor.process(StreamingSpectrum(magnitude.copyOfRange(321, 642), phase.copyOfRange(321, 642)), StreamingCancellationToken())
            assertTrue("The graph must retain temporal state between hops.",
                carriedSecond!!.magnitude.indices.any { abs(carriedSecond!!.magnitude[it] - resetSecond.magnitude[it]) > 1e-5f })
        }
    }

    @Test fun actualModelWaveformParityChunkIndependenceAndReset() {
        val fixture = fixture()
        val cases = fixture.json.getJSONArray("fixtures")
        assertTrue(cases.length() >= 1)
        StreamingEnhancer(fixture.model, fixture.manifest, StreamingEnhancerConfiguration(fixture.manifest.layers, 2)).use { enhancer ->
            for (i in 0 until cases.length()) {
                val case = cases.getJSONObject(i)
                val name = case.getString("name")
                val input = floats(fixture.directory, case.getString("inputFile"))
                val expected = floats(fixture.directory, case.getString("outputFile"))
                assertEquals(case.getInt("samples"), input.size)
                enhancer.reset()
                val actual = enhancer.push(input) + enhancer.finish()
                assertClose(expected, actual, 3e-4f, 0f, "$name waveform")
                if (i == 0 || i == cases.length() - 1) {
                    enhancer.reset()
                    val chunks = intArrayOf(1, 17, 320, 641, 97)
                    val output = ArrayList<Float>()
                    var at = 0; var chunk = 0
                    while (at < input.size) {
                        val end = minOf(at + chunks[chunk++ % chunks.size], input.size)
                        output.addAll(enhancer.push(input.copyOfRange(at, end)).toList()); at = end
                    }
                    output.addAll(enhancer.finish().toList())
                    assertArrayEquals("Real-model output must not depend on caller chunk boundaries.", actual, output.toFloatArray(), 0f)
                }
            }
            enhancer.reset()
            val silent = enhancer.push(FloatArray(961)) + enhancer.finish()
            assertEquals(961, silent.size)
            assertTrue(silent.all(Float::isFinite))
        }
    }

    @Test fun actualOrtCancellationTerminatesAndResetRestoresIndependentOutput() {
        val fixture = fixture()
        val token = StreamingCancellationToken()
        val worker = Executors.newSingleThreadExecutor()
        val enhancer = StreamingEnhancer(fixture.model, fixture.manifest,
            StreamingEnhancerConfiguration(fixture.manifest.layers, 2))
        try {
            val entered = CountDownLatch(1)
            val result = worker.submit<FloatArray> { entered.countDown(); enhancer.push(FloatArray(64_000), token) }
            assertTrue(entered.await(2, TimeUnit.SECONDS))
            // This is an integration check of ORT's terminate hook, not a fake spectral processor.
            Thread.sleep(25)
            val started = System.nanoTime()
            token.cancel()
            val failure = assertThrows(ExecutionException::class.java) { result.get(5, TimeUnit.SECONDS) }
            assertTrue("Cancellation must propagate to the caller.", failure.cause is CancellationException)
            assertTrue("Active ONNX inference should terminate within five seconds.",
                System.nanoTime() - started < TimeUnit.SECONDS.toNanos(5))
            enhancer.reset()
            val case = fixture.json.getJSONArray("fixtures").getJSONObject(0)
            val input = floats(fixture.directory, case.getString("inputFile"))
            val expected = floats(fixture.directory, case.getString("outputFile"))
            assertClose(expected, enhancer.push(input) + enhancer.finish(), 3e-4f, 0f, "post-cancellation reset waveform")
        } finally { token.cancel(); worker.shutdownNow(); enhancer.close() }
    }
}
