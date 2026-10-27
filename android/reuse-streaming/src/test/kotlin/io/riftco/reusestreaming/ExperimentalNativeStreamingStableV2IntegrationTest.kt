package io.riftco.reusestreaming

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class ExperimentalNativeStreamingStableV2IntegrationTest {
    private data class Fixture(val v1: File, val v2: File, val baseline: File,
                               val manifest: StreamingModelManifest, val parity: File, val receipt: JSONObject)
    private fun value(name: String): String? = System.getProperty(name) ?: System.getenv(name)
    private fun fixture(): Fixture {
        val v2 = value("REUSE_NATIVE_STREAMING_STABLE_V2_DIRECTORY")
        assumeTrue("Provide separately validated V2 native artifacts.", v2 != null)
        val parity = File(requireNotNull(value("REUSE_STREAMING_PARITY_DIR")))
        val manifest = StreamingModelManifest.read(File(requireNotNull(value("REUSE_STREAMING_MODEL_MANIFEST"))))
        val receipt = JSONObject(File(parity, "parity.json").readText())
        require(manifest.layers == 8 && manifest.modelSha256 == ExperimentalNativeStreaming.BASELINE_MODEL_SHA256 &&
            receipt.get("passed") == true && receipt.getInt("layers") == 8 &&
            receipt.getString("modelSha256") == manifest.modelSha256 &&
            receipt.getString("checkpointSha256") == StreamingModelManifest.CHECKPOINT_SHA256)
        return Fixture(File(requireNotNull(value("REUSE_NATIVE_STREAMING_DIRECTORY"))), File(v2!!),
            File(requireNotNull(value("REUSE_STREAMING_MODEL_FILE"))), manifest, parity, receipt)
    }
    private fun floats(f: Fixture, name: String): FloatArray {
        require(name.matches(Regex("[A-Za-z0-9_.-]+\\.f32")))
        val raw = File(f.parity, name).readBytes()
        val record = f.receipt.getJSONObject("files").getJSONObject(name)
        val sha = MessageDigest.getInstance("SHA-256").digest(raw).joinToString("") { "%02x".format(it.toInt() and 255) }
        require(raw.size == record.getInt("bytes") && sha == record.getString("sha256"))
        val data = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(data.remaining()).also { data.get(it) }
    }
    private fun v1(f: Fixture, options: StreamingEnhancerConfiguration) = ExperimentalNativeStreaming.create(
        File(f.v1, "reuse-streaming-l8-native-frequency-v1.onnx"),
        File(f.v1, "reuse-streaming-l8-native-frequency-v1.manifest.json"),
        File(f.v1, "build-host/libreuse_streaming_scan.dylib"),
        File(f.v1, "build-host/libreuse_streaming_scan.dylib.json"), options)
    private fun v2(f: Fixture, options: StreamingEnhancerConfiguration) = ExperimentalNativeStreamingStableV2.create(
        File(f.v2, "reuse-streaming-l8-native-stable-v2.onnx"),
        File(f.v2, "reuse-streaming-l8-native-stable-v2.manifest.json"),
        File(f.v2, "build-host/libreuse_streaming_stable_v2.dylib"),
        File(f.v2, "build-host/libreuse_streaming_stable_v2.dylib.json"), options)
    private fun bits(expected: FloatArray, actual: FloatArray) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) if (expected[i].toRawBits() != actual[i].toRawBits()) fail("Replay bits differ at $i")
    }
    private fun waveform(expected: FloatArray, actual: FloatArray, label: String, absolute: Double = .0001) {
        assertEquals(expected.size, actual.size)
        var maximum = 0.0; var errorEnergy = 0.0; var signalEnergy = 0.0
        for (i in expected.indices) {
            assertTrue(actual[i].isFinite())
            val error = actual[i].toDouble() - expected[i]
            maximum = maxOf(maximum, abs(error)); errorEnergy += error * error
            signalEnergy += expected[i].toDouble() * expected[i]
        }
        val relative = sqrt(errorEnergy / maxOf(signalEnergy, 1e-30))
        assertTrue("$label maxAbs=$maximum", maximum < absolute)
        assertTrue("$label relativeRms=$relative", relative < .001)
        println("$label samples=${actual.size} maxAbs=$maximum relativeRms=$relative")
    }

    @Test fun sevenWaveformLengthsMatchV1PublishedCopyAndOracleWithExactV2ChunkReplay() {
        val f = fixture(); val options = StreamingEnhancerConfiguration(8, 2)
        StreamingEnhancer(CopyingStreamingReferenceProcessor(f.baseline, f.manifest, options), options).use { published ->
            v1(f, options).use { frequency ->
                v2(f, options).use { temporal ->
                    val fixtures = f.receipt.getJSONArray("fixtures")
                    assertEquals(7, fixtures.length())
                    repeat(fixtures.length()) { index ->
                        val example = fixtures.getJSONObject(index)
                        val input = floats(f, example.getString("inputFile")); val oracle = floats(f, example.getString("outputFile"))
                        published.reset(); frequency.reset(); temporal.reset()
                        val copy = published.push(input) + published.finish()
                        val v1 = frequency.push(input) + frequency.finish()
                        val v2 = temporal.push(input) + temporal.finish()
                        waveform(copy, v2, "V2 versus published")
                        waveform(v1, v2, "V2 versus V1")
                        waveform(oracle, v2, "V2 versus independent oracle", .0003)
                        temporal.reset()
                        val chunks = example.getJSONArray("chunks")
                        val replay = ArrayList<Float>(); var offset = 0
                        repeat(chunks.length()) { chunk ->
                            val end = offset + chunks.getInt(chunk)
                            replay.addAll(temporal.push(input.copyOfRange(offset, end)).toList()); offset = end
                        }
                        assertEquals(input.size, offset)
                        replay.addAll(temporal.finish().toList()); bits(v2, replay.toFloatArray())
                    }
                }
            }
        }
    }

    @Test fun all31CarriedCachesAndSpectralOutputsMatchPublishedAndV1AcrossReset() {
        val f = fixture(); val options = StreamingEnhancerConfiguration(8, 2)
        val v1 = ExperimentalNativeStreaming.verify(File(f.v1, "reuse-streaming-l8-native-frequency-v1.onnx"),
            File(f.v1, "reuse-streaming-l8-native-frequency-v1.manifest.json"), File(f.v1, "build-host/libreuse_streaming_scan.dylib"),
            File(f.v1, "build-host/libreuse_streaming_scan.dylib.json"), options)
        val modelV2 = File(f.v2, "reuse-streaming-l8-native-stable-v2.onnx")
        val v2 = ExperimentalNativeStreamingStableV2.verify(modelV2,
            File(f.v2, "reuse-streaming-l8-native-stable-v2.manifest.json"), File(f.v2, "build-host/libreuse_streaming_stable_v2.dylib"),
            File(f.v2, "build-host/libreuse_streaming_stable_v2.dylib.json"), options)
        val spec = f.receipt.getJSONObject("spectral")
        val magnitude = floats(f, spec.getString("magnitudeFile")); val phase = floats(f, spec.getString("phaseFile"))
        val oracleMagnitude = floats(f, spec.getString("enhancedMagnitudeFile")); val oraclePhase = floats(f, spec.getString("enhancedPhaseFile"))
        CopyingStreamingReferenceProcessor(f.baseline, f.manifest, options).use { published ->
            OnnxStreamingProcessor(File(f.v1, "reuse-streaming-l8-native-frequency-v1.onnx"), v1.manifest, options, v1.library).use { frequency ->
                OnnxStreamingProcessor(modelV2, v2.manifest, options, v2.library, NativeStreamingVariant.STABLE_FREQUENCY_TEMPORAL_V2).use { temporal ->
                    var first: StreamingSpectrum? = null; var firstState: List<FloatArray>? = null
                    var maximumState = 0.0; var maximumMagnitude = 0.0; var maximumPhase = 0.0
                    repeat(spec.getInt("frames")) { frame ->
                        val offset = frame * 321
                        val input = StreamingSpectrum(magnitude.copyOfRange(offset, offset + 321), phase.copyOfRange(offset, offset + 321))
                        val copy = published.process(input, StreamingCancellationToken()); val v1Output = frequency.process(input, StreamingCancellationToken())
                        val v2Output = temporal.process(input, StreamingCancellationToken())
                        val references = listOf(copy, v1Output, StreamingSpectrum(oracleMagnitude.copyOfRange(offset, offset + 321), oraclePhase.copyOfRange(offset, offset + 321)))
                        for (reference in references) for (i in 0 until 321) {
                            maximumMagnitude = maxOf(maximumMagnitude, abs(v2Output.magnitude[i].toDouble() - reference.magnitude[i]))
                            val delta = v2Output.phase[i].toDouble() - reference.phase[i]
                            maximumPhase = maxOf(maximumPhase, abs(atan2(sin(delta), cos(delta))))
                        }
                        val next = temporal.copyStateForValidation()
                        assertEquals(31, next.size)
                        for (reference in listOf(published.copyStateForValidation(), frequency.copyStateForValidation()))
                            for (cache in reference.indices) for (i in reference[cache].indices) {
                                assertTrue(next[cache][i].isFinite())
                                val delta = abs(next[cache][i].toDouble() - reference[cache][i])
                                maximumState = maxOf(maximumState, delta)
                                if (delta > .0003 + .0002 * abs(reference[cache][i])) fail("V2 cache $cache differs at $i: $delta")
                            }
                        if (frame == 0) { first = v2Output; firstState = next }
                    }
                    assertTrue("V2 magnitude=$maximumMagnitude", maximumMagnitude < .003)
                    assertTrue("V2 wrapped phase=$maximumPhase", maximumPhase < .003)
                    println("V2 spectral/cache maxMagnitude=$maximumMagnitude maxWrappedPhase=$maximumPhase maxState=$maximumState")
                    temporal.reset()
                    val replay = temporal.process(StreamingSpectrum(magnitude.copyOfRange(0, 321), phase.copyOfRange(0, 321)), StreamingCancellationToken())
                    bits(first!!.magnitude, replay.magnitude); bits(first!!.phase, replay.phase)
                    val after = temporal.copyStateForValidation()
                    firstState!!.indices.forEach { bits(firstState!![it], after[it]) }
                }
            }
        }
    }

    @Test fun shortInputAndResetLifecycleRemainUnchanged() {
        val f = fixture(); val options = StreamingEnhancerConfiguration(8, 2)
        StreamingEnhancer(CopyingStreamingReferenceProcessor(f.baseline, f.manifest, options), options).use { published ->
            v2(f, options).use { temporal ->
                for (count in listOf(0, 1, 319, 320, 639)) {
                    published.reset(); temporal.reset()
                    assertEquals(0, published.push(FloatArray(count)).size)
                    assertEquals(0, temporal.push(FloatArray(count)).size)
                    val baselineFailure = assertThrows(StreamingException::class.java) { published.finish() }
                    val candidateFailure = assertThrows(StreamingException::class.java) { temporal.finish() }
                    assertEquals(StreamingException.Reason.TOO_SHORT, baselineFailure.reason)
                    assertEquals(baselineFailure.reason, candidateFailure.reason)
                    // TOO_SHORT leaves both streams open; completing the same 640 samples still works.
                    val copy = published.push(FloatArray(640 - count)) + published.finish()
                    val v2 = temporal.push(FloatArray(640 - count)) + temporal.finish()
                    waveform(copy, v2, "V2 short-to-complete $count")
                    assertEquals(StreamingException.Reason.FINISHED,
                        assertThrows(StreamingException::class.java) { temporal.finish() }.reason)
                }
            }
        }
    }

    @Test fun v1AndStableCustomRegistriesCoexistAndClosingV1LeavesStableUsable() {
        val f = fixture(); val options = StreamingEnhancerConfiguration(8, 2)
        val frequency = v1(f, options)
        try {
            v2(f, options).use { stable ->
                val input = floats(f, "640-input.f32")
                val expected = frequency.push(input) + frequency.finish()
                val actual = stable.push(input) + stable.finish()
                waveform(expected, actual, "Stable versus coexisting V1")
                frequency.close()
                stable.reset()
                bits(actual, stable.push(input) + stable.finish())
            }
        } finally { frequency.close() }
    }
}
