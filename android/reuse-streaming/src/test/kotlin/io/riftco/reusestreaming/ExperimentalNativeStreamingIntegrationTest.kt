package io.riftco.reusestreaming

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

class ExperimentalNativeStreamingIntegrationTest {
    private data class Fixture(val directory: File, val model: File, val manifest: File, val library: File,
                               val receipt: File, val baseline: File, val baselineManifest: StreamingModelManifest,
                               val parity: File, val json: JSONObject)
    private fun value(name: String): String? = System.getProperty(name) ?: System.getenv(name)
    private fun fixture(): Fixture {
        val path = value("REUSE_NATIVE_STREAMING_DIRECTORY")
        assumeTrue("Provide the validated native artifacts for host integration.", path != null)
        val directory = File(path!!)
        val baseline = File(requireNotNull(value("REUSE_STREAMING_MODEL_FILE")))
        val baselineManifest = StreamingModelManifest.read(File(requireNotNull(value("REUSE_STREAMING_MODEL_MANIFEST"))))
        require(baselineManifest.layers == 8)
        val parity = File(requireNotNull(value("REUSE_STREAMING_PARITY_DIR")))
        return Fixture(directory, File(directory, "reuse-streaming-l8-native-frequency-v1.onnx"),
            File(directory, "reuse-streaming-l8-native-frequency-v1.manifest.json"),
            File(directory, "build-host/libreuse_streaming_scan.dylib"),
            File(directory, "build-host/libreuse_streaming_scan.dylib.json"), baseline, baselineManifest,
            parity, JSONObject(File(parity, "parity.json").readText()))
    }
    private fun floats(folder: File, filename: String): FloatArray {
        val buffer = ByteBuffer.wrap(File(folder, filename).readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }
    private fun bits(expected: FloatArray, actual: FloatArray) {
        assertEquals(expected.size, actual.size)
        for (i in expected.indices) if (expected[i].toRawBits() != actual[i].toRawBits())
            fail("FP32 replay bits differ at $i")
    }

    @Test fun nativeFactoryWaveformStaysWithinReleasedToleranceAndResetChunkReplayIsExact() {
        val f = fixture()
        val cases = f.json.getJSONArray("fixtures")
        val input = floats(f.parity, cases.getJSONObject(cases.length() - 1).getString("inputFile"))
        val options = StreamingEnhancerConfiguration(8, 2)
        ExperimentalNativeStreaming.create(f.model, f.manifest, f.library, f.receipt, options).use { native ->
            StreamingEnhancer(f.baseline, f.baselineManifest, options).use { baseline ->
                native.warmUp(); baseline.warmUp()
                val expected = baseline.push(input) + baseline.finish()
                val actual = native.push(input) + native.finish()
                assertEquals(expected.size, actual.size)
                var maximum = 0.0; var errorEnergy = 0.0; var expectedEnergy = 0.0
                for (i in expected.indices) {
                    assertTrue(actual[i].isFinite())
                    val error = actual[i].toDouble() - expected[i]
                    maximum = maxOf(maximum, abs(error)); errorEnergy += error * error
                    expectedEnergy += expected[i].toDouble() * expected[i]
                }
                val relativeRms = sqrt(errorEnergy / maxOf(expectedEnergy, 1e-30))
                assertTrue("waveform max abs=$maximum", maximum < .0001)
                assertTrue("waveform relative RMS=$relativeRms", relativeRms < .001)
                println("native SDK waveform maxAbs=$maximum relativeRms=$relativeRms samples=${actual.size}")
                native.reset()
                val chunks = intArrayOf(1, 17, 320, 641, 97)
                val output = ArrayList<Float>()
                var offset = 0; var index = 0
                while (offset < input.size) {
                    val end = minOf(input.size, offset + chunks[index++ % chunks.size])
                    output.addAll(native.push(input.copyOfRange(offset, end)).toList()); offset = end
                }
                output.addAll(native.finish().toList())
                bits(actual, output.toFloatArray())
            }
        }
    }

    @Test fun nativeSpectraAllCarriedCachesAndResetMatchValidatedFp32Contract() {
        val f = fixture()
        val options = StreamingEnhancerConfiguration(8, 2)
        val verified = ExperimentalNativeStreaming.verify(f.model, f.manifest, f.library, f.receipt, options)
        val data = f.json.getJSONObject("spectral")
        val magnitude = floats(f.parity, data.getString("magnitudeFile"))
        val phase = floats(f.parity, data.getString("phaseFile"))
        OnnxStreamingProcessor(f.model, verified.manifest, options, verified.library).use { native ->
            OnnxStreamingProcessor(f.baseline, f.baselineManifest, options).use { baseline ->
                var firstOutput: StreamingSpectrum? = null; var firstState: List<FloatArray>? = null
                var maxMagnitude = 0.0; var maxPhase = 0.0; var maxState = 0.0
                repeat(data.getInt("frames")) { frame ->
                    val offset = frame * 321
                    val input = StreamingSpectrum(magnitude.copyOfRange(offset, offset + 321), phase.copyOfRange(offset, offset + 321))
                    val expected = baseline.process(input, StreamingCancellationToken())
                    val actual = native.process(input, StreamingCancellationToken())
                    for (i in 0 until 321) {
                        maxMagnitude = maxOf(maxMagnitude, abs(actual.magnitude[i].toDouble() - expected.magnitude[i]))
                        val delta = actual.phase[i].toDouble() - expected.phase[i]
                        maxPhase = maxOf(maxPhase, abs(atan2(sin(delta), cos(delta))))
                    }
                    val before = baseline.copyStateForValidation(); val after = native.copyStateForValidation()
                    assertEquals(31, after.size)
                    for (state in before.indices) for (i in before[state].indices) {
                        val expectedValue = before[state][i]; val actualValue = after[state][i]
                        assertTrue(actualValue.isFinite())
                        val delta = abs(actualValue.toDouble() - expectedValue)
                        maxState = maxOf(maxState, delta)
                        if (delta > .0003 + .0002 * abs(expectedValue)) fail("Native carried cache $state differs at $i: $delta")
                    }
                    if (frame == 0) { firstOutput = actual; firstState = after }
                }
                assertTrue("magnitude=$maxMagnitude", maxMagnitude < .003)
                assertTrue("wrapped phase=$maxPhase", maxPhase < .003)
                println("native SDK spectral maxMagnitude=$maxMagnitude maxWrappedPhase=$maxPhase maxState=$maxState")
                native.reset()
                val firstInput = StreamingSpectrum(magnitude.copyOfRange(0, 321), phase.copyOfRange(0, 321))
                val replay = native.process(firstInput, StreamingCancellationToken())
                bits(firstOutput!!.magnitude, replay.magnitude); bits(firstOutput!!.phase, replay.phase)
                val replayState = native.copyStateForValidation()
                firstState!!.indices.forEach { bits(firstState!![it], replayState[it]) }
            }
        }
    }
}
