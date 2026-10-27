package io.riftco.reusestreaming

import ai.onnxruntime.*
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.CancellationException

class PinnedStreamingIntegrationTest {
    private data class Fixture(val model: File, val manifest: StreamingModelManifest, val directory: File, val json: JSONObject)

    private fun fixture(): Fixture {
        fun value(name: String) = System.getProperty(name) ?: System.getenv(name)
        val model = value("REUSE_STREAMING_MODEL_FILE")
        val manifest = value("REUSE_STREAMING_MODEL_MANIFEST")
        val directory = value("REUSE_STREAMING_PARITY_DIR")
        assumeTrue("Provide the pinned real model/reference paths.", listOf(model, manifest, directory).any { it != null })
        require(listOf(model, manifest, directory).all { !it.isNullOrBlank() })
        val folder = File(directory!!)
        return Fixture(File(model!!), StreamingModelManifest.read(File(manifest!!)), folder,
            JSONObject(File(folder, "parity.json").readText()))
    }

    private fun floats(folder: File, name: String): FloatArray {
        val data = File(folder, name).readBytes()
        val buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(buffer.remaining()).also { buffer.get(it) }
    }

    private fun assertBits(expected: FloatArray, actual: FloatArray, label: String) {
        assertEquals("$label length", expected.size, actual.size)
        for (i in expected.indices)
            if (expected[i].toRawBits() != actual[i].toRawBits())
                fail("$label raw FP32 bits differ at $i: ${expected[i].toRawBits()} != ${actual[i].toRawBits()}")
    }

    @Test fun pinnedSpectraAndEveryCommittedStateMatchTheCopyingReferenceBitForBit() {
        val fixture = fixture()
        val spectral = fixture.json.getJSONObject("spectral")
        val frames = spectral.getInt("frames")
        val magnitude = floats(fixture.directory, spectral.getString("magnitudeFile"))
        val phase = floats(fixture.directory, spectral.getString("phaseFile"))
        val options = StreamingEnhancerConfiguration(fixture.manifest.layers, 2)
        OnnxStreamingProcessor(fixture.model, fixture.manifest, options).use { optimized ->
            CopyingStreamingReferenceProcessor(fixture.model, fixture.manifest, options).use { baseline ->
                repeat(2) { replay ->
                    repeat(frames) { frame ->
                        val offset = frame * 321
                        val input = StreamingSpectrum(magnitude.copyOfRange(offset, offset + 321), phase.copyOfRange(offset, offset + 321))
                        val expected = baseline.process(input, StreamingCancellationToken())
                        val actual = optimized.process(input, StreamingCancellationToken())
                        assertBits(expected.magnitude, actual.magnitude, "magnitude replay $replay frame $frame")
                        assertBits(expected.phase, actual.phase, "phase replay $replay frame $frame")
                        val before = baseline.copyStateForValidation()
                        val after = optimized.copyStateForValidation()
                        assertEquals(before.size, after.size)
                        for (state in before.indices) assertBits(before[state], after[state], "state $state replay $replay frame $frame")
                    }
                    optimized.reset()
                    baseline.reset()
                }
            }
        }
    }

    @Test fun fullWaveformAndArbitraryChunkReplayAreBitIdenticalToTheCopyingReference() {
        val fixture = fixture()
        val cases = fixture.json.getJSONArray("fixtures")
        val example = cases.getJSONObject(cases.length() - 1)
        val input = floats(fixture.directory, example.getString("inputFile"))
        val options = StreamingEnhancerConfiguration(fixture.manifest.layers, 2)
        StreamingEnhancer(fixture.model, fixture.manifest, options).use { optimized ->
            StreamingEnhancer(CopyingStreamingReferenceProcessor(fixture.model, fixture.manifest, options), options).use { baseline ->
                optimized.warmUp()
                baseline.warmUp()
                val expected = baseline.push(input) + baseline.finish()
                val actual = optimized.push(input) + optimized.finish()
                assertBits(expected, actual, "complete waveform")
                optimized.reset()
                val chunks = intArrayOf(1, 17, 320, 641, 97)
                val output = ArrayList<Float>()
                var offset = 0
                var index = 0
                while (offset < input.size) {
                    val end = minOf(input.size, offset + chunks[index++ % chunks.size])
                    output.addAll(optimized.push(input.copyOfRange(offset, end)).toList())
                    offset = end
                }
                output.addAll(optimized.finish().toList())
                assertBits(expected, output.toFloatArray(), "chunked waveform after reset")
            }
        }
    }

    @Test fun pinnedOutputResultDoesNotOwnBuffersAndReuseKeepsTensorIdentity() {
        val fixture = fixture()
        val environment = OrtEnvironment.getEnvironment()
        OrtSession.SessionOptions().use { options ->
            options.setIntraOpNumThreads(2)
            environment.createSession(fixture.model.absolutePath, options).use { session ->
                PinnedStreamingBuffers(environment, fixture.manifest).use { buffers ->
                    val originalInput = buffers.inputs.values.toList()
                    val originalOutput = buffers.outputs.values.toList()
                    repeat(3) {
                        buffers.writeSpectrum(StreamingSpectrum(FloatArray(321), FloatArray(321)))
                        session.run(buffers.inputs, buffers.outputs).use { result ->
                            for (i in 0 until result.size()) assertFalse(result.isResultOwner(i))
                        }
                        assertTrue(buffers.nextIsFinite())
                        if (it == 0) {
                            for (i in originalInput.indices) assertSame(originalInput[i], buffers.inputs.values.toList()[i])
                            for (i in originalOutput.indices) assertSame(originalOutput[i], buffers.outputs.values.toList()[i])
                        }
                        buffers.commitNext()
                    }
                    buffers.reset()
                    for (i in originalInput.indices) assertSame(originalInput[i], buffers.inputs.values.toList()[i])
                    for (i in originalOutput.indices) assertSame(originalOutput[i], buffers.outputs.values.toList()[i])
                    assertTrue(buffers.copyCommittedStateForValidation().all { array -> array.all { it.toRawBits() == 0 } })
                }
            }
        }
    }

    @Test fun aPrecancelledHopNeverChangesCommittedState() {
        val fixture = fixture()
        val options = StreamingEnhancerConfiguration(fixture.manifest.layers, 2)
        OnnxStreamingProcessor(fixture.model, fixture.manifest, options).use { processor ->
            processor.process(StreamingSpectrum(FloatArray(321), FloatArray(321)), StreamingCancellationToken())
            val before = processor.copyStateForValidation()
            val token = StreamingCancellationToken().also { it.cancel() }
            assertThrows(CancellationException::class.java) {
                processor.process(StreamingSpectrum(FloatArray(321), FloatArray(321)), token)
            }
            val after = processor.copyStateForValidation()
            for (state in before.indices) assertBits(before[state], after[state], "cancelled state $state")
        }
    }
}
