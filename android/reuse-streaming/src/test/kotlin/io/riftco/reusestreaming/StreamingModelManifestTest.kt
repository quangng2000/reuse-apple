package io.riftco.reusestreaming

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import java.nio.file.Files
import java.security.MessageDigest
import java.util.concurrent.CancellationException

class StreamingModelManifestTest {
    private fun receipt() = JSONObject(javaClass.getResource("/streaming-manifest-l4.json")!!.readText())
    private fun rejects(edit: (JSONObject) -> Unit) {
        val json = receipt(); edit(json)
        assertThrows(Exception::class.java) { StreamingModelManifest.fromJson(json.toString()) }
    }

    @Test fun exactPinnedReceiptParsesIntoBoundedStateLayout() {
        val manifest = StreamingModelManifest.fromJson(receipt().toString())
        assertEquals(4, manifest.layers)
        assertEquals(23, manifest.states.size)
        assertEquals(29_046_320, manifest.states.sumOf { it.elements * 4 })
        assertEquals(listOf(1L, 2L, 2L, 323L), manifest.states.first().shape)
        assertEquals("next_phase_decoder_list_3_up_conv2_0_conv", manifest.states.last().outputName)
        assertThrows(UnsupportedOperationException::class.java) {
            @Suppress("UNCHECKED_CAST")
            (manifest.states as MutableList<StreamingModelManifest.State>).clear()
        }
    }

    @Test fun offlineIdentityAndChangedSamplingMathAreRejected() {
        listOf("modelId", "checkpointSha256", "sourceRevision", "sourceRepository", "license", "precision", "onnxRuntimeVersion")
            .forEach { field -> rejects { it.put(field, "unsupported") } }
        mapOf("schemaVersion" to 2, "layers" to 30, "lookahead" to 1, "sampleRate" to 48_000,
            "fftSize" to 1024, "hopSize" to 80, "frameSamples" to 1280, "frequencyBins" to 320,
            "checkpointBytes" to 1, "sourceParameterCount" to 1, "activeParameterCount" to 1,
            "frequencyScanCount" to 1, "onnxOpset" to 18).forEach { (field, value) -> rejects { it.put(field, value) } }
        rejects { it.put("researchOnly", false) }
        rejects { it.put("selectedWeightBytesUnchanged", false) }
    }

    @Test fun fractionalStringAndOverflowNumbersAreRejected() {
        rejects { it.put("schemaVersion", 1.5) }
        rejects { it.put("schemaVersion", "1") }
        rejects { it.put("layers", 4_294_967_300L) }
        rejects { it.put("modelBytes", 8_217_659.5) }
        rejects { it.getJSONArray("states").getJSONObject(0).getJSONArray("shape").put(0, 1.1) }
        rejects { it.getJSONArray("states").getJSONObject(0).getJSONArray("shape").put(0, "1") }
    }

    @Test fun unknownDuplicateMissingOrReorderedStateIsRejected() {
        rejects { it.getJSONArray("states").getJSONObject(0).put("key", "unknown_state") }
        rejects { json -> json.getJSONArray("states").put(json.getJSONArray("states").getJSONObject(0)) }
        rejects { it.getJSONArray("states").remove(1) }
        rejects { json ->
            val states = json.getJSONArray("states")
            val first = states.getJSONObject(0); val second = states.getJSONObject(1)
            states.put(0, second); states.put(1, first)
        }
        rejects { it.getJSONArray("states").getJSONObject(0).put("inputName", "magnitude") }
        rejects { it.getJSONArray("states").getJSONObject(0).put("outputName", "bad") }
        rejects { it.getJSONArray("states").getJSONObject(0).put("dtype", "float16") }
        rejects { it.getJSONArray("inputNames").put(0, "unexpected") }
        rejects { it.getJSONArray("outputNames").put(0, "unexpected") }
    }

    @Test fun hugeDynamicOrIncorrectCacheShapesCannotAllocate() {
        for (size in listOf(-1, 0, 4097, Int.MAX_VALUE))
            rejects { it.getJSONArray("states").getJSONObject(0).getJSONArray("shape").put(0, size) }
        rejects { it.getJSONArray("states").getJSONObject(0).getJSONArray("shape").put(0, 2) }
        rejects { it.put("stateBytes", Long.MAX_VALUE) }
        rejects { it.put("modelBytes", 513L * 1024 * 1024) }
        rejects { it.put("modelSha256", "a".repeat(63)) }
        rejects { it.put("modelFile", "../model.onnx") }
        assertThrows(IllegalArgumentException::class.java) { StreamingModelManifest.fromJson(" ".repeat(262_145)) }
    }

    @Test fun shaAndSizeAreVerifiedWithCancellationBeforeRuntimeLoading() {
        val file = Files.createTempFile("streaming-manifest", ".onnx").toFile()
        try {
            val bytes = byteArrayOf(1, 5, 9, 42)
            file.writeBytes(bytes)
            val json = receipt().put("modelBytes", bytes.size).put("modelSha256",
                MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it.toInt() and 255) })
            val manifest = StreamingModelManifest.fromJson(json.toString())
            manifest.verify(file)
            val token = StreamingCancellationToken(); token.cancel()
            assertThrows(CancellationException::class.java) { manifest.verify(file, token) }
            file.writeBytes(byteArrayOf(1, 5, 9, 43))
            assertThrows(IllegalArgumentException::class.java) { manifest.verify(file) }
            file.appendBytes(byteArrayOf(0))
            assertThrows(IllegalArgumentException::class.java) { manifest.verify(file) }
        } finally { file.delete() }
    }

    @Test fun requestedLayerMustMatchExportBeforeOnnxSessionCreation() {
        val manifest = StreamingModelManifest.fromJson(receipt().toString())
        assertThrows(IllegalArgumentException::class.java) {
            StreamingEnhancer(java.io.File("missing-model.onnx"), manifest, StreamingEnhancerConfiguration(layers = 8))
        }
        assertEquals(4, StreamingEnhancerConfiguration().layers)
        listOf(0, 2, 13, 30).forEach { layer ->
            assertThrows(IllegalArgumentException::class.java) { StreamingEnhancerConfiguration(layers = layer) }
        }
        listOf(0, 9).forEach { threads ->
            assertThrows(IllegalArgumentException::class.java) { StreamingEnhancerConfiguration(intraOpThreads = threads) }
        }
    }
}
