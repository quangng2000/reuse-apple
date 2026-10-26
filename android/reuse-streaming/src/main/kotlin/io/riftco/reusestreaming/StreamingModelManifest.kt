// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.Collections

/** Trusted export receipt: pin it in your application; never trust a receipt downloaded beside a model. */
class StreamingModelManifest private constructor(
    val modelSha256: String,
    val modelBytes: Long,
    val modelFile: String,
    val layers: Int,
    val states: List<State>,
) {
    val modelId: String get() = MODEL_ID

    class State internal constructor(val key: String, val inputName: String, val outputName: String,
                                     val shape: List<Long>) {
        internal val elements: Int = shape.fold(1L) { a, b -> a * b }.toInt()
    }

    fun verify(file: File, cancellation: StreamingCancellationToken = StreamingCancellationToken()) {
        require(file.isFile && file.length() == modelBytes) { "Streaming model size check failed." }
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { stream ->
            val buffer = ByteArray(128 * 1024)
            while (true) {
                cancellation.throwIfCancelled()
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        cancellation.throwIfCancelled()
        val actual = digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
        require(actual == modelSha256) { "Streaming model integrity check failed." }
    }

    companion object {
        const val MODEL_ID = "nvidia/realtime-reuse-streaming-fp32"
        const val CHECKPOINT_SHA256 = "6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade"
        const val SOURCE_REVISION = "4c0c186272ca853cd18d832fc5c28777b62f32d1"
        private const val MAX_MANIFEST_BYTES = 262_144

        fun read(file: File): StreamingModelManifest {
            require(file.isFile && file.length() in 1..MAX_MANIFEST_BYTES.toLong()) { "Invalid streaming manifest size." }
            return fromJson(file.readText(Charsets.UTF_8))
        }

        fun fromJson(text: String): StreamingModelManifest {
            require(text.toByteArray(Charsets.UTF_8).size in 1..MAX_MANIFEST_BYTES) { "Invalid streaming manifest size." }
            val json = JSONObject(text)
            val layerCount = integer(json, "layers")
            require(layerCount in 3L..12L) { "Unsupported streaming exit depth." }
            val layers = layerCount.toInt()
            require(integer(json, "schemaVersion") == 1L && string(json, "modelId") == MODEL_ID &&
                string(json, "checkpointSha256") == CHECKPOINT_SHA256 &&
                string(json, "sourceRevision") == SOURCE_REVISION && layers in 3..12 &&
                integer(json, "lookahead") == 0L && string(json, "precision") == "float32" &&
                integer(json, "sampleRate") == 16_000L && integer(json, "fftSize") == 640L &&
                integer(json, "hopSize") == 320L && integer(json, "frameSamples") == 640L &&
                integer(json, "frequencyBins") == 321L && string(json, "license") == "NSCLv1" &&
                json.get("researchOnly") == true && integer(json, "onnxOpset") == 17L &&
                string(json, "onnxRuntimeVersion") == "1.23.2" &&
                integer(json, "checkpointBytes") == 54_055_764L &&
                integer(json, "sourceParameterCount") == 13_475_815L &&
                integer(json, "activeParameterCount") == 1_198_019L + 208_512L * layers &&
                string(json, "sourceRepository") == "nvidia/Real-time_RE-USE" &&
                integer(json, "frequencyScanCount") == layers * 2L &&
                json.get("selectedWeightBytesUnchanged") == true) { "Unsupported streaming export contract." }
            val sha = string(json, "modelSha256")
            val bytes = integer(json, "modelBytes")
            val filename = string(json, "modelFile")
            require(sha.matches(Regex("[0-9a-f]{64}")) && bytes in 1..512L * 1024 * 1024 &&
                filename.matches(Regex("[A-Za-z0-9_.-]+\\.onnx"))) { "Invalid streaming export receipt." }
            val entries = json.getJSONArray("states")
            require(entries.length() in 1..128) { "Missing or excessive streaming states." }
            val states = ArrayList<State>()
            var totalElements = 0L
            for (i in 0 until entries.length()) {
                val state = entries.getJSONObject(i)
                val key = string(state, "key")
                val input = string(state, "inputName")
                val output = string(state, "outputName")
                require(key.matches(Regex("[A-Za-z0-9_.]+")) && key.length <= 200 &&
                    input == "cache_" + key.replace('.', '_') && output == "next_" + key.replace('.', '_') &&
                    string(state, "dtype") == "float32") { "Invalid streaming state name/type." }
                val shapeJson = state.getJSONArray("shape")
                require(shapeJson.length() in 1..4) { "Invalid streaming state rank." }
                val shape = (0 until shapeJson.length()).map { index ->
                    val dimension = shapeJson.get(index)
                    require(dimension is Int || dimension is Long) { "State dimensions must be integers." }
                    (dimension as Number).toLong()
                }
                require(shape.all { it in 1..4096 }) { "Invalid streaming state dimension." }
                val elements = shape.fold(1L) { a, b -> a * b }
                require(elements in 1..8_000_000) { "Excessive streaming state size." }
                totalElements += elements
                require(totalElements <= 32_000_000) { "Excessive streaming state allocation." }
                states.add(State(key, input, output, Collections.unmodifiableList(shape)))
            }
            require(states.map { it.key }.toSet().size == states.size &&
                states.map { it.inputName }.toSet().size == states.size &&
                states.map { it.outputName }.toSet().size == states.size) { "Duplicate streaming state." }
            val expected = expectedStates(layers)
            require(states.size == expected.size && states.zip(expected).all { (actual, pinned) ->
                actual.key == pinned.first && actual.shape == pinned.second
            }) { "Unsupported streaming cache topology." }
            require(integer(json, "stateBytes") == totalElements * 4) { "Invalid streaming state byte count." }
            fun tensorNames(name: String): List<String> {
                val array = json.getJSONArray(name)
                return (0 until array.length()).map { index ->
                    require(array.get(index) is String) { "Invalid tensor name." }
                    array.getString(index)
                }
            }
            require(tensorNames("inputNames") == listOf("magnitude", "phase") + states.map { it.inputName } &&
                tensorNames("outputNames") == listOf("enhanced_magnitude", "enhanced_phase") + states.map { it.outputName }) {
                "Invalid streaming tensor order."
            }
            return StreamingModelManifest(sha, bytes, filename, layers, Collections.unmodifiableList(states))
        }

        private fun integer(json: JSONObject, name: String): Long {
            val value = json.get(name)
            require(value is Int || value is Long) { "$name must be an integer." }
            return (value as Number).toLong()
        }

        private fun string(json: JSONObject, name: String): String {
            val value = json.get(name)
            require(value is String) { "$name must be a string." }
            return value
        }

        /** Layout from the pinned Real-time RE-USE checkpoint; no dynamic/unbounded cache allocation. */
        private fun expectedStates(layers: Int): List<Pair<String, List<Long>>> = buildList {
            fun addState(key: String, vararg shape: Long) { add(key to shape.toList()) }
            fun dense(prefix: String, frequency: Long) {
                for (i in 0..3) addState("$prefix.dense_block.$i.1", 1, 64L * (i + 1), 2L shl i, frequency)
            }
            addState("dense_encoder.dense_conv_1_1.0", 1, 2, 2, 323)
            dense("dense_encoder.dense_block", 323)
            for (i in 0 until layers) {
                val prefix = "TSMamba.$i.time_mamba.forward_blocks"
                addState("$prefix.conv1d", 161, 256, 3)
                addState("$prefix.ssm_state", 161, 256, 16)
            }
            for (head in listOf("mask_decoder_list", "phase_decoder_list")) {
                val prefix = "$head.${layers - 1}"
                dense("$prefix.dense_block", 161)
                addState("$prefix.up_conv2.0.conv", 1, 64, 2, 322)
            }
        }
    }
}
