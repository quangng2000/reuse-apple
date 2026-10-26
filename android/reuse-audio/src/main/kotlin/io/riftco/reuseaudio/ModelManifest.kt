package io.riftco.reuseaudio

import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/** Trusted export receipt. Applications must pin this receipt, rather than trust a downloaded one. */
data class ModelManifest(
    val modelSha256: String,
    val modelBytes: Long,
    val modelFile: String = "reuse-30-fp32.onnx",
    val modelId: String = MODEL_ID,
) {
    init {
        require(modelSha256.matches(Regex("[0-9a-f]{64}"))) { "Invalid ONNX SHA-256." }
        require(modelBytes in 1..512L * 1024 * 1024) { "Invalid ONNX size." }
        require(modelFile.matches(Regex("[A-Za-z0-9_.-]+\\.onnx"))) { "Invalid model filename." }
        require(modelId == MODEL_ID) { "Only the commercial 30-layer FP32 model is supported." }
    }

    fun verify(file: File, cancellation: CancellationToken = CancellationToken()) {
        require(file.isFile && file.length() == modelBytes) { "Model size check failed." }
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
        val actual = digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
        require(actual == modelSha256) { "Model integrity check failed." }
    }

    companion object {
        const val MODEL_ID = "nvidia/reuse-30-fp32"
        const val CHECKPOINT_SHA256 = "06badaf4029b297f97f7fe48d62990b48d5482608210df577354e0d4e0ed70e1"
        const val SOURCE_REVISION = "31f82f2efad1095034492f236e8dac9239ba713b"

        fun read(file: File): ModelManifest {
            require(file.length() in 1..16_384) { "Invalid model manifest size." }
            return fromJson(file.readText(Charsets.UTF_8))
        }

        fun fromJson(text: String): ModelManifest {
            require(text.toByteArray(Charsets.UTF_8).size in 1..16_384) { "Invalid model manifest size." }
            val json = JSONObject(text)
            require(json.getInt("schemaVersion") == 1 &&
                json.getString("checkpointSha256") == CHECKPOINT_SHA256 &&
                json.getString("sourceRevision") == SOURCE_REVISION &&
                json.getInt("numLayers") == 30 && json.getLong("parameterCount") == 9_607_747L &&
                json.getString("precision") == "float32" && json.getInt("sampleRate") == 16_000 &&
                json.getInt("fftSize") == 640 && json.getInt("hopSize") == 80 &&
                json.getInt("windowSamples") == 64_000 && json.getInt("overlapSamples") == 8_000
            ) { "Unsupported model export contract." }
            return ModelManifest(json.getString("modelSha256"), json.getLong("modelBytes"),
                json.getString("modelFile"), json.getString("modelId"))
        }
    }
}
