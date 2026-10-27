// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

import ai.onnxruntime.OrtEnvironment
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * Explicit, local experimental eight-layer FP32 frequency and temporal kernel. It permits FP32 rounding
 * differences from the standard graph; it is not selected by the ordinary enhancer constructor.
 * Supply the exact validated model and native library with trusted, application-bundled receipts.
 * Nothing is downloaded. Android currently supports only the validated arm64-v8a library; the
 * separate host library exists for numerical tests. The streaming model remains NSCLv1 research-only.
 */
object ExperimentalNativeStreamingV2 {
    internal const val MODEL_SHA256 = "1fbd91be9c90568c58ab40e9ce70e07c54dafc599fffb694a84c1f4d3905b11d"
    internal const val BASELINE_MODEL_SHA256 = "18db8d32b638702909de484d2a9cffc97ed09e6454c4caed3f0008fb0aab53e8"
    private const val SOURCE_SHA256 = "a6553337fb2bbdf5592bd44ec96a3a57b437b271490987cbc5298a27725e2196"
    private const val MAX_RECEIPT_BYTES = 262_144L
    private const val COMPILER_FLAGS = "-O3 -fno-fast-math -ffp-contract=off -fvisibility=hidden"
    private val sources = mapOf(
        "native_v2.cpp" to SOURCE_SHA256,
        "CMakeLists.txt" to "2efe800cc04cb9eeba946cf54b538abdadded238815f7a060f85081803e23ca6",
        "build_v2.py" to "45c71780dfca0130604e0e8cd7807c9f99722c7f4977db4a3fed39a2446ef13c",
        "transform_v2.py" to "0899a73d6cdf4a321e2aa5c26dbb99e31c0569463191b65e26bf7bc0e35acdc2",
        "v1FrequencySource" to "1708573ce611abc9da7ab8453fbd51c7635f61dd8f5db77ee14788002f6ac030",
        "v1TransformSource" to "c77c1560266f87b61ef8972c0231c52b789c4294c061959cbc15db38f5327be8",
        "ortHeaderProvenance" to "5abee8adc9f970885827b081f21d3091b4ed8bbb8112a523d80a8d0966119281",
    )
    private data class Library(val filename: String, val bytes: Long, val sha256: String)
    private val libraries = mapOf(
        "arm64-v8a" to Library("libreuse_streaming_native_v2.so", 1_192_152,
            "3013f39b7e97464b415a6b8b3500be0ac0a26b678ea8d7e71f248178a29b8868"),
        "host" to Library("libreuse_streaming_native_v2.dylib", 80_432,
            "050188d3c67092a9079bc9391417837a78c657d59f1b8c84648d7b77bd51a78f"),
    )

    @JvmStatic
    fun create(modelFile: File, manifestFile: File, libraryFile: File, libraryReceiptFile: File,
               configuration: StreamingEnhancerConfiguration): StreamingEnhancer {
        val verified = verify(modelFile, manifestFile, libraryFile, libraryReceiptFile, configuration)
        require(OrtEnvironment.getEnvironment().version == "1.23.2") { "Experimental streaming requires ONNX Runtime 1.23.2." }
        return StreamingEnhancer(OnnxStreamingProcessor(modelFile, verified.manifest, configuration,
            verified.library, NativeStreamingVariant.FREQUENCY_TEMPORAL_V2), configuration)
    }

    internal data class Verified(val manifest: StreamingModelManifest, val library: File)

    /** All identity and integrity checks complete before any custom code is registered or loaded. */
    internal fun verify(modelFile: File, manifestFile: File, libraryFile: File, libraryReceiptFile: File,
                        configuration: StreamingEnhancerConfiguration): Verified {
        require(configuration.layers == 8) { "Experimental native streaming requires exactly eight layers." }
        val manifestText = readReceipt(manifestFile)
        val manifest = StreamingModelManifest.fromJson(manifestText)
        val json = JSONObject(manifestText)
        require(manifest.layers == 8 && manifest.modelSha256 == MODEL_SHA256 && manifest.modelBytes == 11_590_141L &&
            string(json, "executionVariant") == "native-frequency-temporal-v2" &&
            string(json, "baselineModelSha256") == BASELINE_MODEL_SHA256 &&
            string(json, "nativeFrequencyBaselineModelSha256") == ExperimentalNativeStreaming.MODEL_SHA256 &&
            string(json, "customOpDomain") == "ai.reuse.streaming" && string(json, "customOpName") == "FrequencyScanV1" &&
            string(json, "customTemporalOpName") == "TemporalStepV2" &&
            integer(json, "kernelAbi") == 2L && integer(json, "nativeFrequencyScanCount") == 16L &&
            integer(json, "standardFrequencyScanCount") == 0L && integer(json, "nativeTemporalStepCount") == 8L &&
            integer(json, "standardTemporalStepCount") == 0L && json.get("preparedDecay") == true &&
            json.get("preparedGate") == true && json.get("temporalPreparedDecay") == true &&
            json.get("temporalPreparedGate") == true && string(json, "temporalReduction") == "serial16" &&
            string(json, "recurrenceReduction") == "serial16" &&
            json.get("nativeChannelParallel") == true && sourceMap(json.getJSONObject("nativeSources")) == sources) {
            "Unsupported experimental streaming graph contract."
        }
        manifest.verify(modelFile)
        return Verified(manifest, verifyLibrary(libraryFile, libraryReceiptFile))
    }

    internal fun verifyLibrary(libraryFile: File, libraryReceiptFile: File): File {
        val receipt = JSONObject(readReceipt(libraryReceiptFile))
        val library = requireNotNull(libraries[string(receipt, "abi")]) { "Unvalidated experimental native library ABI." }
        require(integer(receipt, "kernelAbi") == 2L && string(receipt, "domain") == "ai.reuse.streaming" &&
            string(receipt, "op") == "FrequencyScanV1" && string(receipt, "temporalOp") == "TemporalStepV2" &&
            string(receipt, "ortVersion") == "1.23.2" &&
            string(receipt, "precision") == "float32" && string(receipt, "sourceSha256") == SOURCE_SHA256 &&
            string(receipt, "compilerFlags") == COMPILER_FLAGS && sourceMap(receipt.getJSONObject("sources")) == sources &&
            symbolNames(receipt) == listOf("RegisterCustomOps", "ReuseStreamingKernelAbiVersion") &&
            string(receipt, "file") == library.filename && integer(receipt, "bytes") == library.bytes &&
            string(receipt, "sha256") == library.sha256) { "Unsupported experimental native library receipt." }
        require(libraryFile.isAbsolute && libraryFile.name == library.filename && libraryFile.isFile &&
            libraryFile.length() == library.bytes) { "Experimental native library size/path check failed." }
        val canonicalLibrary = libraryFile.canonicalFile
        require(sha256(canonicalLibrary) == library.sha256) { "Experimental native library integrity check failed." }
        return canonicalLibrary
    }

    private fun readReceipt(file: File): String {
        require(file.isFile && file.length() in 1..MAX_RECEIPT_BYTES) { "Invalid experimental receipt size." }
        val text = file.readText(Charsets.UTF_8)
        require(text.toByteArray(Charsets.UTF_8).size.toLong() in 1..MAX_RECEIPT_BYTES) { "Invalid experimental receipt size." }
        return text
    }

    private fun sourceMap(json: JSONObject): Map<String, String> = json.keys().asSequence().associateWith { string(json, it) }
    private fun symbolNames(json: JSONObject): List<String> {
        val symbols = json.getJSONArray("verifiedExportedSymbols")
        return (0 until symbols.length()).map { index ->
            val value = symbols.get(index)
            require(value is String) { "Native exported symbol names must be strings." }
            value
        }
    }
    private fun string(json: JSONObject, name: String): String {
        val value = json.get(name)
        require(value is String) { "$name must be a string." }
        return value
    }
    private fun integer(json: JSONObject, name: String): Long {
        val value = json.get(name)
        require(value is Int || value is Long) { "$name must be an integer." }
        return (value as Number).toLong()
    }
    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().buffered().use { stream ->
            val buffer = ByteArray(128 * 1024)
            while (true) {
                val count = stream.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
    }
}
