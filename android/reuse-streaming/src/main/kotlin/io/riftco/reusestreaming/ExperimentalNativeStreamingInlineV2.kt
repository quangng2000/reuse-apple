// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

import ai.onnxruntime.OrtEnvironment
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * Explicit, local experimental eight-layer FP32 frequency and temporal kernel with the pinned ORT MLAS NEON exponential. It permits FP32 rounding
 * differences from the standard graph; it is not selected by the ordinary enhancer constructor.
 * Supply the exact validated model and native library with trusted, application-bundled receipts.
 * Nothing is downloaded. Android currently supports only the validated arm64-v8a library; the
 * separate host library exists for numerical tests. The streaming model remains NSCLv1 research-only.
 */
object ExperimentalNativeStreamingInlineV2 {
    internal const val MODEL_SHA256 = "55d910c30cd7144187e5f358dca6700ae6e7e1051eb8f7758d454f68e9144c11"
    internal const val BASELINE_MODEL_SHA256 = "18db8d32b638702909de484d2a9cffc97ed09e6454c4caed3f0008fb0aab53e8"
    private const val SOURCE_SHA256 = "77f816373c9d84a75d3f5de1379c1add7d6f88ed16dbe3a36c47cade3a91bc44"
    private const val MAX_RECEIPT_BYTES = 262_144L
    private const val COMPILER_FLAGS = "-O3 -fno-fast-math -ffp-contract=off -fvisibility=hidden"
    private val sources = mapOf(
        "native_inline_v2.cpp" to SOURCE_SHA256,
        "CMakeLists.txt" to "04673a3e31d487a0a1d03298974695a3512b00628d2c200883a93f19d09962c8",
        "build_inline_v2.py" to "7f6b5a39666b96d7e5656bdd1069dff9a426afa25fca272956777289467e0bdd",
        "transform_inline_v2.py" to "9d00c182b50b6b143493eae3817d1826610cf9a335de213cd9c509a88b99709b",
        "mlas_exp_neon.h" to "e65204dc9f5a7748201c186cbb7ef713631d2c94c8facc866ef35b13993e3f24",
        "mlas-provenance.json" to "558d85391b7302a8bb4749c280eb2ab3a37a992b3a06a1fa7fd365d8269e0bc5",
        "v1FrequencySource" to "1708573ce611abc9da7ab8453fbd51c7635f61dd8f5db77ee14788002f6ac030",
        "v1TransformSource" to "c77c1560266f87b61ef8972c0231c52b789c4294c061959cbc15db38f5327be8",
        "ortHeaderProvenance" to "5abee8adc9f970885827b081f21d3091b4ed8bbb8112a523d80a8d0966119281",
    )
    private data class Library(val filename: String, val bytes: Long, val sha256: String)
    private val libraries = mapOf(
        "arm64-v8a" to Library("libreuse_streaming_inline_v2.so", 1_198_920,
            "7a6d98e2f2d13991de0a5aa21aebc8879a262ffdcf2fe85d10a0e027aea3be01"),
        "host" to Library("libreuse_streaming_inline_v2.dylib", 80_432,
            "24ffeb225aaa8b7b62d16caf8af22edff4640a93b0e5a3f27cfba9e914fd8163"),
    )

    @JvmStatic
    fun create(modelFile: File, manifestFile: File, libraryFile: File, libraryReceiptFile: File,
               configuration: StreamingEnhancerConfiguration): StreamingEnhancer {
        val verified = verify(modelFile, manifestFile, libraryFile, libraryReceiptFile, configuration)
        require(OrtEnvironment.getEnvironment().version == "1.23.2") { "Experimental streaming requires ONNX Runtime 1.23.2." }
        return StreamingEnhancer(OnnxStreamingProcessor(modelFile, verified.manifest, configuration,
            verified.library, NativeStreamingVariant.INLINE_FREQUENCY_TEMPORAL_V2), configuration)
    }

    internal data class Verified(val manifest: StreamingModelManifest, val library: File)

    /** All identity and integrity checks complete before any custom code is registered or loaded. */
    internal fun verify(modelFile: File, manifestFile: File, libraryFile: File, libraryReceiptFile: File,
                        configuration: StreamingEnhancerConfiguration): Verified {
        require(configuration.layers == 8) { "Experimental native streaming requires exactly eight layers." }
        val manifestText = readReceipt(manifestFile)
        val manifest = StreamingModelManifest.fromJson(manifestText)
        val json = JSONObject(manifestText)
        require(manifest.layers == 8 && manifest.modelSha256 == MODEL_SHA256 && manifest.modelBytes == 11_584_702L &&
            string(json, "executionVariant") == "native-inline-frequency-temporal-v2" &&
            string(json, "baselineModelSha256") == BASELINE_MODEL_SHA256 &&
            string(json, "nativeFrequencyBaselineModelSha256") == ExperimentalNativeStreaming.MODEL_SHA256 &&
            string(json, "nativeTemporalBaselineModelSha256") == ExperimentalNativeStreamingV2.MODEL_SHA256 &&
            string(json, "inlineExp") == "ort-1.23.2-mlas-neon-fp32" &&
            stringArray(json, "nativeTargets") == listOf("arm64-v8a", "host-arm64") &&
            string(json, "customOpDomain") == "ai.reuse.streaming" && string(json, "customOpName") == "FrequencyScanV2" &&
            string(json, "customTemporalOpName") == "TemporalStepV2" &&
            integer(json, "kernelAbi") == 2L && integer(json, "nativeFrequencyScanCount") == 16L &&
            integer(json, "standardFrequencyScanCount") == 0L && integer(json, "nativeTemporalStepCount") == 8L &&
            integer(json, "standardTemporalStepCount") == 0L && json.get("preparedDecay") == false &&
            json.get("preparedGate") == true && json.get("temporalPreparedDecay") == false &&
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
            string(receipt, "op") == "FrequencyScanV2" && string(receipt, "temporalOp") == "TemporalStepV2" &&
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
    private fun symbolNames(json: JSONObject): List<String> = stringArray(json, "verifiedExportedSymbols")
    private fun stringArray(json: JSONObject, name: String): List<String> {
        val symbols = json.getJSONArray(name)
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
