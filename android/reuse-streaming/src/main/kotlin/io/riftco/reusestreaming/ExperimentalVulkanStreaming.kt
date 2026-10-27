// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

import org.json.JSONObject
import java.io.File
import java.security.MessageDigest

/**
 * Explicit experimental eight-layer FP32 Vulkan network, with the shared streaming DSP and framing.
 * The model and recurrent caches stay on the GPU. No ONNX Runtime inference or CPU model fallback
 * is used. The same weights are used; GPU FP32 rounding can differ from the CPU reference.
 * Only the measured v2 bundle is supported; v1 and v3 experiments are rejected.
 * Supply only the pinned application-bundled artifacts. No download occurs here.
 * This separate streaming model remains licensed for noncommercial research under NSCLv1.
 */
object ExperimentalVulkanStreaming {
    // Exact reproducible artifact identities; quality results are recorded separately in validation receipts.
    internal const val BUNDLE_MANIFEST_SHA256 = "951adf47ebece2c27a5af4df22b91c30c8a442b8caa04bb01db102ceb0ac733a"
    internal const val ANDROID_LIBRARY_SHA256 = "507377ed71ff24cebcf1176995d91ee6ffc9b4d358e6cd6252b601c5baf2fcf5"
    internal const val HOST_LIBRARY_SHA256 = "cd03dd83c31a41474d42da76e018e3c0c6271d93b3fb2866d50d45b96e15c75d"
    private val loadLock = Any()
    private var loadedLibrary: String? = null

    /** Uses the measured tiled v2 kernels. intraOpThreads has no effect on Vulkan dispatch. */
    @JvmStatic
    fun create(bundleDirectory: File, libraryFile: File,
               configuration: StreamingEnhancerConfiguration = StreamingEnhancerConfiguration(8, 1)): StreamingEnhancer {
        return open(bundleDirectory, libraryFile, configuration, true)
    }

    /** Compatibility alias for [create]; both select the same tiled v2 kernels. */
    @JvmStatic
    fun createTiled(bundleDirectory: File, libraryFile: File,
                    configuration: StreamingEnhancerConfiguration = StreamingEnhancerConfiguration(8, 1)): StreamingEnhancer {
        return create(bundleDirectory, libraryFile, configuration)
    }

    /** Timestamp-instrumented diagnostics only; scheduling overhead makes these timings unsuitable for normal latency claims. */
    @JvmStatic
    fun createProfiled(bundleDirectory: File, libraryFile: File,
                       configuration: StreamingEnhancerConfiguration = StreamingEnhancerConfiguration(8, 1),
                       tiled: Boolean = true): StreamingEnhancer {
        return open(bundleDirectory, libraryFile, configuration, tiled, true)
    }

    private fun open(bundleDirectory: File, libraryFile: File,
                     configuration: StreamingEnhancerConfiguration, tiled: Boolean, profiled: Boolean = false): StreamingEnhancer {
        val verified = verify(bundleDirectory, libraryFile, configuration)
        synchronized(loadLock) {
            if (loadedLibrary == null) {
                System.load(verified.library.absolutePath)
                loadedLibrary = verified.library.absolutePath
            } else require(loadedLibrary == verified.library.absolutePath) {
                "A different Vulkan library is already loaded."
            }
        }
        return StreamingEnhancer(VulkanStreamingProcessor(verified.directory, tiled, profiled), configuration)
    }

    internal data class Verified(val directory: File, val library: File)
    internal fun verify(directory: File, library: File, configuration: StreamingEnhancerConfiguration): Verified {
        require(configuration.layers == 8) { "Vulkan streaming supports exactly eight layers." }
        require(directory.isAbsolute && directory.isDirectory) { "Missing Vulkan bundle directory." }
        val root = directory.canonicalFile
        val receipt = File(root, "manifest.json")
        require(receipt.isFile && receipt.length() in 1..262_144L &&
            sha256(receipt) == BUNDLE_MANIFEST_SHA256) {
            "Only the validated v2 Vulkan bundle is supported."
        }
        val manifest = JSONObject(receipt.readText(Charsets.UTF_8))
        require(manifest.getInt("schemaVersion") == 1 && manifest.getString("executionBackend") == "native-vulkan-fp32" &&
            manifest.getString("precision") == "float32" && manifest.getInt("layers") == 8 &&
            manifest.getInt("sampleRate") == 16000 && manifest.getInt("hopSize") == 320 &&
            manifest.getInt("fftSize") == 640 && manifest.getInt("stateCount") == 31 &&
            manifest.getString("checkpointSha256") == StreamingModelManifest.CHECKPOINT_SHA256 &&
            manifest.getString("baselineModelSha256") == ExperimentalNativeStreaming.BASELINE_MODEL_SHA256 &&
            manifest.getBoolean("selectedWeightBytesUnchanged")) { "Unsupported Vulkan graph contract." }
        val files = manifest.getJSONObject("files")
        require(files.length() in 3..64 && files.has("plan.bin") && files.has("weights.bin"))
        for (name in files.keys()) {
            require(name.matches(Regex("(?:[A-Za-z0-9_-]+\\.bin|shaders/[A-Za-z0-9_-]+\\.spv)"))) {
                "Invalid Vulkan bundle path."
            }
            val file = File(root, name)
            require(file.canonicalPath == file.absolutePath && file.isFile) { "Vulkan bundle links are not supported." }
            val pin = files.getJSONObject(name)
            require(file.length() == pin.getLong("bytes") && sha256(file) == pin.getString("sha256")) {
                "Vulkan artifact integrity check failed: $name"
            }
        }
        val expectedLibraryHash = when (library.name) {
            "libreuse_streaming_vulkan.so" -> ANDROID_LIBRARY_SHA256
            "libreuse_streaming_vulkan.dylib" -> HOST_LIBRARY_SHA256
            else -> error("Unsupported Vulkan library name.")
        }
        require(library.isAbsolute && library.isFile && library.length() in 1..32L * 1024 * 1024 &&
            sha256(library) == expectedLibraryHash) { "Untrusted Vulkan native library." }
        return Verified(root, library.canonicalFile)
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
