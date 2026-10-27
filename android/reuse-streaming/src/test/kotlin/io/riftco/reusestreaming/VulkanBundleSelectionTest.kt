package io.riftco.reusestreaming

import org.junit.Assert.*
import org.junit.Test
import java.io.File
import java.security.MessageDigest

class VulkanBundleSelectionTest {
    private fun manifest(version: String): ByteArray = requireNotNull(
        javaClass.getResourceAsStream("/vulkan-manifests/$version.json")
    ).use { it.readBytes() }

    private fun rejectBeforeNativeLoad(bytes: ByteArray) {
        val directory = kotlin.io.path.createTempDirectory("vulkan-selection-").toFile()
        try {
            File(directory, "manifest.json").writeBytes(bytes)
            val library = File(directory, "libreuse_streaming_vulkan.so")
            val options = StreamingEnhancerConfiguration(8, 1)
            val factories = listOf<() -> StreamingEnhancer>(
                { ExperimentalVulkanStreaming.create(directory, library, options) },
                { ExperimentalVulkanStreaming.createTiled(directory, library, options) },
                { ExperimentalVulkanStreaming.createProfiled(directory, library, options) }
            )
            factories.forEach { factory ->
                val failure = assertThrows(IllegalArgumentException::class.java) { factory().close() }
                assertEquals("Only the validated v2 Vulkan bundle is supported.", failure.message)
            }
        } finally {
            directory.deleteRecursively()
        }
    }

    @Test fun authenticV1BundleIsRejectedByEveryFactoryBeforeLoadingNativeCode() {
        rejectBeforeNativeLoad(manifest("v1"))
    }

    @Test fun authenticV3BundleIsRejectedByEveryFactoryBeforeLoadingNativeCode() {
        rejectBeforeNativeLoad(manifest("v3"))
    }

    @Test fun modifiedV2ManifestIsRejectedByEveryFactory() {
        val original = manifest("v2")
        val digest = MessageDigest.getInstance("SHA-256").digest(original)
            .joinToString("") { "%02x".format(it.toInt() and 255) }
        assertEquals("951adf47ebece2c27a5af4df22b91c30c8a442b8caa04bb01db102ceb0ac733a", digest)
        rejectBeforeNativeLoad(original + byteArrayOf(0x20))
    }
}
