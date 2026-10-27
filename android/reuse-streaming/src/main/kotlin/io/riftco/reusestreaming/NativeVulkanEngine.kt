// Original Vulkan/JNI adapter; MIT adapter terms in ADAPTER_LICENSE.txt.
package io.riftco.reusestreaming

/** Loaded only after the experimental factory verifies the exact native library. */
internal object NativeVulkanEngine {
    @JvmStatic external fun create(planPath: String, weightsPath: String, shaderDirectory: String): Long
    @JvmStatic external fun createTuned(planPath: String, weightsPath: String, shaderDirectory: String,
                                      tiledMatmul: Boolean, tiledConv: Boolean): Long
    @JvmStatic external fun createProfiled(planPath: String, weightsPath: String, shaderDirectory: String,
                                         tiledMatmul: Boolean, tiledConv: Boolean): Long
    @JvmStatic external fun profileJson(handle: Long): String?
    @JvmStatic external fun frame(handle: Long, magnitude: FloatArray, phase: FloatArray,
                                 outputMagnitude: FloatArray, outputPhase: FloatArray)
    @JvmStatic external fun commit(handle: Long)
    @JvmStatic external fun reset(handle: Long)
    @JvmStatic external fun close(handle: Long)
    @JvmStatic external fun infoJson(handle: Long): String
    @JvmStatic external fun copyState(handle: Long, index: Int): FloatArray
}
