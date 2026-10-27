// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

import java.io.File

/** Fixed eight-layer network on Vulkan; STFT/framing/overlap-add remain the shared FP32 path. */
internal class VulkanStreamingProcessor(directory: File, tiledDenseKernels: Boolean = false, profiled: Boolean = false) : StreamingSpectralProcessor {
    private var handle = if (profiled) NativeVulkanEngine.createProfiled(File(directory, "plan.bin").absolutePath,
        File(directory, "weights.bin").absolutePath, File(directory, "shaders").absolutePath,
        tiledDenseKernels, tiledDenseKernels) else NativeVulkanEngine.createTuned(File(directory, "plan.bin").absolutePath,
        File(directory, "weights.bin").absolutePath, File(directory, "shaders").absolutePath,
        tiledDenseKernels, tiledDenseKernels)
    override val diagnosticsJson: String
        get() = NativeVulkanEngine.infoJson(openHandle())

    override val profileJson: String?
        get() = NativeVulkanEngine.profileJson(openHandle())

    override fun process(input: StreamingSpectrum, cancellation: StreamingCancellationToken): StreamingSpectrum {
        val current = openHandle()
        cancellation.throwIfCancelled()
        val magnitude = FloatArray(321)
        val phase = FloatArray(321)
        // Vulkan cannot cancel an already submitted dispatch. The fence completes before returning,
        // but the recurrent bank is committed only if the result and cancellation checks pass.
        NativeVulkanEngine.frame(current, input.magnitude, input.phase, magnitude, phase)
        cancellation.throwIfCancelled()
        val result = StreamingSpectrum(magnitude, phase)
        cancellation.throwIfCancelled()
        NativeVulkanEngine.commit(current)
        return result
    }

    override fun reset() = NativeVulkanEngine.reset(openHandle())
    internal fun copyStateForValidation(): List<FloatArray> =
        (0 until 31).map { NativeVulkanEngine.copyState(openHandle(), it) }

    override fun close() {
        val current = handle
        if (current != 0L) {
            handle = 0L
            NativeVulkanEngine.close(current)
        }
    }
    private fun openHandle(): Long = handle.also { check(it != 0L) { "Vulkan processor is closed." } }
}
