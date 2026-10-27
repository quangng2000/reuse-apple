// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Two disjoint FP32 state banks. ORT writes the next bank without copying through Java arrays. */
internal class PinnedStreamingBuffers(environment: OrtEnvironment, manifest: StreamingModelManifest) : Closeable {
    private class TensorBuffer(environment: OrtEnvironment, shape: LongArray, val elements: Int) : Closeable {
        val bytes: ByteBuffer = ByteBuffer.allocateDirect(Math.multiplyExact(elements, 4)).order(ByteOrder.nativeOrder())
        val values = bytes.asFloatBuffer()
        val tensor = OnnxTensor.createTensor(environment, values, shape)
        init {
            if (tensor.ownsBuffer() || !tensor.bufferRef.isPresent) {
                tensor.close()
                error("FP32 tensor must reference the supplied direct buffer.")
            }
        }
        fun write(input: FloatArray) {
            require(input.size == elements)
            values.position(0)
            values.put(input)
            values.position(0)
        }
        fun copy(): FloatArray = FloatArray(elements).also {
            values.position(0)
            values.get(it)
            values.position(0)
        }
        override fun close() { tensor.close() }
    }
    private val owned = ArrayList<TensorBuffer>()
    private val magnitudeInput: TensorBuffer
    private val phaseInput: TensorBuffer
    private val magnitudeOutput: TensorBuffer
    private val phaseOutput: TensorBuffer
    private val banks: Array<List<TensorBuffer>>
    private val feeds: Array<Map<String, OnnxTensor>>
    private val pinnedOutputs: Array<Map<String, OnnxTensor>>
    private val nextBytes: Array<Array<ByteBuffer>>
    private val outputCounts: IntArray
    private val allBytes: Array<ByteBuffer>
    private val allCounts: IntArray
    private var committedBank = 0

    init {
        try {
            fun allocate(shape: LongArray, elements: Int): TensorBuffer {
                val buffer = TensorBuffer(environment, shape, elements)
                try { owned.add(buffer) } catch (failure: Throwable) {
                    buffer.close()
                    throw failure
                }
                return buffer
            }
            val spectrumShape = longArrayOf(1, 321, 1)
            magnitudeInput = allocate(spectrumShape, 321)
            phaseInput = allocate(spectrumShape, 321)
            magnitudeOutput = allocate(spectrumShape, 321)
            phaseOutput = allocate(spectrumShape, 321)
            banks = Array(2) { manifest.states.map { allocate(it.shape.toLongArray(), it.elements) } }
            feeds = Array(2) { bank -> LinkedHashMap<String, OnnxTensor>().apply {
                put("magnitude", magnitudeInput.tensor)
                put("phase", phaseInput.tensor)
                manifest.states.forEachIndexed { i, state -> put(state.inputName, banks[bank][i].tensor) }
            } }
            pinnedOutputs = Array(2) { bank -> LinkedHashMap<String, OnnxTensor>().apply {
                put("enhanced_magnitude", magnitudeOutput.tensor)
                put("enhanced_phase", phaseOutput.tensor)
                manifest.states.forEachIndexed { i, state -> put(state.outputName, banks[1 - bank][i].tensor) }
            } }
            nextBytes = Array(2) { bank ->
                (listOf(magnitudeOutput.bytes, phaseOutput.bytes) + banks[1 - bank].map { it.bytes }).toTypedArray()
            }
            outputCounts = intArrayOf(321, 321) + manifest.states.map { it.elements }.toIntArray()
            allBytes = owned.map { it.bytes }.toTypedArray()
            allCounts = owned.map { it.elements }.toIntArray()
        } catch (failure: Throwable) {
            owned.forEach { it.close() }
            throw failure
        }
    }

    val inputs: Map<String, OnnxTensor> get() = feeds[committedBank]
    val outputs: Map<String, OnnxTensor> get() = pinnedOutputs[committedBank]
    fun writeSpectrum(input: StreamingSpectrum) {
        magnitudeInput.write(input.magnitude)
        phaseInput.write(input.phase)
    }
    fun nextIsFinite(): Boolean = NativeStreamingBuffers.allFinite(nextBytes[committedBank], outputCounts)
    fun copySpectrum(): StreamingSpectrum = StreamingSpectrum(magnitudeOutput.copy(), phaseOutput.copy())
    /** The caller commits only after a completed, finite, uncancelled inference. */
    fun commitNext() { committedBank = 1 - committedBank }
    fun reset() {
        NativeStreamingBuffers.zero(allBytes, allCounts)
        committedBank = 0
    }
    /** Explicit diagnostics only; the steady path never copies the large state banks. */
    fun copyCommittedStateForValidation(): List<FloatArray> = banks[committedBank].map { it.copy() }
    override fun close() { owned.forEach { it.close() } }
}
