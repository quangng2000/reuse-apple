// Frozen pre-optimization copying reference for exact FP32 output validation.
// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

import ai.onnxruntime.*
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.CancellationException

/** One ONNX hop with explicit persistent FP32 convolution and recurrent caches. */
internal class CopyingStreamingReferenceProcessor(modelFile: File, private val manifest: StreamingModelManifest,
                                     options: StreamingEnhancerConfiguration) : StreamingSpectralProcessor {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private var state = emptyList<FloatArray>()
    private val spectralShape = longArrayOf(1, 321, 1)

    init {
        require(options.layers == manifest.layers) { "The streaming model was exported for a different exit layer." }
        manifest.verify(modelFile)
        session = OrtSession.SessionOptions().use { config ->
            config.setIntraOpNumThreads(options.intraOpThreads)
            config.setInterOpNumThreads(1)
            config.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            // CPU is the sole registered provider. No FP16, quantization or NNAPI fallback.
            environment.createSession(modelFile.absolutePath, config)
        }
        try {
            val expectedInputs = setOf("magnitude", "phase") + manifest.states.map { it.inputName }
            val expectedOutputs = setOf("enhanced_magnitude", "enhanced_phase") + manifest.states.map { it.outputName }
            require(session.inputNames == expectedInputs && session.outputNames == expectedOutputs) { "Unexpected streaming tensors." }
            for (name in listOf("magnitude", "phase")) validate(session.inputInfo.getValue(name).info, spectralShape)
            for (name in listOf("enhanced_magnitude", "enhanced_phase")) validate(session.outputInfo.getValue(name).info, spectralShape)
            for (spec in manifest.states) {
                validate(session.inputInfo.getValue(spec.inputName).info, spec.shape.toLongArray())
                validate(session.outputInfo.getValue(spec.outputName).info, spec.shape.toLongArray())
            }
            val metadata = session.metadata.customMetadata
            require(metadata["checkpoint_sha256"] == StreamingModelManifest.CHECKPOINT_SHA256 &&
                metadata["source_revision"] == StreamingModelManifest.SOURCE_REVISION &&
                metadata["precision"] == "float32" && metadata["layers"] == manifest.layers.toString() &&
                metadata["lookahead"] == "0" && metadata["license"] == "NSCLv1") { "Invalid streaming graph identity." }
            state = manifest.states.map { FloatArray(it.elements) }
        } catch (error: Throwable) { session.close(); throw error }
    }

    override fun process(input: StreamingSpectrum, cancellation: StreamingCancellationToken): StreamingSpectrum {
        cancellation.throwIfCancelled()
        val tensors = LinkedHashMap<String, OnnxTensor>()
        try {
            tensors["magnitude"] = OnnxTensor.createTensor(environment, FloatBuffer.wrap(input.magnitude), spectralShape)
            tensors["phase"] = OnnxTensor.createTensor(environment, FloatBuffer.wrap(input.phase), spectralShape)
            for ((i, spec) in manifest.states.withIndex())
                tensors[spec.inputName] = OnnxTensor.createTensor(environment, FloatBuffer.wrap(state[i]), spec.shape.toLongArray())
            OrtSession.RunOptions().use { run ->
                cancellation.bind { try { run.setTerminate(true) } catch (_: OrtException) { /* already terminated */ } }
                try {
                    session.run(tensors, run).use { result ->
                        cancellation.throwIfCancelled()
                        fun read(name: String, shape: LongArray, count: Int): FloatArray {
                            val tensor = result.get(name).orElseThrow { IllegalArgumentException("Missing streaming output.") } as? OnnxTensor
                                ?: error("Expected streaming tensor output.")
                            validate(tensor.info, shape)
                            val buffer = tensor.floatBuffer ?: error("Missing streaming FP32 buffer.")
                            require(buffer.remaining() == count) { "Unexpected streaming output count." }
                            return FloatArray(count).also {
                                buffer.get(it)
                                require(it.all(Float::isFinite)) { "Nonfinite streaming model output." }
                            }
                        }
                        val output = StreamingSpectrum(read("enhanced_magnitude", spectralShape, 321),
                            read("enhanced_phase", spectralShape, 321))
                        val next = manifest.states.map { read(it.outputName, it.shape.toLongArray(), it.elements) }
                        cancellation.throwIfCancelled()
                        state = next
                        return output
                    }
                } catch (error: OrtException) {
                    if (cancellation.isCancelled || Thread.currentThread().isInterrupted)
                        throw CancellationException("Streaming enhancement cancelled; reset before continuing.")
                    throw IllegalStateException("Stateful CPU streaming inference failed.", error)
                } finally { cancellation.unbind() }
            }
        } finally { tensors.values.forEach { it.close() } }
    }

    internal fun copyStateForValidation(): List<FloatArray> = state.map { it.copyOf() }
    override fun reset() { state.forEach { it.fill(0f) } }
    override fun close() { session.close(); state = emptyList() }

    private fun validate(info: ValueInfo, expectedShape: LongArray) {
        val tensor = info as? TensorInfo ?: error("Expected FP32 streaming tensor.")
        require(tensor.type == OnnxJavaType.FLOAT && tensor.shape.contentEquals(expectedShape)) {
            "Unexpected streaming tensor type/shape."
        }
    }
}
