// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

import ai.onnxruntime.*
import java.io.File
import java.util.concurrent.CancellationException

/** One unchanged ONNX hop with persistent, disjoint direct FP32 state input/output banks. */
internal class OnnxStreamingProcessor(modelFile: File, private val manifest: StreamingModelManifest,
                                     options: StreamingEnhancerConfiguration,
                                     customOpLibrary: File? = null,
                                     customVariant: NativeStreamingVariant = NativeStreamingVariant.FREQUENCY_V1) : StreamingSpectralProcessor {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private var customSessionOptions: OrtSession.SessionOptions? = null
    private var buffers: PinnedStreamingBuffers? = null
    private var runOptions: OrtSession.RunOptions? = null
    private val spectralShape = longArrayOf(1, 321, 1)
    private val requestedOutputs = emptySet<String>()
    private var closed = false

    init {
        require(options.layers == manifest.layers) { "The streaming model was exported for a different exit layer." }
        manifest.verify(modelFile)
        fun createSession(config: OrtSession.SessionOptions): OrtSession {
            config.setIntraOpNumThreads(options.intraOpThreads)
            config.setInterOpNumThreads(1)
            config.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            // CPU is the sole registered provider. No FP16, quantization or NNAPI fallback.
            customOpLibrary?.let { config.registerCustomOpLibrary(it.absolutePath) }
            return environment.createSession(modelFile.absolutePath, config)
        }
        session = if (customOpLibrary == null) OrtSession.SessionOptions().use { createSession(it) } else {
            // ORT Java unloads registered libraries when SessionOptions closes. Keep the custom
            // options alive until after the session and every callback have been destroyed.
            val config = OrtSession.SessionOptions()
            try { createSession(config).also { customSessionOptions = config } }
            catch (failure: Throwable) { config.close(); throw failure }
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
            if (customOpLibrary != null) require(customVariant.metadata.all { (key, value) -> metadata[key] == value }) {
                "Invalid experimental streaming graph identity."
            }
            buffers = PinnedStreamingBuffers(environment, manifest)
            runOptions = OrtSession.RunOptions()
        } catch (failure: Throwable) {
            buffers?.close()
            buffers = null
            runOptions?.close()
            runOptions = null
            try { session.close() } finally {
                val config = customSessionOptions
                customSessionOptions = null
                config?.close()
            }
            throw failure
        }
    }

    override fun process(input: StreamingSpectrum, cancellation: StreamingCancellationToken): StreamingSpectrum {
        check(!closed) { "Streaming processor is closed." }
        cancellation.throwIfCancelled()
        val storage = checkNotNull(buffers)
        val run = checkNotNull(runOptions)
        run.setTerminate(false)
        storage.writeSpectrum(input)
        cancellation.bind { try { run.setTerminate(true) } catch (_: OrtException) { /* already terminated */ } }
        try {
            cancellation.throwIfCancelled()
            // Every output is supplied by the caller. Result does not own or close these tensors.
            session.run(storage.inputs, requestedOutputs, storage.outputs, run).use { result ->
                check(result.size() == storage.outputs.size) { "Unexpected streaming output count." }
                cancellation.throwIfCancelled()
                require(storage.nextIsFinite()) { "Nonfinite streaming model output." }
                val output = storage.copySpectrum()
                cancellation.throwIfCancelled()
                storage.commitNext()
                return output
            }
        } catch (failure: OrtException) {
            if (cancellation.isCancelled || Thread.currentThread().isInterrupted)
                throw CancellationException("Streaming enhancement cancelled; reset before continuing.")
            throw IllegalStateException("Stateful CPU streaming inference failed.", failure)
        } finally { cancellation.unbind() }
    }

    override fun reset() {
        check(!closed) { "Streaming processor is closed." }
        checkNotNull(runOptions).setTerminate(false)
        checkNotNull(buffers).reset()
    }

    /** Expensive state copies are validation diagnostics, never part of normal processing. */
    internal fun copyStateForValidation(): List<FloatArray> = checkNotNull(buffers).copyCommittedStateForValidation()

    override fun close() {
        if (!closed) {
            closed = true
            try { session.close() } finally {
                val config = customSessionOptions
                customSessionOptions = null
                try { config?.close() } finally {
                    try { runOptions?.close() } finally {
                        runOptions = null
                        val storage = buffers
                        buffers = null
                        storage?.close()
                    }
                }
            }
        }
    }

    private fun validate(info: ValueInfo, expectedShape: LongArray) {
        val tensor = info as? TensorInfo ?: error("Expected FP32 streaming tensor.")
        require(tensor.type == OnnxJavaType.FLOAT && tensor.shape.contentEquals(expectedShape)) {
            "Unexpected streaming tensor type/shape."
        }
    }
}
