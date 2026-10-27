// NVIDIA Real-time RE-USE adaptation; NSCLv1. Noncommercial research and education only.
package io.riftco.reusestreaming

import java.io.Closeable
import java.io.File

/**
 * Stateful Real-time RE-USE, mono 16 kHz FP32. This separate research module uses NSCLv1 weights.
 * Blocking methods serialize on the instance: call from one dedicated worker, never an audio callback.
 * Call reset after finish, inference failure or cancellation. No network access or recording is performed.
 */
class StreamingEnhancer internal constructor(
    private val processor: StreamingSpectralProcessor,
    val configuration: StreamingEnhancerConfiguration,
) : Closeable {
    constructor(modelFile: File, manifest: StreamingModelManifest,
                configuration: StreamingEnhancerConfiguration = StreamingEnhancerConfiguration()) :
        this(OnnxStreamingProcessor(modelFile, manifest, configuration), configuration)

    val layers: Int get() = configuration.layers
    /** Optional execution details for explicit experimental backends; no audio or model data. */
    val backendDiagnosticsJson: String? @Synchronized get() { checkOpen(); return processor.diagnosticsJson }
    /** Last timestamp-instrumented GPU frame, only for an explicitly profiled experimental backend. */
    val backendProfileJson: String? @Synchronized get() { checkOpen(); return processor.profileJson }
    private var session = StreamingSession()
    private var transform = StreamingTransform()
    private var closed = false
    private val timings = ArrayList<Double>()
    /** Completed frame timings for the last operation; warmup is excluded. */
    val lastFrameMilliseconds: DoubleArray @Synchronized get() = timings.toDoubleArray()

    /** Arbitrary chunk sizes; first output waits for 640 input samples. No normalization is applied. */
    @Synchronized
    fun push(samples: FloatArray, cancellation: StreamingCancellationToken = StreamingCancellationToken()): FloatArray {
        checkOpen()
        timings.clear()
        return session.push(samples, cancellation) { render(it, cancellation) }
    }

    /** Reflect and flush EOF once. Combined output contains exactly as many samples as input. */
    @Synchronized
    fun finish(cancellation: StreamingCancellationToken = StreamingCancellationToken()): FloatArray {
        checkOpen()
        timings.clear()
        return session.finish(cancellation) { render(it, cancellation) }
    }

    /** Clears every convolution/recurrent cache and all framing/overlap state. */
    @Synchronized
    fun reset() {
        checkOpen()
        processor.reset()
        transform = StreamingTransform()
        session = StreamingSession()
        timings.clear()
    }

    /** Three disposable zero frames, then a complete reset. Only allowed before any audio. */
    @Synchronized
    fun warmUp(cancellation: StreamingCancellationToken = StreamingCancellationToken()) {
        checkOpen()
        if (!session.isFresh) throw StreamingException(StreamingException.Reason.WARMUP_REQUIRES_RESET)
        try { repeat(3) { render(FloatArray(640), cancellation) } } finally { reset() }
    }

    @Synchronized
    override fun close() {
        if (!closed) { closed = true; processor.close(); timings.clear() }
    }

    private fun render(frame: FloatArray, cancellation: StreamingCancellationToken): FloatArray {
        cancellation.throwIfCancelled()
        val started = System.nanoTime()
        val output = transform.synthesize(processor.process(transform.analyze(frame), cancellation))
        cancellation.throwIfCancelled()
        timings.add((System.nanoTime() - started) / 1_000_000.0)
        return output
    }

    private fun checkOpen() { check(!closed) { "Enhancer is closed." } }

    companion object {
        const val SAMPLE_RATE = 16_000
        const val HOP_SAMPLES = 320
        const val WINDOW_SAMPLES = 640
    }
}

internal interface StreamingSpectralProcessor : Closeable {
    val diagnosticsJson: String? get() = null
    val profileJson: String? get() = null
    fun process(input: StreamingSpectrum, cancellation: StreamingCancellationToken): StreamingSpectrum
    fun reset()
}
