package io.riftco.reuseaudio

import ai.onnxruntime.*
import java.io.Closeable
import java.io.File
import java.nio.FloatBuffer
import java.util.concurrent.CancellationException
import kotlin.math.*

data class InferenceOptions(val intraOpThreads: Int = 2) {
    init { require(intraOpThreads in 1..8) { "Choose 1–8 CPU inference threads." } }
}

data class EnhancementResult(
    val sampleCount: Long,
    val sampleRate: Int,
    val peakBeforeGain: Float,
    val appliedGain: Float,
    val modelId: String,
)

/**
 * Commercial 30-layer RE-USE, CPU FP32 only. All methods block: call on a dedicated worker.
 * One session serializes jobs. close() waits for its active job; cancel the token first.
 * No model/audio networking or recording-library ownership is performed by this SDK.
 */
class ReuseEnhancer(
    modelFile: File,
    val manifest: ModelManifest,
    options: InferenceOptions = InferenceOptions(),
) : Closeable {
    private val processor = OnnxSpectralProcessor(modelFile, manifest, options)
    private var closed = false

    /** Raw waveform, same windowing as Apple ReuseEngine; no peak normalization. */
    @Synchronized
    fun enhance(
        input: FloatArray,
        cancellation: CancellationToken = CancellationToken(),
        progress: (Double) -> Unit = {},
    ): FloatArray {
        check(!closed) { "Enhancer is closed." }
        require(input.isNotEmpty() && input.all(Float::isFinite)) { "Audio must be finite 16 kHz mono PCM." }
        val output = FloatArray(input.size)
        var written = 0
        WindowedEnhancer.render(input.size.toLong(),
            read = { at, count -> input.copyOfRange(at.toInt(), at.toInt() + count) },
            emit = { samples, count -> samples.copyInto(output, written, 0, count); written += count },
            processor = processor, cancellation = cancellation, progress = progress)
        cancellation.throwIfCancelled()
        return output
    }

    /**
     * Bounded WAV/AAC/M4A import, 16 kHz mono normalization, windowed inference, then one global
     * gain min(1, .98/peak). The existing output is unchanged on failure/cancellation. The input
     * and output must be different files. Scratch files are uniquely owned and always removed.
     */
    @Synchronized
    fun enhanceFile(
        input: File,
        output: File,
        scratchDirectory: File,
        cancellation: CancellationToken = CancellationToken(),
        progress: (Double) -> Unit = {},
    ): EnhancementResult {
        check(!closed) { "Enhancer is closed." }
        return FileEnhancement.render(input, output, scratchDirectory, processor, manifest.modelId,
            cancellation, progress)
    }

    @Synchronized
    override fun close() { if (!closed) { closed = true; processor.close() } }
}

internal fun interface SpectralProcessor {
    fun process(input: Spectrum, cancellation: CancellationToken): Spectrum
}

internal class OnnxSpectralProcessor(
    modelFile: File,
    manifest: ModelManifest,
    options: InferenceOptions,
) : SpectralProcessor, Closeable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        manifest.verify(modelFile)
        session = OrtSession.SessionOptions().use { config ->
            config.setIntraOpNumThreads(options.intraOpThreads)
            config.setInterOpNumThreads(1)
            config.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            // CPU is the default and the only provider registered here. Never enable FP16/NNAPI.
            environment.createSession(modelFile.absolutePath, config)
        }
        try {
            require(session.inputNames == setOf("magnitude", "phase") &&
                session.outputNames == setOf("enhanced_magnitude", "enhanced_phase")) { "Unexpected model tensors." }
            for (info in session.inputInfo.values + session.outputInfo.values) {
                val tensor = info.info as? TensorInfo ?: error("Expected float32 model tensors.")
                require(tensor.type == OnnxJavaType.FLOAT && tensor.shape.size == 3 &&
                    tensor.shape[0] == 1L && tensor.shape[1] == 321L && tensor.shape[2] == -1L) {
                    "Expected dynamic [1, 321, frames] float32 model tensors."
                }
            }
        } catch (error: Throwable) { session.close(); throw error }
    }

    override fun process(input: Spectrum, cancellation: CancellationToken): Spectrum {
        cancellation.throwIfCancelled()
        val shape = longArrayOf(1, 321, input.frames.toLong())
        OnnxTensor.createTensor(environment, FloatBuffer.wrap(input.magnitude), shape).use { magnitude ->
            OnnxTensor.createTensor(environment, FloatBuffer.wrap(input.phase), shape).use { phase ->
                OrtSession.RunOptions().use { run ->
                    cancellation.bind { try { run.setTerminate(true) } catch (_: OrtException) { /* still cancelled */ } }
                    try {
                        session.run(mapOf("magnitude" to magnitude, "phase" to phase), run).use { result ->
                            cancellation.throwIfCancelled()
                            fun read(name: String): FloatArray {
                                val value = result.get(name).orElseThrow { IllegalArgumentException("Missing model output.") }
                                val tensor = value as? OnnxTensor ?: error("Expected tensor model output.")
                                val info = tensor.info
                                require(info.type == OnnxJavaType.FLOAT && info.shape.contentEquals(shape)) {
                                    "Unexpected model output length/type."
                                }
                                val buffer = tensor.floatBuffer ?: throw IllegalArgumentException("Missing float32 output.")
                                return FloatArray(321 * input.frames).also { buffer.get(it) }
                            }
                            return Spectrum(input.frames, read("enhanced_magnitude"), read("enhanced_phase"))
                        }
                    } catch (error: OrtException) {
                        if (cancellation.isCancelled) throw CancellationException("Enhancement cancelled.")
                        throw IllegalStateException("CPU inference failed.", error)
                    } finally { cancellation.unbind() }
                }
            }
        }
    }

    override fun close() { session.close() }
}

internal object WindowedEnhancer {
    const val WINDOW = 64_000
    const val OVERLAP = 8_000
    const val STRIDE = WINDOW - OVERLAP

    fun render(
        sampleCount: Long,
        read: (Long, Int) -> FloatArray,
        emit: (FloatArray, Int) -> Unit,
        processor: SpectralProcessor,
        cancellation: CancellationToken,
        progress: (Double) -> Unit,
    ) {
        require(sampleCount > 0)
        val limit = max(1L, sampleCount - OVERLAP)
        val windows = (limit - 1) / STRIDE + 1
        val signal = FloatArray(WINDOW)
        val weight = FloatArray(WINDOW)
        var start = 0L
        var index = 0L
        while (start < limit) {
            cancellation.throwIfCancelled()
            progress(index.toDouble() / windows)
            val valid = min(WINDOW.toLong(), sampleCount - start).toInt()
            val input = read(start, valid)
            require(input.size == valid && input.all(Float::isFinite)) { "Invalid or truncated audio." }
            val padded = input.copyOf(max(640, ((valid + 79) / 80) * 80))
            val spectrum = AudioTransform.analyze(padded, cancellation)
            val processed = processor.process(spectrum, cancellation)
            require(processed.frames == spectrum.frames) { "Model changed the frame count." }
            val output = AudioTransform.synthesize(processed, padded.size, cancellation)
            val last = index == windows - 1
            for (i in 0 until valid) {
                var gain = 1f
                if (index > 0 && i < OVERLAP) gain *= (i + 1).toFloat() / (OVERLAP + 1)
                if (!last && i >= valid - OVERLAP) gain *= (valid - i).toFloat() / (OVERLAP + 1)
                signal[i] += output[i] * gain
                weight[i] += gain
            }
            val stable = if (last) valid else STRIDE
            for (i in 0 until stable) signal[i] /= max(weight[i], 1e-8f)
            cancellation.throwIfCancelled()
            emit(signal, stable)
            val tail = valid - stable
            signal.copyInto(signal, 0, stable, valid)
            weight.copyInto(weight, 0, stable, valid)
            signal.fill(0f, tail); weight.fill(0f, tail)
            start += STRIDE
            index++
        }
        cancellation.throwIfCancelled()
        progress(1.0)
    }
}
