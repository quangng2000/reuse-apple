package io.riftco.reuseaudio

import java.io.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.AtomicMoveNotSupportedException
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import kotlin.math.*

internal data class DecodedAudio(val file: File, val sampleRate: Int, val sampleCount: Long)

internal class FloatWriter(file: File) : Closeable {
    private val output = BufferedOutputStream(FileOutputStream(file), 64 * 1024)
    private val buffer = ByteBuffer.allocate(16 * 1024).order(ByteOrder.LITTLE_ENDIAN)
    var count = 0L
        private set
    fun write(samples: FloatArray, length: Int = samples.size, gain: Float = 1f) {
        require(length in 0..samples.size)
        for (i in 0 until length) {
            val sample = samples[i] * gain
            require(sample.isFinite()) { "Invalid audio samples." }
            if (buffer.remaining() < 4) flushBuffer()
            buffer.putFloat(sample)
        }
        count += length
    }
    private fun flushBuffer() { output.write(buffer.array(), 0, buffer.position()); buffer.clear() }
    override fun close() { try { flushBuffer() } finally { output.close() } }
}

internal class FloatReader(file: File) : Closeable {
    private val input = RandomAccessFile(file, "r")
    val count = input.length() / 4
    init { require(input.length() % 4 == 0L) { "Truncated float PCM." } }
    fun read(at: Long, count: Int): FloatArray {
        require(at >= 0 && count >= 0 && at + count <= this.count)
        val bytes = ByteArray(count * 4)
        input.seek(at * 4); input.readFully(bytes)
        val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(count).also { buffer.get(it); require(it.all(Float::isFinite)) { "Invalid audio samples." } }
    }
    override fun close() { input.close() }
}

internal object WavFiles {
    fun isWave(file: File): Boolean = file.inputStream().use { stream ->
        val bytes = ByteArray(12)
        stream.read(bytes) == 12 && String(bytes, 0, 4, Charsets.US_ASCII) == "RIFF" &&
            String(bytes, 8, 4, Charsets.US_ASCII) == "WAVE"
    }

    fun decode(input: File, target: File, cancellation: CancellationToken): DecodedAudio {
        RandomAccessFile(input, "r").use { file ->
            require(file.length() >= 44) { "Truncated WAV." }
            file.seek(12)
            var format = 0; var channels = 0; var rate = 0; var bits = 0; var blockAlign = 0
            var dataAt = -1L; var dataBytes = 0L
            while (file.filePointer + 8 <= file.length()) {
                val id = ByteArray(4).also(file::readFully).toString(Charsets.US_ASCII)
                val size = Integer.reverseBytes(file.readInt()).toLong() and 0xffff_ffffL
                val at = file.filePointer
                require(size <= file.length() - at) { "Truncated WAV chunk." }
                if (id == "fmt ") {
                    require(size >= 16)
                    format = java.lang.Short.reverseBytes(file.readShort()).toInt() and 65535
                    channels = java.lang.Short.reverseBytes(file.readShort()).toInt() and 65535
                    rate = Integer.reverseBytes(file.readInt())
                    file.skipBytes(4)
                    blockAlign = java.lang.Short.reverseBytes(file.readShort()).toInt() and 65535
                    bits = java.lang.Short.reverseBytes(file.readShort()).toInt() and 65535
                    if (format == 0xfffe && size >= 40) {
                        file.skipBytes(8)
                        format = java.lang.Short.reverseBytes(file.readShort()).toInt() and 65535
                    }
                }
                if (id == "data") { dataAt = at; dataBytes = size }
                file.seek(at + size + (size and 1))
            }
            require(channels in 1..2 && rate in 8_000..192_000 &&
                ((format == 1 && bits in listOf(16, 24, 32)) || (format == 3 && bits == 32)) &&
                blockAlign == channels * (bits / 8) && dataAt >= 0 && dataBytes > 0 &&
                dataBytes % blockAlign == 0L) { "Unsupported WAV; use mono/stereo PCM16/24/32 or float32." }
            file.seek(dataAt)
            val samples = dataBytes / blockAlign
            FloatWriter(target).use { writer ->
                var remaining = samples
                while (remaining > 0) {
                    cancellation.throwIfCancelled()
                    val frames = min(4096, remaining).toInt()
                    val bytes = ByteArray(frames * blockAlign).also(file::readFully)
                    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                    val mono = FloatArray(frames)
                    for (frame in 0 until frames) {
                        var value = 0f
                        repeat(channels) {
                            value += when {
                                format == 3 -> buffer.float
                                bits == 16 -> buffer.short / 32768f
                                bits == 32 -> buffer.int / 2147483648f
                                else -> {
                                    val b0 = buffer.get().toInt() and 255
                                    val b1 = buffer.get().toInt() and 255
                                    val b2 = buffer.get().toInt()
                                    (b0 or (b1 shl 8) or (b2 shl 16)) / 8388608f
                                }
                            }
                        }
                        mono[frame] = value / channels
                    }
                    writer.write(mono); remaining -= frames
                }
            }
            return DecodedAudio(target, rate, samples)
        }
    }

    fun write(raw: File, target: File, gain: Float, cancellation: CancellationToken) {
        val samples = raw.length() / 4
        require(samples in 1..((0xffff_ffffL - 36) / 4)) { "Recording exceeds WAV size limit." }
        val dataBytes = samples * 4
        FileOutputStream(target).use { file ->
            val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            header.put("RIFF".toByteArray()).putInt((dataBytes + 36).toInt()).put("WAVEfmt ".toByteArray())
                .putInt(16).putShort(3).putShort(1).putInt(16_000).putInt(64_000).putShort(4).putShort(32)
                .put("data".toByteArray()).putInt(dataBytes.toInt())
            file.write(header.array())
            val buffer = ByteBuffer.allocate(16 * 1024).order(ByteOrder.LITTLE_ENDIAN)
            FloatReader(raw).use { reader ->
                var at = 0L
                while (at < samples) {
                    cancellation.throwIfCancelled()
                    val data = reader.read(at, min(4096, samples - at).toInt())
                    buffer.clear()
                    for (sample in data) {
                        val output = sample * gain
                        require(output.isFinite()) { "Invalid output audio." }
                        buffer.putFloat(output)
                    }
                    file.write(buffer.array(), 0, buffer.position())
                    at += data.size
                }
            }
            file.fd.sync()
        }
    }
}

internal object PcmResampler {
    /** 64-tap windowed-sinc downsampling/upsampling; never materializes a complete clip. */
    fun normalize(source: DecodedAudio, target: File, cancellation: CancellationToken): DecodedAudio {
        require(source.sampleRate in 8_000..192_000 && source.sampleCount > 0)
        if (source.sampleRate == 16_000) return source
        val count = (source.sampleCount.toDouble() * 16_000 / source.sampleRate).roundToLong().coerceAtLeast(1)
        require(count <= (0xffff_ffffL - 36) / 4) { "Recording exceeds WAV size limit." }
        val ratio = source.sampleRate / 16_000.0
        val cutoff = min(1.0, 1.0 / ratio)
        // At high source rates widen support to retain the same low-pass transition width.
        val half = ceil(32 / cutoff).toInt()
        FloatReader(source.file).use { input ->
            FloatWriter(target).use { writer ->
                var at = 0L
                while (at < count) {
                    cancellation.throwIfCancelled()
                    val outputCount = min(1024L, count - at).toInt()
                    val first = max(0L, floor(at * ratio).toLong() - half)
                    val last = min(source.sampleCount, floor((at + outputCount - 1) * ratio).toLong() + half + 1)
                    val data = input.read(first, (last - first).toInt())
                    val output = FloatArray(outputCount) { i ->
                        val position = (at + i) * ratio
                        val center = floor(position).toLong()
                        var sum = 0.0; var weights = 0.0
                        for (j in center - half + 1..center + half) {
                            if (j < 0 || j >= source.sampleCount) continue
                            val delta = j - position
                            val x = PI * delta * cutoff
                            val sinc = if (abs(x) < 1e-12) 1.0 else sin(x) / x
                            val weight = cutoff * sinc * (0.5 + 0.5 * cos(PI * delta / half))
                            sum += data[(j - first).toInt()] * weight; weights += weight
                        }
                        (sum / weights).toFloat()
                    }
                    writer.write(output); at += outputCount
                }
            }
        }
        return DecodedAudio(target, 16_000, count)
    }
}

internal object FileEnhancement {
    fun render(input: File, output: File, scratchDirectory: File, processor: SpectralProcessor,
               modelId: String, cancellation: CancellationToken, progress: (Double) -> Unit): EnhancementResult {
        require(input.isFile && input.canonicalFile != output.canonicalFile) { "Choose distinct input/output files." }
        require(scratchDirectory.isDirectory || scratchDirectory.mkdirs()) { "Cannot create scratch directory." }
        val parent = output.absoluteFile.parentFile!!
        require(parent.isDirectory) { "Output directory does not exist." }
        val scratch = Files.createTempDirectory(scratchDirectory.toPath(), "reuse-").toFile()
        var staged: File? = null
        try {
            cancellation.throwIfCancelled(); progress(0.0)
            val decodedFile = File(scratch, "decoded.f32")
            val decoded = if (WavFiles.isWave(input)) WavFiles.decode(input, decodedFile, cancellation)
                else AndroidAudioDecoder.decode(input, decodedFile, cancellation)
            cancellation.throwIfCancelled(); progress(0.08)
            val normalized = PcmResampler.normalize(decoded, File(scratch, "normalized.f32"), cancellation)
            progress(0.12)
            var peak = 0f
            val raw = File(scratch, "restored.f32")
            FloatReader(normalized.file).use { reader ->
                FloatWriter(raw).use { writer ->
                    WindowedEnhancer.render(normalized.sampleCount, reader::read, { data, count ->
                        for (i in 0 until count) {
                            require(data[i].isFinite()) { "Model produced invalid audio." }
                            peak = max(peak, abs(data[i]))
                        }
                        writer.write(data, count)
                    }, processor, cancellation) { progress(0.12 + it * 0.80) }
                    require(writer.count == normalized.sampleCount) { "Output length mismatch." }
                }
            }
            val gain = if (peak > .98f) .98f / peak else 1f
            staged = File.createTempFile(".reuse-", ".wav", parent)
            WavFiles.write(raw, staged, gain, cancellation)
            progress(1.0); cancellation.throwIfCancelled()
            try {
                Files.move(staged.toPath(), output.toPath(), StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)
            } catch (_: AtomicMoveNotSupportedException) {
                Files.move(staged.toPath(), output.toPath(), StandardCopyOption.REPLACE_EXISTING)
            }
            // Publishing is the commit point; cancellation after this point belongs to the caller's save phase.
            return EnhancementResult(normalized.sampleCount, 16_000, peak, gain, modelId)
        } finally { staged?.delete(); scratch.deleteRecursively() }
    }
}
