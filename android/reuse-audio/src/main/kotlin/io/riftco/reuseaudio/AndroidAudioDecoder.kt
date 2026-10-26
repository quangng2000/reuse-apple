package io.riftco.reuseaudio

import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.io.File
import java.nio.ByteOrder

/** Native decoder, bounded PCM buffers. Does not record, play, or contact any network. */
internal object AndroidAudioDecoder {
    fun decode(input: File, target: File, cancellation: CancellationToken): DecodedAudio {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(input.absolutePath)
            val track = (0 until extractor.trackCount).firstOrNull {
                extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
            } ?: throw IllegalArgumentException("No audio track found.")
            extractor.selectTrack(track)
            val format = extractor.getTrackFormat(track)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: error("Missing audio format.")
            val delay = if (format.containsKey(MediaFormat.KEY_ENCODER_DELAY)) format.getInteger(MediaFormat.KEY_ENCODER_DELAY) else 0
            val padding = if (format.containsKey(MediaFormat.KEY_ENCODER_PADDING)) format.getInteger(MediaFormat.KEY_ENCODER_PADDING) else 0
            require(delay >= 0 && padding >= 0) { "Invalid audio trim metadata." }
            var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            require(rate in 8_000..192_000 && channels in 1..2) { "Choose mono/stereo audio at 8–192 kHz." }
            codec = MediaCodec.createDecoderByType(mime)
            codec.configure(format, null, null, 0); codec.start()
            var encoding = AudioFormat.ENCODING_PCM_16BIT
            var inputEnded = false; var outputEnded = false
            var lastActivity = System.nanoTime()
            val info = MediaCodec.BufferInfo()
            FloatWriter(target).use { writer ->
                while (!outputEnded) {
                    cancellation.throwIfCancelled()
                    if (!inputEnded) {
                        val index = codec.dequeueInputBuffer(10_000)
                        if (index >= 0) {
                            val buffer = codec.getInputBuffer(index) ?: error("Missing decoder input buffer.")
                            buffer.clear()
                            val bytes = extractor.readSampleData(buffer, 0)
                            if (bytes < 0) {
                                codec.queueInputBuffer(index, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                                inputEnded = true
                            } else {
                                codec.queueInputBuffer(index, 0, bytes, extractor.sampleTime, 0)
                                extractor.advance()
                            }
                            lastActivity = System.nanoTime()
                        }
                    }
                    val index = codec.dequeueOutputBuffer(info, 10_000)
                    if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                        val output = codec.outputFormat
                        val newRate = output.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        val newChannels = output.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        require(writer.count == 0L || (newRate == rate && newChannels == channels)) { "Audio format changed mid-recording." }
                        rate = newRate; channels = newChannels
                        require(rate in 8_000..192_000 && channels in 1..2) { "Unsupported decoder PCM format." }
                        encoding = if (output.containsKey(MediaFormat.KEY_PCM_ENCODING)) output.getInteger(MediaFormat.KEY_PCM_ENCODING)
                            else AudioFormat.ENCODING_PCM_16BIT
                        require(encoding == AudioFormat.ENCODING_PCM_16BIT || encoding == AudioFormat.ENCODING_PCM_FLOAT) {
                            "Decoder must provide PCM16 or float32."
                        }
                        lastActivity = System.nanoTime()
                    } else if (index >= 0) {
                        try {
                            if (info.size > 0 && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                                val buffer = codec.getOutputBuffer(index) ?: error("Missing decoder PCM buffer.")
                                buffer.position(info.offset); buffer.limit(info.offset + info.size)
                                buffer.order(ByteOrder.LITTLE_ENDIAN)
                                val bytes = if (encoding == AudioFormat.ENCODING_PCM_FLOAT) 4 else 2
                                require(info.size % (bytes * channels) == 0) { "Truncated decoder PCM." }
                                val mono = FloatArray(info.size / (bytes * channels))
                                for (frame in mono.indices) {
                                    var sum = 0f
                                    repeat(channels) { sum += if (bytes == 4) buffer.float else buffer.short / 32768f }
                                    mono[frame] = sum / channels
                                }
                                writer.write(mono)
                            }
                            outputEnded = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                        } finally { codec.releaseOutputBuffer(index, false) }
                        lastActivity = System.nanoTime()
                    }
                    check(System.nanoTime() - lastActivity < 60_000_000_000L) { "Audio decoder timed out." }
                }
                require(writer.count > 0) { "The audio recording is empty." }
                // MediaCodec consumes encoder-delay/padding in this unmodified format.
                // Trimming the returned PCM again would remove real audio a second time.
                return DecodedAudio(target, rate, writer.count)
            }
        } finally {
            try { codec?.stop() } catch (_: Exception) { }
            codec?.release(); extractor.release()
        }
    }

}
