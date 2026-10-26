package io.riftco.reuseaudio

import android.content.Context
import android.media.*
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.*
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.*

@RunWith(AndroidJUnit4::class)
class AndroidEnhancementTests {
    private val context: Context = ApplicationProvider.getApplicationContext()
    private val fixtures = InstrumentationRegistry.getInstrumentation().context.assets

    @Test fun nativeAacDecodeAndBoundedFilePipeline() = withDirectory { directory ->
        val input = File(directory, "synthetic.m4a")
        encodeAac(input, 48_000, 2)
        val expectedSamples = expectedAacSamples(input)
        val before = input.readBytes()
        val output = File(directory, "restored.wav")
        var fraction = -1.0
        val result = FileEnhancement.render(input, output, directory, SpectralProcessor { value, _ -> value },
            ModelManifest.MODEL_ID, CancellationToken()) {
            assertTrue(it.isFinite() && it >= fraction && it in 0.0..1.0); fraction = it
        }
        assertEquals((expectedSamples * 16_000.0 / 48_000).roundToLong(), result.sampleCount)
        assertEquals(16_000, result.sampleRate)
        val decoded = WavFiles.decode(output, File(directory, "check.f32"), CancellationToken())
        FloatReader(decoded.file).use { reader ->
            val samples = reader.read(0, decoded.sampleCount.toInt())
            assertTrue(samples.all(Float::isFinite))
            assertTrue(samples.maxOf { abs(it) } > .2f)
        }
        assertArrayEquals(before, input.readBytes())
        assertEquals(1.0, fraction, 0.0)
    }

    @Test fun fullThirtyLayerCpuParityOnAndroid() = withDirectory { directory ->
        val names = fixtures.list("")!!.toSet()
        assumeTrue("Stage generated trusted model + parity assets first.", "reuse-30-fp32.onnx" in names)
        fun asset(name: String): File = File(directory, name).also { target ->
            fixtures.open(name).use { input -> target.outputStream().use(input::copyTo) }
        }
        val model = asset("reuse-30-fp32.onnx")
        val manifest = ModelManifest.read(asset("reuse-30-fp32.manifest.json"))
        val input = floats(asset("640-input.f32"))
        val expected = floats(asset("640-waveform.f32"))
        val spectrum = Spectrum(9, floats(asset("640-magnitude.f32")), floats(asset("640-phase.f32")))
        OnnxSpectralProcessor(model, manifest, InferenceOptions(2)).use { processor ->
            val result = processor.process(spectrum, CancellationToken())
            val waveform = AudioTransform.synthesize(result, input.size, CancellationToken())
            assertEquals(expected.size, waveform.size)
            assertTrue(waveform.indices.maxOf { abs(waveform[it] - expected[it]) } < 5e-5f)
        }
        ReuseEnhancer(model, manifest).use { enhancer ->
            val actual = enhancer.enhance(input)
            assertEquals(input.size, actual.size)
            // Includes the pinned Float32 PocketFFT/Hann path as well as spectral inference.
            val maxError = actual.indices.maxOf { abs(actual[it] - expected[it]) }
            val rmsError = sqrt(actual.indices.sumOf { (actual[it] - expected[it]).toDouble().pow(2) } / actual.size)
            assertTrue("Full PCM max=$maxError rms=$rmsError", maxError < 1e-4f)
            val token = CancellationToken().also { it.cancel() }
            try { enhancer.enhance(input, token); fail() } catch (_: java.util.concurrent.CancellationException) { }
        }
    }

    @Test fun decoderCancellationNeverCreatesDestination() = withDirectory { directory ->
        val input = File(directory, "synthetic.m4a"); encodeAac(input, 44_100, 1)
        val output = File(directory, "restored.wav")
        val token = CancellationToken().also { it.cancel() }
        try {
            FileEnhancement.render(input, output, directory, SpectralProcessor { value, _ -> value },
                ModelManifest.MODEL_ID, token, {})
            fail()
        } catch (_: java.util.concurrent.CancellationException) { }
        assertFalse(output.exists()); assertTrue(input.isFile)
    }

    private fun withDirectory(block: (File) -> Unit) {
        val directory = File(context.cacheDir, "reuse-test-${java.util.UUID.randomUUID()}").also { it.mkdir() }
        try { block(directory) } finally { directory.deleteRecursively() }
    }

    private fun floats(file: File): FloatArray {
        val bytes = file.readBytes()
        return FloatArray(bytes.size / 4).also { ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(it) }
    }

    private fun expectedAacSamples(file: File): Long {
        val extractor = MediaExtractor()
        try {
            extractor.setDataSource(file.absolutePath); extractor.selectTrack(0)
            val format = extractor.getTrackFormat(0)
            val delay = if (format.containsKey(MediaFormat.KEY_ENCODER_DELAY)) format.getInteger(MediaFormat.KEY_ENCODER_DELAY) else 0
            val padding = if (format.containsKey(MediaFormat.KEY_ENCODER_PADDING)) format.getInteger(MediaFormat.KEY_ENCODER_PADDING) else 0
            var packets = 0L
            while (extractor.sampleTime >= 0) { packets++; if (!extractor.advance()) break }
            // This test explicitly encodes AAC-LC: every coded access unit has 1024 PCM frames.
            return packets * 1024 - delay - padding
        } finally { extractor.release() }
    }

    private fun encodeAac(file: File, rate: Int, channels: Int) {
        val format = MediaFormat.createAudioFormat("audio/mp4a-latm", rate, channels)
        format.setInteger(MediaFormat.KEY_AAC_PROFILE, MediaCodecInfo.CodecProfileLevel.AACObjectLC)
        format.setInteger(MediaFormat.KEY_BIT_RATE, 128_000)
        val codec = MediaCodec.createEncoderByType("audio/mp4a-latm")
        val muxer = MediaMuxer(file.absolutePath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var started = false
        try {
            codec.configure(format, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE); codec.start()
            var frame = 0; var ended = false; var done = false; var track = -1
            val info = MediaCodec.BufferInfo()
            var iterations = 0
            while (!done) {
                check(iterations++ < 5000) { "Encoder did not complete." }
                if (!ended) {
                    val index = codec.dequeueInputBuffer(10_000)
                    if (index >= 0) {
                        val buffer = codec.getInputBuffer(index)!!.order(ByteOrder.LITTLE_ENDIAN); buffer.clear()
                        val count = min(1024, rate - frame)
                        for (i in 0 until count) repeat(channels) {
                            buffer.putShort((sin(2 * PI * 440 * (frame + i) / rate) * 12_000).toInt().toShort())
                        }
                        codec.queueInputBuffer(index, 0, count * channels * 2, frame * 1_000_000L / rate,
                            if (count == 0) MediaCodec.BUFFER_FLAG_END_OF_STREAM else 0)
                        frame += count; ended = count == 0
                    }
                }
                val index = codec.dequeueOutputBuffer(info, 10_000)
                if (index == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                    track = muxer.addTrack(codec.outputFormat); muxer.start(); started = true
                } else if (index >= 0) {
                    if (info.size > 0 && info.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG == 0) {
                        assertTrue(started)
                        val buffer = codec.getOutputBuffer(index)!!
                        buffer.position(info.offset); buffer.limit(info.offset + info.size)
                        muxer.writeSampleData(track, buffer, info)
                    }
                    done = info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0
                    codec.releaseOutputBuffer(index, false)
                }
            }
        } finally {
            codec.stop(); codec.release()
            if (started) muxer.stop()
            muxer.release()
        }
    }
}
