package io.riftco.reusestreaming

import org.junit.Assert.*
import org.junit.Test
import java.util.concurrent.CancellationException
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.*

class StreamingEnhancerTest {
    private class IdentityProcessor : StreamingSpectralProcessor {
        var calls = 0
        var resets = 0
        var closed = false
        var action: (() -> Unit)? = null
        override fun process(input: StreamingSpectrum, cancellation: StreamingCancellationToken): StreamingSpectrum {
            calls++
            action?.invoke()
            cancellation.throwIfCancelled()
            return input
        }
        override fun reset() { calls = 0; resets++ }
        override fun close() { closed = true }
    }

    private fun audio(count: Int) = FloatArray(count) { i ->
        (0.11 * sin(i * 0.087) + 0.035 * cos(i * 0.0213) + (i % 13 - 6) * 0.0007).toFloat()
    }
    private fun enhance(input: FloatArray, chunks: IntArray): FloatArray {
        StreamingEnhancer(IdentityProcessor(), StreamingEnhancerConfiguration()).use { enhancer ->
            val output = ArrayList<Float>()
            var at = 0
            var c = 0
            while (at < input.size) {
                val next = minOf(input.size, at + chunks[c++ % chunks.size])
                output.addAll(enhancer.push(input.copyOfRange(at, next)).toList())
                at = next
            }
            output.addAll(enhancer.finish().toList())
            return output.toFloatArray()
        }
    }

    @Test fun reflectionFramingMatchesAppleAtStartupAndEof() {
        val source = StreamingFrames()
        val input = FloatArray(641) { it.toFloat() }
        val frames = ArrayList<FloatArray>()
        source.push(input, 0, 320) { frames.add(it) }
        assertTrue(frames.isEmpty())
        source.push(input, 320, 640) { frames.add(it) }
        assertEquals(2, frames.size)
        assertEquals(320f, frames[0][0]); assertEquals(1f, frames[0][319])
        assertEquals(0f, frames[0][320]); assertEquals(319f, frames[0][639])
        assertArrayEquals(input.copyOf(640), frames[1], 0f)
        source.push(input, 640, 641) { frames.add(it) }
        source.finish { frames.add(it) }
        assertEquals(4, frames.size)
        assertEquals(320f, frames[2][0]); assertEquals(640f, frames[2][320])
        assertEquals(639f, frames[2][321]); assertEquals(321f, frames[2][639])
        assertEquals(640f, frames[3][0]); assertEquals(639f, frames[3][1])
    }

    @Test fun arbitraryChunksAreBitwiseIdenticalAndEofPreservesEverySample() {
        for (count in listOf(640, 641, 799, 959, 960, 961, 32_057)) {
            val input = audio(count)
            val whole = enhance(input, intArrayOf(count))
            assertEquals(count, whole.size)
            assertArrayEquals(whole, enhance(input, intArrayOf(1, 17, 319, 641, 23)), 0f)
            assertArrayEquals(whole, enhance(input, intArrayOf(320)), 0f)
            assertTrue(whole.all(Float::isFinite))
            for (i in input.indices) assertEquals("sample $i at length $count", input[i], whole[i], 2e-6f)
        }
    }

    @Test fun startupIsFortyMillisecondsAndFinishIsOneShot() {
        val processor = IdentityProcessor()
        StreamingEnhancer(processor, StreamingEnhancerConfiguration()).use { enhancer ->
            assertEquals(0, enhancer.push(audio(639)).size)
            assertEquals(0, processor.calls)
            assertEquals(320, enhancer.push(floatArrayOf(0.1f)).size)
            assertEquals(2, processor.calls)
            assertEquals(320, enhancer.finish().size)
            assertReason(StreamingException.Reason.FINISHED) { enhancer.push(floatArrayOf(0f)) }
            assertReason(StreamingException.Reason.FINISHED) { enhancer.finish() }
            enhancer.reset()
            assertEquals(320, enhancer.push(audio(640)).size)
        }
    }

    @Test fun invalidInputAndPrematureEofDoNotChangeTheOpenStream() {
        StreamingEnhancer(IdentityProcessor(), StreamingEnhancerConfiguration()).use { enhancer ->
            val input = audio(640)
            assertEquals(0, enhancer.push(input.copyOf(320)).size)
            assertReason(StreamingException.Reason.TOO_SHORT) { enhancer.finish() }
            for (bad in listOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY))
                assertReason(StreamingException.Reason.NONFINITE_INPUT) { enhancer.push(floatArrayOf(0.1f, bad)) }
            val result = enhancer.push(input.copyOfRange(320, 640)) + enhancer.finish()
            assertArrayEquals(enhance(input, intArrayOf(640)), result, 0f)
        }
    }

    @Test fun resetAndWarmupDiscardEveryCacheAndOverlap() {
        val processor = IdentityProcessor()
        StreamingEnhancer(processor, StreamingEnhancerConfiguration()).use { enhancer ->
            enhancer.warmUp()
            assertEquals(0, processor.calls)
            assertEquals(1, processor.resets)
            assertEquals(0, enhancer.lastFrameMilliseconds.size)
            val input = audio(991)
            val first = enhancer.push(input) + enhancer.finish()
            enhancer.reset()
            val second = enhancer.push(input) + enhancer.finish()
            assertArrayEquals(first, second, 0f)
            enhancer.reset(); enhancer.push(floatArrayOf(1f))
            assertReason(StreamingException.Reason.WARMUP_REQUIRES_RESET) { enhancer.warmUp() }
        }
    }

    @Test fun silenceRemainsFiniteWithoutGainOrNormalization() {
        val output = enhance(FloatArray(17_137), intArrayOf(19, 321, 800))
        assertTrue(output.all { it == 0f && it.isFinite() })
    }

    @Test fun partialCancellationRequiresResetAndDoesNotLeakToNextStream() {
        val processor = IdentityProcessor()
        val token = StreamingCancellationToken()
        StreamingEnhancer(processor, StreamingEnhancerConfiguration()).use { enhancer ->
            processor.action = { if (processor.calls == 3) token.cancel() }
            assertThrows(CancellationException::class.java) { enhancer.push(audio(1600), token) }
            assertReason(StreamingException.Reason.RESET_REQUIRED) { enhancer.push(audio(640)) }
            enhancer.reset(); processor.action = null
            assertArrayEquals(enhance(audio(800), intArrayOf(800)), enhancer.push(audio(800)) + enhancer.finish(), 0f)
        }
    }

    @Test fun cancelledBeforeEntryLeavesFreshStreamUsable() {
        val token = StreamingCancellationToken(); token.cancel()
        StreamingEnhancer(IdentityProcessor(), StreamingEnhancerConfiguration()).use { enhancer ->
            assertThrows(CancellationException::class.java) { enhancer.push(audio(640), token) }
            assertEquals(320, enhancer.push(audio(640)).size)
        }
    }

    @Test fun failedFrameRequiresReset() {
        val processor = IdentityProcessor()
        StreamingEnhancer(processor, StreamingEnhancerConfiguration()).use { enhancer ->
            processor.action = { error("Simulated provider failure") }
            assertThrows(IllegalStateException::class.java) { enhancer.push(audio(640)) }
            assertReason(StreamingException.Reason.RESET_REQUIRED) { enhancer.finish() }
            processor.action = null; enhancer.reset()
            assertEquals(640, (enhancer.push(audio(640)) + enhancer.finish()).size)
        }
    }

    @Test fun operationsSerializeOnOneInstanceAndCloseReleasesProcessor() {
        val active = AtomicInteger()
        val peak = AtomicInteger()
        val processor = object : StreamingSpectralProcessor {
            var closed = false
            override fun process(input: StreamingSpectrum, cancellation: StreamingCancellationToken): StreamingSpectrum {
                val now = active.incrementAndGet(); peak.updateAndGet { maxOf(it, now) }
                try { Thread.sleep(3); return input } finally { active.decrementAndGet() }
            }
            override fun reset() = Unit
            override fun close() { closed = true }
        }
        val enhancer = StreamingEnhancer(processor, StreamingEnhancerConfiguration())
        val executor = Executors.newFixedThreadPool(2)
        try {
            val first = executor.submit<FloatArray> { enhancer.push(audio(640)) }
            val second = executor.submit<FloatArray> { enhancer.push(audio(640)) }
            first.get(5, TimeUnit.SECONDS); second.get(5, TimeUnit.SECONDS)
            assertEquals(1, peak.get())
            enhancer.close(); assertTrue(processor.closed)
            assertThrows(IllegalStateException::class.java) { enhancer.reset() }
        } finally { executor.shutdownNow(); enhancer.close() }
    }

    @Test fun cancellationTokenInterruptsBoundInferenceImmediately() {
        val token = StreamingCancellationToken()
        val called = AtomicInteger()
        token.bind { called.incrementAndGet() }
        token.cancel()
        assertEquals(1, called.get())
        token.unbind(); token.cancel()
        assertEquals(1, called.get())
        token.bind { called.incrementAndGet() }
        assertEquals(2, called.get())
        token.unbind()
    }

    private fun assertReason(reason: StreamingException.Reason, block: () -> Unit) {
        val error = assertThrows(StreamingException::class.java, block)
        assertEquals(reason, error.reason)
    }
}
