// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

/** Reflection, framing and lifecycle independent from the network. */
internal class StreamingSession {
    private enum class Status { OPEN, FINISHED, FAILED }
    private var status = Status.OPEN
    private val frames = StreamingFrames()
    private var received = 0L
    private var emitted = 0L
    private var rendered = 0L
    val isFresh: Boolean get() = status == Status.OPEN && received == 0L

    fun push(samples: FloatArray, cancellation: StreamingCancellationToken,
             render: (FloatArray) -> FloatArray): FloatArray {
        requireOpen()
        if (!samples.all(Float::isFinite)) throw StreamingException(StreamingException.Reason.NONFINITE_INPUT)
        cancellation.throwIfCancelled()
        require(samples.size <= Int.MAX_VALUE - 640) { "Push a smaller PCM chunk." }
        require(received <= Long.MAX_VALUE - samples.size - 640) { "Stream length exceeded." }
        val output = FloatArray(samples.size + 640)
        var written = 0
        try {
            var start = 0
            while (start < samples.size) {
                cancellation.throwIfCancelled()
                val end = minOf(start + 320, samples.size)
                received += end - start
                frames.push(samples, start, end) { frame ->
                    val audio = renderOne(frame, render)
                    audio.copyInto(output, written)
                    written += audio.size
                }
                start = end
            }
            cancellation.throwIfCancelled()
            emitted += written
            return output.copyOf(written)
        } catch (error: Throwable) { status = Status.FAILED; throw error }
    }

    fun finish(cancellation: StreamingCancellationToken, render: (FloatArray) -> FloatArray): FloatArray {
        requireOpen()
        if (received < 640) throw StreamingException(StreamingException.Reason.TOO_SHORT)
        cancellation.throwIfCancelled()
        val remaining = (received - emitted).toInt()
        check(remaining in 0..640) { "Invalid stream framing." }
        val output = FloatArray(remaining)
        var written = 0
        try {
            frames.finish { frame ->
                cancellation.throwIfCancelled()
                val audio = renderOne(frame, render)
                val count = minOf(audio.size, remaining - written)
                audio.copyInto(output, written, 0, count)
                written += count
            }
            cancellation.throwIfCancelled()
            if (written != remaining) throw StreamingException(StreamingException.Reason.INVALID_OUTPUT)
            emitted += written
            status = Status.FINISHED
            return output
        } catch (error: Throwable) { status = Status.FAILED; throw error }
    }

    private fun renderOne(frame: FloatArray, render: (FloatArray) -> FloatArray): FloatArray {
        val audio = render(frame)
        if (audio.size != (if (rendered == 0L) 0 else 320) || !audio.all(Float::isFinite))
            throw StreamingException(StreamingException.Reason.INVALID_OUTPUT)
        rendered++
        return audio
    }

    private fun requireOpen() {
        when (status) {
            Status.OPEN -> Unit
            Status.FINISHED -> throw StreamingException(StreamingException.Reason.FINISHED)
            Status.FAILED -> throw StreamingException(StreamingException.Reason.RESET_REQUIRED)
        }
    }
}

/** Bounded frame FIFO: 1,280 PCM slots and 641 recent samples irrespective of recording length. */
internal class StreamingFrames {
    private val buffer = FloatArray(1280)
    private var count = 0
    private val recent = FloatArray(641)
    private var recentCount = 0
    private var initialized = false
    private var total = 0L
    private var emitted = 0L

    fun push(samples: FloatArray, start: Int, end: Int, emit: (FloatArray) -> Unit) {
        val added = end - start
        require(added in 0..320)
        total += added
        val retained = minOf(recentCount, 641 - added)
        recent.copyInto(recent, 0, recentCount - retained, recentCount)
        samples.copyInto(recent, retained, start, end)
        recentCount = retained + added
        samples.copyInto(buffer, count, start, end)
        count += added
        if (!initialized && count >= 640) {
            buffer.copyInto(buffer, 320, 0, count)
            for (i in 0 until 320) buffer[i] = buffer[640 - i]
            count += 320
            initialized = true
        }
        if (initialized) drain(Long.MAX_VALUE, emit)
    }

    fun finish(emit: (FloatArray) -> Unit) {
        check(initialized)
        val target = 1 + (total + 319) / 320
        for (i in 0 until recentCount - 1) buffer[count + i] = recent[recentCount - 2 - i]
        count += recentCount - 1
        drain(target - emitted, emit)
    }

    private fun drain(limit: Long, emit: (FloatArray) -> Unit) {
        var drained = 0L
        while (count >= 640 && drained < limit) {
            val frame = buffer.copyOf(640)
            buffer.copyInto(buffer, 0, 320, count)
            count -= 320
            emitted++
            drained++
            emit(frame)
        }
    }
}
