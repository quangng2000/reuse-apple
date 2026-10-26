package io.riftco.reuseaudio

import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean

/** One job, never reset. Cancellation also terminates an active ONNX Runtime call. */
class CancellationToken {
    private val cancelled = AtomicBoolean(false)
    private val lock = Any()
    private var interrupt: (() -> Unit)? = null
    val isCancelled: Boolean get() = cancelled.get()

    fun cancel() {
        cancelled.set(true)
        synchronized(lock) { interrupt?.invoke() }
    }

    fun throwIfCancelled() {
        if (cancelled.get() || Thread.currentThread().isInterrupted) {
            throw CancellationException("Enhancement cancelled.")
        }
    }

    internal fun bind(interrupt: () -> Unit) = synchronized(lock) {
        check(this.interrupt == null) { "A cancellation token belongs to one active job." }
        this.interrupt = interrupt
        if (cancelled.get()) interrupt()
    }

    internal fun unbind() = synchronized(lock) { interrupt = null }
}
