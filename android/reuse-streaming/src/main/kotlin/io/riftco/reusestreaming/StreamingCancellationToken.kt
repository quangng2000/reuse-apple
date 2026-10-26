// NVIDIA Real-time RE-USE adaptation; NSCLv1. See bundled LICENSE.txt.
package io.riftco.reusestreaming

import java.util.concurrent.CancellationException
import java.util.concurrent.atomic.AtomicBoolean

/** One cancellation token per active operation. Cancelling also interrupts an active ORT run. */
class StreamingCancellationToken {
    private val cancelled = AtomicBoolean(false)
    private val lock = Any()
    private var interrupt: (() -> Unit)? = null
    val isCancelled: Boolean get() = cancelled.get()

    fun cancel() {
        cancelled.set(true)
        synchronized(lock) { interrupt?.invoke() }
    }

    fun throwIfCancelled() {
        if (isCancelled || Thread.currentThread().isInterrupted)
            throw CancellationException("Streaming enhancement cancelled; reset before continuing.")
    }

    internal fun bind(callback: () -> Unit) = synchronized(lock) {
        check(interrupt == null) { "A cancellation token belongs to one active operation." }
        interrupt = callback
        if (isCancelled) callback()
    }

    internal fun unbind() = synchronized(lock) { interrupt = null }
}
