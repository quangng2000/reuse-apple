// Original transport helper; MIT adapter terms in ADAPTER_LICENSE.txt.
package io.riftco.reusestreaming

import java.nio.ByteBuffer

/** Exact bit checks and zeroing of native-order direct FP32 storage; no floating arithmetic. */
internal object NativeStreamingBuffers {
    init { System.loadLibrary("reusestreaming_dsp") }

    @JvmStatic external fun allFinite(buffers: Array<ByteBuffer>, elementCounts: IntArray): Boolean
    @JvmStatic external fun zero(buffers: Array<ByteBuffer>, elementCounts: IntArray)
}
