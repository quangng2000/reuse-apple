package io.riftco.reusestreaming

import org.junit.Assert.*
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

class NativeStreamingBuffersTest {
    private fun direct(vararg bits: Int): ByteBuffer = ByteBuffer.allocateDirect(bits.size * 4)
        .order(ByteOrder.nativeOrder()).also { bytes -> bits.forEachIndexed { i, value -> bytes.putInt(i * 4, value) } }

    @Test fun finiteCheckPreservesEveryBitIncludingSignedZeroAndSubnormals() {
        val values = intArrayOf(0, Int.MIN_VALUE, 1, 0x007fffff, 0x00800000, 0x3f800000, 0x7f7fffff,
            0x80000001.toInt(), 0xff7fffff.toInt())
        val bytes = direct(*values)
        assertTrue(NativeStreamingBuffers.allFinite(arrayOf(bytes), intArrayOf(values.size)))
        for (i in values.indices) assertEquals(values[i], bytes.getInt(i * 4))
    }

    @Test fun everyNanAndInfinityClassIsRejectedWithoutModifyingStorage() {
        for (bad in intArrayOf(0x7f800000, 0xff800000.toInt(), 0x7f800001, 0x7fc00000, 0xffc01234.toInt())) {
            for (index in 0..2) {
                val values = intArrayOf(0x3f800000, 0x40000000, 0x40400000).also { it[index] = bad }
                val bytes = direct(*values)
                assertFalse(NativeStreamingBuffers.allFinite(arrayOf(bytes), intArrayOf(values.size)))
                for (i in values.indices) assertEquals(values[i], bytes.getInt(i * 4))
            }
        }
    }

    @Test fun allBuffersAreCheckedAndResetWritesPositiveZeroToEveryElement() {
        val first = direct(0x3f800000, Int.MIN_VALUE)
        val second = direct(0x3f800000, 0x7f800000, 1)
        val buffers = arrayOf(first, second)
        val counts = intArrayOf(2, 3)
        assertFalse(NativeStreamingBuffers.allFinite(buffers, counts))
        NativeStreamingBuffers.zero(buffers, counts)
        assertTrue(NativeStreamingBuffers.allFinite(buffers, counts))
        for ((index, buffer) in buffers.withIndex())
            for (i in 0 until counts[index]) assertEquals(0, buffer.getInt(i * 4))
    }

    @Test fun malformedBufferListsFailBeforeAccessOrPartialReset() {
        val first = direct(0x3f800000)
        for ((buffers, counts) in listOf(
            arrayOf(first) to intArrayOf(),
            arrayOf(first) to intArrayOf(-1),
            arrayOf(first) to intArrayOf(2),
            arrayOf(first, ByteBuffer.allocate(4)) to intArrayOf(1, 1),
            emptyArray<ByteBuffer>() to intArrayOf(),
        )) {
            assertThrows(IllegalArgumentException::class.java) { NativeStreamingBuffers.allFinite(buffers, counts) }
            assertThrows(IllegalArgumentException::class.java) { NativeStreamingBuffers.zero(buffers, counts) }
            assertEquals(0x3f800000, first.getInt(0))
        }
    }

    @Test fun resetRejectsReadOnlyDirectStorageWithoutWritingThroughTheView() {
        val original = direct(0x3f800000)
        val view = original.asReadOnlyBuffer()
        assertTrue(NativeStreamingBuffers.allFinite(arrayOf(view), intArrayOf(1)))
        assertThrows(IllegalArgumentException::class.java) { NativeStreamingBuffers.zero(arrayOf(view), intArrayOf(1)) }
        assertEquals(0x3f800000, original.getInt(0))
    }
}
