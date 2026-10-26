package io.riftco.reusestreaming

/** Exact pinned FP32 PocketFFT backend, separate from the commercial SDK library. */
internal object StreamingFourier640 {
    init { System.loadLibrary("reusestreaming_dsp") }
    @JvmStatic external fun forward(real: FloatArray, imaginary: FloatArray)
    @JvmStatic external fun inverse(real: FloatArray, imaginary: FloatArray)
}
